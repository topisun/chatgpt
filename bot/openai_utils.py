import base64
from io import BytesIO
import config
import logging

import tiktoken
import openai


# setup openai client (new 1.x SDK)
client = openai.AsyncOpenAI(
    api_key=config.openai_api_key,
    base_url=config.openai_api_base or None,
)
logger = logging.getLogger(__name__)


OPENAI_COMPLETION_OPTIONS = {
    "temperature": 0.7,
    "top_p": 1,
    "timeout": 60.0,
}

MAX_OUTPUT_TOKENS = 1000

# Web search tool. The model decides on its own when to actually search, so we
# attach it to every request (all chat modes) — mirrors the ChatGPT app.
WEB_SEARCH_TOOL = {"type": "web_search"}

# Newer "reasoning" models (GPT-5.x) differ from classic chat models:
#   - only accept DEFAULT sampling params: no custom temperature / top_p /
#     frequency_penalty / presence_penalty (they error out otherwise)
# Keep the list explicit so it's obvious what differs.
REASONING_MODELS = {
    "gpt-5.5", "gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano",
}


def build_completion_options(model):
    """Per-model request options for the Responses API.

    Responses API uses `max_output_tokens` (not `max_tokens`). Reasoning models
    additionally reject custom sampling params, so they get a minimal set.
    """
    if model in REASONING_MODELS:
        return {
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "timeout": OPENAI_COMPLETION_OPTIONS["timeout"],
        }

    options = dict(OPENAI_COMPLETION_OPTIONS)
    options["max_output_tokens"] = MAX_OUTPUT_TOKENS
    return options


# Models that use the Responses (chat) endpoint
CHAT_COMPLETION_MODELS = {
    "gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4", "gpt-4o", "gpt-4o-mini",
    "gpt-4-1106-preview", "gpt-4-vision-preview",
    "gpt-5.5", "gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano",
}

# Models that can understand images
VISION_MODELS = {
    "gpt-4-vision-preview", "gpt-4o",
    "gpt-5.5", "gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano",
}

# All supported models (chat + legacy completion)
SUPPORTED_MODELS = CHAT_COMPLETION_MODELS | {"text-davinci-003"}


def is_vision_model(model):
    return model in VISION_MODELS


def is_context_length_error(e):
    """True only for "prompt too long" errors.

    The dialog-shortening retry loop must trigger ONLY on real context-overflow
    errors. Every other BadRequest (malformed payload, bad param, etc.) must
    propagate so it's visible instead of silently eating the whole conversation.
    """
    code = getattr(e, "code", None)
    if code == "context_length_exceeded":
        return True
    msg = str(getattr(e, "message", "") or e).lower()
    return "context length" in msg or "maximum context" in msg or "too many tokens" in msg


class ChatGPT:
    def __init__(self, model="gpt-5.5"):
        assert model in SUPPORTED_MODELS, f"Unknown model: {model}"
        self.model = model

    async def send_message(self, message, dialog_messages=[], chat_mode="assistant"):
        if chat_mode not in config.chat_modes.keys():
            raise ValueError(f"Chat mode {chat_mode} is not supported")

        n_dialog_messages_before = len(dialog_messages)
        answer = None
        while answer is None:
            try:
                instructions, input_items = self._generate_input(message, dialog_messages, chat_mode)

                r = await client.responses.create(
                    model=self.model,
                    instructions=instructions,
                    input=input_items,
                    tools=[WEB_SEARCH_TOOL],
                    **build_completion_options(self.model)
                )
                answer = r.output_text
                answer = self._postprocess_answer(answer)
                answer += self._format_sources(r)
                n_input_tokens, n_output_tokens = r.usage.input_tokens, r.usage.output_tokens
            except openai.BadRequestError as e:  # too many tokens
                if not is_context_length_error(e):
                    raise  # not a context-overflow error — surface it, don't eat the dialog
                if len(dialog_messages) == 0:
                    raise ValueError("Dialog messages is reduced to zero, but still has too many tokens to make completion") from e

                # forget first message in dialog_messages
                dialog_messages = dialog_messages[1:]

        n_first_dialog_messages_removed = n_dialog_messages_before - len(dialog_messages)

        return answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed

    async def send_message_stream(self, message, dialog_messages=[], chat_mode="assistant"):
        if chat_mode not in config.chat_modes.keys():
            raise ValueError(f"Chat mode {chat_mode} is not supported")

        n_dialog_messages_before = len(dialog_messages)
        answer = None
        while answer is None:
            try:
                instructions, input_items = self._generate_input(message, dialog_messages, chat_mode)

                stream = await client.responses.create(
                    model=self.model,
                    instructions=instructions,
                    input=input_items,
                    tools=[WEB_SEARCH_TOOL],
                    stream=True,
                    **build_completion_options(self.model)
                )

                answer = ""
                n_input_tokens, n_output_tokens = 0, 0
                final_response = None
                async for event in stream:
                    if event.type == "response.output_text.delta":
                        answer += event.delta
                        n_input_tokens, n_output_tokens = self._count_tokens_from_input(
                            instructions, input_items, answer, model=self.model
                        )
                        n_first_dialog_messages_removed = n_dialog_messages_before - len(dialog_messages)
                        yield "not_finished", answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed
                    elif event.type == "response.completed":
                        final_response = event.response

                answer = self._postprocess_answer(answer)
                if final_response is not None:
                    answer += self._format_sources(final_response)
                    if final_response.usage is not None:
                        n_input_tokens = final_response.usage.input_tokens
                        n_output_tokens = final_response.usage.output_tokens

            except openai.BadRequestError as e:  # too many tokens
                if not is_context_length_error(e):
                    raise  # not a context-overflow error — surface it, don't eat the dialog
                if len(dialog_messages) == 0:
                    raise e

                # forget first message in dialog_messages
                dialog_messages = dialog_messages[1:]

        n_first_dialog_messages_removed = n_dialog_messages_before - len(dialog_messages)
        yield "finished", answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed  # sending final answer

    async def send_vision_message(
        self,
        message,
        dialog_messages=[],
        chat_mode="assistant",
        image_buffer: BytesIO = None,
    ):
        n_dialog_messages_before = len(dialog_messages)
        answer = None
        while answer is None:
            try:
                if self.model in VISION_MODELS:
                    instructions, input_items = self._generate_input(
                        message, dialog_messages, chat_mode, image_buffer
                    )
                    r = await client.responses.create(
                        model=self.model,
                        instructions=instructions,
                        input=input_items,
                        tools=[WEB_SEARCH_TOOL],
                        **build_completion_options(self.model)
                    )
                    answer = r.output_text
                else:
                    raise ValueError(f"Unsupported model: {self.model}")

                answer = self._postprocess_answer(answer)
                answer += self._format_sources(r)
                n_input_tokens, n_output_tokens = (
                    r.usage.input_tokens,
                    r.usage.output_tokens,
                )
            except openai.BadRequestError as e:  # too many tokens
                if not is_context_length_error(e):
                    raise  # not a context-overflow error — surface it, don't eat the dialog
                if len(dialog_messages) == 0:
                    raise ValueError(
                        "Dialog messages is reduced to zero, but still has too many tokens to make completion"
                    ) from e

                # forget first message in dialog_messages
                dialog_messages = dialog_messages[1:]

        n_first_dialog_messages_removed = n_dialog_messages_before - len(
            dialog_messages
        )

        return (
            answer,
            (n_input_tokens, n_output_tokens),
            n_first_dialog_messages_removed,
        )

    async def send_vision_message_stream(
        self,
        message,
        dialog_messages=[],
        chat_mode="assistant",
        image_buffer: BytesIO = None,
    ):
        n_dialog_messages_before = len(dialog_messages)
        answer = None
        while answer is None:
            try:
                if self.model in VISION_MODELS:
                    instructions, input_items = self._generate_input(
                        message, dialog_messages, chat_mode, image_buffer
                    )

                    stream = await client.responses.create(
                        model=self.model,
                        instructions=instructions,
                        input=input_items,
                        tools=[WEB_SEARCH_TOOL],
                        stream=True,
                        **build_completion_options(self.model),
                    )

                    answer = ""
                    n_input_tokens, n_output_tokens = 0, 0
                    final_response = None
                    async for event in stream:
                        if event.type == "response.output_text.delta":
                            answer += event.delta
                            (
                                n_input_tokens,
                                n_output_tokens,
                            ) = self._count_tokens_from_input(
                                instructions, input_items, answer, model=self.model
                            )
                            n_first_dialog_messages_removed = (
                                n_dialog_messages_before - len(dialog_messages)
                            )
                            yield "not_finished", answer, (
                                n_input_tokens,
                                n_output_tokens,
                            ), n_first_dialog_messages_removed
                        elif event.type == "response.completed":
                            final_response = event.response

                answer = self._postprocess_answer(answer)
                if final_response is not None:
                    answer += self._format_sources(final_response)
                    if final_response.usage is not None:
                        n_input_tokens = final_response.usage.input_tokens
                        n_output_tokens = final_response.usage.output_tokens

            except openai.BadRequestError as e:  # too many tokens
                if not is_context_length_error(e):
                    raise  # not a context-overflow error — surface it, don't eat the dialog
                if len(dialog_messages) == 0:
                    raise e
                # forget first message in dialog_messages
                dialog_messages = dialog_messages[1:]

        n_first_dialog_messages_removed = n_dialog_messages_before - len(dialog_messages)
        yield "finished", answer, (
            n_input_tokens,
            n_output_tokens,
        ), n_first_dialog_messages_removed

    def _generate_prompt(self, message, dialog_messages, chat_mode):
        prompt = config.chat_modes[chat_mode]["prompt_start"]
        prompt += "\n\n"

        # add chat context
        if len(dialog_messages) > 0:
            prompt += "Chat:\n"
            for dialog_message in dialog_messages:
                prompt += f"User: {dialog_message['user']}\n"
                prompt += f"Assistant: {dialog_message['bot']}\n"

        # current message
        prompt += f"User: {message}\n"
        prompt += "Assistant: "

        return prompt

    def _encode_image(self, image_buffer: BytesIO) -> bytes:
        return base64.b64encode(image_buffer.read()).decode("utf-8")

    def _normalize_user_content(self, content):
        """Convert persisted dialog history into Responses API input content.

        History is stored (bot.py) as Chat-Completions-style blocks:
        ``{"type": "text", "text": ...}`` / ``{"type": "image", "image": <base64>}``.
        The Responses API rejects those — user content must use ``input_text`` /
        ``input_image`` blocks (a plain string is also fine and passes through).
        Without this conversion every turn with history 400s, which the retry
        loop misreads as "too long" and drops the whole conversation.
        """
        if isinstance(content, str):
            return content

        normalized = []
        for block in content:
            block_type = block.get("type")
            if block_type in ("text", "input_text"):
                normalized.append({"type": "input_text", "text": block.get("text", "")})
            elif block_type in ("image", "input_image"):
                image = block.get("image") or block.get("image_url")
                if isinstance(image, str) and not image.startswith("data:"):
                    image = f"data:image/jpeg;base64,{image}"
                normalized.append({"type": "input_image", "image_url": image, "detail": "high"})
        return normalized

    def _generate_input(self, message, dialog_messages, chat_mode, image_buffer: BytesIO = None):
        """Build (instructions, input) for the Responses API.

        The system prompt goes into `instructions`; the dialog history and the
        current user message become the `input` list. Image content uses the
        Responses `input_text` / `input_image` content blocks.
        """
        instructions = config.chat_modes[chat_mode]["prompt_start"]

        input_items = []
        for dialog_message in dialog_messages:
            input_items.append({"role": "user", "content": self._normalize_user_content(dialog_message["user"])})
            input_items.append({"role": "assistant", "content": dialog_message["bot"]})

        if image_buffer is not None:
            input_items.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": message,
                        },
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{self._encode_image(image_buffer)}",
                            "detail": "high",
                        },
                    ],
                }
            )
        else:
            input_items.append({"role": "user", "content": message})

        return instructions, input_items

    def _format_sources(self, response):
        """Build a "Sources" footer from url_citation annotations on the response."""
        sources = []
        seen = set()
        for output in getattr(response, "output", None) or []:
            if getattr(output, "type", None) != "message":
                continue
            for content in getattr(output, "content", None) or []:
                for annotation in getattr(content, "annotations", None) or []:
                    if getattr(annotation, "type", None) != "url_citation":
                        continue
                    url = getattr(annotation, "url", None)
                    if not url or url in seen:
                        continue
                    seen.add(url)
                    title = getattr(annotation, "title", None) or url
                    sources.append((title, url))

        if not sources:
            return ""

        footer = "\n\n📎 Источники:\n"
        footer += "\n".join(f"• {title} — {url}" for title, url in sources)
        return footer

    def _postprocess_answer(self, answer):
        answer = answer.strip()
        return answer

    def _count_tokens_from_input(self, instructions, input_items, answer, model="gpt-3.5-turbo"):
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # tiktoken may not know newer models yet; o200k_base is used by gpt-4o and gpt-5.x
            encoding = tiktoken.get_encoding("o200k_base")

        if model == "gpt-3.5-turbo-16k":
            tokens_per_message = 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
        elif model == "gpt-3.5-turbo":
            tokens_per_message = 4
        elif model == "gpt-4o-mini":
            tokens_per_message = 0.1
        else:
            # gpt-4, gpt-4o, gpt-5.x and other chat models
            tokens_per_message = 3

        # input: instructions (system prompt) + dialog/user items
        n_input_tokens = tokens_per_message + len(encoding.encode(instructions or ""))
        for item in input_items:
            n_input_tokens += tokens_per_message
            content = item["content"]
            if isinstance(content, list):
                for sub_message in content:
                    if sub_message.get("type") == "input_text":
                        n_input_tokens += len(encoding.encode(sub_message["text"]))
                    elif sub_message.get("type") == "input_image":
                        pass
            else:
                n_input_tokens += len(encoding.encode(content))

        n_input_tokens += 2

        # output
        n_output_tokens = 1 + len(encoding.encode(answer))

        return n_input_tokens, n_output_tokens

    def _count_tokens_from_prompt(self, prompt, answer, model="text-davinci-003"):
        encoding = tiktoken.encoding_for_model(model)

        n_input_tokens = len(encoding.encode(prompt)) + 1
        n_output_tokens = len(encoding.encode(answer))

        return n_input_tokens, n_output_tokens


async def transcribe_audio(audio_file) -> str:
    r = await client.audio.transcriptions.create(model="whisper-1", file=audio_file)
    return r.text or ""


# Latest available image model. gpt-image-1 always returns base64 (no URLs) and
# only accepts these sizes (plus "auto"); anything else 400s.
IMAGE_MODEL = "gpt-image-1"
IMAGE_SIZES = {"1024x1024", "1024x1536", "1536x1024", "auto"}
IMAGE_QUALITIES = {"low", "medium", "high", "auto"}


async def generate_images(prompt, n_images=1, size="1024x1024", quality="medium"):
    """Generate images with gpt-image-1 and return them as raw PNG bytes.

    Returns a list of `bytes` (decoded base64) rather than URLs, because
    gpt-image-1 only ever returns b64_json.
    """
    if size not in IMAGE_SIZES:
        size = "1024x1024"
    if quality not in IMAGE_QUALITIES:
        quality = "medium"
    r = await client.images.generate(
        model=IMAGE_MODEL, prompt=prompt, n=n_images, size=size, quality=quality
    )
    return [base64.b64decode(item.b64_json) for item in r.data]


async def is_content_acceptable(prompt):
    r = await client.moderations.create(input=prompt)
    categories = r.results[0].categories
    return not any(categories.model_dump().values())
