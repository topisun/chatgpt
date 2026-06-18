"""
Tests for bot/openai_utils.py — focused on the request payload, since that's
where model differences bite (e.g. Responses API uses `max_output_tokens`, and
reasoning GPT-5.x models reject custom sampling params). No real network calls:
the AsyncOpenAI client is mocked.

Run:  python3 -m unittest discover -s tests
"""
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
BOT_DIR = REPO_ROOT / "bot"
CONFIG_DIR = REPO_ROOT / "config"

# Make `import openai_utils` / `import config` resolve to the bot package.
sys.path.insert(0, str(BOT_DIR))

import yaml  # noqa: E402

# Load the REAL models.yml so the regression tests catch config/code drift.
with open(CONFIG_DIR / "models.yml") as f:
    REAL_MODELS = yaml.safe_load(f)

# Stub out `config` so importing openai_utils doesn't need real secrets/services.
_fake_config = types.ModuleType("config")
_fake_config.openai_api_key = "test-key"
_fake_config.openai_api_base = None
_fake_config.chat_modes = {"assistant": {"prompt_start": "You are an assistant."}}
_fake_config.models = REAL_MODELS
sys.modules["config"] = _fake_config

import openai  # noqa: E402
import openai_utils  # noqa: E402


def _fake_response(content="Hallo", input_tokens=10, output_tokens=5, sources=None):
    """Mimic the shape openai 1.x Responses API returns from responses.create().

    `sources` is an optional list of (title, url) used to build url_citation
    annotations on the output text content block.
    """
    annotations = []
    for title, url in (sources or []):
        annotations.append(
            types.SimpleNamespace(type="url_citation", title=title, url=url)
        )
    text_block = types.SimpleNamespace(
        type="output_text", text=content, annotations=annotations
    )
    message = types.SimpleNamespace(type="message", content=[text_block])
    usage = types.SimpleNamespace(
        input_tokens=input_tokens, output_tokens=output_tokens
    )
    return types.SimpleNamespace(
        output_text=content, output=[message], usage=usage
    )


# Sampling params that reasoning models (GPT-5.x) reject — they only allow defaults.
SAMPLING_PARAMS = ("temperature", "top_p", "frequency_penalty", "presence_penalty")


def _make_bad_request(message, code=None):
    """A real openai.BadRequestError instance without needing an httpx response.

    `except openai.BadRequestError` matches by type, so __new__ + attribute set
    is enough to exercise the retry/propagation logic.
    """
    e = openai.BadRequestError.__new__(openai.BadRequestError)
    e.code = code
    e.message = message
    return e


class BuildCompletionOptionsTest(unittest.TestCase):
    def test_all_models_use_max_output_tokens(self):
        # Responses API renamed the cap to `max_output_tokens` for every model.
        for model in ["gpt-5.5", "gpt-5.4", "gpt-4o", "gpt-4o-mini", "gpt-4"]:
            opts = openai_utils.build_completion_options(model)
            self.assertIn("max_output_tokens", opts, model)
            self.assertNotIn("max_tokens", opts, model)
            self.assertNotIn("max_completion_tokens", opts, model)

    def test_reasoning_models_send_no_custom_sampling_params(self):
        # Regression for: "'temperature' does not support 0.7 with this model".
        for model in ["gpt-5.5", "gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano"]:
            opts = openai_utils.build_completion_options(model)
            for param in SAMPLING_PARAMS:
                self.assertNotIn(param, opts, f"{model} must not send '{param}'")

    def test_legacy_models_keep_temperature(self):
        opts = openai_utils.build_completion_options("gpt-4o")
        self.assertEqual(opts["temperature"], 0.7)


class ModelConfigRegressionTest(unittest.TestCase):
    """Guards against adding a model to models.yml that the code can't run."""

    def test_available_models_are_supported(self):
        for model in REAL_MODELS["available_text_models"]:
            self.assertIn(
                model,
                openai_utils.SUPPORTED_MODELS,
                f"{model} is in available_text_models but not SUPPORTED_MODELS",
            )

    def test_available_models_have_required_info(self):
        for model in REAL_MODELS["available_text_models"]:
            info = REAL_MODELS["info"][model]
            for field in ("name", "description", "price_per_1000_input_tokens",
                          "price_per_1000_output_tokens", "scores"):
                self.assertIn(field, info, f"{model} missing '{field}'")

    def test_default_model_is_a_new_one(self):
        # The product decision: default must be the newest flagship.
        self.assertEqual(REAL_MODELS["available_text_models"][0], "gpt-5.5")


class VisionModelTest(unittest.TestCase):
    def test_gpt5_and_4o_support_vision(self):
        for model in ["gpt-5.5", "gpt-5.4", "gpt-4o", "gpt-4-vision-preview"]:
            self.assertTrue(openai_utils.is_vision_model(model), model)

    def test_text_only_models_have_no_vision(self):
        for model in ["gpt-3.5-turbo", "gpt-4", "text-davinci-003"]:
            self.assertFalse(openai_utils.is_vision_model(model), model)


class SendMessagePayloadTest(unittest.IsolatedAsyncioTestCase):
    """End-to-end-ish: verify the exact kwargs that reach the Responses API.

    This is the test that would have caught the `max_tokens` bug.
    """

    async def _capture_create_kwargs(self, model, response=None):
        with mock.patch.object(
            openai_utils.client.responses,
            "create",
            new=mock.AsyncMock(return_value=response or _fake_response()),
        ) as create:
            chat = openai_utils.ChatGPT(model=model)
            answer, tokens, _ = await chat.send_message(
                "Привет", dialog_messages=[], chat_mode="assistant"
            )
        self.assertTrue(create.called)
        return create.call_args.kwargs, answer, tokens

    async def test_gpt5_sends_max_output_tokens(self):
        kwargs, _, _ = await self._capture_create_kwargs("gpt-5.5")
        self.assertEqual(kwargs["model"], "gpt-5.5")
        self.assertIn("max_output_tokens", kwargs)
        self.assertNotIn("max_tokens", kwargs)

    async def test_gpt5_sends_no_custom_sampling_params(self):
        kwargs, _, _ = await self._capture_create_kwargs("gpt-5.5")
        for param in SAMPLING_PARAMS:
            self.assertNotIn(param, kwargs, f"gpt-5.5 must not send '{param}'")

    async def test_web_search_tool_is_attached(self):
        kwargs, _, _ = await self._capture_create_kwargs("gpt-5.5")
        self.assertIn({"type": "web_search"}, kwargs["tools"])

    async def test_system_prompt_goes_to_instructions(self):
        kwargs, _, _ = await self._capture_create_kwargs("gpt-5.5")
        self.assertEqual(kwargs["instructions"], "You are an assistant.")
        # No legacy `messages`; user turn lives in `input`.
        self.assertNotIn("messages", kwargs)
        self.assertEqual(kwargs["input"][-1], {"role": "user", "content": "Привет"})

    async def test_usage_uses_responses_field_names(self):
        _, _, (n_in, n_out) = await self._capture_create_kwargs("gpt-5.5")
        self.assertEqual((n_in, n_out), (10, 5))

    async def test_sources_footer_appended(self):
        resp = _fake_response(
            content="В Берлине +15°C.",
            sources=[("Gismeteo", "https://gismeteo.ru/berlin")],
        )
        _, answer, _ = await self._capture_create_kwargs("gpt-5.5", response=resp)
        self.assertIn("📎 Источники:", answer)
        self.assertIn("https://gismeteo.ru/berlin", answer)

    async def test_no_sources_no_footer(self):
        _, answer, _ = await self._capture_create_kwargs("gpt-5.5")
        self.assertNotIn("Источники", answer)


@unittest.skipUnless(
    os.environ.get("RUN_LIVE_OPENAI_TESTS") == "1",
    "live API test — set RUN_LIVE_OPENAI_TESTS=1 (uses real key, costs money)",
)
class LiveOpenAITest(unittest.IsolatedAsyncioTestCase):
    """Real calls to OpenAI. This is the ONLY layer that catches server-side
    validation errors like the temperature/max_tokens ones — mocks can't.

    Imports the real `config` (un-stubs the fake one) so the actual API key
    is used. Sends a tiny prompt to every default model.
    """

    async def asyncSetUp(self):
        del sys.modules["config"]          # drop the fake stub
        sys.modules.pop("openai_utils", None)
        import importlib
        self.ou = importlib.import_module("openai_utils")  # real config + key

    async def test_every_available_model_completes(self):
        for model in REAL_MODELS["available_text_models"]:
            with self.subTest(model=model):
                chat = self.ou.ChatGPT(model=model)
                answer, _, _ = await chat.send_message(
                    "Say 'ok'.", dialog_messages=[], chat_mode="assistant"
                )
                self.assertTrue(answer)

    async def test_holds_a_conversation_across_turns(self):
        """THE core health check: the bot must remember context turn-to-turn.

        Sends history in the exact shape bot.py persists it. If history is sent
        in a format the API rejects, the retry loop drops it (removed > 0) and
        the model forgets the name — exactly the bug this guards against.
        """
        chat = self.ou.ChatGPT(model="gpt-5.5")
        a1, _, _ = await chat.send_message(
            "Запомни: меня зовут Антон. Ответь одним словом.",
            dialog_messages=[], chat_mode="assistant",
        )
        self.assertTrue(a1)
        history = [{
            "user": [{"type": "text", "text": "Запомни: меня зовут Антон. Ответь одним словом."}],
            "bot": a1,
        }]
        a2, _, removed = await chat.send_message(
            "Как меня зовут? Ответь одним словом.",
            dialog_messages=history, chat_mode="assistant",
        )
        self.assertEqual(removed, 0, "history was dropped — the conversation regression is back")
        self.assertIn("Антон", a2)

    async def test_web_search_returns_live_data(self):
        # Asks something only answerable with a live search.
        chat = self.ou.ChatGPT(model="gpt-5.5")
        answer, _, _ = await chat.send_message(
            "What is today's weather in Berlin? Search the web.",
            dialog_messages=[], chat_mode="assistant",
        )
        self.assertTrue(answer, "web search produced no answer at all")
        # Whether the model attaches url_citation annotations is up to the model;
        # only assert the footer plumbing when it actually returned citations.
        if "Источники" not in answer:
            self.skipTest("model answered without url_citation annotations this run")

    async def test_image_generation_returns_png_bytes(self):
        """Health check for image generation: gpt-image-1 must return real PNG bytes."""
        images = await self.ou.generate_images(
            "a small red circle on a white background",
            n_images=1, size="1024x1024", quality="low",
        )
        self.assertEqual(len(images), 1)
        self.assertIsInstance(images[0], (bytes, bytearray))
        self.assertTrue(bytes(images[0]).startswith(b"\x89PNG"), "not a PNG image")


class DialogHistoryInputTest(unittest.IsolatedAsyncioTestCase):
    """Regression for the "doesn't hold a conversation" bug.

    Persisted history is stored as Chat-Completions blocks
    ({"type": "text", ...} / {"type": "image", ...}). The Responses API rejects
    those (400 invalid_value), and the old code misread that 400 as "too many
    tokens" and silently wiped the dialog. History must reach the API as
    input_text / input_image instead.
    """

    async def _capture_input(self, dialog_messages):
        with mock.patch.object(
            openai_utils.client.responses,
            "create",
            new=mock.AsyncMock(return_value=_fake_response()),
        ) as create:
            chat = openai_utils.ChatGPT(model="gpt-5.5")
            await chat.send_message(
                "Как меня зовут?", dialog_messages=dialog_messages, chat_mode="assistant"
            )
        return create.call_args.kwargs["input"]

    async def test_text_history_becomes_input_text(self):
        history = [{"user": [{"type": "text", "text": "Меня зовут Антон"}], "bot": "Привет!"}]
        input_items = await self._capture_input(history)
        user_turn = input_items[0]
        self.assertEqual(user_turn["role"], "user")
        self.assertEqual(user_turn["content"][0]["type"], "input_text")
        self.assertEqual(user_turn["content"][0]["text"], "Меня зовут Антон")

    async def test_no_chat_completions_block_types_leak(self):
        history = [{"user": [{"type": "text", "text": "hi"}], "bot": "hello"}]
        for item in await self._capture_input(history):
            content = item["content"]
            if isinstance(content, list):
                for block in content:
                    self.assertNotIn(block.get("type"), ("text", "image"),
                                     "leaked a Chat-Completions block type into Responses input")

    async def test_image_history_becomes_input_image(self):
        history = [{"user": [
            {"type": "text", "text": "что на фото?"},
            {"type": "image", "image": "QUJDREVG"},
        ], "bot": "кошка"}]
        blocks = (await self._capture_input(history))[0]["content"]
        types = [b["type"] for b in blocks]
        self.assertIn("input_text", types)
        self.assertIn("input_image", types)
        img = next(b for b in blocks if b["type"] == "input_image")
        self.assertTrue(img["image_url"].startswith("data:image/jpeg;base64,"))

    async def test_plain_string_history_passes_through(self):
        history = [{"user": "просто строка", "bot": "ок"}]
        self.assertEqual((await self._capture_input(history))[0]["content"], "просто строка")


class BadRequestHandlingTest(unittest.IsolatedAsyncioTestCase):
    """The dialog-shortening retry must fire ONLY on real context overflow."""

    def test_is_context_length_error_classifier(self):
        self.assertTrue(openai_utils.is_context_length_error(
            _make_bad_request("...", code="context_length_exceeded")))
        self.assertTrue(openai_utils.is_context_length_error(
            _make_bad_request("This model's maximum context length is 8192 tokens")))
        self.assertFalse(openai_utils.is_context_length_error(
            _make_bad_request("Invalid value: 'text'", code="invalid_value")))

    async def test_non_context_error_propagates_and_keeps_dialog(self):
        create = mock.AsyncMock(
            side_effect=_make_bad_request("Invalid value: 'text'", code="invalid_value"))
        with mock.patch.object(openai_utils.client.responses, "create", new=create):
            chat = openai_utils.ChatGPT(model="gpt-5.5")
            with self.assertRaises(openai.BadRequestError):
                await chat.send_message(
                    "hi", dialog_messages=[{"user": "a", "bot": "b"}], chat_mode="assistant")
        self.assertEqual(create.call_count, 1, "must NOT retry/strip on a non-context error")

    async def test_context_error_shortens_then_succeeds(self):
        create = mock.AsyncMock(side_effect=[
            _make_bad_request("maximum context length exceeded", code="context_length_exceeded"),
            _fake_response(content="ok"),
        ])
        with mock.patch.object(openai_utils.client.responses, "create", new=create):
            chat = openai_utils.ChatGPT(model="gpt-5.5")
            answer, _, removed = await chat.send_message(
                "hi", dialog_messages=[{"user": "a", "bot": "b"}], chat_mode="assistant")
        self.assertEqual(answer, "ok")
        self.assertEqual(removed, 1)
        self.assertEqual(create.call_count, 2)


class GenerateImagesTest(unittest.IsolatedAsyncioTestCase):
    """Image generation uses gpt-image-1 and returns decoded PNG bytes (no URLs)."""

    def _fake_images_response(self, b64_list):
        data = [types.SimpleNamespace(b64_json=b) for b in b64_list]
        return types.SimpleNamespace(data=data)

    async def _call(self, **kwargs):
        import base64
        png = base64.b64encode(b"\x89PNG-fake").decode()
        create = mock.AsyncMock(return_value=self._fake_images_response([png]))
        with mock.patch.object(openai_utils.client.images, "generate", new=create):
            result = await openai_utils.generate_images("a cat", **kwargs)
        return create.call_args.kwargs, result

    async def test_uses_gpt_image_1_and_decodes_bytes(self):
        kwargs, result = await self._call()
        self.assertEqual(kwargs["model"], "gpt-image-1")
        self.assertEqual(result, [b"\x89PNG-fake"])

    async def test_invalid_size_and_quality_are_coerced(self):
        # gpt-image-1 would 400 on dall-e sizes; we must never forward them.
        kwargs, _ = await self._call(size="512x512", quality="ultra")
        self.assertIn(kwargs["size"], openai_utils.IMAGE_SIZES)
        self.assertIn(kwargs["quality"], openai_utils.IMAGE_QUALITIES)

    async def test_valid_size_and_quality_pass_through(self):
        kwargs, _ = await self._call(size="1536x1024", quality="high")
        self.assertEqual(kwargs["size"], "1536x1024")
        self.assertEqual(kwargs["quality"], "high")


class UnknownModelTest(unittest.TestCase):
    def test_rejects_unknown_model(self):
        with self.assertRaises(AssertionError):
            openai_utils.ChatGPT(model="gpt-does-not-exist")


if __name__ == "__main__":
    unittest.main()
