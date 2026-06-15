"""
Render model output (standard Markdown) into Telegram MarkdownV2.

Models reply in GitHub-flavored Markdown (``**bold**``, ``# headings``,
``- bullets``, fenced code blocks). Telegram does not understand that syntax, so
without conversion users see raw ``**...**`` and ``#`` in their chats.

We delegate the actual conversion to ``telegramify-markdown``, a maintained
library that turns Markdown into properly escaped Telegram MarkdownV2 (handling
all the escaping edge cases so we don't hand-roll a fragile replacement map).
"""

import logging

try:
    import telegramify_markdown
except ImportError:  # pragma: no cover - dependency missing only in bare checkouts
    telegramify_markdown = None

logger = logging.getLogger(__name__)

# Telegram message hard limit
MAX_MESSAGE_LENGTH = 4096


def render(text: str) -> str:
    """Convert Markdown ``text`` to Telegram MarkdownV2.

    Falls back to the original text if conversion fails (e.g. on partial Markdown
    produced mid-stream); the caller already retries without ``parse_mode`` on a
    Telegram BadRequest, so a rare bad frame degrades gracefully.
    """
    if not text or telegramify_markdown is None:
        return text
    try:
        return telegramify_markdown.markdownify(text)
    except Exception as e:
        logger.warning("telegramify_markdown failed, sending raw text: %s", e)
        return text
