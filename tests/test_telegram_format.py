"""
Tests for bot/telegram_format.py — Markdown -> Telegram MarkdownV2 rendering.

Skipped automatically where ``telegramify-markdown`` isn't installed (e.g. a bare
local checkout); runs inside the Docker image / CI where requirements are present.

Run:  python3 -m unittest discover -s tests
"""
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BOT_DIR = REPO_ROOT / "bot"
sys.path.insert(0, str(BOT_DIR))

try:
    import telegramify_markdown  # noqa: F401
    HAS_LIB = True
except ImportError:
    HAS_LIB = False

import telegram_format


@unittest.skipUnless(HAS_LIB, "telegramify-markdown not installed")
class RenderTests(unittest.TestCase):
    def test_bold_heading_not_left_raw(self):
        out = telegram_format.render("# Title\n\n**bold** text")
        # the literal markdown markers must not survive verbatim
        self.assertNotIn("**bold**", out)
        self.assertNotIn("# Title", out)

    def test_special_chars_escaped(self):
        # MarkdownV2 reserved chars must be escaped, not left bare
        out = telegram_format.render("cost is 5.0 (USD)!")
        self.assertIn("\\.", out)

    def test_code_block_preserved(self):
        out = telegram_format.render("```python\nprint('hi')\n```")
        self.assertIn("print", out)


class FallbackTests(unittest.TestCase):
    def test_empty_passthrough(self):
        self.assertEqual(telegram_format.render(""), "")
        self.assertEqual(telegram_format.render(None), None)


if __name__ == "__main__":
    unittest.main()
