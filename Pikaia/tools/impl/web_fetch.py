"""
web_fetch
---------
Fetch a URL and return clean, readable text (HTML stripped).

params:
    url       : str        - URL to fetch
    max_chars : int | None - truncation limit (default: 8000)
    timeout   : int | None - seconds (default: 15)

returns:
    url       : str
    content   : str   - cleaned text
    truncated : bool
"""

from __future__ import annotations

import html
import re
import urllib.request
import urllib.error
from typing import Any


# Tags whose entire content (including children) we discard
_DISCARD_TAGS = re.compile(
    r"<(script|style|noscript|nav|footer|header|aside|form|iframe)[^>]*>.*?</\1>",
    re.IGNORECASE | re.DOTALL,
)
_TAG_RE    = re.compile(r"<[^>]+>")
_WS_RE     = re.compile(r"[ \t]+")
_BLANK_RE  = re.compile(r"\n{3,}")


def _strip_html(raw: str) -> str:
    text = _DISCARD_TAGS.sub(" ", raw)
    text = _TAG_RE.sub(" ", text)
    text = html.unescape(text)
    # Normalise whitespace
    lines = [_WS_RE.sub(" ", line).strip() for line in text.splitlines()]
    lines = [l for l in lines if l]
    text  = "\n".join(lines)
    text  = _BLANK_RE.sub("\n\n", text)
    return text.strip()


def run(params: dict, context: dict) -> dict[str, Any]:
    url       = params["url"]
    max_chars = params.get("max_chars", 8000)
    timeout   = params.get("timeout", 15)

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; AgentBot/1.0; "
            "+https://github.com/anthropics/claude-code)"
        )
    }
    req = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return {"url": url, "content": f"HTTP {exc.code}: {exc.reason}", "truncated": False}
    except urllib.error.URLError as exc:
        return {"url": url, "content": f"URL error: {exc.reason}", "truncated": False}
    except Exception as exc:
        return {"url": url, "content": f"Error: {exc}", "truncated": False}

    content   = _strip_html(raw)
    truncated = False
    if max_chars and len(content) > max_chars:
        content   = content[:max_chars]
        truncated = True

    return {"url": url, "content": content, "truncated": truncated}
