"""
web_search
----------
Search the web using DuckDuckGo Lite (no API key required).
Returns titles, URLs, and snippets for the top results.

Uses DuckDuckGo's HTML endpoint (html.duckduckgo.com) which is stable,
free, and requires no authentication.

params:
    query       : str   - search query
    max_results : int   - max results to return (default: 10)

returns:
    results : list[{title, url, snippet}]
    query   : str
    count   : int

SCHEMA (self-registering)
"""

from __future__ import annotations

import html as html_mod
import re
import urllib.parse
import urllib.request
import urllib.error
from typing import Any

SCHEMA = {
    "name": "web_search",
    "description": (
        "Search the web using DuckDuckGo. "
        "Returns titles, URLs, and snippets for the top results. "
        "No API key required."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum results to return (default: 10)",
            },
        },
        "required": ["query"],
    },
}

_DDG_URL    = "https://html.duckduckgo.com/html/"
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

# Patterns to extract results from DuckDuckGo HTML
_RESULT_BLOCK = re.compile(
    r'<div class="result[^"]*"[^>]*>(.*?)</div>\s*</div>',
    re.DOTALL | re.IGNORECASE,
)
_TITLE_RE    = re.compile(r'<a[^>]+class="result__a"[^>]*>(.*?)</a>', re.DOTALL | re.IGNORECASE)
_URL_RE      = re.compile(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"', re.IGNORECASE)
_SNIPPET_RE  = re.compile(r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>', re.DOTALL | re.IGNORECASE)
_TAG_RE      = re.compile(r"<[^>]+>")


def _clean(text: str) -> str:
    text = _TAG_RE.sub("", text)
    return html_mod.unescape(text).strip()


def _resolve_ddg_url(href: str) -> str:
    """DuckDuckGo result links are wrapped in a redirect — extract the real URL."""
    if href.startswith("//duckduckgo.com/l/"):
        # Parse uddg= param
        parsed = urllib.parse.urlparse("https:" + href)
        qs     = urllib.parse.parse_qs(parsed.query)
        uddg   = qs.get("uddg", [""])[0]
        return urllib.parse.unquote(uddg) if uddg else href
    if href.startswith("/"):
        return "https://duckduckgo.com" + href
    return href


def run(params: dict, context: dict) -> dict[str, Any]:
    query       = params["query"]
    max_results = int(params.get("max_results", 10))

    data = urllib.parse.urlencode({"q": query, "kl": "us-en"}).encode()
    headers = {
        "User-Agent":   _USER_AGENT,
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept":       "text/html",
    }
    req = urllib.request.Request(_DDG_URL, data=data, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return {"results": [], "query": query, "count": 0,
                "error": f"HTTP {exc.code}: {exc.reason}"}
    except Exception as exc:
        return {"results": [], "query": query, "count": 0, "error": str(exc)}

    results: list[dict] = []
    for block in _RESULT_BLOCK.finditer(raw):
        if len(results) >= max_results:
            break
        block_html = block.group(1)

        title_m   = _TITLE_RE.search(block_html)
        url_m     = _URL_RE.search(block_html)
        snippet_m = _SNIPPET_RE.search(block_html)

        if not title_m or not url_m:
            continue

        title   = _clean(title_m.group(1))
        url     = _resolve_ddg_url(url_m.group(1))
        snippet = _clean(snippet_m.group(1)) if snippet_m else ""

        if not title or not url:
            continue

        results.append({"title": title, "url": url, "snippet": snippet})

    return {"results": results, "query": query, "count": len(results)}
