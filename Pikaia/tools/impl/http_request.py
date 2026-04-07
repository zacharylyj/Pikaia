"""
http_request
------------
Generic REST calls.

params:
    method  : str              - HTTP method (GET, POST, PUT, PATCH, DELETE)
    url     : str              - full URL
    headers : dict | None      - request headers
    body    : Any | None       - request body (dict → JSON-encoded, str → raw)
    timeout : int | None       - seconds (default: 30)

returns:
    status_code : int
    headers     : dict
    body        : str | dict   - dict if response Content-Type is JSON
    ok          : bool         - True if 200-299
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


def run(params: dict, context: dict) -> dict[str, Any]:
    method  = params.get("method", "GET").upper()
    url     = params["url"]
    headers = dict(params.get("headers") or {})
    body    = params.get("body")
    timeout = params.get("timeout", 30)

    # Encode body
    encoded_body: bytes | None = None
    if body is not None:
        if isinstance(body, (dict, list)):
            encoded_body = json.dumps(body).encode()
            headers.setdefault("Content-Type", "application/json")
        elif isinstance(body, str):
            encoded_body = body.encode()
        elif isinstance(body, bytes):
            encoded_body = body

    req = urllib.request.Request(
        url,
        data=encoded_body,
        headers=headers,
        method=method,
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw_body = resp.read().decode("utf-8", errors="replace")
            resp_headers = dict(resp.headers)
            status_code  = resp.status

    except urllib.error.HTTPError as exc:
        raw_body     = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        resp_headers = dict(exc.headers) if exc.headers else {}
        status_code  = exc.code

    except urllib.error.URLError as exc:
        return {
            "status_code": 0,
            "headers":     {},
            "body":        str(exc.reason),
            "ok":          False,
        }

    # Try to parse JSON response
    parsed_body: Any = raw_body
    content_type = resp_headers.get("Content-Type", "") or resp_headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            parsed_body = json.loads(raw_body)
        except json.JSONDecodeError:
            pass

    return {
        "status_code": status_code,
        "headers":     resp_headers,
        "body":        parsed_body,
        "ok":          200 <= status_code < 300,
    }
