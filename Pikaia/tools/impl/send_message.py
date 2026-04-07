"""
send_message
------------
Send a message via Telegram, Discord, or Slack.

Channel credentials are read from config["messaging"] which looks like:
{
  "telegram": { "bot_token": "...", "chat_id": "..." },
  "discord":  { "webhook_url": "..." },
  "slack":    { "webhook_url": "..." }
}

Active interfaces come from config["interfaces"] list.

params:
    channel    : str         - "telegram" | "discord" | "slack"
    message    : str         - text to send
    parse_mode : str | None  - Telegram parse mode ("Markdown", "HTML") — optional

returns:
    sent    : bool
    channel : str
    detail  : str   - status detail or error message
"""

from __future__ import annotations

import json
import urllib.request
import urllib.error
from typing import Any


def run(params: dict, context: dict) -> dict[str, Any]:
    channel    = params["channel"].lower()
    message    = params["message"]
    parse_mode = params.get("parse_mode")
    config     = context.get("config", {})

    interfaces = config.get("interfaces", [])
    if channel not in interfaces:
        return {
            "sent":    False,
            "channel": channel,
            "detail":  f"Channel '{channel}' is not in config.interfaces: {interfaces}",
        }

    messaging = config.get("messaging", {})
    creds     = messaging.get(channel)
    if not creds:
        return {
            "sent":    False,
            "channel": channel,
            "detail":  f"No credentials found for channel '{channel}' in config.messaging",
        }

    try:
        if channel == "telegram":
            return _send_telegram(message, creds, parse_mode)
        if channel == "discord":
            return _send_discord(message, creds)
        if channel == "slack":
            return _send_slack(message, creds)
        return {"sent": False, "channel": channel, "detail": f"Unknown channel: {channel}"}
    except Exception as exc:
        return {"sent": False, "channel": channel, "detail": str(exc)}


# ------------------------------------------------------------------
# Channel implementations
# ------------------------------------------------------------------

def _send_telegram(message: str, creds: dict, parse_mode: str | None) -> dict:
    token   = creds["bot_token"]
    chat_id = creds["chat_id"]
    url     = f"https://api.telegram.org/bot{token}/sendMessage"
    payload: dict = {"chat_id": chat_id, "text": message}
    if parse_mode:
        payload["parse_mode"] = parse_mode
    resp = _post_json(url, payload)
    ok   = resp.get("ok", False)
    return {
        "sent":    ok,
        "channel": "telegram",
        "detail":  resp.get("description", "ok") if not ok else "ok",
    }


def _send_discord(message: str, creds: dict) -> dict:
    url     = creds["webhook_url"]
    payload = {"content": message}
    _post_json(url, payload, expect_json=False)
    return {"sent": True, "channel": "discord", "detail": "ok"}


def _send_slack(message: str, creds: dict) -> dict:
    url     = creds["webhook_url"]
    payload = {"text": message}
    _post_json(url, payload, expect_json=False)
    return {"sent": True, "channel": "slack", "detail": "ok"}


def _post_json(url: str, payload: dict, expect_json: bool = True) -> dict:
    body    = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    req     = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=15) as resp:
        raw = resp.read().decode()
    if expect_json:
        return json.loads(raw)
    return {}
