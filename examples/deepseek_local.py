"""
examples/deepseek_local.py
--------------------------
Standalone example: chat with DeepSeek-R1 1.5B locally.
No Pikaia orchestration stack needed — runs the provider directly.

Setup (pick ONE):
  Option A — Ollama  (recommended, ~600 MB RAM, fast startup)
      pip install ollama            # Python SDK (optional, not used here)
      ollama pull deepseek-r1:1.5b  # downloads ~1.1 GB model
      # make sure Ollama is running (background service)

  Option B — HuggingFace transformers  (no Ollama required)
      pip install transformers torch accelerate
      # model downloads automatically on first run (~1.1 GB)

Run:
    cd Pikaia
    python ../examples/deepseek_local.py
    python ../examples/deepseek_local.py --show-thinking
    python ../examples/deepseek_local.py --backend transformers
    python ../examples/deepseek_local.py --prompt "Explain gradient descent"
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

# ---------------------------------------------------------------------------
# Path bootstrap so we can import the provider without installing the package
# ---------------------------------------------------------------------------
_PIKAIA_DIR = Path(__file__).resolve().parent.parent / "Pikaia"
if str(_PIKAIA_DIR) not in sys.path:
    sys.path.insert(0, str(_PIKAIA_DIR))

from tools.providers.deepseek_local import (  # noqa: E402
    Adapter,
    OLLAMA_MODEL,
    HF_MODEL_ID,
    _TransformersBackend,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a helpful, concise assistant. "
    "Think step-by-step when solving problems, but keep final answers brief."
)

_DIVIDER = "-" * 60


def _print_response(resp: dict, show_thinking: bool) -> None:
    """Pretty-print a parsed provider response."""
    thinking = resp.get("thinking", "")
    content  = resp.get("content", "")
    tin      = resp.get("tokens_in", 0)
    tout     = resp.get("tokens_out", 0)

    if show_thinking and thinking:
        print(f"\n{_DIVIDER}")
        print("THINKING (chain-of-thought):")
        print(_DIVIDER)
        print(textwrap.fill(thinking, width=80))

    print(f"\n{_DIVIDER}")
    print("ANSWER:")
    print(_DIVIDER)
    print(content or "(empty response)")
    print(f"\n[tokens: {tin} in / {tout} out]")


def _force_transformers(adapter: Adapter, request: dict) -> dict:
    """Bypass Ollama and call the transformers backend directly."""
    messages   = request.pop("_messages_for_transformers", [])
    max_tokens = request.get("options", {}).get("num_predict", 1024)
    return _TransformersBackend.generate(messages, max_tokens)


# ---------------------------------------------------------------------------
# Single-shot demo
# ---------------------------------------------------------------------------

def run_single(prompt: str, show_thinking: bool, force_transformers: bool) -> None:
    adapter = Adapter(api_key=None, model_id=OLLAMA_MODEL)

    messages = [{"role": "user", "content": prompt}]
    request  = adapter.build_request(
        system     = _SYSTEM_PROMPT,
        messages   = messages,
        max_tokens = 1024,
    )

    print(f"\nModel  : {OLLAMA_MODEL} ({'transformers' if force_transformers else 'Ollama → transformers fallback'})")
    print(f"Prompt : {prompt}")

    if force_transformers:
        raw = _force_transformers(adapter, request)
    else:
        raw = adapter.call(request)

    resp = adapter.parse_response(raw)
    _print_response(resp, show_thinking)


# ---------------------------------------------------------------------------
# Interactive chat loop
# ---------------------------------------------------------------------------

def run_chat(show_thinking: bool, force_transformers: bool) -> None:
    adapter  = Adapter(api_key=None, model_id=OLLAMA_MODEL)
    history: list[dict] = []
    backend  = "transformers" if force_transformers else "Ollama → transformers fallback"

    print(f"\nDeepSeek-R1 1.5B  [{backend}]")
    print("Type your message, or 'exit' / Ctrl-C to quit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "/exit"):
            print("Bye!")
            break

        history.append({"role": "user", "content": user_input})

        request = adapter.build_request(
            system     = _SYSTEM_PROMPT,
            messages   = history,
            max_tokens = 1024,
        )

        print("DeepSeek: ", end="", flush=True)
        try:
            if force_transformers:
                raw = _force_transformers(adapter, request)
            else:
                raw = adapter.call(request)

            resp    = adapter.parse_response(raw)
            content = resp.get("content", "")
            tin     = resp.get("tokens_in", 0)
            tout    = resp.get("tokens_out", 0)

            if show_thinking and resp.get("thinking"):
                print(f"\n[thinking: {len(resp['thinking'])} chars]")

            print(content)
            print(f"  [{tin}→{tout} tokens]\n")

            history.append({"role": "assistant", "content": content})

        except RuntimeError as exc:
            print(f"\nError: {exc}\n")
            history.pop()   # don't add failed turn to history


# ---------------------------------------------------------------------------
# Smoke test (no user interaction)
# ---------------------------------------------------------------------------

def run_smoke_test() -> None:
    """Quick non-interactive test to verify the provider works."""
    print("Running smoke test…")
    adapter = Adapter(api_key=None, model_id=OLLAMA_MODEL)
    request = adapter.build_request(
        system     = "You are a concise assistant.",
        messages   = [{"role": "user", "content": "Say exactly: Hello from DeepSeek"}],
        max_tokens = 64,
    )
    try:
        raw  = adapter.call(request)
        resp = adapter.parse_response(raw)
        print(f"Response : {resp['content'][:120]!r}")
        print(f"Tokens   : {resp['tokens_in']} in / {resp['tokens_out']} out")
        print("Smoke test PASSED.")
    except RuntimeError as exc:
        print(f"Smoke test FAILED: {exc}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chat with DeepSeek-R1 1.5B locally (Ollama or transformers)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Quick start (Ollama):
          ollama pull deepseek-r1:1.5b
          python examples/deepseek_local.py

        Quick start (transformers, no Ollama):
          pip install transformers torch accelerate
          python examples/deepseek_local.py --backend transformers

        Show reasoning trace:
          python examples/deepseek_local.py --show-thinking

        Non-interactive smoke test:
          python examples/deepseek_local.py --smoke-test
        """),
    )
    parser.add_argument("--prompt",           metavar="TEXT",
                        help="Single prompt (non-interactive)")
    parser.add_argument("--show-thinking",    action="store_true",
                        help="Print the <think> chain-of-thought before the answer")
    parser.add_argument("--backend",          choices=["ollama", "transformers"],
                        default="ollama",
                        help="Force a specific backend (default: ollama with auto-fallback)")
    parser.add_argument("--smoke-test",       action="store_true",
                        help="Quick non-interactive verification test")
    args = parser.parse_args()

    force_tf = args.backend == "transformers"

    if args.smoke_test:
        run_smoke_test()
    elif args.prompt:
        run_single(args.prompt, args.show_thinking, force_tf)
    else:
        run_chat(args.show_thinking, force_tf)


if __name__ == "__main__":
    main()
