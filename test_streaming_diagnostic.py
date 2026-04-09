"""Diagnostic script: does Instructor's create_partial with Anthropic yield
incremental snapshots where a list grows over time, or does it buffer
everything and yield the complete result at the end?

Usage:
    python test_streaming_diagnostic.py
"""

import time

import instructor

from autocompleter.config import load_config
from autocompleter.suggestion_engine import SuggestionItem, SuggestionList


def main() -> None:
    config = load_config()
    api_key = config.anthropic_api_key
    if not api_key:
        print("ERROR: No Anthropic API key found.")
        print("Set ANTHROPIC_API_KEY in your environment or in .env")
        return

    model = "anthropic/claude-haiku-4-5-20251001"
    print(f"Creating Instructor client with model: {model}")
    client = instructor.from_provider(model, api_key=api_key)

    messages = [
        {
            "role": "user",
            "content": (
                "Suggest 3 fun weekend activities. "
                "Return exactly 3 suggestions."
            ),
        },
    ]

    print()
    print("=" * 70)
    print("Calling client.create_partial(response_model=SuggestionList, ...)")
    print("=" * 70)
    print()

    t_start = time.time()
    snapshot_count = 0

    stream = client.create_partial(
        response_model=SuggestionList,
        messages=messages,
        max_tokens=500,
    )

    for partial in stream:
        snapshot_count += 1
        t_now = time.time()
        elapsed = t_now - t_start

        items = partial.suggestions or []

        # Extract texts, handling SuggestionItem, dict, or None
        texts: list[str] = []
        for s in items:
            if s is None:
                continue
            if isinstance(s, dict):
                text = s.get("text", "")
            else:
                text = getattr(s, "text", None) or ""
            texts.append(text)

        num_suggestions = len(texts)

        print(f"[snapshot {snapshot_count:3d}]  "
              f"t={elapsed:6.3f}s  "
              f"suggestions={num_suggestions}")
        for i, text in enumerate(texts):
            truncated = text[:40] + ("..." if len(text) > 40 else "")
            print(f"    [{i}] {truncated!r}")
        print()

    t_end = time.time()
    total_elapsed = t_end - t_start

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total snapshots yielded:  {snapshot_count}")
    print(f"Total elapsed time:       {total_elapsed:.3f}s")
    print()

    if snapshot_count <= 1:
        print("RESULT: create_partial buffered everything and yielded "
              "a single complete result.")
    else:
        print("RESULT: create_partial yielded incremental snapshots "
              f"({snapshot_count} total).")
        print("        The list grew over time as the LLM streamed tokens.")


if __name__ == "__main__":
    main()
