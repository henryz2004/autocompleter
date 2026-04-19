"""Phase A worker loop.

Consumes trigger messages on stdin and emits inject commands on stdout, all as
UTF-8 JSON lines terminated by `\\n`. Phase A behavior is trivial: reverse the
caller's `before_cursor` and send it back as an inject command. This is the
substrate smoke test — no LLM, no overlay, no context assembly. Phase B will
replace `_compute_suggestion()` with the real pipeline.

Protocol summary (see plan for the authoritative spec):

    Swift → Python (stdin)
        {"seq": <int>, "type": "trigger", "trigger_type": "manual",
         "timestamp": <float>, "app_name": "...", "app_pid": <int>,
         "window_title": "...", "source_url": "...",
         "focused_element": {
             "role": "...", "value": "...",
             "insertion_point": <int|null>, "selection_length": <int>,
             "placeholder_detected": <bool>,
             "before_cursor": "...", "after_cursor": "...",
             "position": [x, y], "size": [w, h]},
         "caret_position": {"x": <float>, "y": <float>, "height": <float>},
         "ax_subtree": {"bottom_up": "...", "top_down": "..."},
         "conversation_turns": [...],
         "cross_app_snapshots": [...]}

    Python → Swift (stdout)
        {"seq": <int>, "type": "inject",
         "text": "...",
         "strategy": "ax_value"|"clipboard"|"keystrokes"|"cdp"|"auto",
         "requires_focused_element": <bool>,
         "in_reply_to": <int>}

    Python → Swift (housekeeping, optional)
        {"seq": <int>, "type": "log", "level": "info"|"warning"|..., "message": "..."}
        {"seq": <int>, "type": "pong", "in_reply_to": <int>}
        {"seq": <int>, "type": "error", "message": "...", "in_reply_to": <int|null>}

stdout is the command channel. All human/diagnostic output goes to stderr so
the parent (Swift) can tee it into the log file.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any


PROTOCOL_VERSION = 1

# Module-level lock so _emit() calls from background threads don't interleave
# JSON lines on stdout. Introduced for trajectory_query, which runs its LLM
# call on a worker thread so stdin dispatch isn't blocked.
_EMIT_LOCK = threading.Lock()


class WorkerState:
    """Minimal worker state. No LLM, no overlay — Phase A is a smoke loop.

    The trajectory-intent-probe experiment adds a lazily-initialized LLM
    client and a background-thread pool so `trajectory_query` handling
    doesn't block the stdin loop. Kept private to the worker to avoid
    tangling the Phase A path.
    """

    def __init__(self) -> None:
        self._seq = 0
        self._seq_lock = threading.Lock()
        # Lazy LLM bits for trajectory_intent. Only constructed the first
        # time a trajectory_query arrives, so the Phase A smoke path pays
        # zero cost.
        self._intent_client = None
        self._intent_model: str | None = None
        self._intent_eval_log: Path | None = None
        self._intent_lock = threading.Lock()

    def next_seq(self) -> int:
        with self._seq_lock:
            self._seq += 1
            return self._seq

    def get_intent_llm(self):
        """Return (client, model, eval_log_path) for the intent probe.

        Hardcoded to Groq + `qwen/qwen3-32b`. We deliberately do NOT honor
        `AUTOCOMPLETER_LLM_MODEL` / `AUTOCOMPLETER_LLM_BASE_URL` here —
        Henry's `.env` overrides those for the main suggestion pipeline
        (often pointing at Cerebras-flavored model names), and the intent
        probe should be isolated from that. Override via
        `AUTOCOMPLETER_INTENT_MODEL` if you want to iterate.

        Returns (None, None, None) on any config error so the caller can
        log and skip.
        """
        with self._intent_lock:
            if self._intent_client is not None:
                return self._intent_client, self._intent_model, self._intent_eval_log
            try:
                from .config import load_config
                import openai

                config = load_config()
                api_key = os.environ.get("GROQ_API_KEY", "")
                if not api_key:
                    return None, None, None
                base_url = os.environ.get(
                    "AUTOCOMPLETER_INTENT_BASE_URL",
                    "https://api.groq.com/openai/v1",
                )
                model = os.environ.get(
                    "AUTOCOMPLETER_INTENT_MODEL",
                    "qwen/qwen3-32b",
                )
                self._intent_client = openai.OpenAI(api_key=api_key, base_url=base_url)
                self._intent_model = model
                self._intent_eval_log = config.data_dir / "trajectory_eval.jsonl"
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[worker] trajectory intent LLM init failed: {exc!r}",
                    file=sys.stderr,
                    flush=True,
                )
                return None, None, None
            return self._intent_client, self._intent_model, self._intent_eval_log


def _emit(msg: dict[str, Any]) -> None:
    """Write a JSON-line message to stdout and flush immediately.

    If the Swift parent has died, stdout writes raise BrokenPipeError — in
    that case the worker should exit cleanly rather than dump a stack trace.
    """
    try:
        # Background threads (trajectory-intent handler) also call _emit, so
        # serialize writes to keep JSON lines intact. The single-threaded
        # Phase A path pays the cost of an uncontended lock acquire, which
        # is negligible.
        with _EMIT_LOCK:
            sys.stdout.write(json.dumps(msg, ensure_ascii=False, separators=(",", ":")))
            sys.stdout.write("\n")
            sys.stdout.flush()
    except BrokenPipeError:
        # Parent is gone. Close stderr so Python's atexit doesn't re-raise,
        # then exit quietly.
        try:
            sys.stderr.close()
        except Exception:
            pass
        os._exit(0)


def _log(state: WorkerState, level: str, message: str) -> None:
    _emit({
        "seq": state.next_seq(),
        "type": "log",
        "level": level,
        "message": message,
    })


def _error(state: WorkerState, message: str, in_reply_to: int | None) -> None:
    _emit({
        "seq": state.next_seq(),
        "type": "error",
        "message": message,
        "in_reply_to": in_reply_to,
    })


def _compute_suggestion(trigger: dict[str, Any]) -> str | None:
    """Phase A stand-in for the real suggestion pipeline.

    Returns the reverse of whatever sits immediately before the cursor. If
    there's no cursor or no text to reverse, return None so Swift sees nothing
    to inject.
    """
    focused = trigger.get("focused_element") or {}
    before = focused.get("before_cursor") or ""
    if not before.strip():
        return None
    # Reverse the final word for a visible but harmless smoke signal.
    trailing_ws_len = len(before) - len(before.rstrip())
    trailing_ws = before[len(before) - trailing_ws_len:] if trailing_ws_len else ""
    body = before[: len(before) - trailing_ws_len] if trailing_ws_len else before
    last_space = body.rfind(" ")
    word = body[last_space + 1:] if last_space >= 0 else body
    if not word:
        return None
    return word[::-1] + trailing_ws


def _handle_trigger(state: WorkerState, msg: dict[str, Any]) -> None:
    in_reply_to = int(msg.get("seq") or 0)
    suggestion = _compute_suggestion(msg)
    if suggestion is None:
        _log(state, "info", "trigger: nothing to suggest (empty before_cursor)")
        return

    _emit({
        "seq": state.next_seq(),
        "type": "inject",
        "text": suggestion,
        "strategy": "auto",
        "requires_focused_element": True,
        "in_reply_to": in_reply_to,
    })


def _handle_ping(state: WorkerState, msg: dict[str, Any]) -> None:
    _emit({
        "seq": state.next_seq(),
        "type": "pong",
        "in_reply_to": int(msg.get("seq") or 0),
    })


def _handle_trajectory_query(state: WorkerState, msg: dict[str, Any]) -> None:
    """Experimental: call a fast LLM to guess user intent from the ring buffer.

    Runs on a background thread so the stdin dispatch loop is never blocked
    by the LLM call (which can take ~200-500ms even on Cerebras).
    """
    in_reply_to = int(msg.get("seq") or 0)
    snapshots = msg.get("snapshots") or []
    if not isinstance(snapshots, list):
        _error(state, "trajectory_query.snapshots must be a list", in_reply_to)
        return
    _log(state, "info",
         f"trajectory_query received seq={in_reply_to} snapshots={len(snapshots)}")

    def run() -> None:
        try:
            from .trajectory_intent import infer_intent

            client, model, eval_log = state.get_intent_llm()
            if client is None or model is None or eval_log is None:
                _log(state, "warning",
                     "trajectory_query: intent LLM not configured (set fallback API key)")
                return
            guess = infer_intent(
                snapshots,
                client=client,
                model=model,
                eval_log_path=eval_log,
            )
            if guess is None:
                _log(state, "warning", "trajectory_query: LLM call failed (see eval log)")
                return
            _emit({
                "seq": state.next_seq(),
                "type": "trajectory_guess",
                "in_reply_to": in_reply_to,
                "intent_sentence": guess.intent_sentence,
                "confidence": guess.confidence,
                "model_used": guess.model_used,
                "latency_ms": guess.latency_ms,
            })
        except Exception as exc:  # noqa: BLE001 — keep the worker alive
            _error(state, f"trajectory handler crashed: {exc}", in_reply_to)
            print(f"[worker] trajectory handler exception: {exc!r}",
                  file=sys.stderr, flush=True)

    t = threading.Thread(target=run, name="trajectory-intent", daemon=True)
    t.start()


def _dispatch(state: WorkerState, msg: dict[str, Any]) -> None:
    kind = msg.get("type")
    if kind == "trigger":
        _handle_trigger(state, msg)
    elif kind == "ping":
        _handle_ping(state, msg)
    elif kind == "trajectory_query":
        _handle_trajectory_query(state, msg)
    elif kind == "inject_result":
        # Swift reporting whether the last inject succeeded. Phase A just logs.
        ok = msg.get("ok")
        reason = msg.get("reason") or ""
        _log(state, "info" if ok else "warning",
             f"inject_result ok={ok} reason={reason!r}")
    else:
        _error(state, f"unknown message type: {kind!r}", int(msg.get("seq") or 0))


def run_worker_loop() -> int:
    """Read JSON lines from stdin until EOF, dispatching to handlers.

    Returns an exit code. stdin EOF is treated as a clean shutdown (the parent
    closed our input pipe). Any other error is fatal and surfaces via stderr so
    Swift's stderr-tail can show it in the diagnostic pane.
    """
    state = WorkerState()
    _log(state, "info", f"worker ready (protocol=v{PROTOCOL_VERSION}, pid={time_pid()})")

    # Line-buffered read. We intentionally do NOT use asyncio — stdin is a
    # single pipe and handlers are synchronous. If Phase B wants concurrency,
    # introduce a thread pool here.
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as exc:
            _error(state, f"invalid json: {exc}", None)
            continue
        if not isinstance(msg, dict):
            _error(state, f"expected JSON object, got {type(msg).__name__}", None)
            continue
        try:
            _dispatch(state, msg)
        except Exception as exc:  # noqa: BLE001 — keep the worker alive
            _error(state, f"handler crashed: {exc}", int(msg.get("seq") or 0))
            print(f"[worker] handler exception: {exc!r}", file=sys.stderr, flush=True)

    _log(state, "info", "stdin closed — exiting")
    return 0


def time_pid() -> str:
    """Small helper for the ready-log. Kept here to avoid an os import up top
    in the hot-path parts of the module."""
    import os
    return f"{int(time.time())}:{os.getpid()}"
