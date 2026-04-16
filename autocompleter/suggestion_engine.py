"""Suggestion Engine - generates contextual completions via LLM API.

Receives current input + sliced context from the context store,
calls an external LLM API with short max token limit, and returns
1-3 short completions. Includes debouncing to avoid excessive API calls.

Supports both blocking (generate_suggestions) and streaming
(generate_suggestions_stream) modes. The streaming mode uses the raw
provider SDK with plain-text JSON output, parsing incrementally to
yield Suggestion objects one at a time as they complete.
"""

from __future__ import annotations

import enum
import json
import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, Generator, Optional

from pydantic import BaseModel, Field

from .config import Config
from .prompts import build_messages as _build_messages

# ---- Shell / terminal app detection ----
# Used by input_observer and app.py for injection strategy and
# placeholder detection — NOT for prompt selection (removed).

_SHELL_APP_NAMES: frozenset[str] = frozenset({
    "Terminal", "iTerm2", "iTerm", "Warp",
    "Alacritty", "kitty", "Hyper", "WezTerm",
})


def is_shell_app(app_name: str) -> bool:
    """Return True if *app_name* is a known terminal emulator."""
    return app_name in _SHELL_APP_NAMES


class SuggestionItem(BaseModel):
    """Pydantic model for a single suggestion."""
    text: str = Field(description="The suggestion text the user would type")


class SuggestionList(BaseModel):
    """Pydantic model wrapping multiple suggestions for reliable count."""
    suggestions: list[SuggestionItem] = Field(
        ...,
        min_length=2,
        description="List of distinct suggestions",
    )


class AutocompleteMode(enum.Enum):
    """Determines how context is assembled and what kind of suggestion to generate."""
    CONTINUATION = "continuation"  # User has draft text, predict next words
    REPLY = "reply"               # Input is empty/short, suggest a full response

logger = logging.getLogger(__name__)


MAX_PREVIEW_LENGTH = 80


def build_messages(
    mode: AutocompleteMode,
    context: str,
    num_suggestions: int = 3,
    max_suggestion_lines: int = 10,
    streaming: bool = False,
    source_app: str = "",
    prompt_placeholder_aware: bool = False,
) -> tuple[str, str]:
    """Build (system_prompt, user_message) for an LLM call.

    Shared by SuggestionEngine and the benchmark harness so prompt
    construction stays in one place.

    Args:
        mode: CONTINUATION or REPLY.
        context: Assembled context string.
        num_suggestions: Number of suggestions to request.
        max_suggestion_lines: Max lines per suggestion (reply mode only).
        streaming: If True, append STREAMING_JSON_INSTRUCTION to the
            system prompt (used by the streaming / benchmark paths).
        source_app: Name of the source application.

    Returns:
        (system_prompt, user_message) tuple.
    """
    return _build_messages(
        mode=mode,
        context=context,
        num_suggestions=num_suggestions,
        max_suggestion_lines=max_suggestion_lines,
        streaming=streaming,
        source_app=source_app,
        prompt_placeholder_aware=prompt_placeholder_aware,
        streaming_json_instruction=STREAMING_JSON_INSTRUCTION,
    )


def _normalize_continuation_spacing(before_cursor: str | None, text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return stripped
    if before_cursor and not before_cursor.endswith(" "):
        return " " + stripped
    return stripped


def _normalize_similarity_text(text: str) -> str:
    return " ".join(text.lower().split()).strip()


def _filter_avoided_text(
    text: str,
    avoid_texts: set[str] | None = None,
) -> str:
    if avoid_texts and _normalize_similarity_text(text) in avoid_texts:
        return ""
    return text


def _strip_repeated_prefix(text: str, before_cursor: str | None) -> str:
    suggestion = text.strip()
    if not suggestion or not before_cursor:
        return suggestion

    draft_candidates: list[str] = []
    for candidate in (before_cursor, before_cursor.rstrip(), before_cursor.strip()):
        normalized = candidate.strip()
        if normalized and normalized not in draft_candidates:
            draft_candidates.append(normalized)

    lowered = suggestion.lower()
    for candidate in sorted(draft_candidates, key=len, reverse=True):
        if lowered.startswith(candidate.lower()):
            suggestion = suggestion[len(candidate):].lstrip(" ,")
            break
    return suggestion.strip()


def _postprocess_continuation_text(
    text: str,
    before_cursor: str | None,
    index: int,
    avoid_texts: set[str] | None = None,
) -> str:
    stripped = _strip_repeated_prefix(text, before_cursor)
    result = _normalize_continuation_spacing(before_cursor, stripped or text)
    del index
    return _filter_avoided_text(result, avoid_texts)


def postprocess_suggestion_texts(
    texts: list[str],
    mode: AutocompleteMode,
    before_cursor: str | None,
    avoid_texts: list[str] | None = None,
) -> list[str]:
    avoid = {
        _normalize_similarity_text(text)
        for text in (avoid_texts or [])
        if _normalize_similarity_text(text)
    }
    if mode != AutocompleteMode.CONTINUATION:
        return [_filter_avoided_text(text, avoid) for text in texts]
    return [
        _postprocess_continuation_text(text, before_cursor, i, avoid)
        for i, text in enumerate(texts)
    ]


def postprocess_suggestion_text(
    text: str,
    mode: AutocompleteMode,
    before_cursor: str | None,
    index: int,
    avoid_texts: list[str] | None = None,
) -> str:
    avoid = {
        _normalize_similarity_text(item)
        for item in (avoid_texts or [])
        if _normalize_similarity_text(item)
    }
    if mode != AutocompleteMode.CONTINUATION:
        return _filter_avoided_text(text, avoid)
    return _postprocess_continuation_text(text, before_cursor, index, avoid)


@dataclass
class Suggestion:
    text: str
    index: int

    @property
    def preview(self) -> str:
        """Return a truncated single-line preview of the full text.

        Takes the first line (or first sentence), strips whitespace,
        and truncates to MAX_PREVIEW_LENGTH characters with an ellipsis
        if the full text is longer or multi-line.
        """
        # Take the first line
        first_line = self.text.split("\n", 1)[0].strip()
        is_multiline = "\n" in self.text
        needs_truncation = len(first_line) > MAX_PREVIEW_LENGTH or is_multiline
        if needs_truncation:
            truncated = first_line[:MAX_PREVIEW_LENGTH].rstrip()
            return truncated + "..."
        return first_line


STREAMING_JSON_INSTRUCTION = """

CRITICAL: You MUST respond with a JSON object in this exact format:
{"suggestions": [{"text": "suggestion 1"}, {"text": "suggestion 2"}, {"text": "suggestion 3"}]}
Output ONLY the raw JSON object. No other text, no markdown, no code blocks."""


class SuggestionEngine:
    def __init__(self, config: Config):
        self.config = config
        self._last_request_time: float = 0.0
        self._client = None  # Instructor client (for blocking path)
        self._raw_client = None  # Raw provider client (for streaming path)
        self._rate_limited_until: float = 0.0  # timestamp when cooldown expires

    def _get_client(self):
        if self._client is None:
            import instructor

            if self.config.effective_llm_provider == "anthropic":
                import anthropic
                raw = anthropic.Anthropic(api_key=self.config.anthropic_api_key)
                self._client = instructor.from_anthropic(raw)
            else:
                import openai
                kwargs: dict = {"api_key": self.config.effective_openai_api_key}
                if self.config.effective_llm_base_url:
                    kwargs["base_url"] = self.config.effective_llm_base_url
                raw = openai.OpenAI(**kwargs)
                self._client = instructor.from_openai(raw)
        return self._client

    def _get_raw_client(self):
        """Get a raw provider client for true text streaming."""
        if self._raw_client is None:
            if self.config.effective_llm_provider == "anthropic":
                import anthropic
                self._raw_client = anthropic.Anthropic(
                    api_key=self.config.anthropic_api_key,
                )
            else:
                import openai
                kwargs: dict = {"api_key": self.config.effective_openai_api_key}
                if self.config.effective_llm_base_url:
                    kwargs["base_url"] = self.config.effective_llm_base_url
                self._raw_client = openai.OpenAI(**kwargs)
        return self._raw_client

    _fallback_client = None  # Cached fallback provider client
    _fallback_client_lock = threading.Lock()

    def _get_fallback_client(self):
        """Get a raw client for the fallback provider (thread-safe)."""
        if self._fallback_client is not None:
            return self._fallback_client
        with self._fallback_client_lock:
            # Double-check after acquiring lock
            if self._fallback_client is not None:
                return self._fallback_client
            if not self.config.effective_fallback_api_key:
                return None
            import openai
            kwargs: dict = {"api_key": self.config.effective_fallback_api_key}
            if self.config.effective_fallback_base_url:
                kwargs["base_url"] = self.config.effective_fallback_base_url
            self._fallback_client = openai.OpenAI(**kwargs)
        return self._fallback_client

    # Cooldown duration (seconds) after a rate-limit hit
    _RATE_LIMIT_COOLDOWN = 30.0

    @property
    def is_rate_limited(self) -> bool:
        """True if we're in a rate-limit cooldown period."""
        return time.time() < self._rate_limited_until

    def _is_rate_limit_error(self, exc: Exception) -> bool:
        """Check if an exception is a rate-limit (429) error."""
        # OpenAI SDK
        try:
            import openai
            if isinstance(exc, openai.RateLimitError):
                return True
        except ImportError:
            pass
        # Anthropic SDK
        try:
            import anthropic
            if isinstance(exc, anthropic.RateLimitError):
                return True
        except ImportError:
            pass
        # Generic: check for status_code attribute
        status = getattr(exc, "status_code", None)
        if status == 429:
            return True
        return False

    def _handle_rate_limit(self) -> None:
        """Set cooldown after a rate-limit hit."""
        self._rate_limited_until = time.time() + self._RATE_LIMIT_COOLDOWN
        logger.warning(
            f"Rate limited — cooling down for {self._RATE_LIMIT_COOLDOWN:.0f}s"
        )

    def can_request(self) -> bool:
        """Check if enough time has passed since the last request (debounce)."""
        if self.is_rate_limited:
            return False
        elapsed_ms = (time.time() - self._last_request_time) * 1000
        return elapsed_ms >= self.config.debounce_ms

    def generate_suggestions(
        self,
        current_input: str,
        context: str,
        app_name: str = "Unknown",
        mode: Optional[AutocompleteMode] = None,
        before_cursor: Optional[str] = None,
        feedback_stats: Optional[dict] = None,
        negative_patterns: Optional[list[str]] = None,
        prompt_placeholder_aware: bool = False,
    ) -> list[Suggestion]:
        """Generate completion suggestions using the configured LLM.

        Args:
            current_input: The text the user has typed so far.
            context: Sliced context from the context store.
            app_name: Name of the app where the user is typing.
            mode: Explicit mode override. If None, inferred from before_cursor.
            before_cursor: Text before the cursor. Used for mode detection
                when mode is None. Falls back to current_input if not provided.
            feedback_stats: Optional dict from ContextStore.get_feedback_stats().
                Used to adjust temperature based on accept rate.
            negative_patterns: Optional list of recently dismissed suggestion
                texts. Appended to the system prompt to avoid similar suggestions.

        Returns:
            A list of Suggestion objects.
        """
        if self.is_rate_limited:
            remaining = self._rate_limited_until - time.time()
            logger.debug(f"Rate limited, {remaining:.0f}s remaining")
            return [Suggestion(text=f"Rate limited — retry in {remaining:.0f}s", index=0)]

        if not self.can_request():
            logger.debug("Debounce: skipping request (too soon)")
            return []

        if not current_input.strip() and not context.strip():
            return []

        self._last_request_time = time.time()

        if mode is None:
            mode = detect_mode(
                before_cursor=before_cursor if before_cursor is not None else current_input,
            )

        system, user_msg = build_messages(
            mode=mode,
            context=context,
            num_suggestions=self.config.num_suggestions,
            max_suggestion_lines=getattr(self.config, "max_suggestion_lines", 10),
            streaming=False,
            source_app=app_name,
            prompt_placeholder_aware=prompt_placeholder_aware,
        )

        if mode == AutocompleteMode.CONTINUATION:
            temperature = self.config.continuation_temperature
        else:
            temperature = self.config.reply_temperature

        max_tokens = self.config.max_tokens

        # Adjust temperature based on feedback stats
        if feedback_stats is not None:
            temperature = adjust_temperature(temperature, feedback_stats.get("accept_rate", 0.5))

        # Append negative patterns to system prompt
        if negative_patterns:
            avoided = "\n".join(f"- {p}" for p in negative_patterns)
            system += (
                "\n\nThe user recently dismissed these suggestions. "
                "Avoid generating similar completions:\n" + avoided
            )

        logger.debug(
            f"--- LLM REQUEST ({self.config.effective_llm_provider}/{self.config.effective_llm_model}, "
            f"mode={mode.value}, temp={temperature}, max_tok={max_tokens}) ---"
        )
        logger.debug(f"System prompt ({len(system)} chars): {system[:200]!r}...")
        logger.debug(f"User message ({len(user_msg)} chars):")
        for line in user_msg.splitlines():
            logger.debug(f"  | {line}")

        try:
            results = self._call_llm(
                system, user_msg, temperature=temperature, max_tokens=max_tokens,
            )
            if prompt_placeholder_aware:
                processed = postprocess_suggestion_texts(
                    [suggestion.text for suggestion in results],
                    mode=mode,
                    before_cursor=before_cursor if before_cursor is not None else current_input,
                    avoid_texts=negative_patterns,
                )
                for suggestion, text in zip(results, processed):
                    suggestion.text = text
            logger.debug(f"LLM returned {len(results)} suggestions")
            return results
        except Exception as exc:
            if self._is_rate_limit_error(exc):
                self._handle_rate_limit()
                return [Suggestion(text="Rate limited — try again later", index=0)]
            logger.exception("Error generating suggestions")
            return [Suggestion(text="LLM error — try again", index=0)]

    def _call_llm(
        self, system: str, user_msg: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> list[Suggestion]:
        """Call the LLM via Instructor and return parsed suggestions."""
        client = self._get_client()
        result = client.create(
            response_model=SuggestionList,
            max_retries=2,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return [
            Suggestion(text=item.text, index=i)
            for i, item in enumerate(result.suggestions)
        ]

    # ---- Streaming API ----

    def generate_suggestions_stream(
        self,
        current_input: str,
        context: str,
        app_name: str = "Unknown",
        mode: Optional[AutocompleteMode] = None,
        before_cursor: Optional[str] = None,
        feedback_stats: Optional[dict] = None,
        negative_patterns: Optional[list[str]] = None,
        temperature_boost: float = 0.0,
        event_callback: Optional[Callable[[str, dict | None], None]] = None,
        prompt_placeholder_aware: bool = False,
    ) -> Generator[Suggestion, None, None]:
        """Generate completion suggestions via streaming, yielding each as it completes.

        Yields Suggestion objects one at a time as they become available
        from the LLM via raw SDK text streaming with incremental JSON parsing.

        Args:
            current_input: The text the user has typed so far.
            context: Sliced context from the context store.
            app_name: Name of the app where the user is typing.
            mode: Explicit mode override. If None, inferred from before_cursor.
            before_cursor: Text before the cursor. Used for mode detection
                when mode is None. Falls back to current_input if not provided.
            feedback_stats: Optional dict from ContextStore.get_feedback_stats().
            negative_patterns: Optional list of recently dismissed suggestion texts.
            temperature_boost: Added to the base temperature (e.g. 0.3 on
                regenerate) to increase output diversity. Clamped to [0, 2].

        Yields:
            Suggestion objects, in the order they are completed by the LLM.
        """
        if self.is_rate_limited:
            remaining = self._rate_limited_until - time.time()
            logger.debug(f"Rate limited, {remaining:.0f}s remaining")
            yield Suggestion(text=f"Rate limited — retry in {remaining:.0f}s", index=0)
            return

        if not self.can_request():
            logger.debug("Debounce: skipping streaming request (too soon)")
            return

        if not current_input.strip() and not context.strip():
            return

        self._last_request_time = time.time()

        if mode is None:
            mode = detect_mode(
                before_cursor=before_cursor if before_cursor is not None else current_input,
            )

        system, user_msg = build_messages(
            mode=mode,
            context=context,
            num_suggestions=self.config.num_suggestions,
            max_suggestion_lines=getattr(self.config, "max_suggestion_lines", 10),
            streaming=True,
            source_app=app_name,
            prompt_placeholder_aware=prompt_placeholder_aware,
        )

        if mode == AutocompleteMode.CONTINUATION:
            temperature = self.config.continuation_temperature
        else:
            temperature = self.config.reply_temperature

        max_tokens = self.config.max_tokens

        # Apply feedback-based temperature adjustment
        if feedback_stats is not None:
            temperature = adjust_temperature(temperature, feedback_stats.get("accept_rate", 0.5))

        # Apply temperature boost (e.g. on regenerate for more diversity)
        if temperature_boost:
            temperature = min(temperature + temperature_boost, 2.0)

        # Append negative patterns to system prompt
        if negative_patterns:
            avoided = "\n".join(f"- {p}" for p in negative_patterns)
            if temperature_boost > 0:
                # Regenerate: stronger diversity instruction
                system += (
                    "\n\nIMPORTANT: The user is regenerating because they want "
                    "DIFFERENT suggestions. Do NOT repeat or closely paraphrase "
                    "any of these previous suggestions — take a completely "
                    "different angle, tone, or approach:\n" + avoided
                )
            else:
                system += (
                    "\n\nAvoid generating suggestions similar to these recently "
                    "dismissed completions:\n" + avoided
                )

        logger.debug(
            f"--- LLM STREAM REQUEST ({self.config.effective_llm_provider}/{self.config.effective_llm_model}, "
            f"mode={mode.value}, temp={temperature}, max_tok={max_tokens}) ---"
        )
        if event_callback is not None:
            event_callback(
                "request_built",
                {
                    "provider": self.config.effective_llm_provider,
                    "base_url": self.config.effective_llm_base_url,
                    "model": self.config.effective_llm_model,
                    "fallback_provider": self.config.effective_fallback_provider,
                    "fallback_base_url": self.config.effective_fallback_base_url,
                    "fallback_model": self.config.effective_fallback_model,
                    "mode": mode.value,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "num_suggestions": self.config.num_suggestions,
                    "streaming": True,
                    "system_prompt": system,
                    "user_prompt": user_msg,
                    "feedback_stats": feedback_stats or {},
                    "negative_patterns": negative_patterns or [],
                    "temperature_boost": temperature_boost,
                    "prompt_placeholder_aware": prompt_placeholder_aware,
                },
            )

        try:
            for suggestion in self._call_llm_stream(
                system,
                user_msg,
                temperature=temperature,
                max_tokens=max_tokens,
                event_callback=event_callback,
            ):
                if prompt_placeholder_aware:
                    suggestion.text = postprocess_suggestion_text(
                        suggestion.text,
                        mode=mode,
                        before_cursor=before_cursor if before_cursor is not None else current_input,
                        index=suggestion.index,
                        avoid_texts=negative_patterns if temperature_boost > 0 else None,
                    )
                if suggestion.text.strip():
                    yield suggestion
        except Exception as exc:
            if self._is_rate_limit_error(exc):
                self._handle_rate_limit()
                yield Suggestion(text="Rate limited — try again later", index=0)
            else:
                logger.exception("Error during streaming suggestions")
            return

    # ---- Streaming helpers ----

    @staticmethod
    def _stream_openai_to_queue(
        client,
        model: str,
        system: str,
        user_msg: str,
        temperature: Optional[float],
        max_tokens: Optional[int],
        out_q: queue.Queue,
        cancel: threading.Event,
        tag: str,
        extra_body: Optional[dict] = None,
    ) -> None:
        """Run an OpenAI-compatible streaming call, pushing suggestions to *out_q*.

        Each item pushed is ``(tag, Suggestion)`` or ``(tag, None)`` for end-of-stream.
        On error pushes ``(tag, Exception)``.  Stops early if *cancel* is set.
        """
        try:
            create_kwargs: dict = dict(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_msg},
                ],
                stream=True,
                temperature=temperature,
                max_tokens=max_tokens or 1024,
            )
            if extra_body:
                create_kwargs["extra_body"] = extra_body
            response = client.chat.completions.create(**create_kwargs)
            json_buf = ""
            last_yielded = 0
            for chunk in response:
                if cancel.is_set():
                    # The other provider won — stop reading
                    try:
                        response.close()
                    except Exception:
                        pass
                    return
                delta = chunk.choices[0].delta if chunk.choices else None
                content = delta.content if delta else None
                if content:
                    json_buf += content
                    complete = _extract_complete_suggestions(json_buf)
                    while len(complete) > last_yielded:
                        s = complete[last_yielded]
                        out_q.put((tag, Suggestion(text=s, index=last_yielded)))
                        last_yielded += 1
            # Signal stream complete
            out_q.put((tag, None))
        except Exception as exc:
            out_q.put((tag, exc))

    @staticmethod
    def _stream_anthropic_to_queue(
        client,
        model: str,
        system: str,
        user_msg: str,
        temperature: Optional[float],
        max_tokens: Optional[int],
        out_q: queue.Queue,
        cancel: threading.Event,
        tag: str,
    ) -> None:
        """Run an Anthropic streaming call, pushing suggestions to *out_q*."""
        try:
            with client.messages.stream(
                model=model,
                system=system,
                messages=[{"role": "user", "content": user_msg}],
                temperature=temperature,
                max_tokens=max_tokens or 1024,
            ) as stream:
                json_buf = ""
                last_yielded = 0
                for text_chunk in stream.text_stream:
                    if cancel.is_set():
                        return
                    json_buf += text_chunk
                    complete = _extract_complete_suggestions(json_buf)
                    while len(complete) > last_yielded:
                        s = complete[last_yielded]
                        out_q.put((tag, Suggestion(text=s, index=last_yielded)))
                        last_yielded += 1
            out_q.put((tag, None))
        except Exception as exc:
            out_q.put((tag, exc))

    def _call_llm_stream(
        self, system: str, user_msg: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        event_callback: Optional[Callable[[str, dict | None], None]] = None,
    ) -> Generator[Suggestion, None, None]:
        """Stream suggestions with timeout-based provider escalation.

        Starts the primary provider immediately.  If no suggestion arrives
        within ``escalation_timeout_ms``, fires the fallback provider in
        parallel.  Whichever provider yields first wins — the loser is
        cancelled via a ``threading.Event``.

        Falls back to the blocking ``_call_llm`` path if both streams fail.
        """
        # NOTE: system already includes STREAMING_JSON_INSTRUCTION via
        # build_messages(streaming=True).
        out_q: queue.Queue = queue.Queue()
        cancel_primary = threading.Event()
        cancel_fallback = threading.Event()

        # Compute extra_body for providers that need it (e.g. Groq reasoning_effort)
        def _extra_body_for_url(base_url: str) -> dict | None:
            if not base_url:
                return None
            from urllib.parse import urlparse
            hostname = urlparse(base_url).hostname or ""
            if hostname.endswith("groq.com"):
                return {"reasoning_effort": "none"}
            return None

        # Start primary provider
        primary_client = self._get_raw_client()
        if self.config.effective_llm_provider == "anthropic":
            primary_fn = self._stream_anthropic_to_queue
        else:
            primary_fn = self._stream_openai_to_queue

        primary_extra = _extra_body_for_url(self.config.effective_llm_base_url)
        primary_args = [
            primary_client, self.config.effective_llm_model, system, user_msg,
            temperature, max_tokens, out_q, cancel_primary, "primary",
        ]
        if primary_fn == self._stream_openai_to_queue and primary_extra:
            primary_args.append(primary_extra)

        primary_thread = threading.Thread(
            target=primary_fn,
            args=tuple(primary_args),
            daemon=True,
        )
        primary_thread.start()

        # Pre-compute fallback extra_body for Qwen3 thinking-mode suppression
        fallback_extra = _extra_body_for_url(self.config.effective_fallback_base_url)

        def _start_fallback(client) -> threading.Thread:
            """Create and start a fallback streaming thread."""
            args = (
                client, self.config.effective_fallback_model,
                system, user_msg, temperature, max_tokens,
                out_q, cancel_fallback, "fallback",
            )
            if fallback_extra:
                args = args + (fallback_extra,)
            t = threading.Thread(
                target=self._stream_openai_to_queue,
                args=args, daemon=True,
            )
            t.start()
            return t

        escalation_timeout = self.config.escalation_timeout_ms / 1000.0
        max_wall_time = 15.0  # Hard cap to prevent infinite loops
        wall_start = time.time()
        fallback_started = False
        winner = None  # Which provider yielded first
        loser_cancel = None

        try:
            while time.time() - wall_start < max_wall_time:
                # Determine timeout: use escalation timeout until fallback
                # starts, then wait indefinitely for whichever is active.
                if not fallback_started and winner is None:
                    timeout = escalation_timeout
                else:
                    timeout = 5.0  # generous timeout for remaining suggestions

                try:
                    tag, item = out_q.get(timeout=timeout)
                except queue.Empty:
                    if not fallback_started and winner is None:
                        # Primary didn't respond in time — escalate
                        fallback_client = self._get_fallback_client()
                        if fallback_client is not None:
                            logger.info(
                                f"Primary provider didn't respond in "
                                f"{self.config.escalation_timeout_ms}ms, "
                                f"escalating to fallback "
                                f"({self.config.effective_fallback_model})"
                            )
                            if event_callback is not None:
                                event_callback(
                                    "fallback_started",
                                    {
                                        "reason": "timeout",
                                        "model": self.config.effective_fallback_model,
                                    },
                                )
                            _start_fallback(fallback_client)
                            fallback_started = True
                            continue
                        else:
                            # No fallback configured — keep waiting
                            continue
                    else:
                        # Both providers active but no response — give up
                        logger.warning("Both providers timed out")
                        break

                # Handle errors
                if isinstance(item, Exception):
                    if winner is not None and tag != winner:
                        # The loser errored — ignore
                        continue
                    if self._is_rate_limit_error(item):
                        if tag == "primary" and not fallback_started:
                            # Primary rate-limited — escalate immediately
                            logger.info("Primary rate-limited, escalating")
                            fallback_client = self._get_fallback_client()
                            if fallback_client is not None:
                                if event_callback is not None:
                                    event_callback(
                                        "fallback_started",
                                        {
                                            "reason": "rate_limit",
                                            "model": self.config.effective_fallback_model,
                                        },
                                    )
                                _start_fallback(fallback_client)
                                fallback_started = True
                                cancel_primary.set()
                                continue
                        self._handle_rate_limit()
                        yield Suggestion(
                            text="Rate limited — try again later", index=0,
                        )
                        return
                    logger.warning(
                        f"Stream error from {tag}: {item}", exc_info=item,
                    )
                    if not fallback_started and tag == "primary":
                        # Primary failed — try fallback
                        fallback_client = self._get_fallback_client()
                        if fallback_client is not None:
                            logger.info("Primary failed, escalating to fallback")
                            if event_callback is not None:
                                event_callback(
                                    "fallback_started",
                                    {
                                        "reason": "error",
                                        "model": self.config.effective_fallback_model,
                                    },
                                )
                            _start_fallback(fallback_client)
                            fallback_started = True
                            continue
                    break

                # Handle end-of-stream
                if item is None:
                    if tag == winner:
                        # Winner finished — done
                        break
                    # One stream ended — if no winner yet and fallback is
                    # running, wait for the other
                    if winner is None and fallback_started:
                        continue
                    break

                # Handle suggestion
                suggestion = item
                if winner is None:
                    winner = tag
                    loser_cancel = (
                        cancel_fallback if tag == "primary" else cancel_primary
                    )
                    loser_cancel.set()
                    logger.info(
                        f"Provider '{tag}' won the race "
                        f"(model={self.config.effective_llm_model if tag == 'primary' else self.config.effective_fallback_model})"
                    )
                    if event_callback is not None:
                        event_callback(
                            "winner",
                            {
                                "provider": tag,
                                "model": (
                                    self.config.effective_llm_model
                                    if tag == "primary"
                                    else self.config.effective_fallback_model
                                ),
                            },
                        )

                if tag == winner:
                    logger.debug(
                        f"Stream yielding suggestion [{suggestion.index}] "
                        f"from {tag}: {suggestion.text[:80]!r}"
                    )
                    yield suggestion
                # else: discard suggestions from the loser

            # If no suggestions were yielded, fall back to blocking path
            if winner is None:
                logger.warning(
                    "Streams produced no suggestions, falling back to blocking"
                )
                cancel_primary.set()
                cancel_fallback.set()
                try:
                    results = self._call_llm(
                        system, user_msg,
                        temperature=temperature, max_tokens=max_tokens,
                    )
                    for suggestion in results:
                        yield suggestion
                except Exception:
                    logger.exception("Blocking fallback also failed")
                    yield Suggestion(text="LLM error — try again", index=0)

        except Exception as exc:
            cancel_primary.set()
            cancel_fallback.set()
            if self._is_rate_limit_error(exc):
                self._handle_rate_limit()
                yield Suggestion(text="Rate limited — try again later", index=0)
                return
            logger.warning(
                "Streaming with escalation failed, falling back to blocking",
                exc_info=True,
            )
            try:
                results = self._call_llm(
                    system, user_msg,
                    temperature=temperature, max_tokens=max_tokens,
                )
                for suggestion in results:
                    yield suggestion
            except Exception as fallback_exc:
                if self._is_rate_limit_error(fallback_exc):
                    self._handle_rate_limit()
                    yield Suggestion(
                        text="Rate limited — try again later", index=0,
                    )
                else:
                    logger.exception("Blocking fallback also failed")
                    yield Suggestion(text="LLM error — try again", index=0)


# ---- Mode detection (module-level for reuse) ----

MODE_THRESHOLD_CHARS = 3


def detect_mode(
    before_cursor: str,
    current_input: str | None = None,
) -> AutocompleteMode:
    """Determine autocomplete mode from the text before the cursor.

    Uses before_cursor (text to the left of the caret) rather than the full
    field value, so a user with the cursor at the start of a long paragraph
    correctly gets REPLY mode.

    Continuation: before_cursor has meaningful draft text (>= MODE_THRESHOLD_CHARS).
    Reply: before_cursor is empty or very short (< MODE_THRESHOLD_CHARS).

    Args:
        before_cursor: Text before the cursor position.
        current_input: Deprecated — ignored if before_cursor is provided.
            Kept for backwards compatibility; callers should migrate to
            passing before_cursor explicitly.
    """
    text = before_cursor
    if len(text.strip()) >= MODE_THRESHOLD_CHARS:
        return AutocompleteMode.CONTINUATION
    return AutocompleteMode.REPLY


def adjust_temperature(base_temp: float, accept_rate: float) -> float:
    """Adjust LLM temperature based on suggestion accept rate.

    If accept rate is below 0.3, lower temperature by 0.1 (more conservative).
    If accept rate is above 0.7, raise by 0.05 (more creative).
    Otherwise, keep the base temperature unchanged.

    Result is clamped to [0.1, 1.0].
    """
    if accept_rate < 0.3:
        temp = base_temp - 0.1
    elif accept_rate > 0.7:
        temp = base_temp + 0.05
    else:
        temp = base_temp
    return max(0.1, min(1.0, temp))


def _strip_think_tags(text: str) -> str:
    """Strip Qwen3 ``<think>...</think>`` reasoning blocks from LLM output.

    Qwen3 models default to thinking mode on some providers, emitting a
    ``<think>`` block **before** the actual JSON.  We only strip blocks
    that precede the first ``{`` (the JSON payload start) so legitimate
    ``<think>`` text inside suggestion content is never touched.
    """
    import re
    # Find where the JSON payload begins
    json_start = text.find("{")
    if json_start == -1:
        # No JSON yet — if the whole buffer is a <think> block, return ""
        stripped = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
        if stripped.lstrip().startswith("<think>"):
            return ""  # incomplete <think> still streaming
        return stripped
    # Only strip <think> blocks in the prefix before JSON
    prefix = text[:json_start]
    payload = text[json_start:]
    prefix = re.sub(r"<think>.*?</think>\s*", "", prefix, flags=re.DOTALL)
    # If prefix still has an incomplete <think>, drop it
    if "<think>" in prefix:
        prefix = ""
    return prefix + payload


def _extract_complete_suggestions(json_buf: str) -> list[str]:
    """Extract complete suggestion texts from a growing JSON buffer.

    Scans for complete ``{"text": "..."}`` objects inside the
    ``"suggestions"`` array.  Incomplete objects (still being streamed)
    are skipped, so only fully-received suggestions are returned.

    Automatically strips Qwen3 ``<think>`` blocks that some providers
    emit before the JSON payload.
    """
    json_buf = _strip_think_tags(json_buf)
    marker = '"suggestions"'
    idx = json_buf.find(marker)
    if idx == -1:
        return []

    bracket_idx = json_buf.find("[", idx + len(marker))
    if bracket_idx == -1:
        return []

    texts: list[str] = []
    pos = bracket_idx + 1

    while pos < len(json_buf):
        # Skip whitespace and commas
        while pos < len(json_buf) and json_buf[pos] in " \t\n\r,":
            pos += 1
        if pos >= len(json_buf) or json_buf[pos] == "]":
            break
        if json_buf[pos] != "{":
            break

        # Find matching closing brace
        depth = 0
        in_string = False
        escape_next = False
        obj_end = -1
        for i in range(pos, len(json_buf)):
            c = json_buf[i]
            if escape_next:
                escape_next = False
                continue
            if c == "\\" and in_string:
                escape_next = True
                continue
            if c == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    obj_end = i
                    break

        if obj_end == -1:
            break  # Incomplete object — still streaming

        obj_str = json_buf[pos : obj_end + 1]
        try:
            obj = json.loads(obj_str)
            if isinstance(obj, dict) and "text" in obj:
                texts.append(obj["text"])
        except json.JSONDecodeError:
            pass
        pos = obj_end + 1

    return texts
