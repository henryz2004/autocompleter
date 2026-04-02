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
import time
from dataclasses import dataclass
from typing import Generator, Optional

from pydantic import BaseModel, Field

from .config import Config

# ---- Shell / terminal app detection ----

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

SYSTEM_PROMPT_COMPLETION = """\
You are a ghostwriter continuing the user's text. Write AS the user, in \
their voice, continuing their exact train of thought. Your output will be \
spliced directly onto what they have already written.

Rules:
- Continue the same sentence, paragraph, or thought — pick up exactly \
where the cursor is
- If the text before the cursor is mid-sentence or mid-question, CONTINUE \
the sentence/question — do NOT answer it or provide advice
- Write in the SAME voice, person, and perspective as the existing text \
(if they write "I think...", continue as "I", never comment from outside)
- MATCH THE USER'S EXACT TONE: if they write casually (lowercase, slang, \
abbreviations), continue casually. If formal, stay formal. Mirror their \
style precisely — word choice, punctuation, capitalization, everything.
- Do not restate, summarize, or comment on text before the cursor
- Do not introduce new topics or tangents
- Keep completions SHORT — a few words to half a sentence
- Never generate more than ~15 words per completion
- Vary direction across suggestions: branch the thought in different \
plausible ways (e.g. different word choices, different next clauses, \
different emphasis)
- Output ONLY the continuation text, nothing else
"""

SYSTEM_PROMPT_REPLY = """\
You are a message suggestion assistant. You suggest messages the user \
might type next.

Rules:
- If the user has a "Draft so far", output ONLY the remaining text to \
COMPLETE their draft — your output will be appended directly after what \
they already typed. Do NOT repeat any part of the draft.
- If there is no draft or the draft is empty, suggest complete replies \
to the latest message
- If there is no conversation yet, suggest conversation starters \
appropriate for the app
- MATCH THE USER'S EXACT TONE from the conversation. If the user writes \
casually (lowercase, no punctuation, slang, abbreviations), suggest in \
that same casual style. If formal, stay formal. Mirror how the USER \
actually writes, not the assistant.
- Keep suggestions SHORT — 1 sentence max, like a real text message
- Focus on the most recent messages in the conversation — older messages \
are background context, not topics to address directly
- Vary approach across suggestions (e.g. agree, ask follow-up, push back) \
but keep them relevant to the latest exchange
- Do not repeat or quote content already visible
- Output ONLY the message text, no meta-commentary or descriptions
- Never refuse to generate suggestions
"""

USER_PROMPT_TEMPLATE_COMPLETION = """\
{context}

Continue writing from the cursor position as the same author. Generate \
exactly {num_suggestions} distinct, short, natural continuations (a few words each).\
"""

USER_PROMPT_TEMPLATE_REPLY = """\
{context}

Generate exactly {num_suggestions} distinct suggestions for what the \
user might type next. If a "Draft so far" is shown, output ONLY the \
remaining text to complete it (your output is appended directly). \
If there is no draft, suggest complete replies or conversation starters. \
Keep each suggestion to 1 short sentence max — like a real text message.\
"""

# ---- Shell-specific prompts ----

SYSTEM_PROMPT_SHELL_COMPLETION = """\
You are a shell command completion assistant. Complete the command the \
user is currently typing in their terminal.

Rules:
- Output ONLY the remaining text to complete the command — your output \
will be spliced directly onto what they have already typed
- Suggest flags, arguments, file paths, or subcommands as appropriate
- Keep completions SHORT — finish the current command, don't chain new ones
- Use the command history for context (e.g. repeat similar flags, \
continue a workflow)
- Match the user's shell style (aliases, flag styles, quoting conventions)
- Never output explanations, comments, or prose — only shell syntax
"""

SYSTEM_PROMPT_SHELL_REPLY = """\
You are a shell command suggestion assistant. Suggest commands the user \
might want to run next in their terminal.

Rules:
- Suggest complete, runnable shell commands
- Use the command history and output to infer what the user is doing \
and suggest logical next steps
- If a previous command failed, suggest a fix or alternative
- Keep suggestions short — one command per suggestion (pipes are OK)
- Never output explanations or prose — only shell commands
"""

USER_PROMPT_TEMPLATE_SHELL_COMPLETION = """\
{context}

Complete the command being typed. Generate exactly {num_suggestions} \
distinct completions. Output ONLY the remaining text to append \
(not the part already typed).\
"""

USER_PROMPT_TEMPLATE_SHELL_REPLY = """\
{context}

Suggest exactly {num_suggestions} distinct commands the user might want \
to run next. Each suggestion should be a complete, runnable command.\
"""


MAX_PREVIEW_LENGTH = 80


def build_messages(
    mode: AutocompleteMode,
    context: str,
    num_suggestions: int = 3,
    max_suggestion_lines: int = 10,
    streaming: bool = False,
    source_app: str = "",
    shell_mode: Optional[bool] = None,
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
        source_app: Name of the source application. If a shell app,
            shell-specific prompts are used instead of the generic ones.
        shell_mode: Explicit override for shell prompt selection. If None,
            inferred from source_app. Pass False to force generic prompts
            even for shell apps (e.g. when inside a TUI like Claude Code).

    Returns:
        (system_prompt, user_message) tuple.
    """
    ctx = context or "(no context yet)"

    use_shell = shell_mode if shell_mode is not None else is_shell_app(source_app)
    if use_shell:
        # Shell-specific prompts
        if mode == AutocompleteMode.CONTINUATION:
            system = SYSTEM_PROMPT_SHELL_COMPLETION
            user_msg = USER_PROMPT_TEMPLATE_SHELL_COMPLETION.format(
                context=ctx, num_suggestions=num_suggestions,
            )
        else:
            system = SYSTEM_PROMPT_SHELL_REPLY
            user_msg = USER_PROMPT_TEMPLATE_SHELL_REPLY.format(
                context=ctx, num_suggestions=num_suggestions,
            )
    elif mode == AutocompleteMode.CONTINUATION:
        system = SYSTEM_PROMPT_COMPLETION
        user_msg = USER_PROMPT_TEMPLATE_COMPLETION.format(
            context=ctx, num_suggestions=num_suggestions,
        )
    else:
        system = SYSTEM_PROMPT_REPLY
        user_msg = USER_PROMPT_TEMPLATE_REPLY.format(
            context=ctx, num_suggestions=num_suggestions,
            max_suggestion_lines=max_suggestion_lines,
        )
    if streaming:
        system += STREAMING_JSON_INSTRUCTION
    return system, user_msg


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

            if self.config.llm_provider == "anthropic":
                import anthropic
                raw = anthropic.Anthropic(api_key=self.config.anthropic_api_key)
                self._client = instructor.from_anthropic(raw)
            else:
                import openai
                kwargs: dict = {"api_key": self.config.openai_api_key}
                if self.config.llm_base_url:
                    kwargs["base_url"] = self.config.llm_base_url
                raw = openai.OpenAI(**kwargs)
                self._client = instructor.from_openai(raw)
        return self._client

    def _get_raw_client(self):
        """Get a raw provider client for true text streaming."""
        if self._raw_client is None:
            if self.config.llm_provider == "anthropic":
                import anthropic
                self._raw_client = anthropic.Anthropic(
                    api_key=self.config.anthropic_api_key,
                )
            else:
                import openai
                kwargs: dict = {"api_key": self.config.openai_api_key}
                if self.config.llm_base_url:
                    kwargs["base_url"] = self.config.llm_base_url
                self._raw_client = openai.OpenAI(**kwargs)
        return self._raw_client

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
        shell_mode: Optional[bool] = None,
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
            shell_mode: Explicit override for shell prompt selection. If None,
                inferred from app_name. Pass False to force generic prompts.

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

        _use_shell = shell_mode if shell_mode is not None else is_shell_app(app_name)

        system, user_msg = build_messages(
            mode=mode,
            context=context,
            num_suggestions=self.config.num_suggestions,
            max_suggestion_lines=getattr(self.config, "max_suggestion_lines", 10),
            streaming=False,
            source_app=app_name,
            shell_mode=_use_shell,
        )

        if mode == AutocompleteMode.CONTINUATION:
            temperature = self.config.continuation_temperature
        else:
            temperature = self.config.reply_temperature

        # Override temperature for shell apps (lower = more precise commands)
        if _use_shell:
            temperature = 0.2 if mode == AutocompleteMode.CONTINUATION else 0.5

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
            f"--- LLM REQUEST ({self.config.llm_provider}/{self.config.llm_model}, "
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
        shell_mode: Optional[bool] = None,
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
            shell_mode: Explicit override for shell prompt selection. If None,
                inferred from app_name. Pass False to force generic prompts.

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

        _use_shell = shell_mode if shell_mode is not None else is_shell_app(app_name)

        system, user_msg = build_messages(
            mode=mode,
            context=context,
            num_suggestions=self.config.num_suggestions,
            max_suggestion_lines=getattr(self.config, "max_suggestion_lines", 10),
            streaming=True,
            source_app=app_name,
            shell_mode=_use_shell,
        )

        if mode == AutocompleteMode.CONTINUATION:
            temperature = self.config.continuation_temperature
        else:
            temperature = self.config.reply_temperature

        # Override temperature for shell apps
        if _use_shell:
            temperature = 0.2 if mode == AutocompleteMode.CONTINUATION else 0.5

        max_tokens = self.config.max_tokens

        # Apply feedback-based temperature adjustment
        if feedback_stats is not None:
            temperature = adjust_temperature(temperature, feedback_stats.get("accept_rate", 0.5))

        # Append negative patterns to system prompt
        if negative_patterns:
            avoided = "\n".join(f"- {p}" for p in negative_patterns)
            system += (
                "\n\nAvoid generating suggestions similar to these recently "
                "dismissed completions:\n" + avoided
            )

        logger.debug(
            f"--- LLM STREAM REQUEST ({self.config.llm_provider}/{self.config.llm_model}, "
            f"mode={mode.value}, temp={temperature}, max_tok={max_tokens}) ---"
        )

        try:
            yield from self._call_llm_stream(
                system, user_msg, temperature=temperature, max_tokens=max_tokens,
            )
        except Exception as exc:
            if self._is_rate_limit_error(exc):
                self._handle_rate_limit()
                yield Suggestion(text="Rate limited — try again later", index=0)
            else:
                logger.exception("Error during streaming suggestions")
            return

    def _call_llm_stream(
        self, system: str, user_msg: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Generator[Suggestion, None, None]:
        """Stream suggestions via raw provider SDK for true incremental delivery.

        Uses plain-text JSON output (not tool calls) to bypass Anthropic's
        tool_use key-value buffering and Instructor's sync list() collection.
        Parses the growing JSON buffer incrementally and yields each Suggestion
        as soon as its JSON object is complete.

        Falls back to the blocking _call_llm path on error or if the stream
        produces no suggestions.
        """
        client = self._get_raw_client()
        # NOTE: system already includes STREAMING_JSON_INSTRUCTION via
        # build_messages(streaming=True); no need to append it again.
        json_system = system

        try:
            json_buf = ""
            last_yielded = 0

            if self.config.llm_provider == "anthropic":
                with client.messages.stream(
                    model=self.config.llm_model,
                    system=json_system,
                    messages=[{"role": "user", "content": user_msg}],
                    temperature=temperature,
                    max_tokens=max_tokens or 1024,
                ) as stream:
                    for text_chunk in stream.text_stream:
                        json_buf += text_chunk
                        complete = _extract_complete_suggestions(json_buf)
                        while len(complete) > last_yielded:
                            s = complete[last_yielded]
                            logger.debug(
                                f"Stream yielding suggestion [{last_yielded}]: {s[:80]!r}"
                            )
                            yield Suggestion(text=s, index=last_yielded)
                            last_yielded += 1
            else:
                # OpenAI
                response = client.chat.completions.create(
                    model=self.config.llm_model,
                    messages=[
                        {"role": "system", "content": json_system},
                        {"role": "user", "content": user_msg},
                    ],
                    stream=True,
                    temperature=temperature,
                    max_tokens=max_tokens or 1024,
                )
                for chunk in response:
                    delta = chunk.choices[0].delta if chunk.choices else None
                    content = delta.content if delta else None
                    if content:
                        json_buf += content
                        complete = _extract_complete_suggestions(json_buf)
                        while len(complete) > last_yielded:
                            s = complete[last_yielded]
                            logger.debug(
                                f"Stream yielding suggestion [{last_yielded}]: {s[:80]!r}"
                            )
                            yield Suggestion(text=s, index=last_yielded)
                            last_yielded += 1

            if last_yielded == 0:
                logger.warning(
                    "Stream produced no suggestions, falling back to blocking"
                )
                results = self._call_llm(
                    system, user_msg, temperature=temperature, max_tokens=max_tokens,
                )
                for suggestion in results:
                    yield suggestion

        except Exception as exc:
            if self._is_rate_limit_error(exc):
                self._handle_rate_limit()
                yield Suggestion(text="Rate limited — try again later", index=0)
                return
            logger.warning(
                "Raw streaming failed, falling back to blocking", exc_info=True,
            )
            try:
                results = self._call_llm(
                    system, user_msg, temperature=temperature, max_tokens=max_tokens,
                )
                for suggestion in results:
                    yield suggestion
            except Exception as fallback_exc:
                if self._is_rate_limit_error(fallback_exc):
                    self._handle_rate_limit()
                    yield Suggestion(text="Rate limited — try again later", index=0)
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


def _extract_complete_suggestions(json_buf: str) -> list[str]:
    """Extract complete suggestion texts from a growing JSON buffer.

    Scans for complete ``{"text": "..."}`` objects inside the
    ``"suggestions"`` array.  Incomplete objects (still being streamed)
    are skipped, so only fully-received suggestions are returned.
    """
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
