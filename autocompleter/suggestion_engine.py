"""Suggestion Engine - generates contextual completions via LLM API.

Receives current input + sliced context from the context store,
calls an external LLM API with short max token limit, and returns
1-3 short completions. Includes debouncing to avoid excessive API calls.

Supports both blocking (generate_suggestions) and streaming
(generate_suggestions_stream) modes. The streaming mode yields
Suggestion objects one at a time as delimiters are encountered in
the token stream, enabling the overlay to update incrementally.
"""

from __future__ import annotations

import enum
import logging
import time
from dataclasses import dataclass
from typing import Generator, Optional

from .config import Config


class AutocompleteMode(enum.Enum):
    """Determines how context is assembled and what kind of suggestion to generate."""
    CONTINUATION = "continuation"  # User has draft text, predict next words
    REPLY = "reply"               # Input is empty/short, suggest a full response

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_COMPLETION = """\
You are a text completion engine. Complete text at the cursor position only.

Rules:
- Generate exactly {num_suggestions} distinct completions
- Complete naturally from the cursor — do not restate text before the cursor
- Do not introduce new topics or tangents
- Preserve the existing formatting, tone, and style
- Keep completion length proportional to what's already written
- Each completion should be short (a few words to 1-2 sentences max)
- Separate each completion with the delimiter: ---SUGGESTION---
- Output ONLY the completions separated by the delimiter, nothing else
"""

SYSTEM_PROMPT_REPLY = """\
You are a conversational reply assistant. Suggest messages the user might \
send as their next response in the conversation.

Rules:
- Generate exactly {num_suggestions} distinct reply suggestions
- Respond to the latest message in the conversation
- Match the tone and formality of the conversation
- Do not invent facts or context not present in the conversation
- Keep length proportional to the conversation thread
- Vary intent across suggestions (e.g. agree, ask follow-up, provide info)
- Do not repeat or quote content already in the conversation
- For longer conversations or email-like contexts, suggestions may span \
multiple sentences or paragraphs (up to ~{max_suggestion_lines} lines)
- Separate each suggestion with the delimiter: ---SUGGESTION---
- Output ONLY the suggestions separated by the delimiter, nothing else
"""

USER_PROMPT_TEMPLATE_COMPLETION = """\
{context}

Complete the text at the cursor position. Generate {num_suggestions} \
short, natural completions.\
"""

USER_PROMPT_TEMPLATE_REPLY = """\
{context}

Generate {num_suggestions} short reply suggestions the user might send next.\
"""


MAX_PREVIEW_LENGTH = 80


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


class SuggestionEngine:
    def __init__(self, config: Config):
        self.config = config
        self._last_request_time: float = 0.0
        self._anthropic_client = None
        self._openai_client = None

    def _get_anthropic_client(self):
        if self._anthropic_client is None:
            import anthropic

            self._anthropic_client = anthropic.Anthropic(
                api_key=self.config.anthropic_api_key,
                timeout=10.0,
            )
        return self._anthropic_client

    def _get_openai_client(self):
        if self._openai_client is None:
            import openai

            self._openai_client = openai.OpenAI(
                api_key=self.config.openai_api_key,
                timeout=10.0,
            )
        return self._openai_client

    def can_request(self) -> bool:
        """Check if enough time has passed since the last request (debounce)."""
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

        num = self.config.num_suggestions
        ctx = context or "(no context yet)"

        if mode == AutocompleteMode.CONTINUATION:
            system = SYSTEM_PROMPT_COMPLETION.format(num_suggestions=num)
            user_msg = USER_PROMPT_TEMPLATE_COMPLETION.format(
                context=ctx, num_suggestions=num,
            )
            temperature = self.config.continuation_temperature
            max_tokens = self.config.continuation_max_tokens
        else:
            max_lines = getattr(self.config, "max_suggestion_lines", 10)
            system = SYSTEM_PROMPT_REPLY.format(
                num_suggestions=num, max_suggestion_lines=max_lines,
            )
            user_msg = USER_PROMPT_TEMPLATE_REPLY.format(
                context=ctx, num_suggestions=num,
            )
            temperature = self.config.reply_temperature
            max_tokens = self.config.reply_max_tokens

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
            if self.config.llm_provider == "anthropic":
                results = self._call_anthropic(
                    system, user_msg, temperature=temperature, max_tokens=max_tokens,
                )
            elif self.config.llm_provider == "openai":
                results = self._call_openai(
                    system, user_msg, temperature=temperature, max_tokens=max_tokens,
                )
            else:
                logger.error(f"Unknown LLM provider: {self.config.llm_provider}")
                return []
            logger.debug(f"LLM returned {len(results)} suggestions")
            return results
        except Exception:
            logger.exception("Error generating suggestions")
            return []

    def _call_anthropic(
        self, system: str, user_msg: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> list[Suggestion]:
        client = self._get_anthropic_client()
        response = client.messages.create(
            model=self.config.llm_model,
            max_tokens=max_tokens or self.config.max_tokens,
            temperature=temperature if temperature is not None else self.config.temperature,
            system=system,
            messages=[{"role": "user", "content": user_msg}],
        )
        if not response.content:
            return []
        text = response.content[0].text
        return self._parse_suggestions(text)

    def _call_openai(
        self, system: str, user_msg: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> list[Suggestion]:
        client = self._get_openai_client()
        response = client.chat.completions.create(
            model=self.config.llm_model,
            max_tokens=max_tokens or self.config.max_tokens,
            temperature=temperature if temperature is not None else self.config.temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
        )
        if not response.choices:
            return []
        text = response.choices[0].message.content or ""
        return self._parse_suggestions(text)

    @staticmethod
    def _parse_suggestions(raw_text: str) -> list[Suggestion]:
        """Parse the LLM response into individual suggestions."""
        parts = raw_text.split("---SUGGESTION---")
        suggestions = []
        for i, part in enumerate(parts):
            text = part.strip()
            if text:
                suggestions.append(Suggestion(text=text, index=i))
        return suggestions

    # ---- Streaming API ----

    _DELIMITER = "---SUGGESTION---"

    def generate_suggestions_stream(
        self,
        current_input: str,
        context: str,
        app_name: str = "Unknown",
        mode: Optional[AutocompleteMode] = None,
        before_cursor: Optional[str] = None,
        feedback_stats: Optional[dict] = None,
        negative_patterns: Optional[list[str]] = None,
    ) -> Generator[Suggestion, None, None]:
        """Generate completion suggestions via streaming, yielding each as it completes.

        Yields Suggestion objects one at a time as they become available
        from the LLM token stream. Each suggestion is emitted as soon as the
        ``---SUGGESTION---`` delimiter (or end-of-stream) is encountered.

        Args:
            current_input: The text the user has typed so far.
            context: Sliced context from the context store.
            app_name: Name of the app where the user is typing.
            mode: Explicit mode override. If None, inferred from before_cursor.
            before_cursor: Text before the cursor. Used for mode detection
                when mode is None. Falls back to current_input if not provided.
            feedback_stats: Optional dict from ContextStore.get_feedback_stats().
            negative_patterns: Optional list of recently dismissed suggestion texts.

        Yields:
            Suggestion objects, in the order they are completed by the LLM.
        """
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

        num = self.config.num_suggestions
        ctx = context or "(no context yet)"

        if mode == AutocompleteMode.CONTINUATION:
            system = SYSTEM_PROMPT_COMPLETION.format(num_suggestions=num)
            user_msg = USER_PROMPT_TEMPLATE_COMPLETION.format(
                context=ctx, num_suggestions=num,
            )
            temperature = self.config.continuation_temperature
            max_tokens = self.config.continuation_max_tokens
        else:
            max_lines = getattr(self.config, "max_suggestion_lines", 10)
            system = SYSTEM_PROMPT_REPLY.format(
                num_suggestions=num, max_suggestion_lines=max_lines,
            )
            user_msg = USER_PROMPT_TEMPLATE_REPLY.format(
                context=ctx, num_suggestions=num,
            )
            temperature = self.config.reply_temperature
            max_tokens = self.config.reply_max_tokens

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
            if self.config.llm_provider == "anthropic":
                yield from self._call_anthropic_stream(
                    system, user_msg, temperature=temperature, max_tokens=max_tokens,
                )
            elif self.config.llm_provider == "openai":
                yield from self._call_openai_stream(
                    system, user_msg, temperature=temperature, max_tokens=max_tokens,
                )
            else:
                logger.error(f"Unknown LLM provider: {self.config.llm_provider}")
                return
        except Exception:
            logger.exception("Error during streaming suggestions")
            return

    def _call_anthropic_stream(
        self, system: str, user_msg: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Generator[Suggestion, None, None]:
        """Stream tokens from Anthropic and yield suggestions at delimiters."""
        client = self._get_anthropic_client()
        suggestion_index = 0
        buffer = ""

        try:
            with client.messages.stream(
                model=self.config.llm_model,
                max_tokens=max_tokens or self.config.max_tokens,
                temperature=temperature if temperature is not None else self.config.temperature,
                system=system,
                messages=[{"role": "user", "content": user_msg}],
            ) as stream:
                for text in stream.text_stream:
                    buffer += text
                    # Check for complete delimiters in the buffer
                    while self._DELIMITER in buffer:
                        before, _, buffer = buffer.partition(self._DELIMITER)
                        text_stripped = before.strip()
                        if text_stripped:
                            logger.debug(
                                f"Stream yielding suggestion [{suggestion_index}]: "
                                f"{text_stripped[:80]!r}"
                            )
                            yield Suggestion(text=text_stripped, index=suggestion_index)
                            suggestion_index += 1
        except Exception:
            logger.exception("Error during Anthropic stream")
            # Fall through to yield whatever is in the buffer

        # Yield any remaining text after the stream ends
        remaining = buffer.strip()
        if remaining:
            logger.debug(
                f"Stream yielding final suggestion [{suggestion_index}]: "
                f"{remaining[:80]!r}"
            )
            yield Suggestion(text=remaining, index=suggestion_index)

    def _call_openai_stream(
        self, system: str, user_msg: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Generator[Suggestion, None, None]:
        """Stream tokens from OpenAI and yield suggestions at delimiters."""
        client = self._get_openai_client()
        suggestion_index = 0
        buffer = ""

        try:
            response = client.chat.completions.create(
                model=self.config.llm_model,
                max_tokens=max_tokens or self.config.max_tokens,
                temperature=temperature if temperature is not None else self.config.temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_msg},
                ],
                stream=True,
            )
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    buffer += token
                    # Check for complete delimiters in the buffer
                    while self._DELIMITER in buffer:
                        before, _, buffer = buffer.partition(self._DELIMITER)
                        text_stripped = before.strip()
                        if text_stripped:
                            logger.debug(
                                f"Stream yielding suggestion [{suggestion_index}]: "
                                f"{text_stripped[:80]!r}"
                            )
                            yield Suggestion(text=text_stripped, index=suggestion_index)
                            suggestion_index += 1
        except Exception:
            logger.exception("Error during OpenAI stream")
            # Fall through to yield whatever is in the buffer

        # Yield any remaining text after the stream ends
        remaining = buffer.strip()
        if remaining:
            logger.debug(
                f"Stream yielding final suggestion [{suggestion_index}]: "
                f"{remaining[:80]!r}"
            )
            yield Suggestion(text=remaining, index=suggestion_index)


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
