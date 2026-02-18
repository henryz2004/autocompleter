"""Suggestion Engine - generates contextual completions via LLM API.

Receives current input + sliced context from the context store,
calls an external LLM API with short max token limit, and returns
1-3 short completions. Includes debouncing to avoid excessive API calls.
"""

from __future__ import annotations

import enum
import logging
import time
from dataclasses import dataclass
from typing import Optional

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


@dataclass
class Suggestion:
    text: str
    index: int


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
    ) -> list[Suggestion]:
        """Generate completion suggestions using the configured LLM.

        Args:
            current_input: The text the user has typed so far.
            context: Sliced context from the context store.
            app_name: Name of the app where the user is typing.
            mode: Explicit mode override. If None, inferred from current_input.

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
            mode = detect_mode(current_input)

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
            system = SYSTEM_PROMPT_REPLY.format(num_suggestions=num)
            user_msg = USER_PROMPT_TEMPLATE_REPLY.format(
                context=ctx, num_suggestions=num,
            )
            temperature = self.config.reply_temperature
            max_tokens = self.config.reply_max_tokens

        try:
            if self.config.llm_provider == "anthropic":
                return self._call_anthropic(
                    system, user_msg, temperature=temperature, max_tokens=max_tokens,
                )
            elif self.config.llm_provider == "openai":
                return self._call_openai(
                    system, user_msg, temperature=temperature, max_tokens=max_tokens,
                )
            else:
                logger.error(f"Unknown LLM provider: {self.config.llm_provider}")
                return []
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


# ---- Mode detection (module-level for reuse) ----

MODE_THRESHOLD_CHARS = 5


def detect_mode(current_input: str) -> AutocompleteMode:
    """Determine autocomplete mode from the current input text.

    Continuation: input has meaningful draft text (>= MODE_THRESHOLD_CHARS).
    Reply: input is empty or very short (< MODE_THRESHOLD_CHARS).
    """
    if len(current_input.strip()) >= MODE_THRESHOLD_CHARS:
        return AutocompleteMode.CONTINUATION
    return AutocompleteMode.REPLY
