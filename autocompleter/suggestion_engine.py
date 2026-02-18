"""Suggestion Engine - generates contextual completions via LLM API.

Receives current input + sliced context from the context store,
calls an external LLM API with short max token limit, and returns
1-3 short completions. Includes debouncing to avoid excessive API calls.
"""

import logging
import time
from dataclasses import dataclass

from .config import Config

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a contextual autocomplete assistant. Your job is to suggest short, \
relevant completions for text the user is currently typing in a chat interface.

Rules:
- Generate exactly {num_suggestions} distinct suggestions
- Each suggestion should be 1-2 sentences maximum
- Suggestions should naturally continue or complete the user's current input
- Use the provided context to make suggestions relevant to the conversation
- Do not repeat what the user has already typed
- Do not include meta-commentary, just the completion text
- Separate each suggestion with the delimiter: ---SUGGESTION---
- Output ONLY the suggestions separated by the delimiter, nothing else
"""

USER_PROMPT_TEMPLATE = """\
Context from the current session:
{context}

Currently typing in: {app_name}
Current input so far:
{current_input}

Generate {num_suggestions} short completions for this input.\
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
                api_key=self.config.anthropic_api_key
            )
        return self._anthropic_client

    def _get_openai_client(self):
        if self._openai_client is None:
            import openai

            self._openai_client = openai.OpenAI(
                api_key=self.config.openai_api_key
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
    ) -> list[Suggestion]:
        """Generate completion suggestions using the configured LLM.

        Args:
            current_input: The text the user has typed so far.
            context: Sliced context from the context store.
            app_name: Name of the app where the user is typing.

        Returns:
            A list of Suggestion objects.
        """
        if not self.can_request():
            logger.debug("Debounce: skipping request (too soon)")
            return []

        if not current_input.strip():
            return []

        self._last_request_time = time.time()

        system = SYSTEM_PROMPT.format(
            num_suggestions=self.config.num_suggestions
        )
        user_msg = USER_PROMPT_TEMPLATE.format(
            context=context or "(no context yet)",
            app_name=app_name,
            current_input=current_input,
            num_suggestions=self.config.num_suggestions,
        )

        try:
            if self.config.llm_provider == "anthropic":
                return self._call_anthropic(system, user_msg)
            elif self.config.llm_provider == "openai":
                return self._call_openai(system, user_msg)
            else:
                logger.error(f"Unknown LLM provider: {self.config.llm_provider}")
                return []
        except Exception:
            logger.exception("Error generating suggestions")
            return []

    def _call_anthropic(
        self, system: str, user_msg: str
    ) -> list[Suggestion]:
        client = self._get_anthropic_client()
        response = client.messages.create(
            model=self.config.llm_model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            system=system,
            messages=[{"role": "user", "content": user_msg}],
        )
        text = response.content[0].text
        return self._parse_suggestions(text)

    def _call_openai(self, system: str, user_msg: str) -> list[Suggestion]:
        client = self._get_openai_client()
        response = client.chat.completions.create(
            model=self.config.llm_model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
        )
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
