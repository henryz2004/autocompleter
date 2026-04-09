"""App-specific conversation extractors for chat-like UIs.

Provides a plugin-like system where known apps (Gemini, ChatGPT, Claude Desktop,
Slack, iMessage) have dedicated extractors for pulling structured conversation
turns from their Accessibility API trees. Falls back to a generic heuristic
for unknown apps.

Also defines ``ConversationTurn`` (the shared data class) and the
``_collect_child_text`` helper so that both this module and
``input_observer`` can use them without a circular import.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from .ax_utils import ax_get_attribute, ax_get_children

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared data class
# ---------------------------------------------------------------------------

@dataclass
class ConversationTurn:
    """A single message in a conversation."""
    speaker: str
    text: str
    timestamp: str = ""  # e.g. "10:05 PM", "12/14/25, 11:22 AM", "Yesterday at 8:47 PM"


# ---------------------------------------------------------------------------
# Shared AX text-collection helper
# ---------------------------------------------------------------------------

# Roles to skip entirely during child-text collection (UI chrome)
_CHILD_TEXT_SKIP_ROLES = frozenset({
    "AXToolbar", "AXMenuBar", "AXMenu", "AXMenuItem",
    "AXButton", "AXScrollBar", "AXSlider", "AXIncrementor",
    "AXPopUpButton", "AXCheckBox", "AXRadioButton",
    "AXTabGroup", "AXTab",
})

_MAX_CHILD_TEXT_SIBLINGS = 80  # Max children to iterate per node

# Subroles to skip entirely (UI chrome that isn't conversation content)
_CHILD_TEXT_SKIP_SUBROLES = frozenset({
    "AXDocumentNote",        # Footer disclaimers ("Claude/ChatGPT can make mistakes...")
    "AXApplicationStatus",   # Status/loading indicators
    "AXApplicationAlert",    # Alert banners
})

# Common AI disclaimer patterns to filter from extracted conversation text.
# These appear as footer text in chat apps but sometimes lack proper AX subroles.
_AI_DISCLAIMER_PATTERNS = (
    "can make mistakes",
    "can be inaccurate",
    "may produce inaccurate",
    "double-check responses",
    "verify important information",
)


def _is_ai_disclaimer(text: str) -> bool:
    """Return True if text looks like an AI disclaimer footer."""
    lower = text.strip().lower()
    if len(lower) > 200:
        return False
    return any(pat in lower for pat in _AI_DISCLAIMER_PATTERNS)


def _collect_child_text(
    element, max_depth: int = 5, max_chars: int = 2000, depth: int = 0
) -> str:
    """Collect text from child elements.

    Concatenates adjacent AXStaticText siblings (handles Electron apps that
    fragment text into single-character elements). Separates text from
    different container elements with newlines. Skips UI chrome roles.
    """
    if depth > max_depth:
        return ""

    parts: list[str] = []
    total = 0
    children = ax_get_children(element)
    if children:
        # Buffer adjacent AXStaticText values to concatenate fragments
        text_buffer: list[str] = []
        for child in children[:_MAX_CHILD_TEXT_SIBLINGS]:
            if total >= max_chars:
                break
            role = ax_get_attribute(child, "AXRole") or ""
            if role in _CHILD_TEXT_SKIP_ROLES:
                continue
            if role == "AXStaticText":
                val = ax_get_attribute(child, "AXValue")
                if isinstance(val, str) and val.strip():
                    text_buffer.append(val)
                    total += len(val)
                else:
                    # Electron apps (ChatGPT) put text in AXDescription
                    desc = ax_get_attribute(child, "AXDescription")
                    if isinstance(desc, str) and desc.strip():
                        text_buffer.append(desc)
                        total += len(desc)
            else:
                # Flush buffered adjacent text fragments
                if text_buffer:
                    joined = "".join(text_buffer).strip()
                    if joined:
                        parts.append(joined)
                    text_buffer = []
                # Skip chrome subroles (disclaimers, status indicators, alerts)
                if role == "AXGroup":
                    subrole = ax_get_attribute(child, "AXSubrole") or ""
                    if subrole in _CHILD_TEXT_SKIP_SUBROLES:
                        continue
                child_text = _collect_child_text(
                    child, max_depth, max_chars - total, depth + 1
                )
                if child_text:
                    parts.append(child_text)
                    total += len(child_text)
        # Flush remaining buffer
        if text_buffer:
            joined = "".join(text_buffer).strip()
            if joined:
                parts.append(joined)

    return "\n".join(parts)


class ConversationExtractor(ABC):
    """Base class for app-specific conversation extractors."""

    app_names: tuple[str, ...] = ()  # App names this extractor handles

    @abstractmethod
    def extract(
        self, window_element, max_turns: int = 15
    ) -> Optional[list[ConversationTurn]]:
        """Extract conversation turns from the window.

        Returns a list of ConversationTurn objects, or None if extraction fails
        or the window does not contain a recognizable conversation.
        """
        ...


class GenericExtractor(ConversationExtractor):
    """Generic conversation extractor using heuristics.

    Walks the AX tree looking for message-like groups: containers with child
    elements where one is short (speaker name, <= 40 chars) and another is
    longer (message body, > 5 chars).

    This is the fallback used for unknown apps.
    """

    app_names: tuple[str, ...] = ()  # Matches no specific app; used as fallback

    _MAX_MSG_VISITS = 600
    _MAX_CHILDREN_PER_NODE = 50

    def extract(
        self, window_element, max_turns: int = 15
    ) -> Optional[list[ConversationTurn]]:
        try:
            turns = self._walk_for_messages(window_element, max_turns, max_depth=8)
            if len(turns) >= 2:
                return turns
        except Exception:
            logger.debug("Generic conversation extraction failed", exc_info=True)
        return None

    def _walk_for_messages(
        self,
        element,
        max_turns: int,
        max_depth: int,
        depth: int = 0,
        _visits: Optional[list] = None,
    ) -> list[ConversationTurn]:
        """Recursively search for message-like group elements."""
        if _visits is None:
            _visits = [0]
        _visits[0] += 1
        if _visits[0] > self._MAX_MSG_VISITS:
            return []
        if depth > max_depth:
            return []

        role = ax_get_attribute(element, "AXRole") or ""
        turns: list[ConversationTurn] = []

        # Look for groups/cells that might be message containers
        if role in {"AXGroup", "AXCell", "AXRow"}:
            turn = self._try_parse_message_group(element)
            if turn is not None:
                turns.append(turn)
                if len(turns) >= max_turns:
                    return turns

        children = ax_get_children(element)
        if children:
            for child in children[: self._MAX_CHILDREN_PER_NODE]:
                if len(turns) >= max_turns:
                    break
                child_turns = self._walk_for_messages(
                    child, max_turns - len(turns), max_depth, depth + 1, _visits
                )
                turns.extend(child_turns)

        return turns

    def _try_parse_message_group(self, element) -> Optional[ConversationTurn]:
        """Check if an AX element looks like a chat message container.

        Heuristic: a group containing at least two text children where one
        is short (likely a speaker name, <= 40 chars) and another is longer
        (likely the message body, > 5 chars).
        """
        children = ax_get_children(element)
        if not children or len(children) < 2:
            return None

        texts: list[tuple[str, str]] = []  # (role, value)
        for child in children[:10]:  # limit scan
            child_role = ax_get_attribute(child, "AXRole") or ""
            if child_role == "AXStaticText":
                val = ax_get_attribute(child, "AXValue")
                if isinstance(val, str) and val.strip():
                    texts.append((child_role, val.strip()))
                else:
                    # Fallback: Electron/native apps (ChatGPT 5.x, Slack)
                    # often put text in AXDescription instead of AXValue
                    desc = ax_get_attribute(child, "AXDescription")
                    if isinstance(desc, str) and desc.strip():
                        texts.append((child_role, desc.strip()))
            elif child_role in {"AXGroup", "AXTextArea"}:
                # Nested group might contain the message text
                nested_text = _collect_child_text(child, max_depth=3, max_chars=500)
                if nested_text.strip():
                    texts.append((child_role, nested_text.strip()))

        if len(texts) < 2:
            return None

        # Heuristic: find a short text (speaker) and a longer text (body)
        speaker = None
        body = None
        for _, text in texts:
            if len(text) <= 40 and speaker is None:
                speaker = text
            elif len(text) > 5 and body is None:
                body = text

        if speaker and body:
            return ConversationTurn(speaker=speaker, text=body)
        return None


class GeminiExtractor(ConversationExtractor):
    """Extractor for Google Gemini (web and desktop app).

    Gemini's AX tree structure:
    - AXLandmarkMain > conversation area
    - Contains a heading "Conversation with Gemini"
    - Message groups are siblings of the heading inside a container group
    - User messages and model responses alternate as AXGroup containers
    """

    app_names: tuple[str, ...] = ("Google Gemini", "Gemini")

    def extract(
        self, window_element, max_turns: int = 15
    ) -> Optional[list[ConversationTurn]]:
        try:
            # Find the main landmark area
            main_area = self._find_landmark_main(window_element, max_depth=20)
            if main_area is None:
                return None

            # Look for the conversation heading
            heading_parent = self._find_conversation_container(main_area, max_depth=8)
            if heading_parent is None:
                return None

            # Extract message groups from the container
            turns = self._extract_messages_from_container(heading_parent, max_turns)
            if turns and len(turns) >= 1:
                return turns
        except Exception:
            logger.debug("Gemini conversation extraction failed", exc_info=True)

        # Fall back to generic extraction for resilience
        return ActionDelimitedExtractor().extract(window_element, max_turns)

    def _find_landmark_main(
        self, element, max_depth: int, depth: int = 0
    ):
        """Find the AXLandmarkMain element."""
        if depth > max_depth:
            return None
        sub_role = ax_get_attribute(element, "AXSubrole") or ""
        if sub_role == "AXLandmarkMain":
            return element
        role = ax_get_attribute(element, "AXRole") or ""
        if role == "AXGroup":
            role_desc = ax_get_attribute(element, "AXRoleDescription") or ""
            if role_desc == "main":
                return element
        children = ax_get_children(element)
        if children:
            for child in children[:50]:
                result = self._find_landmark_main(child, max_depth, depth + 1)
                if result is not None:
                    return result
        return None

    def _find_conversation_container(
        self, element, max_depth: int, depth: int = 0
    ):
        """Find the container that holds the conversation heading and message groups.

        Looks for a heading containing "Conversation with Gemini" and returns
        the parent container that also holds the message groups.
        """
        if depth > max_depth:
            return None

        children = ax_get_children(element)
        if not children:
            return None

        # Check if any child is the conversation heading
        for child in children[:50]:
            role = ax_get_attribute(child, "AXRole") or ""
            if role == "AXHeading":
                heading_text = (
                    ax_get_attribute(child, "AXValue")
                    or ax_get_attribute(child, "AXTitle")
                    or ""
                )
                if not heading_text:
                    heading_text = _collect_child_text(child, max_depth=2, max_chars=200)
                if "Conversation with Gemini" in heading_text:
                    return element  # Return the parent container

        # Recurse into children
        for child in children[:50]:
            result = self._find_conversation_container(child, max_depth, depth + 1)
            if result is not None:
                return result
        return None

    def _extract_messages_from_container(
        self, container, max_turns: int
    ) -> list[ConversationTurn]:
        """Extract user/model messages using 'You said'/'Gemini said' headings.

        Gemini's AX tree groups messages into pair containers.  Each pair
        contains an AXHeading with title starting with ``You said`` (user
        message) and another with ``Gemini said`` (model response).  We
        recursively find these headings and extract text from the appropriate
        sibling/child elements.

        Falls back to index-based alternation if no speaker headings are found.
        """
        children = ax_get_children(container)
        if not children:
            return []

        # Find the child group with the most sub-children (the message list).
        # Unwrap single-child wrappers: if a group has exactly one child that
        # is also a group, use the inner group's child count instead.
        best_group = None
        best_count = 0
        for child in children[:50]:
            role = ax_get_attribute(child, "AXRole") or ""
            if role in {"AXGroup", "AXList"}:
                grandchildren = ax_get_children(child)
                count = len(grandchildren) if grandchildren else 0
                candidate = child
                # Unwrap single-child wrapper
                if count == 1 and grandchildren:
                    inner = grandchildren[0]
                    inner_role = ax_get_attribute(inner, "AXRole") or ""
                    if inner_role in {"AXGroup", "AXList"}:
                        inner_children = ax_get_children(inner)
                        inner_count = len(inner_children) if inner_children else 0
                        if inner_count > count:
                            candidate = inner
                            count = inner_count
                if count > best_count:
                    best_count = count
                    best_group = candidate

        if best_group is None or best_count == 0:
            return []

        msg_children = ax_get_children(best_group)
        if not msg_children:
            return []

        # Heading-based extraction: find "You said" / "Gemini said" headings
        turns: list[ConversationTurn] = []
        for msg_child in msg_children:
            if len(turns) >= max_turns:
                break
            self._extract_turns_from_pair(msg_child, turns, max_turns)

        # Fallback: if no headings were found, use index-based alternation
        # with marker stripping as a safety net.
        if not turns:
            for i, msg_child in enumerate(msg_children):
                if len(turns) >= max_turns:
                    break
                text = _collect_child_text(msg_child, max_depth=5, max_chars=1000)
                text = self._strip_speaker_markers(text)
                if not text or _is_ai_disclaimer(text):
                    continue
                speaker = "User" if i % 2 == 0 else "Gemini"
                turns.append(ConversationTurn(speaker=speaker, text=text))

        return turns

    def _extract_turns_from_pair(
        self, element, turns: list[ConversationTurn], max_turns: int,
    ) -> None:
        """Extract User and Gemini turns from a message pair element.

        Recursively searches for AXHeading elements whose title starts with
        ``You said`` or ``Gemini said``, then extracts the appropriate text.
        """
        headings: list[tuple] = []  # (heading, title, parent, sibling_idx)
        self._find_speaker_headings(element, headings, max_depth=12)

        if headings:
            user_count = sum(1 for _, t, _, _ in headings if t.startswith("You said"))
            gemini_count = sum(1 for _, t, _, _ in headings if t.startswith("Gemini said"))
            logger.debug(
                "[Gemini] _extract_turns_from_pair: %d headings "
                "(%d 'You said', %d 'Gemini said')",
                len(headings), user_count, gemini_count,
            )

        for heading, title, parent, sibling_idx in headings:
            if len(turns) >= max_turns:
                break

            if title.startswith("You said"):
                text = self._extract_user_text_from_heading(heading, title)
                if text and not _is_ai_disclaimer(text):
                    turns.append(ConversationTurn(speaker="User", text=text))

            elif title.startswith("Gemini said"):
                text = self._extract_response_after_heading(heading, parent, sibling_idx)
                if text and not _is_ai_disclaimer(text):
                    turns.append(ConversationTurn(speaker="Gemini", text=text))

    def _find_speaker_headings(
        self,
        element,
        results: list[tuple],
        max_depth: int,
        depth: int = 0,
        _visits: Optional[list] = None,
    ) -> None:
        """Recursively find AXHeading elements with 'You said' or 'Gemini said' titles."""
        if _visits is None:
            _visits = [0]
        _visits[0] += 1
        if _visits[0] > 300 or depth > max_depth:
            return

        children = ax_get_children(element)
        if not children:
            return

        for i, child in enumerate(children[:50]):
            role = ax_get_attribute(child, "AXRole") or ""
            if role == "AXHeading":
                title = (
                    ax_get_attribute(child, "AXTitle")
                    or ax_get_attribute(child, "AXValue")
                    or ""
                ).strip()
                if title.startswith("You said") or title.startswith("Gemini said"):
                    results.append((child, title, element, i))
            else:
                self._find_speaker_headings(
                    child, results, max_depth, depth + 1, _visits,
                )

    @staticmethod
    def _extract_user_text_from_heading(heading, title: str) -> str:
        """Extract user message text from a 'You said' heading.

        Prefers the AXGroup child of the heading (which has multi-line content)
        over the title-derived text (which may be single-line).
        """
        # Try to get text from the AXGroup child (skipping the "You said" AXStaticText)
        children = ax_get_children(heading) or []
        for child in children:
            child_role = ax_get_attribute(child, "AXRole") or ""
            if child_role == "AXGroup":
                child_text = _collect_child_text(child, max_depth=3, max_chars=1000)
                if child_text and child_text.strip():
                    return child_text.strip()

        # Fallback: strip "You said " prefix from the heading title
        prefix = "You said "
        if title.startswith(prefix):
            return title[len(prefix):].strip()
        if title == "You said":
            return ""
        return ""

    @staticmethod
    def _extract_response_after_heading(heading, parent, heading_idx: int) -> str:
        """Extract Gemini response text from siblings following a 'Gemini said' heading.

        ``heading_idx`` is the heading's position within ``parent``'s child list,
        captured by ``_find_speaker_headings`` at discovery time.  We cannot
        rediscover it via ``child is heading`` because live AX API calls return
        fresh Python wrapper objects each call, so identity comparison fails.
        """
        parent_children = ax_get_children(parent) or []

        # Sanity-check the index — children list should not have changed since
        # discovery, but if it has, fall back gracefully rather than misattribute
        # text to the wrong speaker.
        if heading_idx >= len(parent_children):
            return ""
        anchor_role = ax_get_attribute(parent_children[heading_idx], "AXRole") or ""
        if anchor_role != "AXHeading":
            return ""

        # Collect text from subsequent siblings until next heading or end
        parts: list[str] = []
        for sibling in parent_children[heading_idx + 1:]:
            sib_role = ax_get_attribute(sibling, "AXRole") or ""
            if sib_role == "AXHeading":
                break
            text = _collect_child_text(sibling, max_depth=5, max_chars=1000)
            if text and text.strip():
                parts.append(text.strip())

        return "\n".join(parts)

    @staticmethod
    def _strip_speaker_markers(text: str) -> str:
        """Strip known Gemini UI speaker markers from text (safety net)."""
        text = text.strip()
        for prefix in ("You said\n", "Gemini said\n", "You said ", "Gemini said "):
            if text.startswith(prefix):
                text = text[len(prefix):]
        for suffix in ("\nGemini said", "\nYou said"):
            if text.endswith(suffix):
                text = text[: -len(suffix)]
        return text.strip()


class SlackExtractor(ConversationExtractor):
    """Extractor for Slack.

    Slack's AX tree uses AXList > AXGroup pattern where message groups have
    AXRoleDescription set to "message".
    """

    app_names: tuple[str, ...] = ("Slack",)

    _MAX_VISITS = 600

    def extract(
        self, window_element, max_turns: int = 15
    ) -> Optional[list[ConversationTurn]]:
        try:
            turns = self._find_message_groups(window_element, max_turns, max_depth=12)
            if turns and len(turns) >= 1:
                return turns
        except Exception:
            logger.debug("Slack conversation extraction failed", exc_info=True)

        # Fall back to generic extraction for resilience
        return ActionDelimitedExtractor().extract(window_element, max_turns)

    def _find_message_groups(
        self,
        element,
        max_turns: int,
        max_depth: int,
        depth: int = 0,
        _visits: Optional[list] = None,
    ) -> list[ConversationTurn]:
        """Search for elements with AXRoleDescription 'message'."""
        if _visits is None:
            _visits = [0]
        _visits[0] += 1
        if _visits[0] > self._MAX_VISITS or depth > max_depth:
            return []

        turns: list[ConversationTurn] = []
        role_desc = ax_get_attribute(element, "AXRoleDescription") or ""

        if role_desc == "message":
            turn = self._parse_slack_message(element)
            if turn is not None:
                turns.append(turn)
                if len(turns) >= max_turns:
                    return turns

        children = ax_get_children(element)
        if children:
            for child in children[:50]:
                if len(turns) >= max_turns:
                    break
                child_turns = self._find_message_groups(
                    child, max_turns - len(turns), max_depth, depth + 1, _visits
                )
                turns.extend(child_turns)

        return turns

    def _parse_slack_message(self, element) -> Optional[ConversationTurn]:
        """Parse a Slack message group.

        Slack messages typically have:
        - A child with the sender's name (short text, often a button or static text)
        - A child group containing the message body
        """
        children = ax_get_children(element)
        if not children:
            return None

        speaker = None
        body = None

        for child in children[:15]:
            child_role = ax_get_attribute(child, "AXRole") or ""

            # Speaker name is often in a button or static text
            if child_role in {"AXButton", "AXStaticText", "AXLink"}:
                val = ax_get_attribute(child, "AXValue") or ax_get_attribute(
                    child, "AXTitle"
                ) or ""
                val = val.strip()
                if val and len(val) <= 50 and speaker is None:
                    speaker = val
            elif child_role in {"AXGroup", "AXTextArea"}:
                text = _collect_child_text(child, max_depth=4, max_chars=1000)
                text = text.strip()
                if text and len(text) > 3 and body is None:
                    body = text

        if speaker and body:
            return ConversationTurn(speaker=speaker, text=body)
        # If we found a body but no speaker, still return with "Unknown"
        if body and not speaker:
            return ConversationTurn(speaker="Unknown", text=body)
        return None


class ChatGPTExtractor(ConversationExtractor):
    """Extractor for ChatGPT (web and desktop app).

    ChatGPT uses article-like containers. Messages are in AXGroup elements
    that may have AXRoleDescription "article" or contain nested groups with
    text content. Messages alternate between user and assistant.
    """

    app_names: tuple[str, ...] = ("ChatGPT",)

    _MAX_VISITS = 600

    def extract(
        self, window_element, max_turns: int = 15
    ) -> Optional[list[ConversationTurn]]:
        try:
            # Strategy 1: article-based layout (older ChatGPT versions)
            articles = self._find_articles(window_element, max_depth=15)
            if articles:
                turns = self._articles_to_turns(articles, max_turns)
                if turns:
                    return turns

            # Strategy 2: SectionList-based layout (ChatGPT 5.x+)
            # Messages are AXGroup children of an AXSectionList, each
            # containing AXStaticText with content in AXDescription.
            message_groups = self._find_section_list_messages(
                window_element, max_depth=15,
            )
            if message_groups:
                turns = self._articles_to_turns(message_groups, max_turns)
                if turns:
                    return turns

        except Exception:
            logger.debug("ChatGPT conversation extraction failed", exc_info=True)

        # Strategy 3: fall back to generic extraction so layout changes
        # don't completely break conversation parsing.
        return ActionDelimitedExtractor().extract(window_element, max_turns)

    def _articles_to_turns(
        self, containers: list, max_turns: int,
    ) -> Optional[list[ConversationTurn]]:
        """Convert a list of message containers into ConversationTurn objects.

        Two-pass approach for speaker detection:
        1. Check which containers have action buttons (Sources, feedback, etc.)
        2. If any container has buttons → has_button = ChatGPT, no_button = User
           If none have buttons → fall back to alternation (even=User, odd=ChatGPT)
        """
        # First pass: collect text and detect buttons
        entries: list[tuple[str, bool]] = []  # (text, has_button)
        for container in containers:
            if len(entries) >= max_turns:
                break
            text = _collect_child_text(container, max_depth=5, max_chars=1000)
            text = text.strip()
            if not text:
                continue
            has_button = self._has_action_buttons(container)
            entries.append((text, has_button))

        any_has_buttons = any(hb for _, hb in entries)

        # Second pass: assign speakers
        turns: list[ConversationTurn] = []
        for i, (text, has_button) in enumerate(entries):
            if any_has_buttons:
                # Structural detection: buttons → assistant, no buttons → user
                speaker = "ChatGPT" if has_button else "User"
            else:
                # No structural cues — fall back to alternation
                speaker = "User" if i % 2 == 0 else "ChatGPT"
            turns.append(ConversationTurn(speaker=speaker, text=text))

        if turns and len(turns) >= 1:
            return turns
        return None

    @staticmethod
    def _has_action_buttons(container) -> bool:
        """Check if a message container has action buttons (Sources, feedback, etc.).

        ChatGPT responses typically contain extra child elements like
        "Sources" buttons alongside the text, while user messages don't.
        """
        children = ax_get_children(container) or []
        for child in children[:10]:
            child_role = ax_get_attribute(child, "AXRole") or ""
            if child_role == "AXButton":
                return True
            # Also check grandchildren (common in ChatGPT 5.x: AXGroup > AXButton)
            if child_role == "AXGroup":
                grandchildren = ax_get_children(child) or []
                for gc in grandchildren[:10]:
                    gc_role = ax_get_attribute(gc, "AXRole") or ""
                    if gc_role == "AXButton":
                        return True
        return False

    def _find_articles(
        self,
        element,
        max_depth: int,
        depth: int = 0,
        _visits: Optional[list] = None,
    ) -> list:
        """Find article-like containers in the AX tree."""
        if _visits is None:
            _visits = [0]
        _visits[0] += 1
        if _visits[0] > self._MAX_VISITS or depth > max_depth:
            return []

        articles: list = []
        role_desc = ax_get_attribute(element, "AXRoleDescription") or ""
        role = ax_get_attribute(element, "AXRole") or ""

        if role_desc == "article" or (
            role == "AXGroup"
            and ax_get_attribute(element, "AXSubrole") == "AXArticle"
        ):
            articles.append(element)
            return articles  # Don't recurse into articles

        children = ax_get_children(element)
        if children:
            for child in children[:50]:
                child_articles = self._find_articles(
                    child, max_depth, depth + 1, _visits
                )
                articles.extend(child_articles)

        return articles

    def _find_section_list_messages(
        self,
        element,
        max_depth: int,
        depth: int = 0,
        _visits: Optional[list] = None,
    ) -> list:
        """Find message groups inside AXSectionList (ChatGPT 5.x+ layout).

        The structure is:
            AXList (subrole=AXSectionList)
              AXGroup  ← message container (may be empty separator)
              AXGroup  ← message container with content
              ...

        Returns the non-empty AXGroup children of the first AXSectionList
        that contains AXStaticText descendants (i.e. actual text content,
        not sidebar buttons).
        """
        if _visits is None:
            _visits = [0]
        _visits[0] += 1
        if _visits[0] > self._MAX_VISITS or depth > max_depth:
            return []

        role = ax_get_attribute(element, "AXRole") or ""
        subrole = ax_get_attribute(element, "AXSubrole") or ""

        if role == "AXList" and subrole == "AXSectionList":
            # Found a section list — check if it has text content
            children = ax_get_children(element) or []
            groups = []
            for child in children[:100]:
                child_role = ax_get_attribute(child, "AXRole") or ""
                if child_role == "AXGroup":
                    grandchildren = ax_get_children(child) or []
                    if grandchildren and self._has_static_text(child, max_depth=3):
                        groups.append(child)
            if groups:
                return groups
            # No text content — this is likely a sidebar list, keep searching

        children = ax_get_children(element)
        if children:
            for child in children[:50]:
                result = self._find_section_list_messages(
                    child, max_depth, depth + 1, _visits
                )
                if result:
                    return result

        return []

    @staticmethod
    def _has_static_text(element, max_depth: int = 3, depth: int = 0) -> bool:
        """Check if an element has any AXStaticText descendants."""
        if depth > max_depth:
            return False
        children = ax_get_children(element) or []
        for child in children[:20]:
            child_role = ax_get_attribute(child, "AXRole") or ""
            if child_role == "AXStaticText":
                return True
            if ChatGPTExtractor._has_static_text(child, max_depth, depth + 1):
                return True
        return False


class ClaudeDesktopExtractor(ConversationExtractor):
    """Extractor for Claude Desktop app.

    Claude Desktop uses a flat list of sibling groups inside a conversation
    container.  Each message (user or assistant) is one or more content groups
    followed by a ``Message actions`` group (AXApplicationGroup with
    AXDescription 'Message actions').

    Role detection:
    - Assistant "Message actions" groups contain feedback buttons
      ('Give positive feedback' / 'Give negative feedback').
    - User "Message actions" groups only contain a 'Copy' button.

    Assistant messages may include a collapsible "Thought process" section
    that is skipped so only the actual response text is captured.
    """

    app_names: tuple[str, ...] = ("Claude",)

    _MAX_VISITS = 800

    def extract(
        self, window_element, max_turns: int = 15
    ) -> Optional[list[ConversationTurn]]:
        try:
            container = self._find_message_container(window_element)
            if container is not None:
                turns = self._parse_conversation(container, max_turns)
                if turns:
                    return turns

            # Fallback to action-delimited (which itself falls back to generic)
            return ActionDelimitedExtractor().extract(window_element, max_turns)
        except Exception:
            logger.debug(
                "Claude Desktop conversation extraction failed", exc_info=True,
            )
            return None

    # -- tree search --

    def _find_message_container(
        self, element, depth: int = 0, _visits: Optional[list] = None
    ):
        """Find the group whose direct children include 'Message actions' groups."""
        if _visits is None:
            _visits = [0]
        _visits[0] += 1
        if _visits[0] > self._MAX_VISITS or depth > 15:
            return None

        children = ax_get_children(element) or []
        for child in children[:50]:
            subrole = ax_get_attribute(child, "AXSubrole") or ""
            desc = ax_get_attribute(child, "AXDescription") or ""
            if subrole == "AXApplicationGroup" and desc == "Message actions":
                return element

        for child in children[:20]:
            result = self._find_message_container(child, depth + 1, _visits)
            if result is not None:
                return result
        return None

    # -- conversation parsing --

    def _parse_conversation(
        self, container, max_turns: int
    ) -> Optional[list[ConversationTurn]]:
        """Walk the container's children, grouping content by 'Message actions' delimiters."""
        children = ax_get_children(container) or []
        turns: list[ConversationTurn] = []
        pending: list = []  # content groups for the current message

        for child in children:
            subrole = ax_get_attribute(child, "AXSubrole") or ""
            desc = ax_get_attribute(child, "AXDescription") or ""

            if subrole == "AXApplicationGroup" and desc == "Message actions":
                if pending:
                    is_assistant = self._actions_have_feedback(child)
                    text = self._extract_message_text(pending, is_assistant)
                    if text.strip():
                        speaker = "Claude" if is_assistant else "User"
                        turns.append(ConversationTurn(speaker=speaker, text=text.strip()))
                    pending = []
                continue

            pending.append(child)

        return turns[-max_turns:] if turns else None

    # -- role detection --

    @staticmethod
    def _actions_have_feedback(actions_group) -> bool:
        """Return True if the 'Message actions' group contains feedback buttons."""
        for child in ax_get_children(actions_group) or []:
            child_desc = ax_get_attribute(child, "AXDescription") or ""
            if "feedback" in child_desc.lower():
                return True
            for gc in ax_get_children(child) or []:
                gc_desc = ax_get_attribute(gc, "AXDescription") or ""
                if "feedback" in gc_desc.lower():
                    return True
        return False

    # -- text extraction --

    def _extract_message_text(
        self, groups: list, is_assistant: bool
    ) -> str:
        """Collect text from a sequence of content groups for one message."""
        parts: list[str] = []
        for group in groups:
            if is_assistant:
                text = self._extract_assistant_response(group)
            else:
                text = _collect_child_text(group, max_depth=6, max_chars=1500)
            if text and text.strip():
                parts.append(text.strip())
        return "\n".join(parts)

    @staticmethod
    def _extract_assistant_response(group) -> str:
        """Extract the response portion of an assistant message, skipping thinking."""
        children = ax_get_children(group) or []
        if not children:
            return _collect_child_text(group, max_depth=6, max_chars=1500)

        # Unwrap single-child wrapper
        inner = children[0] if len(children) == 1 else group
        inner_children = ax_get_children(inner) or []

        # Detect whether this group contains a "Thought process" section
        has_thinking = False
        past_thinking = False
        response_parts: list[str] = []

        for i, child in enumerate(inner_children):
            role = ax_get_attribute(child, "AXRole") or ""
            title = ax_get_attribute(child, "AXTitle") or ""

            # Thinking section: AXButton followed by AXApplicationStatus sibling.
            # Older Claude used title="Thought process"; newer versions use the
            # thinking summary as the dynamic button title.
            if role == "AXButton":
                if title == "Thought process":
                    has_thinking = True
                    continue
                if i + 1 < len(inner_children):
                    next_sub = ax_get_attribute(inner_children[i + 1], "AXSubrole") or ""
                    if next_sub == "AXApplicationStatus":
                        has_thinking = True
                        continue

            if has_thinking and not past_thinking:
                # Still inside the thinking section — look for "Done" marker.
                # Check sub-children individually since "Done" may be nested
                # alongside thinking text in the same group.
                sub_children = ax_get_children(child) or []
                for sc in sub_children:
                    sc_text = _collect_child_text(sc, max_depth=2, max_chars=50)
                    if sc_text.strip() == "Done":
                        past_thinking = True
                        break
                if not past_thinking:
                    # Also check the child directly (legacy layout)
                    child_text = _collect_child_text(child, max_depth=2, max_chars=50)
                    if child_text.strip() == "Done":
                        past_thinking = True
                continue

            if past_thinking or not has_thinking:
                text = _collect_child_text(child, max_depth=6, max_chars=1500)
                if text and text.strip():
                    response_parts.append(text.strip())

        if response_parts:
            return "\n".join(response_parts)

        # No thinking section detected — collect everything
        return _collect_child_text(group, max_depth=6, max_chars=1500)


class ActionDelimitedExtractor(ConversationExtractor):
    """General extractor for chat UIs where messages are separated by action groups.

    Many Electron/web-based chat apps structure their AX trees with message
    content groups interleaved with action button groups (copy, react, reply,
    share, feedback, etc.).  This extractor finds such containers and uses the
    action groups as delimiters to segment the conversation.

    Role detection heuristic:
    - Action groups containing feedback/like/dislike buttons → assistant message
    - Action groups with only copy/share/reply buttons → user message

    This is used as the default fallback before GenericExtractor.  It does NOT
    perform app-specific filtering (e.g. skipping thinking sections); that
    belongs in dedicated extractors like ClaudeDesktopExtractor.
    """

    app_names: tuple[str, ...] = ()  # Fallback only, not registered for any app

    _MAX_VISITS = 800
    _MIN_ACTION_GROUPS = 2  # Need ≥2 action groups to qualify as a conversation

    # Keywords in button AXDescription that identify an "action delimiter" group
    _ACTION_KEYWORDS = frozenset({
        "copy", "react", "reply", "share", "forward", "delete",
        "feedback", "thumbs", "like", "dislike", "bookmark",
        "pin", "thread", "more actions", "message actions",
    })

    # Subset of keywords that signal an assistant/bot message
    _ASSISTANT_KEYWORDS = frozenset({
        "feedback", "thumbs", "like", "dislike", "regenerate",
        "positive", "negative",
    })

    def extract(
        self, window_element, max_turns: int = 15
    ) -> Optional[list[ConversationTurn]]:
        try:
            container = self._find_action_container(window_element)
            if container is not None:
                turns = self._parse_conversation(container, max_turns)
                if turns:
                    return turns
        except Exception:
            logger.debug("Action-delimited extraction failed", exc_info=True)

        # Fall back to generic heuristic
        return GenericExtractor().extract(window_element, max_turns)

    # -- container search --

    def _find_action_container(
        self, element, depth: int = 0, _visits: Optional[list] = None
    ):
        """Find a group with ≥ _MIN_ACTION_GROUPS action-group children."""
        if _visits is None:
            _visits = [0]
        _visits[0] += 1
        if _visits[0] > self._MAX_VISITS or depth > 15:
            return None

        children = ax_get_children(element) or []

        action_count = 0
        for child in children[:60]:
            if self._is_action_group(child):
                action_count += 1
                if action_count >= self._MIN_ACTION_GROUPS:
                    return element

        for child in children[:20]:
            result = self._find_action_container(child, depth + 1, _visits)
            if result is not None:
                return result
        return None

    def _is_action_group(self, element) -> bool:
        """Check if an element looks like a message-action delimiter.

        Matches AXApplicationGroup elements whose description or whose button
        children's descriptions contain action-related keywords.
        """
        subrole = ax_get_attribute(element, "AXSubrole") or ""
        if subrole != "AXApplicationGroup":
            return False

        # Check the group's own description
        desc = (ax_get_attribute(element, "AXDescription") or "").lower()
        if any(kw in desc for kw in self._ACTION_KEYWORDS):
            return True

        # Check immediate button children (shallow — max depth 2)
        return self._has_action_buttons(element, max_depth=2)

    def _has_action_buttons(self, element, max_depth: int, depth: int = 0) -> bool:
        """Check if element contains buttons with action-like descriptions."""
        if depth > max_depth:
            return False
        children = ax_get_children(element) or []
        for child in children[:10]:
            role = ax_get_attribute(child, "AXRole") or ""
            if role == "AXButton":
                btn_desc = (ax_get_attribute(child, "AXDescription") or "").lower()
                if any(kw in btn_desc for kw in self._ACTION_KEYWORDS):
                    return True
            if self._has_action_buttons(child, max_depth, depth + 1):
                return True
        return False

    # -- conversation parsing --

    def _parse_conversation(
        self, container, max_turns: int
    ) -> Optional[list[ConversationTurn]]:
        """Group children by action-group delimiters into conversation turns."""
        children = ax_get_children(container) or []
        turns: list[ConversationTurn] = []
        pending: list = []

        for child in children:
            if self._is_action_group(child):
                if pending:
                    is_assistant = self._is_assistant_actions(child)
                    text = self._extract_text_from_groups(pending)
                    if text.strip():
                        speaker = "Assistant" if is_assistant else "User"
                        turns.append(ConversationTurn(
                            speaker=speaker, text=text.strip(),
                        ))
                    pending = []
                continue

            pending.append(child)

        return turns[-max_turns:] if turns else None

    # -- role detection --

    def _is_assistant_actions(self, action_group) -> bool:
        """Return True if the action group suggests an assistant/bot message."""
        return self._has_assistant_indicators(action_group, max_depth=3)

    def _has_assistant_indicators(
        self, element, max_depth: int, depth: int = 0
    ) -> bool:
        if depth > max_depth:
            return False
        desc = (ax_get_attribute(element, "AXDescription") or "").lower()
        if any(kw in desc for kw in self._ASSISTANT_KEYWORDS):
            return True
        children = ax_get_children(element) or []
        for child in children[:10]:
            if self._has_assistant_indicators(child, max_depth, depth + 1):
                return True
        return False

    # -- text extraction --

    @staticmethod
    def _extract_text_from_groups(groups: list) -> str:
        """Collect text from a sequence of content groups."""
        parts: list[str] = []
        for group in groups:
            text = _collect_child_text(group, max_depth=6, max_chars=1500)
            if text and text.strip():
                parts.append(text.strip())
        return "\n".join(parts)


class IMessageExtractor(ConversationExtractor):
    """Extractor for iMessage (Messages.app).

    iMessage's actual AX tree structure (macOS 15+):

    ::

        AXGroup desc="Messages" rdesc="collection"        ← active chat messages
          AXGroup desc="Your iMessage, Okkk, 8:26 PM"     ← sent message
            AXGroup desc="Your iMessage, Okkk, 8:26 PM"   ← extra wrapper
              AXTextArea val="Okkk"                        ← message body
          AXGroup desc="Jinpaaaa, can't sleep, 10:09 PM"  ← received message
            AXGroup desc="Jinpaaaa, can't sleep, 10:09 PM"
              AXTextArea val="can't sleep"
        AXGroup desc="Conversations" rdesc="collection"    ← sidebar (other chats)

    The AXTextArea may be a direct child or nested inside one or more
    AXGroup wrappers (observed on macOS 15).  ``_find_textarea_value``
    handles variable wrapper depth.

    We find the ``desc="Messages"`` collection, iterate its direct
    children, parse the parent ``AXDescription`` to determine
    sent/received, and extract the child ``AXTextArea`` value for the
    clean message body.
    """

    app_names: tuple[str, ...] = ("Messages",)

    _MAX_VISITS = 600

    def extract(
        self, window_element, max_turns: int = 15
    ) -> Optional[list[ConversationTurn]]:
        try:
            messages_collection = self._find_messages_collection(
                window_element, max_depth=10
            )
            if messages_collection is None:
                return None

            turns = self._extract_turns(messages_collection, max_turns)
            if turns and len(turns) >= 1:
                return turns
        except Exception:
            logger.debug("iMessage conversation extraction failed", exc_info=True)

        # Fall back to generic extraction for resilience
        return ActionDelimitedExtractor().extract(window_element, max_turns)

    def _find_messages_collection(
        self,
        element,
        max_depth: int,
        depth: int = 0,
        _visits: Optional[list] = None,
    ):
        """Find the AXGroup with desc='Messages' and rdesc='collection'."""
        if _visits is None:
            _visits = [0]
        _visits[0] += 1
        if _visits[0] > self._MAX_VISITS or depth > max_depth:
            return None

        role = ax_get_attribute(element, "AXRole") or ""
        if role == "AXGroup":
            rdesc = ax_get_attribute(element, "AXRoleDescription") or ""
            if rdesc == "collection":
                desc = ax_get_attribute(element, "AXDescription") or ""
                if desc == "Messages":
                    return element

        children = ax_get_children(element)
        if children:
            for child in children[:50]:
                result = self._find_messages_collection(
                    child, max_depth, depth + 1, _visits
                )
                if result is not None:
                    return result
        return None

    def _extract_turns(
        self, messages_collection, max_turns: int
    ) -> list[ConversationTurn]:
        """Extract turns from the Messages collection's direct children."""
        children = ax_get_children(messages_collection)
        if not children:
            return []

        turns: list[ConversationTurn] = []
        for child in children:
            if len(turns) >= max_turns:
                break

            role = ax_get_attribute(child, "AXRole") or ""
            if role != "AXGroup":
                continue

            desc = ax_get_attribute(child, "AXDescription") or ""
            if not desc:
                continue

            # Skip separator/timestamp rows: their first child is
            # AXStaticText (e.g. "Today 10:09 PM", "Delivered").
            child_children = ax_get_children(child)
            if child_children:
                first_child_role = ax_get_attribute(child_children[0], "AXRole") or ""
                if first_child_role == "AXStaticText":
                    continue

            # Determine speaker from parent desc
            if desc.startswith("Your iMessage, ") or desc.startswith("Your message, "):
                speaker = "Me"
            else:
                speaker = "Them"

            # Extract clean message body from child AXTextArea (may be
            # nested inside one or more AXGroup wrappers)
            body = self._find_textarea_value(child)
            if not body:
                # Fallback: extract body from desc by stripping prefix and timestamp
                # desc format: "Your iMessage, <body>, <time>" or "<contact>, <body>, <time>"
                parts = desc.split(", ", 1)
                if len(parts) > 1:
                    body = parts[1].rsplit(", ", 1)[0]
            if not body or len(body) < 2:
                continue

            turns.append(ConversationTurn(speaker=speaker, text=body))

        return turns

    @staticmethod
    def _find_textarea_value(element, max_depth: int = 4, depth: int = 0) -> str:
        """Find the AXValue of the first AXTextArea descendant.

        Handles variable wrapper depth — the AXTextArea may be a direct
        child or nested inside one or more AXGroup wrappers.  Only
        recurses into AXGroup elements to avoid wandering into unrelated
        subtrees.
        """
        if depth > max_depth:
            return ""
        children = ax_get_children(element)
        if not children:
            return ""
        for child in children:
            role = ax_get_attribute(child, "AXRole") or ""
            if role == "AXTextArea":
                val = ax_get_attribute(child, "AXValue")
                if isinstance(val, str) and val.strip():
                    return val.strip()
            # Recurse into AXGroup wrappers only
            if role == "AXGroup":
                result = IMessageExtractor._find_textarea_value(
                    child, max_depth, depth + 1
                )
                if result:
                    return result
        return ""


class WhatsAppExtractor(ConversationExtractor):
    """Extractor for WhatsApp (macOS Catalyst app).

    WhatsApp's AX tree structure:

    ::

        AXGroup rdesc="Nav bar"
          AXHeading desc="Chat Name"            ← contact/group name
        AXGroup rdesc="table" desc="Messages in chat with Chat Name"
          AXStaticText desc="Your message, ..., Sent to ..., Red"  ← sent
          AXStaticText desc="Message from X, ..., Received in G"   ← received (group)
          AXStaticText desc="message, ..., Received from ..."      ← received (1:1)
          AXHeading                                                ← date separator
          AXButton desc="N unread messages"                        ← unread marker

    All metadata is in the ``AXDescription`` field.  The ``AXValue`` for
    messages is always an empty string.  U+200E (LTR mark) characters
    appear as prefixes throughout descriptions and are stripped during
    parsing.

    Description format per message type (after stripping U+200E):
    - Sent: ``Your message, <text>, <ts>, Sent to <recipient>, Red|Blue``
    - Received (group): ``Message from <sender>, <text>, <ts>, Received in <group>``
    - Received (1:1): ``message, <text>, <ts>, Received from <phone>``
    - Media: ``Photo, <ts>, Received from <phone>`` (skipped)
    - Edited messages append ``, Edited`` at the very end.
    """

    app_names: tuple[str, ...] = ("\u200eWhatsApp", "WhatsApp")

    _MAX_VISITS = 600

    # Timestamp pattern at end of description segment:
    # ", March4,at4:10 PM"  or  ", May4,2025at8:14 AM"
    _TIMESTAMP_RE = re.compile(
        r",\s*[A-Z][a-z]{2,8}\d{1,2},(?:\d{4})?at\d{1,2}:\d{2}\s[AP]M$"
    )

    # Phone number with VoiceOver-style commas: "+ 1,8 4 0,2 1 8,1 9 0 0"
    _PHONE_RE = re.compile(r"^\+\s*\d[\d,\s]*\d")

    # System/chrome messages to skip
    _SKIP_PREFIXES = (
        "Use WhatsApp",
        "end-to-end encrypted",
        "Messages you send",
    )

    # Media message prefixes (no useful text content)
    _MEDIA_PREFIXES = (
        "Photo,", "Video,", "Voice message,", "Sticker,",
        "Album with", "GIF,", "Document,", "Contact card,",
        "Location,", "Live location,",
    )

    def extract(
        self, window_element, max_turns: int = 15
    ) -> Optional[list[ConversationTurn]]:
        try:
            table = self._find_message_table(window_element, max_depth=12)
            if table is None:
                return None

            chat_name = self._get_chat_name(window_element)
            turns = self._extract_turns(table, max_turns, chat_name)
            if turns and len(turns) >= 1:
                return turns
        except Exception:
            logger.debug("WhatsApp conversation extraction failed", exc_info=True)
        return None

    # -- tree search --

    def _find_message_table(
        self, element, max_depth: int, depth: int = 0,
        _visits: Optional[list] = None,
    ):
        """Find the AXGroup with roleDescription='table' and description
        starting with 'Messages in chat with'."""
        if _visits is None:
            _visits = [0]
        _visits[0] += 1
        if _visits[0] > self._MAX_VISITS or depth > max_depth:
            return None

        role = ax_get_attribute(element, "AXRole") or ""
        if role == "AXGroup":
            rdesc = ax_get_attribute(element, "AXRoleDescription") or ""
            if rdesc == "table":
                desc = (
                    ax_get_attribute(element, "AXDescription") or ""
                ).replace("\u200e", "")
                if desc.startswith("Messages in chat with"):
                    return element

        children = ax_get_children(element)
        if children:
            for child in children[:50]:
                result = self._find_message_table(
                    child, max_depth, depth + 1, _visits,
                )
                if result is not None:
                    return result
        return None

    def _get_chat_name(self, window_element) -> str:
        """Extract chat name from the Nav bar heading."""
        heading = self._find_nav_heading(window_element, max_depth=10)
        if heading:
            desc = (
                ax_get_attribute(heading, "AXDescription") or ""
            ).replace("\u200e", "").strip()
            return desc
        return ""

    # Sidebar nav bar heading labels that should NOT be used as chat names.
    _SIDEBAR_HEADINGS = frozenset({"Chats", "Calls", "Updates", "Communities"})

    def _find_nav_heading(
        self, element, max_depth: int, depth: int = 0,
        _visits: Optional[list] = None,
    ):
        """Find the AXHeading inside the chat-panel Nav bar.

        WhatsApp has two Nav bars: one in the sidebar (heading = "Chats")
        and one in the chat panel (heading = contact/group name).  We
        collect all Nav bar headings and return the first whose label is
        not a known sidebar heading.
        """
        headings: list = []
        self._collect_nav_headings(element, headings, max_depth)
        # Return the first heading that isn't a sidebar label
        for heading in headings:
            desc = (
                ax_get_attribute(heading, "AXDescription") or ""
            ).replace("\u200e", "").strip()
            if desc and desc not in self._SIDEBAR_HEADINGS:
                return heading
        # Fallback: return the last heading found (likely the chat panel)
        return headings[-1] if headings else None

    def _collect_nav_headings(
        self, element, results: list, max_depth: int,
        depth: int = 0, _visits: Optional[list] = None,
    ) -> None:
        """Collect AXHeading elements from all Nav bar groups."""
        if _visits is None:
            _visits = [0]
        _visits[0] += 1
        if _visits[0] > self._MAX_VISITS or depth > max_depth:
            return

        rdesc = ax_get_attribute(element, "AXRoleDescription") or ""
        if rdesc == "Nav bar":
            children = ax_get_children(element) or []
            for child in children:
                if (ax_get_attribute(child, "AXRole") or "") == "AXHeading":
                    results.append(child)
            return  # Don't recurse into nav bars

        children = ax_get_children(element)
        if children:
            for child in children[:50]:
                self._collect_nav_headings(
                    child, results, max_depth, depth + 1, _visits,
                )

    # -- turn extraction --

    def _extract_turns(
        self, table, max_turns: int, chat_name: str,
    ) -> list[ConversationTurn]:
        """Extract conversation turns from the message table's children."""
        children = ax_get_children(table)
        if not children:
            return []

        turns: list[ConversationTurn] = []
        for child in children:
            if len(turns) >= max_turns:
                break

            role = ax_get_attribute(child, "AXRole") or ""
            if role != "AXStaticText":
                continue

            desc = (
                ax_get_attribute(child, "AXDescription") or ""
            ).replace("\u200e", "").strip()
            if not desc:
                continue

            # Skip system/chrome messages
            if any(desc.startswith(p) for p in self._SKIP_PREFIXES):
                continue

            # Skip media messages (no useful text)
            if any(desc.startswith(p) for p in self._MEDIA_PREFIXES):
                continue

            turn = self._parse_message(desc, chat_name)
            if turn is not None:
                turns.append(turn)

        return turns

    # -- message parsing --

    def _parse_message(
        self, desc: str, chat_name: str,
    ) -> Optional[ConversationTurn]:
        """Parse a single WhatsApp message description into a ConversationTurn."""
        # Strip ", Edited" suffix
        if desc.endswith(", Edited"):
            desc = desc[: -len(", Edited")]

        if desc.startswith("Your message, "):
            text = self._parse_sent(desc)
            if text:
                return ConversationTurn(speaker="Me", text=text)

        elif desc.startswith("Message from "):
            sender, text = self._parse_received_group(desc)
            if text:
                return ConversationTurn(
                    speaker=sender or "Unknown", text=text,
                )

        elif desc.startswith("message, "):
            text = self._parse_received_dm(desc)
            if text:
                return ConversationTurn(
                    speaker=chat_name or "Them", text=text,
                )

        return None

    def _parse_sent(self, desc: str) -> Optional[str]:
        """Parse: ``Your message, <text>, <ts>, Sent to <r>, Red|Blue``"""
        remainder = desc[len("Your message, "):]

        # Strip delivery status
        for status in (", Red", ", Blue"):
            if remainder.endswith(status):
                remainder = remainder[: -len(status)]
                break

        # Find ", Sent to " from right
        idx = remainder.rfind(", Sent to ")
        if idx == -1:
            return None

        before_sent = remainder[:idx]
        return self._strip_timestamp(before_sent)

    def _parse_received_group(self, desc: str) -> tuple[Optional[str], Optional[str]]:
        """Parse: ``Message from <sender>, <text>, <ts>, Received in <group>``"""
        remainder = desc[len("Message from "):]

        # Find ", Received in " or ", Received from " from right
        for suffix in (", Received in ", ", Received from "):
            idx = remainder.rfind(suffix)
            if idx != -1:
                break
        else:
            return None, None

        before_recv = remainder[:idx]
        before_ts = self._strip_timestamp(before_recv)
        if not before_ts:
            return None, None

        sender, text = self._split_sender_text(before_ts)
        return sender, text

    def _parse_received_dm(self, desc: str) -> Optional[str]:
        """Parse: ``message, <text>, <ts>, Received from <phone>``"""
        remainder = desc[len("message, "):]

        idx = remainder.rfind(", Received from ")
        if idx == -1:
            return None

        before_recv = remainder[:idx]
        return self._strip_timestamp(before_recv)

    def _strip_timestamp(self, text: str) -> Optional[str]:
        """Remove trailing timestamp from a description segment."""
        match = self._TIMESTAMP_RE.search(text)
        if match:
            text = text[: match.start()]
        return text.strip() or None

    def _split_sender_text(self, s: str) -> tuple[str, str]:
        """Split ``sender, text`` where sender may be a phone number with commas.

        Phone numbers use VoiceOver comma-separated digit groups
        (``+ 1,8 4 0,2 1 8,1 9 0 0``).  For phone senders, match the
        leading phone pattern; for named contacts, split at first ``, ``.
        """
        phone_match = self._PHONE_RE.match(s)
        if phone_match:
            sender = phone_match.group().strip()
            rest = s[phone_match.end():]
            if rest.startswith(", "):
                return sender, rest[2:]
            return sender, rest.lstrip(", ")

        # Regular contact name: split at first ", "
        idx = s.find(", ")
        if idx != -1:
            return s[:idx], s[idx + 2:]
        return s, ""


# ---------------------------------------------------------------------------
# Discord
# ---------------------------------------------------------------------------

class DiscordExtractor(ConversationExtractor):
    """Extractor for Discord (Electron desktop app).

    Discord's AX tree exposes messages inside an ``AXList`` with
    ``subrole="AXContentList"`` and ``description`` starting with
    ``"Messages in"``.

    Each message is an ``AXGroup`` with:
    - ``subrole="AXDocumentArticle"``
    - ``roleDescription="message"``
    - ``title`` in the format: ``"<Speaker> , <text> , <timestamp>"``

    The title is the most reliable source: it contains the speaker name,
    message text, and timestamp in a comma-separated format.  The child
    tree also contains these as separate AX elements but the title is
    simpler to parse.

    Date separators are ``AXSplitter`` elements whose ``description``
    contains the date string (e.g. ``"December 14, 2025"``).
    """

    app_names: tuple[str, ...] = ("Discord",)

    _MAX_VISITS = 1200  # Discord's sidebar is deep — needs ~900 visits

    # Timestamp patterns at end of title:
    # "12/14/25, 11:22 AM" or "10:05 PM" or "Yesterday at 8:47 PM"
    _TS_FULL_RE = re.compile(
        r",\s*\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}\s[AP]M$"
    )
    _TS_SHORT_RE = re.compile(
        r",\s*\d{1,2}:\d{2}\s[AP]M$"
    )
    _TS_RELATIVE_RE = re.compile(
        r",\s*(Yesterday|Today)\s+at\s+\d{1,2}:\d{2}\s[AP]M$"
    )

    # Discord thread role badges appended to speaker names in title strings.
    # e.g. "Bankim OP Original Poster" → "Bankim"
    # Order matters: longer/more specific patterns first.
    _BADGE_SUFFIXES = (
        " OP Original Poster",
        " Original Poster",
        " OP",
    )

    # Discord server tags appended to display names.
    # e.g. "MGpai Server Tag: PALM" → "MGpai"
    _SERVER_TAG_RE = re.compile(r"\s+Server Tag:\s+\S+$")

    def extract(
        self, window_element, max_turns: int = 15
    ) -> Optional[list[ConversationTurn]]:
        try:
            # Primary: detect current user from the status bar
            current_username = self._get_current_username(window_element)
            # Display name may differ from username in server channels
            display_name = self._get_current_display_name(window_element) if current_username else ""

            msg_list = self._find_message_list(window_element, max_depth=25)
            if msg_list is not None:
                # Fallback: infer from DM header image when status bar
                # isn't available (e.g. AX tree truncated)
                dm_target = "" if current_username else self._get_dm_target(msg_list)
                turns = self._extract_turns(
                    msg_list, max_turns,
                    current_username=current_username,
                    dm_target=dm_target,
                    display_name=display_name,
                )
                if turns:
                    return turns
        except Exception:
            logger.debug("Discord conversation extraction failed", exc_info=True)

        # Fall back to action-delimited → generic
        return ActionDelimitedExtractor().extract(window_element, max_turns)

    # -- tree search --

    def _find_message_list(
        self, element, max_depth: int, depth: int = 0,
        _visits: Optional[list] = None,
    ):
        """Find the AXList with description starting with 'Messages in'."""
        if _visits is None:
            _visits = [0]
        _visits[0] += 1
        if _visits[0] > self._MAX_VISITS or depth > max_depth:
            return None

        role = ax_get_attribute(element, "AXRole") or ""
        if role == "AXList":
            subrole = ax_get_attribute(element, "AXSubrole") or ""
            if subrole == "AXContentList":
                desc = ax_get_attribute(element, "AXDescription") or ""
                if desc.startswith("Messages in"):
                    return element

        children = ax_get_children(element)
        if children:
            for child in children[:50]:
                result = self._find_message_list(
                    child, max_depth, depth + 1, _visits,
                )
                if result is not None:
                    return result
        return None

    # -- current user detection (status bar) --

    def _get_current_username(self, window_element) -> str:
        """Detect the current user's Discord username from the status bar.

        Discord's bottom-left status bar is an ``AXGroup`` with
        ``subrole="AXLandmarkRegion"`` and ``description`` containing
        ``"User status"``.  The username is the first ``AXStaticText``
        value within it.  This is always present regardless of which
        channel or DM is open.
        """
        region = self._find_status_region(window_element, max_depth=15)
        if region is None:
            logger.debug("[Discord] Status region NOT found (visit cap or depth limit)")
            return ""
        username = self._first_static_text_value(region, max_depth=8)
        logger.debug("[Discord] Status bar username=%r", username)
        return username

    def _get_current_display_name(self, window_element) -> str:
        """Detect the current user's display name from the status bar.

        Discord's status bar may show a display name that differs from
        the username. The display name appears as the second
        ``AXStaticText`` value within the status region. Returns empty
        string if not found or if it matches the username.
        """
        region = self._find_status_region(window_element, max_depth=15)
        if region is None:
            return ""
        texts = self._all_static_text_values(region, max_depth=8, max_results=3)
        # First text is the username; second (if present) is display name
        if len(texts) >= 2 and texts[1] != texts[0]:
            return texts[1]
        return ""

    def _all_static_text_values(
        self, element, max_depth: int, max_results: int = 3,
        depth: int = 0, _results: list[str] | None = None,
    ) -> list[str]:
        """Collect all AXStaticText values within *element*."""
        if _results is None:
            _results = []
        if depth > max_depth or len(_results) >= max_results:
            return _results
        role = ax_get_attribute(element, "AXRole") or ""
        if role == "AXStaticText":
            value = ax_get_attribute(element, "AXValue") or ""
            if value.strip():
                _results.append(value.strip())
        children = ax_get_children(element) or []
        for child in children[:20]:
            if len(_results) >= max_results:
                break
            self._all_static_text_values(child, max_depth, max_results, depth + 1, _results)
        return _results

    def _find_status_region(
        self, element, max_depth: int, depth: int = 0,
        _visits: Optional[list] = None,
    ):
        """Find the 'User status and settings' landmark region."""
        if _visits is None:
            _visits = [0]
        _visits[0] += 1
        if _visits[0] > self._MAX_VISITS or depth > max_depth:
            return None

        role = ax_get_attribute(element, "AXRole") or ""
        if role == "AXGroup":
            subrole = ax_get_attribute(element, "AXSubrole") or ""
            if subrole == "AXLandmarkRegion":
                desc = ax_get_attribute(element, "AXDescription") or ""
                if "User status" in desc:
                    logger.debug(
                        "[Discord] Found status region after %d visits at depth %d",
                        _visits[0], depth,
                    )
                    return element

        children = ax_get_children(element) or []
        for child in children[:50]:
            result = self._find_status_region(
                child, max_depth, depth + 1, _visits,
            )
            if result is not None:
                return result
        return None

    def _first_static_text_value(
        self, element, max_depth: int, depth: int = 0,
    ) -> str:
        """Return the first AXStaticText value found within *element*."""
        if depth > max_depth:
            return ""
        role = ax_get_attribute(element, "AXRole") or ""
        if role == "AXStaticText":
            value = ax_get_attribute(element, "AXValue") or ""
            if value.strip():
                return value.strip()
        children = ax_get_children(element) or []
        for child in children[:20]:
            val = self._first_static_text_value(child, max_depth, depth + 1)
            if val:
                return val
        return ""

    # -- DM target detection (fallback) --

    def _get_dm_target(self, msg_list) -> str:
        """Detect the DM target name from the message-list header.

        The first child of the message list in a 1:1 DM is a header group
        containing an ``AXImage`` whose ``description`` is the contact name
        (e.g. ``"Bankim"``).  Returns the name, or ``""`` if not found.
        """
        children = ax_get_children(msg_list) or []
        if not children:
            return ""
        header = children[0]
        return self._find_header_image_name(header, max_depth=3)

    def _find_header_image_name(
        self, element, max_depth: int, depth: int = 0,
    ) -> str:
        """Find the AXImage in the DM header and return its description."""
        if depth > max_depth:
            return ""
        role = ax_get_attribute(element, "AXRole") or ""
        if role == "AXImage":
            desc = ax_get_attribute(element, "AXDescription") or ""
            if desc and len(desc) <= 50:
                return desc
        children = ax_get_children(element) or []
        for child in children[:10]:
            name = self._find_header_image_name(child, max_depth, depth + 1)
            if name:
                return name
        return ""

    # -- turn extraction --

    def _extract_turns(
        self, msg_list, max_turns: int,
        current_username: str = "", dm_target: str = "",
        display_name: str = "",
    ) -> list[ConversationTurn]:
        """Extract conversation turns from the message list's children.

        Walks the children of the AXContentList, looking for AXGroup
        elements with ``roleDescription="message"``.  Uses the ``title``
        attribute (format: ``"Speaker , text , timestamp"``) for fast
        parsing, falling back to child-tree text extraction when needed.

        Speaker attribution (in priority order):

        1. **Status bar** (*current_username* / *display_name*): If the
           speaker matches the current user's username or display name,
           replace with ``"You"``.  Works for both DMs and server channels.
        2. **DM header** (*dm_target*): Fallback for 1:1 DMs when the
           status bar is unavailable — speakers that don't match the
           DM target are replaced with ``"You"``.
        """
        logger.debug(
            "[Discord] _extract_turns: username=%r display_name=%r dm_target=%r",
            current_username, display_name, dm_target,
        )
        children = ax_get_children(msg_list) or []
        turns: list[ConversationTurn] = []

        for child in children:
            if len(turns) >= max_turns:
                break
            turn = self._parse_message_element(child)
            if turn is not None:
                # Label the current user's messages as "You"
                if current_username and (
                    turn.speaker == current_username
                    or (display_name and turn.speaker == display_name)
                ):
                    turn = ConversationTurn(speaker="You", text=turn.text, timestamp=turn.timestamp)
                elif dm_target and turn.speaker != dm_target:
                    turn = ConversationTurn(speaker="You", text=turn.text, timestamp=turn.timestamp)
                turns.append(turn)

        return turns

    def _parse_message_element(self, element) -> Optional[ConversationTurn]:
        """Parse a single Discord message from the AX tree.

        Discord wraps each message in:
          AXGroup (outer, no special subrole)
            └─ AXGroup (subrole=AXDocumentArticle, roleDescription="message",
                        title="Speaker , text , timestamp")
        """
        # The message element itself, or its first AXDocumentArticle child
        msg_el = self._find_article(element)
        if msg_el is None:
            return None

        title = ax_get_attribute(msg_el, "AXTitle") or ""
        if not title:
            return None

        return self._parse_title(title, msg_el)

    def _find_article(self, element):
        """Find the AXDocumentArticle message element (may be nested 1 deep)."""
        if (ax_get_attribute(element, "AXRoleDescription") or "") == "message":
            return element
        children = ax_get_children(element) or []
        for child in children[:5]:
            if (ax_get_attribute(child, "AXRoleDescription") or "") == "message":
                return child
        return None

    def _parse_title(self, title: str, msg_el) -> Optional[ConversationTurn]:
        """Parse the title string: ``"Speaker , text , timestamp"``

        The title uses `` , `` (space-comma-space) as separator.
        The timestamp is always last.  We extract it for context, then
        split on the first `` , `` to get speaker and text.
        """
        # Extract timestamp before stripping it
        timestamp = ""
        for ts_re in (self._TS_FULL_RE, self._TS_RELATIVE_RE, self._TS_SHORT_RE):
            m = ts_re.search(title)
            if m:
                # Strip leading ", " from the matched timestamp
                timestamp = m.group().lstrip(", ").strip()
                break

        # Strip trailing timestamp
        text = self._TS_FULL_RE.sub("", title)
        text = self._TS_RELATIVE_RE.sub("", text)
        text = self._TS_SHORT_RE.sub("", text)

        # Split on first " , " → speaker, rest
        sep = " , "
        idx = text.find(sep)
        if idx == -1:
            return None

        speaker = text[:idx].strip()
        body = text[idx + len(sep):].strip()

        # Strip Discord thread role badges from speaker name
        for badge in self._BADGE_SUFFIXES:
            if speaker.endswith(badge):
                speaker = speaker[: -len(badge)].strip()
                break

        # Strip Discord server tags: "MGpai Server Tag: PALM" → "MGpai"
        speaker = self._SERVER_TAG_RE.sub("", speaker).strip()

        if not speaker or not body:
            # Title didn't have useful text — try child-tree extraction
            body = self._extract_body_from_children(msg_el)
            if not body:
                return None

        # Skip sticker / media-only messages
        if body.startswith("Sticker,") or body.startswith("Image"):
            return None

        # Clean up "(edited)" suffix — may include an expanded timestamp
        # e.g. "(edited) Monday, December 15, 2025 at 10:08 PM"
        body = re.sub(r"\s*\(edited\)(?:\s+\w+,\s.+)?$", "", body)

        return ConversationTurn(speaker=speaker, text=body, timestamp=timestamp)

    def _extract_body_from_children(self, msg_el) -> str:
        """Fallback: extract message body from the child tree.

        Looks for AXGroup children (skipping the AXHeading which has the
        speaker/timestamp) and collects text.
        """
        children = ax_get_children(msg_el) or []
        parts: list[str] = []
        for child in children:
            role = ax_get_attribute(child, "AXRole") or ""
            if role == "AXHeading":
                continue  # speaker + timestamp header
            text = _collect_child_text(child, max_depth=5, max_chars=1000)
            if text and text.strip():
                parts.append(text.strip())
        return "\n".join(parts)


# --- Registry ---

# All known extractors
_ALL_EXTRACTORS: list[ConversationExtractor] = [
    GeminiExtractor(),
    SlackExtractor(),
    ChatGPTExtractor(),
    ClaudeDesktopExtractor(),
    IMessageExtractor(),
    WhatsAppExtractor(),
    DiscordExtractor(),
]

# Map from app name to extractor instance
_EXTRACTORS: dict[str, ConversationExtractor] = {}
for _ext in _ALL_EXTRACTORS:
    for _name in _ext.app_names:
        _EXTRACTORS[_name] = _ext

# Default fallback: action-delimited (tries GenericExtractor internally)
_FALLBACK_EXTRACTOR = ActionDelimitedExtractor()

# Browser app names — these need window-title-based dispatch instead of
# a single hardcoded extractor.
_BROWSER_APP_NAMES: frozenset[str] = frozenset({
    "Google Chrome", "Chrome", "Microsoft Edge", "Safari", "Arc",
    "Brave Browser", "Vivaldi", "Opera",
})

# (keywords_to_match_in_title, extractor_instance) — checked in order.
_WINDOW_TITLE_KEYWORDS: list[tuple[tuple[str, ...], ConversationExtractor]] = [
    (("Gemini",), GeminiExtractor()),
    (("ChatGPT",), ChatGPTExtractor()),
    (("Claude",), ClaudeDesktopExtractor()),
]


def get_extractor(app_name: str, window_title: str = "") -> ConversationExtractor:
    """Return the app-specific extractor for the given app name.

    If no specific extractor is registered for the app, returns the
    ActionDelimitedExtractor as a fallback (which itself falls back to
    GenericExtractor).

    For browser apps, the *window_title* is checked against known keywords
    to pick the right chat-UI extractor (e.g. Gemini-in-Chrome →
    GeminiExtractor).
    """
    # Direct app-name match first (Slack, iMessage, ChatGPT desktop, etc.)
    if app_name in _EXTRACTORS:
        return _EXTRACTORS[app_name]
    # For browsers, check window title keywords
    if app_name in _BROWSER_APP_NAMES and window_title:
        title_lower = window_title.lower()
        for keywords, extractor in _WINDOW_TITLE_KEYWORDS:
            if any(kw.lower() in title_lower for kw in keywords):
                return extractor
    return _FALLBACK_EXTRACTOR
