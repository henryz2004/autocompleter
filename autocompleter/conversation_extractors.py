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
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from .ax_utils import ax_get_attribute

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared data class
# ---------------------------------------------------------------------------

@dataclass
class ConversationTurn:
    """A single message in a conversation."""
    speaker: str
    text: str


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
    children = ax_get_attribute(element, "AXChildren")
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

        children = ax_get_attribute(element, "AXChildren")
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
        children = ax_get_attribute(element, "AXChildren")
        if not children or len(children) < 2:
            return None

        texts: list[tuple[str, str]] = []  # (role, value)
        for child in children[:10]:  # limit scan
            child_role = ax_get_attribute(child, "AXRole") or ""
            if child_role == "AXStaticText":
                val = ax_get_attribute(child, "AXValue")
                if isinstance(val, str) and val.strip():
                    texts.append((child_role, val.strip()))
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

    app_names: tuple[str, ...] = ("Google Gemini", "Gemini", "Google Chrome", "Chrome")

    def extract(
        self, window_element, max_turns: int = 15
    ) -> Optional[list[ConversationTurn]]:
        try:
            # Find the main landmark area
            main_area = self._find_landmark_main(window_element, max_depth=10)
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
        return None

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
        children = ax_get_attribute(element, "AXChildren")
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

        children = ax_get_attribute(element, "AXChildren")
        if not children:
            return None

        # Check if any child is the conversation heading
        for child in children[:50]:
            role = ax_get_attribute(child, "AXRole") or ""
            if role == "AXHeading":
                heading_text = ax_get_attribute(child, "AXValue") or ""
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
        """Extract alternating user/model messages from the conversation container.

        After the heading, message groups alternate: user, model, user, model...
        We find the sibling group with the most children (the actual message list).
        """
        children = ax_get_attribute(container, "AXChildren")
        if not children:
            return []

        # Find the child group with the most sub-children (the message list)
        best_group = None
        best_count = 0
        for child in children[:50]:
            role = ax_get_attribute(child, "AXRole") or ""
            if role in {"AXGroup", "AXList"}:
                grandchildren = ax_get_attribute(child, "AXChildren")
                count = len(grandchildren) if grandchildren else 0
                if count > best_count:
                    best_count = count
                    best_group = child

        if best_group is None or best_count == 0:
            # No message list group found — conversation is empty
            return []

        msg_children = ax_get_attribute(best_group, "AXChildren")
        if not msg_children:
            return []

        turns: list[ConversationTurn] = []
        # Gemini alternates: user (even index), model (odd index)
        for i, msg_child in enumerate(msg_children):
            if len(turns) >= max_turns:
                break
            text = _collect_child_text(msg_child, max_depth=5, max_chars=1000)
            text = text.strip()
            if not text:
                continue
            speaker = "User" if i % 2 == 0 else "Gemini"
            turns.append(ConversationTurn(speaker=speaker, text=text))

        return turns


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
        return None

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

        children = ax_get_attribute(element, "AXChildren")
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
        children = ax_get_attribute(element, "AXChildren")
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
            articles = self._find_articles(window_element, max_depth=15)
            if not articles:
                return None

            turns: list[ConversationTurn] = []
            for i, article in enumerate(articles):
                if len(turns) >= max_turns:
                    break
                text = _collect_child_text(article, max_depth=5, max_chars=1000)
                text = text.strip()
                if not text:
                    continue
                # ChatGPT alternates: user (even), assistant (odd)
                speaker = "User" if i % 2 == 0 else "ChatGPT"
                turns.append(ConversationTurn(speaker=speaker, text=text))

            if turns and len(turns) >= 1:
                return turns
        except Exception:
            logger.debug("ChatGPT conversation extraction failed", exc_info=True)
        return None

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

        children = ax_get_attribute(element, "AXChildren")
        if children:
            for child in children[:50]:
                child_articles = self._find_articles(
                    child, max_depth, depth + 1, _visits
                )
                articles.extend(child_articles)

        return articles


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
            logger.debug("Claude Desktop conversation extraction failed", exc_info=True)
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

        children = ax_get_attribute(element, "AXChildren") or []
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
        children = ax_get_attribute(container, "AXChildren") or []
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
        for child in ax_get_attribute(actions_group, "AXChildren") or []:
            child_desc = ax_get_attribute(child, "AXDescription") or ""
            if "feedback" in child_desc.lower():
                return True
            for gc in ax_get_attribute(child, "AXChildren") or []:
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
        children = ax_get_attribute(group, "AXChildren") or []
        if not children:
            return _collect_child_text(group, max_depth=6, max_chars=1500)

        # Unwrap single-child wrapper
        inner = children[0] if len(children) == 1 else group
        inner_children = ax_get_attribute(inner, "AXChildren") or []

        # Detect whether this group contains a "Thought process" section
        has_thinking = False
        past_thinking = False
        response_parts: list[str] = []

        for child in inner_children:
            role = ax_get_attribute(child, "AXRole") or ""
            title = ax_get_attribute(child, "AXTitle") or ""

            # "Thought process" button marks start of thinking section
            if role == "AXButton" and title == "Thought process":
                has_thinking = True
                continue

            if has_thinking and not past_thinking:
                # Still inside the thinking section — look for "Done" marker
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

        children = ax_get_attribute(element, "AXChildren") or []

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
        children = ax_get_attribute(element, "AXChildren") or []
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
        children = ax_get_attribute(container, "AXChildren") or []
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
        children = ax_get_attribute(element, "AXChildren") or []
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

    iMessage uses AXTable > AXRow pattern with AXCell children containing
    message text. Messages have role descriptions or subroles indicating
    sent vs received.
    """

    app_names: tuple[str, ...] = ("Messages",)

    _MAX_VISITS = 600

    def extract(
        self, window_element, max_turns: int = 15
    ) -> Optional[list[ConversationTurn]]:
        try:
            turns = self._find_imessage_rows(window_element, max_turns, max_depth=10)
            if turns and len(turns) >= 1:
                return turns
        except Exception:
            logger.debug("iMessage conversation extraction failed", exc_info=True)
        return None

    def _find_imessage_rows(
        self,
        element,
        max_turns: int,
        max_depth: int,
        depth: int = 0,
        _visits: Optional[list] = None,
    ) -> list[ConversationTurn]:
        """Find message rows in iMessage's AX tree."""
        if _visits is None:
            _visits = [0]
        _visits[0] += 1
        if _visits[0] > self._MAX_VISITS or depth > max_depth:
            return []

        turns: list[ConversationTurn] = []
        role = ax_get_attribute(element, "AXRole") or ""

        if role in {"AXRow", "AXCell"}:
            turn = self._parse_imessage_row(element)
            if turn is not None:
                turns.append(turn)
                if len(turns) >= max_turns:
                    return turns

        children = ax_get_attribute(element, "AXChildren")
        if children:
            for child in children[:50]:
                if len(turns) >= max_turns:
                    break
                child_turns = self._find_imessage_rows(
                    child, max_turns - len(turns), max_depth, depth + 1, _visits
                )
                turns.extend(child_turns)

        return turns

    def _parse_imessage_row(self, element) -> Optional[ConversationTurn]:
        """Parse an iMessage row into a ConversationTurn."""
        text = _collect_child_text(element, max_depth=4, max_chars=500)
        text = text.strip()
        if not text or len(text) < 3:
            return None

        # iMessage uses subrole or description to indicate sent vs received
        subrole = ax_get_attribute(element, "AXSubrole") or ""
        desc = ax_get_attribute(element, "AXDescription") or ""

        if "sent" in subrole.lower() or "sent" in desc.lower():
            speaker = "Me"
        elif "received" in subrole.lower() or "received" in desc.lower():
            speaker = "Them"
        else:
            # Default heuristic: check for visual indicators
            speaker = "Unknown"

        return ConversationTurn(speaker=speaker, text=text)


# --- Registry ---

# All known extractors
_ALL_EXTRACTORS: list[ConversationExtractor] = [
    GeminiExtractor(),
    SlackExtractor(),
    ChatGPTExtractor(),
    ClaudeDesktopExtractor(),
    IMessageExtractor(),
]

# Map from app name to extractor instance
_EXTRACTORS: dict[str, ConversationExtractor] = {}
for _ext in _ALL_EXTRACTORS:
    for _name in _ext.app_names:
        _EXTRACTORS[_name] = _ext

# Default fallback: action-delimited (tries GenericExtractor internally)
_FALLBACK_EXTRACTOR = ActionDelimitedExtractor()


def get_extractor(app_name: str) -> ConversationExtractor:
    """Return the app-specific extractor for the given app name.

    If no specific extractor is registered for the app, returns the
    ActionDelimitedExtractor as a fallback (which itself falls back to
    GenericExtractor).
    """
    return _EXTRACTORS.get(app_name, _FALLBACK_EXTRACTOR)
