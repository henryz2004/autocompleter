"""Tests for conversation_extractors module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from autocompleter.conversation_extractors import (
    ActionDelimitedExtractor,
    ChatGPTExtractor,
    ClaudeDesktopExtractor,
    CodexExtractor,
    ConversationExtractor,
    ConversationTurn,
    DiscordExtractor,
    GeminiExtractor,
    GenericExtractor,
    IMessageExtractor,
    SlackExtractor,
    _collect_child_text,
    get_extractor,
)


# ---------------------------------------------------------------------------
# Helper to build mock AX elements
# ---------------------------------------------------------------------------

def make_ax_element(
    role: str = "",
    value: str | None = None,
    children: list | None = None,
    subrole: str = "",
    role_description: str = "",
    description: str = "",
    title: str = "",
) -> MagicMock:
    """Build a mock AX element with configurable attributes."""
    elem = MagicMock()
    attrs = {
        "AXRole": role,
        "AXValue": value,
        "AXChildren": children,
        "AXSubrole": subrole,
        "AXRoleDescription": role_description,
        "AXDescription": description,
        "AXTitle": title,
    }

    def get_attr(el, attr):
        if el is elem:
            return attrs.get(attr)
        return None

    # The mock carries its own attribute lookup; we'll patch ax_get_attribute
    # globally but route through this helper.
    elem._ax_attrs = attrs
    return elem


def _ax_get_attribute_dispatcher(element, attribute):
    """Dispatcher for patched ax_get_attribute using mock _ax_attrs."""
    attrs = getattr(element, "_ax_attrs", None)
    if attrs is not None:
        return attrs.get(attribute)
    return None


def _ax_get_children_dispatcher(element):
    """Dispatcher for patched ax_get_children using mock _ax_attrs."""
    attrs = getattr(element, "_ax_attrs", None)
    if attrs is not None:
        return attrs.get("AXChildren") or []
    return []


@pytest.fixture(autouse=True)
def patch_ax():
    """Patch ax_get_attribute and ax_get_children for all tests in this module."""
    with patch(
        "autocompleter.conversation_extractors.ax_get_attribute",
        side_effect=_ax_get_attribute_dispatcher,
    ), patch(
        "autocompleter.conversation_extractors.ax_get_children",
        side_effect=_ax_get_children_dispatcher,
    ):
        yield


# Also patch _collect_child_text in the extractors module so it uses our mock
# elements' AXChildren/AXValue structure.
def _mock_collect_child_text(element, max_depth=5, max_chars=2000, depth=0):
    """Simple mock of _collect_child_text that recursively gathers AXValue."""
    value = _ax_get_attribute_dispatcher(element, "AXValue")
    if isinstance(value, str) and value.strip():
        return value.strip()

    children = _ax_get_attribute_dispatcher(element, "AXChildren")
    if not children:
        return ""

    parts = []
    for child in children:
        child_text = _mock_collect_child_text(child, max_depth, max_chars, depth + 1)
        if child_text:
            parts.append(child_text)
    return "\n".join(parts)


@pytest.fixture(autouse=True)
def patch_collect_child_text():
    """Patch _collect_child_text used inside conversation_extractors."""
    with patch(
        "autocompleter.conversation_extractors._collect_child_text",
        side_effect=_mock_collect_child_text,
    ):
        yield


# ===========================================================================
# Tests for get_extractor registry
# ===========================================================================

class TestGetExtractor:
    def test_returns_gemini_extractor_for_google_gemini(self):
        ext = get_extractor("Google Gemini")
        assert isinstance(ext, GeminiExtractor)

    def test_returns_gemini_extractor_for_gemini(self):
        ext = get_extractor("Gemini")
        assert isinstance(ext, GeminiExtractor)

    def test_returns_slack_extractor(self):
        ext = get_extractor("Slack")
        assert isinstance(ext, SlackExtractor)

    def test_returns_chatgpt_extractor(self):
        ext = get_extractor("ChatGPT")
        assert isinstance(ext, ChatGPTExtractor)

    def test_returns_claude_desktop_extractor(self):
        ext = get_extractor("Claude")
        assert isinstance(ext, ClaudeDesktopExtractor)

    def test_returns_codex_extractor(self):
        ext = get_extractor("Codex")
        assert isinstance(ext, CodexExtractor)

    def test_returns_imessage_extractor(self):
        ext = get_extractor("Messages")
        assert isinstance(ext, IMessageExtractor)

    def test_returns_discord_extractor(self):
        ext = get_extractor("Discord")
        assert isinstance(ext, DiscordExtractor)

    def test_returns_action_delimited_for_unknown_app(self):
        ext = get_extractor("SomeRandomApp")
        assert isinstance(ext, ActionDelimitedExtractor)

    def test_returns_action_delimited_for_empty_string(self):
        ext = get_extractor("")
        assert isinstance(ext, ActionDelimitedExtractor)


# ===========================================================================
# Tests for GenericExtractor
# ===========================================================================

class TestGenericExtractor:
    def test_extracts_speaker_and_body(self):
        """Short speaker + long body inside AXGroup -> ConversationTurn."""
        extractor = GenericExtractor()

        # Build: window > group1(speaker + body), group2(speaker + body)
        msg1 = make_ax_element(
            role="AXGroup",
            children=[
                make_ax_element(role="AXStaticText", value="Alice"),
                make_ax_element(role="AXStaticText", value="Hey, how are you doing today?"),
            ],
        )
        msg2 = make_ax_element(
            role="AXGroup",
            children=[
                make_ax_element(role="AXStaticText", value="Bob"),
                make_ax_element(role="AXStaticText", value="I'm doing great, thanks for asking!"),
            ],
        )
        window = make_ax_element(role="AXWindow", children=[msg1, msg2])

        turns = extractor.extract(window)
        assert turns is not None
        assert len(turns) == 2
        assert turns[0].speaker == "Alice"
        assert "how are you" in turns[0].text
        assert turns[1].speaker == "Bob"
        assert "doing great" in turns[1].text

    def test_returns_none_with_too_few_turns(self):
        """GenericExtractor returns None when fewer than 2 turns found."""
        extractor = GenericExtractor()

        msg = make_ax_element(
            role="AXGroup",
            children=[
                make_ax_element(role="AXStaticText", value="Alice"),
                make_ax_element(role="AXStaticText", value="Hello there, friend!"),
            ],
        )
        window = make_ax_element(role="AXWindow", children=[msg])

        turns = extractor.extract(window)
        assert turns is None

    def test_returns_none_with_insufficient_text_children(self):
        """Group with fewer than 2 text children should not produce a turn."""
        extractor = GenericExtractor()

        # Only one text child
        msg = make_ax_element(
            role="AXGroup",
            children=[
                make_ax_element(role="AXStaticText", value="lonely text"),
            ],
        )
        window = make_ax_element(role="AXWindow", children=[msg])

        turns = extractor.extract(window)
        assert turns is None

    def test_max_turns_limiting(self):
        """Extraction should respect max_turns parameter."""
        extractor = GenericExtractor()

        messages = []
        for i in range(10):
            msg = make_ax_element(
                role="AXGroup",
                children=[
                    make_ax_element(role="AXStaticText", value=f"User{i}"),
                    make_ax_element(
                        role="AXStaticText",
                        value=f"This is message number {i} with enough text",
                    ),
                ],
            )
            messages.append(msg)

        window = make_ax_element(role="AXWindow", children=messages)

        turns = extractor.extract(window, max_turns=3)
        assert turns is not None
        assert len(turns) == 3

    def test_nested_message_body_in_group(self):
        """Message body inside a nested AXGroup should be collected."""
        extractor = GenericExtractor()

        msg1 = make_ax_element(
            role="AXGroup",
            children=[
                make_ax_element(role="AXStaticText", value="Alice"),
                make_ax_element(
                    role="AXGroup",
                    children=[
                        make_ax_element(
                            role="AXStaticText",
                            value="This is a nested message body with enough text",
                        ),
                    ],
                ),
            ],
        )
        msg2 = make_ax_element(
            role="AXGroup",
            children=[
                make_ax_element(role="AXStaticText", value="Bob"),
                make_ax_element(
                    role="AXGroup",
                    children=[
                        make_ax_element(
                            role="AXStaticText",
                            value="Another nested message body here as well",
                        ),
                    ],
                ),
            ],
        )
        window = make_ax_element(role="AXWindow", children=[msg1, msg2])

        turns = extractor.extract(window)
        assert turns is not None
        assert len(turns) == 2
        assert turns[0].speaker == "Alice"
        assert "nested message body" in turns[0].text

    def test_graceful_failure_returns_none(self):
        """Exception during extraction should return None, not crash."""
        extractor = GenericExtractor()

        # Element that raises when accessing attributes
        bad_element = MagicMock()
        bad_element._ax_attrs = {"AXRole": "AXWindow"}
        # Make AXChildren raise an exception
        def raise_on_children(el, attr):
            if attr == "AXChildren":
                raise RuntimeError("AX API error")
            return getattr(el, "_ax_attrs", {}).get(attr)

        with patch(
            "autocompleter.conversation_extractors.ax_get_attribute",
            side_effect=raise_on_children,
        ):
            turns = extractor.extract(bad_element)
            assert turns is None


# ===========================================================================
# Tests for GeminiExtractor
# ===========================================================================

class TestGeminiExtractor:
    def _build_gemini_tree(self, messages: list[str]) -> MagicMock:
        """Build a mock AX tree mimicking Gemini's structure.

        AXWindow > AXGroup(AXLandmarkMain) > container > [heading, message_list_group]
        """
        # Message groups
        msg_groups = []
        for text in messages:
            msg_group = make_ax_element(
                role="AXGroup",
                children=[
                    make_ax_element(role="AXStaticText", value=text),
                ],
            )
            msg_groups.append(msg_group)

        # The heading
        heading = make_ax_element(
            role="AXHeading",
            value="Conversation with Gemini",
        )

        # The message list group (sibling of heading, has the most children)
        message_list = make_ax_element(
            role="AXGroup",
            children=msg_groups,
        )

        # Container holding heading + message list
        container = make_ax_element(
            role="AXGroup",
            children=[heading, message_list],
        )

        # Landmark main
        landmark = make_ax_element(
            role="AXGroup",
            subrole="AXLandmarkMain",
            children=[container],
        )

        # Window
        window = make_ax_element(
            role="AXWindow",
            children=[landmark],
        )
        return window

    def test_extracts_gemini_conversation(self):
        extractor = GeminiExtractor()
        window = self._build_gemini_tree([
            "How do I sort a list in Python?",
            "You can use the sorted() function or list.sort() method.",
            "What about reverse sorting?",
            "Pass reverse=True to either sorted() or list.sort().",
        ])

        turns = extractor.extract(window)
        assert turns is not None
        assert len(turns) == 4
        assert turns[0].speaker == "User"
        assert "sort a list" in turns[0].text
        assert turns[1].speaker == "Gemini"
        assert "sorted()" in turns[1].text
        assert turns[2].speaker == "User"
        assert turns[3].speaker == "Gemini"

    def test_empty_conversation_returns_none(self):
        """Gemini with no messages should return None."""
        extractor = GeminiExtractor()
        window = self._build_gemini_tree([])

        turns = extractor.extract(window)
        assert turns is None

    def test_no_landmark_main_returns_none(self):
        """Window without AXLandmarkMain should return None."""
        extractor = GeminiExtractor()
        window = make_ax_element(
            role="AXWindow",
            children=[
                make_ax_element(role="AXGroup", children=[]),
            ],
        )

        turns = extractor.extract(window)
        assert turns is None

    def test_no_conversation_heading_returns_none(self):
        """Landmark without conversation heading should return None."""
        extractor = GeminiExtractor()
        landmark = make_ax_element(
            role="AXGroup",
            subrole="AXLandmarkMain",
            children=[
                make_ax_element(role="AXHeading", value="Some Other Page"),
            ],
        )
        window = make_ax_element(role="AXWindow", children=[landmark])

        turns = extractor.extract(window)
        assert turns is None

    def test_max_turns_limiting(self):
        extractor = GeminiExtractor()
        messages = [f"Message {i} with enough content" for i in range(10)]
        window = self._build_gemini_tree(messages)

        turns = extractor.extract(window, max_turns=3)
        assert turns is not None
        assert len(turns) == 3

    def test_graceful_failure(self):
        """Exception in Gemini extractor returns None."""
        extractor = GeminiExtractor()

        bad_element = MagicMock()
        bad_element._ax_attrs = {"AXRole": "AXWindow"}

        def raise_always(el, attr):
            if attr == "AXChildren":
                raise RuntimeError("boom")
            return getattr(el, "_ax_attrs", {}).get(attr)

        with patch(
            "autocompleter.conversation_extractors.ax_get_attribute",
            side_effect=raise_always,
        ):
            turns = extractor.extract(bad_element)
            assert turns is None


# ===========================================================================
# Tests for SlackExtractor
# ===========================================================================

class TestCodexExtractor:
    def test_extracts_user_and_assistant_turns_from_codex_layout(self):
        extractor = CodexExtractor()

        transcript = make_ax_element(
            role="AXGroup",
            children=[
                make_ax_element(
                    role="AXGroup",
                    children=[make_ax_element(role="AXStaticText", value="can you check this?")],
                ),
                make_ax_element(role="AXStaticText", value="9:31 PM"),
                make_ax_element(role="AXButton", description="Copy message"),
                make_ax_element(role="AXButton", description="Fork from this message"),
                make_ax_element(
                    role="AXGroup",
                    children=[make_ax_element(role="AXButton", title="13 previous messages")],
                ),
                make_ax_element(
                    role="AXGroup",
                    children=[make_ax_element(role="AXStaticText", value="Yes, I checked the logs.")],
                ),
                make_ax_element(
                    role="AXList",
                    children=[make_ax_element(role="AXStaticText", value="Everything looks healthy.")],
                ),
            ],
        )
        window = make_ax_element(role="AXWindow", children=[transcript])

        turns = extractor.extract(window)
        assert turns is not None
        assert [(t.speaker, t.timestamp) for t in turns] == [
            ("User", ""),
            ("Assistant", "9:31 PM"),
        ]
        assert "can you check this?" in turns[0].text
        assert "checked the logs" in turns[1].text
        assert "Everything looks healthy." in turns[1].text


class TestGeminiNestedWrappers:
    def test_extracts_headings_from_deeply_wrapped_conversation_branch(self):
        extractor = GeminiExtractor()

        conversation_branch = make_ax_element(
            role="AXGroup",
            children=[
                make_ax_element(
                    role="AXGroup",
                    children=[
                        make_ax_element(
                            role="AXGroup",
                            children=[
                                make_ax_element(
                                    role="AXHeading",
                                    title="You said temporary env vars on mac",
                                ),
                            ],
                        ),
                        make_ax_element(
                            role="AXGroup",
                            children=[
                                make_ax_element(
                                    role="AXGroup",
                                    children=[
                                        make_ax_element(
                                            role="AXHeading",
                                            title="Gemini said",
                                        ),
                                        make_ax_element(
                                            role="AXGroup",
                                            children=[
                                                make_ax_element(
                                                    role="AXStaticText",
                                                    value="Use VAR=value command for one-shot execution.",
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        )

        container = make_ax_element(
            role="AXGroup",
            children=[
                make_ax_element(role="AXHeading", value="Conversation with Gemini"),
                conversation_branch,
                make_ax_element(
                    role="AXGroup",
                    children=[make_ax_element(role="AXStaticText", value="Ask Gemini")],
                ),
            ],
        )

        turns = extractor._extract_messages_from_container(container, max_turns=6)
        assert [(t.speaker, t.text) for t in turns] == [
            ("User", "temporary env vars on mac"),
            ("Gemini", "Use VAR=value command for one-shot execution."),
        ]


class TestSlackExtractor:
    def _build_slack_message(self, sender: str, body: str) -> MagicMock:
        """Build a mock Slack message group with AXRoleDescription 'message'."""
        return make_ax_element(
            role="AXGroup",
            role_description="message",
            children=[
                make_ax_element(role="AXButton", value=sender, title=sender),
                make_ax_element(
                    role="AXGroup",
                    children=[
                        make_ax_element(role="AXStaticText", value=body),
                    ],
                ),
            ],
        )

    def test_extracts_slack_messages(self):
        extractor = SlackExtractor()

        msg1 = self._build_slack_message("Alice", "Hey team, standup in 5")
        msg2 = self._build_slack_message("Bob", "On my way!")
        msg3 = self._build_slack_message("Carol", "Same, just finishing a test")

        message_list = make_ax_element(
            role="AXList",
            children=[msg1, msg2, msg3],
        )
        window = make_ax_element(role="AXWindow", children=[message_list])

        turns = extractor.extract(window)
        assert turns is not None
        assert len(turns) == 3
        assert turns[0].speaker == "Alice"
        assert "standup" in turns[0].text
        assert turns[1].speaker == "Bob"
        assert turns[2].speaker == "Carol"

    def test_message_without_speaker_uses_unknown(self):
        """Slack message with body but no speaker should use 'Unknown'."""
        extractor = SlackExtractor()

        msg = make_ax_element(
            role="AXGroup",
            role_description="message",
            children=[
                make_ax_element(
                    role="AXGroup",
                    children=[
                        make_ax_element(
                            role="AXStaticText",
                            value="A message with no identifiable sender",
                        ),
                    ],
                ),
            ],
        )
        window = make_ax_element(role="AXWindow", children=[msg])

        turns = extractor.extract(window)
        assert turns is not None
        assert len(turns) == 1
        assert turns[0].speaker == "Unknown"

    def test_no_messages_returns_none(self):
        extractor = SlackExtractor()
        window = make_ax_element(
            role="AXWindow",
            children=[
                make_ax_element(role="AXGroup", children=[]),
            ],
        )

        turns = extractor.extract(window)
        assert turns is None

    def test_max_turns_limiting(self):
        extractor = SlackExtractor()
        messages = [
            self._build_slack_message(f"User{i}", f"Message number {i} text")
            for i in range(10)
        ]
        message_list = make_ax_element(role="AXList", children=messages)
        window = make_ax_element(role="AXWindow", children=[message_list])

        turns = extractor.extract(window, max_turns=4)
        assert turns is not None
        assert len(turns) == 4

    def test_graceful_failure(self):
        extractor = SlackExtractor()

        bad = MagicMock()
        bad._ax_attrs = {"AXRole": "AXWindow"}

        def raise_on_children(el, attr):
            if attr == "AXChildren":
                raise RuntimeError("AX failure")
            return getattr(el, "_ax_attrs", {}).get(attr)

        with patch(
            "autocompleter.conversation_extractors.ax_get_attribute",
            side_effect=raise_on_children,
        ):
            turns = extractor.extract(bad)
            assert turns is None


# ===========================================================================
# Tests for ChatGPTExtractor
# ===========================================================================

class TestChatGPTExtractor:
    def _build_chatgpt_tree(self, messages: list[str]) -> MagicMock:
        """Build a mock AX tree mimicking ChatGPT's article-based structure."""
        articles = []
        for text in messages:
            article = make_ax_element(
                role="AXGroup",
                subrole="AXArticle",
                role_description="article",
                children=[
                    make_ax_element(role="AXStaticText", value=text),
                ],
            )
            articles.append(article)

        container = make_ax_element(role="AXGroup", children=articles)
        window = make_ax_element(role="AXWindow", children=[container])
        return window

    def test_extracts_chatgpt_messages(self):
        extractor = ChatGPTExtractor()
        window = self._build_chatgpt_tree([
            "What is the capital of France?",
            "The capital of France is Paris.",
        ])

        turns = extractor.extract(window)
        assert turns is not None
        assert len(turns) == 2
        assert turns[0].speaker == "User"
        assert "capital of France" in turns[0].text
        assert turns[1].speaker == "ChatGPT"
        assert "Paris" in turns[1].text

    def test_no_articles_returns_none(self):
        extractor = ChatGPTExtractor()
        window = make_ax_element(
            role="AXWindow",
            children=[
                make_ax_element(role="AXGroup", children=[]),
            ],
        )

        turns = extractor.extract(window)
        assert turns is None

    def test_max_turns_limiting(self):
        extractor = ChatGPTExtractor()
        messages = [f"Message {i} with some text content" for i in range(8)]
        window = self._build_chatgpt_tree(messages)

        turns = extractor.extract(window, max_turns=3)
        assert turns is not None
        assert len(turns) == 3

    def test_skips_empty_articles(self):
        """Articles with no text content should be skipped."""
        extractor = ChatGPTExtractor()

        articles = [
            make_ax_element(
                role="AXGroup",
                subrole="AXArticle",
                role_description="article",
                children=[],  # empty article
            ),
            make_ax_element(
                role="AXGroup",
                subrole="AXArticle",
                role_description="article",
                children=[
                    make_ax_element(role="AXStaticText", value="Actual content here"),
                ],
            ),
        ]
        container = make_ax_element(role="AXGroup", children=articles)
        window = make_ax_element(role="AXWindow", children=[container])

        turns = extractor.extract(window)
        assert turns is not None
        assert len(turns) == 1
        assert "Actual content" in turns[0].text

    def test_graceful_failure(self):
        extractor = ChatGPTExtractor()
        bad = MagicMock()
        bad._ax_attrs = {"AXRole": "AXWindow"}

        def raise_always(el, attr):
            if attr == "AXChildren":
                raise RuntimeError("boom")
            return getattr(el, "_ax_attrs", {}).get(attr)

        with patch(
            "autocompleter.conversation_extractors.ax_get_attribute",
            side_effect=raise_always,
        ):
            turns = extractor.extract(bad)
            assert turns is None


# ===========================================================================
# Tests for ActionDelimitedExtractor
# ===========================================================================

def _make_action_group(button_descs: list[str]):
    """Build a mock AXApplicationGroup with the given button descriptions."""
    buttons = [
        make_ax_element(role="AXButton", description=desc)
        for desc in button_descs
    ]
    return make_ax_element(
        role="AXGroup",
        subrole="AXApplicationGroup",
        children=[
            make_ax_element(role="AXGroup", children=buttons),
        ],
    )


def _make_content_group(*lines: str):
    """Build a mock content group containing AXStaticText children."""
    return make_ax_element(
        role="AXGroup",
        children=[
            make_ax_element(
                role="AXGroup",
                children=[
                    make_ax_element(role="AXStaticText", value=line)
                    for line in lines
                ],
            ),
        ],
    )


class TestActionDelimitedExtractor:
    def test_extracts_user_and_assistant(self):
        """Messages separated by action groups are extracted correctly."""
        extractor = ActionDelimitedExtractor()

        container = make_ax_element(
            role="AXGroup",
            children=[
                _make_content_group("Hello there"),
                _make_action_group(["Copy"]),
                _make_content_group("Hi! How can I help?"),
                _make_action_group(["Copy", "Give positive feedback", "Give negative feedback"]),
            ],
        )
        window = make_ax_element(role="AXWindow", children=[container])

        turns = extractor.extract(window)
        assert turns is not None
        assert len(turns) == 2
        assert turns[0].speaker == "User"
        assert turns[0].text == "Hello there"
        assert turns[1].speaker == "Assistant"
        assert turns[1].text == "Hi! How can I help?"

    def test_multiple_turns(self):
        """Multiple conversation turns are extracted in order."""
        extractor = ActionDelimitedExtractor()

        container = make_ax_element(
            role="AXGroup",
            children=[
                _make_content_group("Question 1"),
                _make_action_group(["Copy"]),
                _make_content_group("Answer 1"),
                _make_action_group(["Copy", "Like", "Dislike"]),
                _make_content_group("Question 2"),
                _make_action_group(["Copy"]),
                _make_content_group("Answer 2"),
                _make_action_group(["Copy", "Thumbs up"]),
            ],
        )
        window = make_ax_element(role="AXWindow", children=[container])

        turns = extractor.extract(window)
        assert turns is not None
        assert len(turns) == 4
        assert [t.speaker for t in turns] == ["User", "Assistant", "User", "Assistant"]

    def test_requires_min_action_groups(self):
        """A container with only one action group is not treated as conversation."""
        extractor = ActionDelimitedExtractor()

        container = make_ax_element(
            role="AXGroup",
            children=[
                _make_content_group("Some text"),
                _make_action_group(["Copy"]),
            ],
        )
        window = make_ax_element(role="AXWindow", children=[container])

        # Falls through to GenericExtractor, which needs ≥2 speaker+body pairs
        turns = extractor.extract(window)
        assert turns is None

    def test_action_group_detected_by_button_description(self):
        """Action groups without explicit description are detected by button keywords."""
        extractor = ActionDelimitedExtractor()

        # Action groups have no AXDescription, but buttons have action keywords
        action1 = make_ax_element(
            role="AXGroup",
            subrole="AXApplicationGroup",
            children=[
                make_ax_element(role="AXButton", description="Copy"),
                make_ax_element(role="AXButton", description="Reply"),
            ],
        )
        action2 = make_ax_element(
            role="AXGroup",
            subrole="AXApplicationGroup",
            children=[
                make_ax_element(role="AXButton", description="Copy"),
                make_ax_element(role="AXButton", description="React"),
                make_ax_element(role="AXButton", description="Thumbs up"),
            ],
        )

        container = make_ax_element(
            role="AXGroup",
            children=[
                _make_content_group("User message"),
                action1,
                _make_content_group("Bot response"),
                action2,
            ],
        )
        window = make_ax_element(role="AXWindow", children=[container])

        turns = extractor.extract(window)
        assert turns is not None
        assert len(turns) == 2
        assert turns[0].speaker == "User"
        assert turns[1].speaker == "Assistant"

    def test_non_application_group_not_treated_as_delimiter(self):
        """Regular AXGroups with buttons should not be mistaken for action delimiters."""
        extractor = ActionDelimitedExtractor()

        # A regular group (no AXApplicationGroup subrole) with buttons
        not_action = make_ax_element(
            role="AXGroup",
            children=[
                make_ax_element(role="AXButton", description="Copy"),
            ],
        )

        container = make_ax_element(
            role="AXGroup",
            children=[
                _make_content_group("Some text"),
                not_action,
                _make_content_group("More text"),
            ],
        )
        window = make_ax_element(role="AXWindow", children=[container])

        # No action groups found → falls through to generic
        turns = extractor.extract(window)
        assert turns is None

    def test_falls_back_to_generic(self):
        """When no action-delimited container found, falls back to GenericExtractor."""
        extractor = ActionDelimitedExtractor()

        msg1 = make_ax_element(
            role="AXGroup",
            children=[
                make_ax_element(role="AXStaticText", value="Alice"),
                make_ax_element(role="AXStaticText", value="Hey, how are you?"),
            ],
        )
        msg2 = make_ax_element(
            role="AXGroup",
            children=[
                make_ax_element(role="AXStaticText", value="Bob"),
                make_ax_element(role="AXStaticText", value="Doing great, thanks!"),
            ],
        )
        window = make_ax_element(role="AXWindow", children=[msg1, msg2])

        turns = extractor.extract(window)
        assert turns is not None
        assert len(turns) == 2

    def test_max_turns_limiting(self):
        """Respects max_turns parameter."""
        extractor = ActionDelimitedExtractor()

        children = []
        for i in range(10):
            children.append(_make_content_group(f"Message {i}"))
            children.append(_make_action_group(["Copy"]))

        container = make_ax_element(role="AXGroup", children=children)
        window = make_ax_element(role="AXWindow", children=[container])

        turns = extractor.extract(window, max_turns=3)
        assert turns is not None
        assert len(turns) == 3

    def test_get_extractor_returns_action_delimited_for_unknown(self):
        """get_extractor returns ActionDelimitedExtractor for unknown apps."""
        ext = get_extractor("SomeRandomApp")
        assert isinstance(ext, ActionDelimitedExtractor)


# ===========================================================================
# Tests for ClaudeDesktopExtractor
# ===========================================================================

def _make_claude_message_actions(has_feedback: bool = False):
    """Build a mock 'Message actions' group for Claude Desktop tests."""
    children = [
        make_ax_element(
            role="AXGroup",
            children=[
                make_ax_element(role="AXButton", description="Copy"),
            ],
        ),
    ]
    if has_feedback:
        children.append(
            make_ax_element(role="AXGroup", children=[
                make_ax_element(role="AXButton", description="Give positive feedback"),
            ]),
        )
        children.append(
            make_ax_element(role="AXGroup", children=[
                make_ax_element(role="AXButton", description="Give negative feedback"),
            ]),
        )
    return make_ax_element(
        role="AXGroup",
        subrole="AXApplicationGroup",
        description="Message actions",
        children=children,
    )


def _make_claude_user_msg(*lines: str):
    """Build a mock user message content group for Claude Desktop."""
    return make_ax_element(
        role="AXGroup",
        children=[
            make_ax_element(
                role="AXGroup",
                children=[
                    make_ax_element(
                        role="AXGroup",
                        children=[
                            make_ax_element(role="AXStaticText", value=line)
                            for line in lines
                        ],
                    ),
                ],
            ),
        ],
    )


def _make_claude_assistant_msg(response_text: str, thinking_text: str = ""):
    """Build a mock assistant message content group for Claude Desktop."""
    inner_children = []

    if thinking_text:
        # Thought process button + thinking content + Done
        inner_children.append(
            make_ax_element(role="AXButton", title="Thought process")
        )
        inner_children.append(
            make_ax_element(
                role="AXGroup",
                children=[
                    make_ax_element(role="AXStaticText", value=thinking_text),
                ],
            ),
        )
        inner_children.append(
            make_ax_element(
                role="AXGroup",
                children=[
                    make_ax_element(role="AXStaticText", value="Done"),
                ],
            ),
        )

    # Response text
    inner_children.append(
        make_ax_element(
            role="AXGroup",
            children=[
                make_ax_element(role="AXStaticText", value=response_text),
            ],
        ),
    )

    return make_ax_element(
        role="AXGroup",
        children=[
            make_ax_element(role="AXGroup", children=inner_children),
        ],
    )


class TestClaudeDesktopExtractor:
    def test_extracts_user_and_assistant(self):
        """Claude Desktop should extract user and assistant turns."""
        extractor = ClaudeDesktopExtractor()

        container = make_ax_element(
            role="AXGroup",
            children=[
                _make_claude_user_msg("Hello Claude"),
                _make_claude_message_actions(has_feedback=False),
                _make_claude_assistant_msg("Hello! How can I help you today?"),
                _make_claude_message_actions(has_feedback=True),
            ],
        )
        # Wrap in enough nesting that _find_message_container can find it
        window = make_ax_element(role="AXWindow", children=[container])

        turns = extractor.extract(window)
        assert turns is not None
        assert len(turns) == 2
        assert turns[0].speaker == "User"
        assert turns[0].text == "Hello Claude"
        assert turns[1].speaker == "Claude"
        assert turns[1].text == "Hello! How can I help you today?"

    def test_skips_thinking_section(self):
        """Assistant thinking sections should be excluded from extracted text."""
        extractor = ClaudeDesktopExtractor()

        container = make_ax_element(
            role="AXGroup",
            children=[
                _make_claude_user_msg("What is 2+2?"),
                _make_claude_message_actions(has_feedback=False),
                _make_claude_assistant_msg(
                    response_text="The answer is 4.",
                    thinking_text="Let me think about this math problem...",
                ),
                _make_claude_message_actions(has_feedback=True),
            ],
        )
        window = make_ax_element(role="AXWindow", children=[container])

        turns = extractor.extract(window)
        assert turns is not None
        assert len(turns) == 2
        assert "thinking" not in turns[1].text.lower()
        assert "The answer is 4." in turns[1].text

    def test_multi_line_user_message(self):
        """User messages with multiple lines are captured."""
        extractor = ClaudeDesktopExtractor()

        container = make_ax_element(
            role="AXGroup",
            children=[
                _make_claude_user_msg("Line one", "Line two", "Line three"),
                _make_claude_message_actions(has_feedback=False),
            ],
        )
        window = make_ax_element(role="AXWindow", children=[container])

        turns = extractor.extract(window)
        assert turns is not None
        assert len(turns) == 1
        assert "Line one" in turns[0].text
        assert "Line three" in turns[0].text

    def test_multiple_turns(self):
        """Multiple conversation turns are extracted in order."""
        extractor = ClaudeDesktopExtractor()

        container = make_ax_element(
            role="AXGroup",
            children=[
                _make_claude_user_msg("First question"),
                _make_claude_message_actions(has_feedback=False),
                _make_claude_assistant_msg("First answer"),
                _make_claude_message_actions(has_feedback=True),
                _make_claude_user_msg("Follow up"),
                _make_claude_message_actions(has_feedback=False),
                _make_claude_assistant_msg("Second answer"),
                _make_claude_message_actions(has_feedback=True),
            ],
        )
        window = make_ax_element(role="AXWindow", children=[container])

        turns = extractor.extract(window)
        assert turns is not None
        assert len(turns) == 4
        assert turns[0].speaker == "User"
        assert turns[1].speaker == "Claude"
        assert turns[2].speaker == "User"
        assert turns[3].speaker == "Claude"

    def test_falls_back_to_generic(self):
        """If no 'Message actions' found, falls back to GenericExtractor."""
        extractor = ClaudeDesktopExtractor()

        msg1 = make_ax_element(
            role="AXGroup",
            children=[
                make_ax_element(role="AXStaticText", value="Henry"),
                make_ax_element(role="AXStaticText", value="Can you help me with this code?"),
            ],
        )
        msg2 = make_ax_element(
            role="AXGroup",
            children=[
                make_ax_element(role="AXStaticText", value="Claude"),
                make_ax_element(role="AXStaticText", value="Of course! I'd be happy to help."),
            ],
        )
        window = make_ax_element(role="AXWindow", children=[msg1, msg2])

        turns = extractor.extract(window)
        assert turns is not None
        assert len(turns) == 2


# ===========================================================================
# Tests for IMessageExtractor
# ===========================================================================

class TestIMessageExtractor:
    """Tests for the rewritten IMessageExtractor that uses the real
    iMessage AX tree structure: AXGroup(desc="Messages", rdesc="collection")
    containing AXGroup children with desc-based speaker detection and
    AXTextArea children for message body.
    """

    def _build_imessage_tree(
        self, messages: list[tuple[str, str]], *, sidebar: bool = False
    ) -> MagicMock:
        """Build a mock iMessage AX tree.

        ``messages`` is a list of (desc, body) tuples.  desc should look
        like ``"Your iMessage, Hey, 8:26 PM"`` for sent messages or
        ``"Alice, Hey there, 10:09 PM"`` for received.

        Mirrors the real macOS 15 structure where each message group has
        an extra AXGroup wrapper around the AXTextArea.
        """
        msg_children = []
        for desc, body in messages:
            # Real structure: AXGroup(desc) > AXGroup(desc) > AXTextArea(val)
            msg_group = make_ax_element(
                role="AXGroup",
                description=desc,
                children=[
                    make_ax_element(
                        role="AXGroup",
                        description=desc,
                        children=[
                            make_ax_element(role="AXTextArea", value=body),
                        ],
                    ),
                ],
            )
            msg_children.append(msg_group)

        messages_collection = make_ax_element(
            role="AXGroup",
            description="Messages",
            role_description="collection",
            children=msg_children,
        )

        parts = [messages_collection]
        if sidebar:
            sidebar_collection = make_ax_element(
                role="AXGroup",
                description="Conversations",
                role_description="collection",
                children=[
                    make_ax_element(role="AXStaticText", value="Emma, Pinned"),
                ],
            )
            parts.append(sidebar_collection)

        window = make_ax_element(role="AXWindow", children=parts)
        return window

    def test_extracts_sent_and_received(self):
        extractor = IMessageExtractor()
        window = self._build_imessage_tree([
            ("Your iMessage, Hey are you free tonight?, 8:26 PM", "Hey are you free tonight?"),
            ("Alice, Yeah what's up?, 8:27 PM", "Yeah what's up?"),
        ])

        turns = extractor.extract(window)
        assert turns is not None
        assert len(turns) == 2
        assert turns[0].speaker == "Me"
        assert "free tonight" in turns[0].text
        assert turns[1].speaker == "Them"
        assert "what's up" in turns[1].text

    def test_your_message_prefix_also_works(self):
        """'Your message, ...' prefix (SMS fallback) should also be Me."""
        extractor = IMessageExtractor()
        window = self._build_imessage_tree([
            ("Your message, Hello, 9:00 AM", "Hello"),
        ])

        turns = extractor.extract(window)
        assert turns is not None
        assert turns[0].speaker == "Me"

    def test_skips_separator_rows(self):
        """Timestamp/separator rows (first child is AXStaticText) are skipped."""
        extractor = IMessageExtractor()

        separator = make_ax_element(
            role="AXGroup",
            description="Today 10:09 PM",
            children=[
                make_ax_element(role="AXStaticText", value="Today 10:09 PM"),
            ],
        )
        msg = make_ax_element(
            role="AXGroup",
            description="Your iMessage, Hey, 10:10 PM",
            children=[
                make_ax_element(
                    role="AXGroup",
                    description="Your iMessage, Hey, 10:10 PM",
                    children=[
                        make_ax_element(role="AXTextArea", value="Hey"),
                    ],
                ),
            ],
        )
        messages_collection = make_ax_element(
            role="AXGroup",
            description="Messages",
            role_description="collection",
            children=[separator, msg],
        )
        window = make_ax_element(role="AXWindow", children=[messages_collection])

        turns = extractor.extract(window)
        assert turns is not None
        assert len(turns) == 1
        assert turns[0].speaker == "Me"
        assert turns[0].text == "Hey"

    def test_no_messages_collection_returns_none(self):
        """Window without the Messages collection returns None."""
        extractor = IMessageExtractor()
        window = make_ax_element(
            role="AXWindow",
            children=[
                make_ax_element(role="AXGroup", children=[]),
            ],
        )

        turns = extractor.extract(window)
        assert turns is None

    def test_empty_messages_collection_returns_none(self):
        """Messages collection with no children returns None."""
        extractor = IMessageExtractor()
        messages_collection = make_ax_element(
            role="AXGroup",
            description="Messages",
            role_description="collection",
            children=[],
        )
        window = make_ax_element(role="AXWindow", children=[messages_collection])

        turns = extractor.extract(window)
        assert turns is None

    def test_direct_textarea_child_still_works(self):
        """AXTextArea as direct child (no wrapper) should still extract."""
        extractor = IMessageExtractor()

        msg = make_ax_element(
            role="AXGroup",
            description="Your iMessage, Hey, 8:00 PM",
            children=[
                make_ax_element(role="AXTextArea", value="Hey"),
            ],
        )
        messages_collection = make_ax_element(
            role="AXGroup",
            description="Messages",
            role_description="collection",
            children=[msg],
        )
        window = make_ax_element(role="AXWindow", children=[messages_collection])

        turns = extractor.extract(window)
        assert turns is not None
        assert len(turns) == 1
        assert turns[0].text == "Hey"

    def test_desc_fallback_when_no_textarea(self):
        """When no AXTextArea exists, body should be parsed from desc."""
        extractor = IMessageExtractor()

        # Message group with an AXGroup child (not AXStaticText, so not
        # treated as separator) but no AXTextArea anywhere
        msg = make_ax_element(
            role="AXGroup",
            description="Your iMessage, Hello world, 9:00 PM",
            children=[
                make_ax_element(role="AXGroup", children=[]),
            ],
        )
        messages_collection = make_ax_element(
            role="AXGroup",
            description="Messages",
            role_description="collection",
            children=[msg],
        )
        window = make_ax_element(role="AXWindow", children=[messages_collection])

        turns = extractor.extract(window)
        assert turns is not None
        assert len(turns) == 1
        assert turns[0].speaker == "Me"
        assert turns[0].text == "Hello world"

    def test_max_turns_limiting(self):
        extractor = IMessageExtractor()
        messages = [
            (
                f"Your iMessage, Msg {i}, {i}:00 PM" if i % 2 == 0
                else f"Alice, Msg {i}, {i}:00 PM",
                f"Message number {i} text",
            )
            for i in range(10)
        ]
        window = self._build_imessage_tree(messages)

        turns = extractor.extract(window, max_turns=4)
        assert turns is not None
        assert len(turns) == 4

    def test_graceful_failure(self):
        extractor = IMessageExtractor()
        bad = MagicMock()
        bad._ax_attrs = {"AXRole": "AXWindow"}

        def raise_always(el, attr):
            if attr == "AXChildren":
                raise RuntimeError("boom")
            return getattr(el, "_ax_attrs", {}).get(attr)

        with patch(
            "autocompleter.conversation_extractors.ax_get_attribute",
            side_effect=raise_always,
        ):
            turns = extractor.extract(bad)
            assert turns is None


# ===========================================================================
# Integration-style test: input_observer uses extractors
# ===========================================================================

class TestInputObserverIntegration:
    """Verify that InputObserver._extract_conversation_turns delegates to extractors."""

    def test_delegates_to_app_specific_extractor(self):
        """Calling _extract_conversation_turns with app_name should use the right extractor."""
        from autocompleter.input_observer import InputObserver

        observer = InputObserver()

        # Build a Slack-style window
        msg = make_ax_element(
            role="AXGroup",
            role_description="message",
            children=[
                make_ax_element(role="AXButton", value="Alice", title="Alice"),
                make_ax_element(
                    role="AXGroup",
                    children=[
                        make_ax_element(role="AXStaticText", value="Hey there!"),
                    ],
                ),
            ],
        )
        window = make_ax_element(role="AXWindow", children=[msg])

        # Also patch ax_get_attribute in input_observer module
        with patch(
            "autocompleter.input_observer.ax_get_attribute",
            side_effect=_ax_get_attribute_dispatcher,
        ):
            turns = observer._extract_conversation_turns(
                window, max_turns=15, app_name="Slack"
            )

        assert turns is not None
        assert len(turns) == 1
        assert turns[0].speaker == "Alice"

    def test_falls_back_to_generic_for_unknown_app(self):
        """Unknown app should use GenericExtractor."""
        from autocompleter.input_observer import InputObserver

        observer = InputObserver()

        msg1 = make_ax_element(
            role="AXGroup",
            children=[
                make_ax_element(role="AXStaticText", value="User1"),
                make_ax_element(role="AXStaticText", value="First message body text here"),
            ],
        )
        msg2 = make_ax_element(
            role="AXGroup",
            children=[
                make_ax_element(role="AXStaticText", value="User2"),
                make_ax_element(role="AXStaticText", value="Second message body text here"),
            ],
        )
        window = make_ax_element(role="AXWindow", children=[msg1, msg2])

        with patch(
            "autocompleter.input_observer.ax_get_attribute",
            side_effect=_ax_get_attribute_dispatcher,
        ):
            turns = observer._extract_conversation_turns(
                window, max_turns=15, app_name="UnknownChatApp"
            )

        assert turns is not None
        assert len(turns) == 2


# ===========================================================================
# Tests for _collect_child_text subrole filtering
# ===========================================================================

class TestCollectChildTextSubroleFiltering:
    """Tests for subrole-based filtering in _collect_child_text.

    These tests use the REAL _collect_child_text (not the mock) to verify
    that AXGroup elements with chrome subroles are skipped.
    """

    def test_skips_document_note_groups(self):
        """AXGroup with AXDocumentNote subrole should be skipped (disclaimers)."""
        # Use the real _collect_child_text by re-patching over the autouse mock
        with patch(
            "autocompleter.conversation_extractors._collect_child_text",
            wraps=_collect_child_text,
        ):
            container = make_ax_element(
                role="AXGroup",
                children=[
                    make_ax_element(role="AXStaticText", value="Hello world"),
                    make_ax_element(
                        role="AXGroup",
                        subrole="AXDocumentNote",
                        children=[
                            make_ax_element(
                                role="AXStaticText",
                                value="Claude is AI and can make mistakes",
                            ),
                        ],
                    ),
                ],
            )
            result = _collect_child_text(container)
            assert "Hello world" in result
            assert "can make mistakes" not in result

    def test_skips_application_status_groups(self):
        """AXGroup with AXApplicationStatus subrole should be skipped."""
        with patch(
            "autocompleter.conversation_extractors._collect_child_text",
            wraps=_collect_child_text,
        ):
            container = make_ax_element(
                role="AXGroup",
                children=[
                    make_ax_element(role="AXStaticText", value="Actual content"),
                    make_ax_element(
                        role="AXGroup",
                        subrole="AXApplicationStatus",
                        children=[
                            make_ax_element(
                                role="AXStaticText", value="Loading...",
                            ),
                        ],
                    ),
                ],
            )
            result = _collect_child_text(container)
            assert "Actual content" in result
            assert "Loading" not in result

    def test_skips_application_alert_groups(self):
        """AXGroup with AXApplicationAlert subrole should be skipped."""
        with patch(
            "autocompleter.conversation_extractors._collect_child_text",
            wraps=_collect_child_text,
        ):
            container = make_ax_element(
                role="AXGroup",
                children=[
                    make_ax_element(role="AXStaticText", value="Real text"),
                    make_ax_element(
                        role="AXGroup",
                        subrole="AXApplicationAlert",
                        children=[
                            make_ax_element(
                                role="AXStaticText", value="Alert banner",
                            ),
                        ],
                    ),
                ],
            )
            result = _collect_child_text(container)
            assert "Real text" in result
            assert "Alert banner" not in result

    def test_does_not_skip_normal_groups(self):
        """AXGroup without a skip subrole should still be recursed into."""
        with patch(
            "autocompleter.conversation_extractors._collect_child_text",
            wraps=_collect_child_text,
        ):
            container = make_ax_element(
                role="AXGroup",
                children=[
                    make_ax_element(role="AXStaticText", value="Top level"),
                    make_ax_element(
                        role="AXGroup",
                        subrole="",
                        children=[
                            make_ax_element(
                                role="AXStaticText", value="Nested text",
                            ),
                        ],
                    ),
                ],
            )
            result = _collect_child_text(container)
            assert "Top level" in result
            assert "Nested text" in result


class TestClaudeDesktopDisclaimerFiltering:
    """Test that ClaudeDesktopExtractor filters disclaimers via subrole
    rather than hardcoded text matching.
    """

    def _build_user_message(self, text: str):
        return make_ax_element(
            role="AXGroup",
            children=[
                make_ax_element(
                    role="AXGroup",
                    children=[
                        make_ax_element(
                            role="AXGroup",
                            children=[
                                make_ax_element(role="AXStaticText", value=text),
                            ],
                        ),
                    ],
                ),
            ],
        )

    def _build_disclaimer(self, text: str = "Claude is AI and can make mistakes"):
        """Build a disclaimer element with AXDocumentNote subrole."""
        return make_ax_element(
            role="AXGroup",
            children=[
                make_ax_element(
                    role="AXGroup",
                    subrole="AXDocumentNote",
                    children=[
                        make_ax_element(role="AXStaticText", value=text),
                    ],
                ),
            ],
        )

    def _build_message_actions(self):
        return make_ax_element(
            role="AXGroup",
            subrole="AXApplicationGroup",
            description="Message actions",
            children=[
                make_ax_element(role="AXButton", title="Copy"),
            ],
        )

    def _build_window(self, container_children):
        conversation_container = make_ax_element(
            role="AXGroup", children=container_children,
        )
        web_area = make_ax_element(
            role="AXWebArea", title="Claude",
            children=[conversation_container],
        )
        return make_ax_element(role="AXWindow", children=[web_area])

    def test_disclaimer_not_extracted_as_turn(self):
        """Disclaimer with AXDocumentNote subrole should not produce a turn."""
        extractor = ClaudeDesktopExtractor()
        window = self._build_window([
            self._build_user_message("Hello Claude"),
            self._build_message_actions(),
            self._build_disclaimer(),
        ])

        # Use the real _collect_child_text so subrole filtering is active
        with patch(
            "autocompleter.conversation_extractors._collect_child_text",
            wraps=_collect_child_text,
        ):
            turns = extractor.extract(window)
        assert turns is not None
        assert len(turns) == 1
        assert turns[0].speaker == "User"
        assert "Hello Claude" in turns[0].text
        # Disclaimer should not appear
        for turn in turns:
            assert "can make mistakes" not in turn.text


# ===========================================================================
# Tests for DiscordExtractor
# ===========================================================================

class TestDiscordExtractor:
    """Unit tests for DiscordExtractor.

    Discord's AX tree:
      AXWindow
        ├─ ... > AXList(subrole=AXContentList, desc="Messages in <name>")
        │   └─ AXGroup (header with AXImage desc=contact_name)
        │   └─ AXGroup > AXGroup(subrole=AXDocumentArticle, rdesc="message",
        │                         title="Speaker , text , timestamp")
        │   └─ AXSplitter (date separator, desc="December 14, 2025")
        └─ AXGroup(subrole=AXLandmarkRegion, desc="User status and settings")
            └─ ... > AXStaticText(value="<current_username>")
    """

    def _build_discord_message(
        self, speaker: str, text: str, timestamp: str = "10:05 PM",
    ) -> MagicMock:
        """Build a mock Discord message element.

        Returns an AXGroup wrapping the article element.
        """
        title = f"{speaker} , {text} , {timestamp}"
        article = make_ax_element(
            role="AXGroup",
            subrole="AXDocumentArticle",
            role_description="message",
            title=title,
            children=[
                make_ax_element(
                    role="AXHeading",
                    children=[
                        make_ax_element(role="AXStaticText", value=speaker),
                        make_ax_element(role="AXStaticText", value=timestamp),
                    ],
                ),
                make_ax_element(
                    role="AXGroup",
                    children=[
                        make_ax_element(role="AXStaticText", value=text),
                    ],
                ),
            ],
        )
        # Outer wrapper group (as seen in real AX tree)
        return make_ax_element(role="AXGroup", children=[article])

    def _build_discord_header(self, contact_name: str) -> MagicMock:
        """Build the DM header element with an AXImage bearing the contact name."""
        return make_ax_element(
            role="AXGroup",
            children=[
                make_ax_element(
                    role="AXImage",
                    description=contact_name,
                ),
                make_ax_element(
                    role="AXStaticText",
                    value=f"This is the beginning of your direct message history with {contact_name}.",
                ),
            ],
        )

    def _build_status_bar(self, username: str) -> MagicMock:
        """Build the 'User status and settings' landmark region."""
        return make_ax_element(
            role="AXGroup",
            subrole="AXLandmarkRegion",
            description="User status and settings",
            children=[
                make_ax_element(
                    role="AXGroup",
                    children=[
                        make_ax_element(role="AXStaticText", value=username),
                    ],
                ),
                make_ax_element(
                    role="AXButton",
                    description="Manage profile and status",
                ),
            ],
        )

    def _build_date_separator(self, date_text: str = "December 14, 2025") -> MagicMock:
        """Build a date separator (AXSplitter)."""
        return make_ax_element(
            role="AXSplitter",
            description=date_text,
        )

    def _build_discord_tree(
        self,
        messages: list,
        contact_name: str = "Bankim",
        *,
        current_username: str = "henryz2004",
        include_header: bool = True,
        include_separator: bool = False,
        include_status_bar: bool = True,
    ) -> MagicMock:
        """Build a full mock Discord AX tree.

        ``messages`` is a list of (speaker, text) or (speaker, text, timestamp) tuples.
        """
        msg_list_children = []

        if include_header:
            msg_list_children.append(self._build_discord_header(contact_name))

        if include_separator:
            msg_list_children.append(self._build_date_separator())

        for msg in messages:
            if len(msg) == 3:
                speaker, text, ts = msg
                msg_list_children.append(self._build_discord_message(speaker, text, ts))
            else:
                speaker, text = msg
                msg_list_children.append(self._build_discord_message(speaker, text))

        msg_list = make_ax_element(
            role="AXList",
            subrole="AXContentList",
            description=f"Messages in {contact_name}",
            children=msg_list_children,
        )

        window_children = [msg_list]
        if include_status_bar:
            window_children.append(self._build_status_bar(current_username))

        window = make_ax_element(role="AXWindow", children=window_children)
        return window

    # -- Basic extraction --

    def test_extracts_basic_conversation(self):
        """Basic Discord DM extraction with speaker attribution via status bar."""
        extractor = DiscordExtractor()
        window = self._build_discord_tree([
            ("Bankim", "hi.."),
            ("henryz2004", "hey! i saw your issue regarding billing"),
            ("Bankim", "thank you for the solution."),
        ])

        turns = extractor.extract(window)
        assert turns is not None
        assert len(turns) == 3
        assert turns[0].speaker == "Bankim"
        assert turns[0].text == "hi.."
        assert turns[1].speaker == "You"
        assert "billing" in turns[1].text
        assert turns[2].speaker == "Bankim"
        assert "solution" in turns[2].text

    def test_you_attribution_via_status_bar(self):
        """Status bar username detection labels current user's messages as 'You'."""
        extractor = DiscordExtractor()
        window = self._build_discord_tree(
            [
                ("henryz2004", "Message one"),
                ("Bankim", "Message two"),
                ("henryz2004", "Message three"),
            ],
            contact_name="Bankim",
            current_username="henryz2004",
        )

        turns = extractor.extract(window)
        assert turns is not None
        speakers = [t.speaker for t in turns]
        assert speakers == ["You", "Bankim", "You"]

    def test_dm_target_fallback_without_status_bar(self):
        """Without status bar, falls back to DM header image for attribution."""
        extractor = DiscordExtractor()
        window = self._build_discord_tree(
            [
                ("henryz2004", "Message one"),
                ("Bankim", "Message two"),
                ("henryz2004", "Message three"),
            ],
            contact_name="Bankim",
            include_status_bar=False,
        )

        turns = extractor.extract(window)
        assert turns is not None
        speakers = [t.speaker for t in turns]
        # DM target fallback: non-Bankim → "You"
        assert speakers == ["You", "Bankim", "You"]

    def test_no_header_no_status_bar_raw_names(self):
        """Without header or status bar, raw speaker names are used."""
        extractor = DiscordExtractor()
        window = self._build_discord_tree(
            [
                ("Alice", "Hello"),
                ("Bob", "Hi there"),
            ],
            contact_name="Alice",
            include_header=False,
            include_status_bar=False,
        )

        turns = extractor.extract(window)
        assert turns is not None
        assert len(turns) == 2
        assert turns[0].speaker == "Alice"
        assert turns[1].speaker == "Bob"

    def test_group_channel_with_status_bar(self):
        """In group channels, status bar detects current user among many."""
        extractor = DiscordExtractor()
        window = self._build_discord_tree(
            [
                ("Alice", "Hey everyone"),
                ("myuser", "Hi Alice!"),
                ("Bob", "What's up"),
                ("myuser", "Not much"),
            ],
            contact_name="general",
            current_username="myuser",
            include_header=False,  # no DM header in group channels
        )

        turns = extractor.extract(window)
        assert turns is not None
        speakers = [t.speaker for t in turns]
        assert speakers == ["Alice", "You", "Bob", "You"]

    # -- Timestamp parsing --

    def test_strips_short_timestamp(self):
        """Short timestamps like '10:05 PM' are stripped from text."""
        extractor = DiscordExtractor()

        msg_list = make_ax_element(
            role="AXList",
            subrole="AXContentList",
            description="Messages in Test",
            children=[
                self._build_discord_header("Test"),
                self._build_discord_message("Alice", "hello", "3:45 PM"),
            ],
        )
        window = make_ax_element(role="AXWindow", children=[msg_list])

        turns = extractor.extract(window)
        assert turns is not None
        assert turns[0].text == "hello"
        assert "PM" not in turns[0].text

    def test_strips_full_timestamp(self):
        """Full timestamps like '12/14/25, 11:22 AM' are stripped."""
        extractor = DiscordExtractor()

        # Build a message with full timestamp in title
        title = "Alice , hey there , 12/14/25, 11:22 AM"
        article = make_ax_element(
            role="AXGroup",
            subrole="AXDocumentArticle",
            role_description="message",
            title=title,
            children=[],
        )
        wrapper = make_ax_element(role="AXGroup", children=[article])

        msg_list = make_ax_element(
            role="AXList",
            subrole="AXContentList",
            description="Messages in Alice",
            children=[
                self._build_discord_header("Alice"),
                wrapper,
            ],
        )
        window = make_ax_element(role="AXWindow", children=[msg_list])

        turns = extractor.extract(window)
        assert turns is not None
        assert turns[0].text == "hey there"

    # -- Edited messages --

    def test_strips_edited_suffix(self):
        """'(edited)' suffix should be removed from message body."""
        extractor = DiscordExtractor()

        title = "Alice , some edited message (edited) , 10:05 PM"
        article = make_ax_element(
            role="AXGroup",
            subrole="AXDocumentArticle",
            role_description="message",
            title=title,
            children=[],
        )
        wrapper = make_ax_element(role="AXGroup", children=[article])

        msg_list = make_ax_element(
            role="AXList",
            subrole="AXContentList",
            description="Messages in Alice",
            children=[
                self._build_discord_header("Alice"),
                wrapper,
            ],
        )
        window = make_ax_element(role="AXWindow", children=[msg_list])

        turns = extractor.extract(window)
        assert turns is not None
        assert "(edited)" not in turns[0].text
        assert turns[0].text == "some edited message"

    def test_strips_edited_with_expanded_timestamp(self):
        """'(edited) Monday, December 15, 2025 at 10:08 PM' suffix should be removed."""
        extractor = DiscordExtractor()

        title = "Alice , updated text (edited) Monday, December 15, 2025 at 10:08 PM , 10:05 PM"
        article = make_ax_element(
            role="AXGroup",
            subrole="AXDocumentArticle",
            role_description="message",
            title=title,
            children=[],
        )
        wrapper = make_ax_element(role="AXGroup", children=[article])

        msg_list = make_ax_element(
            role="AXList",
            subrole="AXContentList",
            description="Messages in Alice",
            children=[
                self._build_discord_header("Alice"),
                wrapper,
            ],
        )
        window = make_ax_element(role="AXWindow", children=[msg_list])

        turns = extractor.extract(window)
        assert turns is not None
        assert "(edited)" not in turns[0].text
        assert "Monday" not in turns[0].text
        assert turns[0].text == "updated text"

    # -- Sticker / media skipping --

    def test_skips_sticker_messages(self):
        """Messages starting with 'Sticker,' should be skipped."""
        extractor = DiscordExtractor()

        title_sticker = "Alice , Sticker, pepe , 10:05 PM"
        article_sticker = make_ax_element(
            role="AXGroup",
            subrole="AXDocumentArticle",
            role_description="message",
            title=title_sticker,
            children=[],
        )
        sticker_wrapper = make_ax_element(role="AXGroup", children=[article_sticker])

        title_normal = "Alice , hello , 10:06 PM"
        article_normal = make_ax_element(
            role="AXGroup",
            subrole="AXDocumentArticle",
            role_description="message",
            title=title_normal,
            children=[],
        )
        normal_wrapper = make_ax_element(role="AXGroup", children=[article_normal])

        msg_list = make_ax_element(
            role="AXList",
            subrole="AXContentList",
            description="Messages in Alice",
            children=[
                self._build_discord_header("Alice"),
                sticker_wrapper,
                normal_wrapper,
            ],
        )
        window = make_ax_element(role="AXWindow", children=[msg_list])

        turns = extractor.extract(window)
        assert turns is not None
        assert len(turns) == 1
        assert turns[0].text == "hello"

    def test_skips_image_messages(self):
        """Messages starting with 'Image' should be skipped."""
        extractor = DiscordExtractor()

        title = "Bob , Image , 10:05 PM"
        article = make_ax_element(
            role="AXGroup",
            subrole="AXDocumentArticle",
            role_description="message",
            title=title,
            children=[],
        )
        wrapper = make_ax_element(role="AXGroup", children=[article])

        msg_list = make_ax_element(
            role="AXList",
            subrole="AXContentList",
            description="Messages in Bob",
            children=[
                self._build_discord_header("Bob"),
                wrapper,
            ],
        )
        window = make_ax_element(role="AXWindow", children=[msg_list])

        turns = extractor.extract(window)
        # Only media message → no turns → falls back to ActionDelimited
        # which also finds nothing → returns None
        assert turns is None

    # -- Date separators --

    def test_skips_date_separators(self):
        """AXSplitter date separators should not produce turns."""
        extractor = DiscordExtractor()
        window = self._build_discord_tree(
            [
                ("Alice", "first message"),
                ("Bob", "second message"),
            ],
            contact_name="Alice",
            include_separator=True,
        )

        turns = extractor.extract(window)
        assert turns is not None
        # Separators are not AXGroup with roleDescription="message", so skipped
        assert len(turns) == 2

    # -- Empty / edge cases --

    def test_no_message_list_falls_back(self):
        """Without a message list, should fall back to ActionDelimited."""
        extractor = DiscordExtractor()
        window = make_ax_element(
            role="AXWindow",
            children=[
                make_ax_element(role="AXGroup", children=[]),
            ],
        )

        turns = extractor.extract(window)
        # ActionDelimited also finds nothing → None
        assert turns is None

    def test_empty_message_list_falls_back(self):
        """Message list with no messages falls back."""
        extractor = DiscordExtractor()

        msg_list = make_ax_element(
            role="AXList",
            subrole="AXContentList",
            description="Messages in Empty",
            children=[],
        )
        window = make_ax_element(role="AXWindow", children=[msg_list])

        turns = extractor.extract(window)
        assert turns is None

    def test_message_without_title_skipped(self):
        """Message article without a title attribute should be skipped."""
        extractor = DiscordExtractor()

        article = make_ax_element(
            role="AXGroup",
            subrole="AXDocumentArticle",
            role_description="message",
            title="",  # empty title
            children=[],
        )
        wrapper = make_ax_element(role="AXGroup", children=[article])

        msg_list = make_ax_element(
            role="AXList",
            subrole="AXContentList",
            description="Messages in Test",
            children=[
                self._build_discord_header("Test"),
                wrapper,
            ],
        )
        window = make_ax_element(role="AXWindow", children=[msg_list])

        turns = extractor.extract(window)
        assert turns is None

    # -- max_turns --

    def test_max_turns_limiting(self):
        """Extraction should respect max_turns parameter."""
        extractor = DiscordExtractor()
        messages = [(f"User{i}", f"Message number {i}") for i in range(10)]
        window = self._build_discord_tree(messages, contact_name="User0")

        turns = extractor.extract(window, max_turns=4)
        assert turns is not None
        assert len(turns) == 4

    # -- Fallback body extraction --

    def test_fallback_body_from_children(self):
        """When title has no text body, fall back to child-tree extraction."""
        extractor = DiscordExtractor()

        # Title with speaker but empty body after split
        title = "Alice ,  , 10:05 PM"
        article = make_ax_element(
            role="AXGroup",
            subrole="AXDocumentArticle",
            role_description="message",
            title=title,
            children=[
                make_ax_element(role="AXHeading", children=[]),
                make_ax_element(
                    role="AXGroup",
                    children=[
                        make_ax_element(role="AXStaticText", value="fallback body text"),
                    ],
                ),
            ],
        )
        wrapper = make_ax_element(role="AXGroup", children=[article])

        msg_list = make_ax_element(
            role="AXList",
            subrole="AXContentList",
            description="Messages in Alice",
            children=[
                self._build_discord_header("Alice"),
                wrapper,
            ],
        )
        window = make_ax_element(role="AXWindow", children=[msg_list])

        turns = extractor.extract(window)
        assert turns is not None
        assert turns[0].text == "fallback body text"

    # -- Timestamp preservation --

    def test_timestamp_short_format(self):
        """Short timestamp (e.g. '10:05 PM') is preserved in ConversationTurn."""
        extractor = DiscordExtractor()
        tree = self._build_discord_tree(
            messages=[("Alice", "hello", "10:05 PM")],
            current_username="Bob",
        )
        turns = extractor.extract(tree)
        assert turns is not None
        assert turns[0].timestamp == "10:05 PM"

    def test_timestamp_full_format(self):
        """Full timestamp (e.g. '12/14/25, 11:22 AM') is preserved."""
        extractor = DiscordExtractor()
        tree = self._build_discord_tree(
            messages=[("Alice", "hello", "12/14/25, 11:22 AM")],
            current_username="Bob",
        )
        turns = extractor.extract(tree)
        assert turns is not None
        assert turns[0].timestamp == "12/14/25, 11:22 AM"

    def test_timestamp_relative_yesterday(self):
        """Relative timestamp ('Yesterday at 8:47 PM') is preserved."""
        extractor = DiscordExtractor()
        tree = self._build_discord_tree(
            messages=[("Alice", "hello", "Yesterday at 8:47 PM")],
            current_username="Bob",
        )
        turns = extractor.extract(tree)
        assert turns is not None
        assert turns[0].timestamp == "Yesterday at 8:47 PM"
        # Message body should NOT contain the timestamp
        assert "Yesterday" not in turns[0].text

    def test_timestamp_relative_today(self):
        """Relative timestamp ('Today at 3:22 PM') is preserved."""
        extractor = DiscordExtractor()
        tree = self._build_discord_tree(
            messages=[("Alice", "hello", "Today at 3:22 PM")],
            current_username="Bob",
        )
        turns = extractor.extract(tree)
        assert turns is not None
        assert turns[0].timestamp == "Today at 3:22 PM"
        assert "Today" not in turns[0].text

    def test_timestamp_preserved_after_you_relabeling(self):
        """Timestamp survives 'You' relabeling via status bar."""
        extractor = DiscordExtractor()
        tree = self._build_discord_tree(
            messages=[("Bob", "my message", "10:05 PM")],
            current_username="Bob",
        )
        turns = extractor.extract(tree)
        assert turns is not None
        assert turns[0].speaker == "You"
        assert turns[0].timestamp == "10:05 PM"

    # -- Speaker badge cleanup --

    def test_strips_op_original_poster_badge(self):
        """'Bankim OP Original Poster' should become 'Bankim'."""
        extractor = DiscordExtractor()
        window = self._build_discord_tree(
            messages=[("Bankim OP Original Poster", "some text")],
            contact_name="Account Issue",
            include_header=False,
            include_status_bar=False,
        )
        turns = extractor.extract(window)
        assert turns is not None
        assert turns[0].speaker == "Bankim"

    def test_strips_op_only_badge(self):
        """'Alice OP' should become 'Alice'."""
        extractor = DiscordExtractor()
        window = self._build_discord_tree(
            messages=[("Alice OP", "thread starter message")],
            contact_name="Thread",
            include_header=False,
            include_status_bar=False,
        )
        turns = extractor.extract(window)
        assert turns is not None
        assert turns[0].speaker == "Alice"

    def test_strips_original_poster_badge(self):
        """'Bob Original Poster' should become 'Bob'."""
        extractor = DiscordExtractor()
        window = self._build_discord_tree(
            messages=[("Bob Original Poster", "message")],
            contact_name="Thread",
            include_header=False,
            include_status_bar=False,
        )
        turns = extractor.extract(window)
        assert turns is not None
        assert turns[0].speaker == "Bob"

    def test_no_badge_unchanged(self):
        """Normal speaker names without badges should be left unchanged."""
        extractor = DiscordExtractor()
        window = self._build_discord_tree(
            messages=[("Charlie", "hello")],
            contact_name="Thread",
            include_header=False,
            include_status_bar=False,
        )
        turns = extractor.extract(window)
        assert turns is not None
        assert turns[0].speaker == "Charlie"

    def test_badge_stripped_before_you_relabeling(self):
        """Badge should be stripped before 'You' relabeling in threads."""
        extractor = DiscordExtractor()
        # Status bar says username is "henryz2004"
        # Message has "henryz2004 OP Original Poster" as speaker
        window = self._build_discord_tree(
            messages=[
                ("henryz2004 OP Original Poster", "my thread message"),
                ("Alice", "reply to thread"),
            ],
            contact_name="Account Issue",
            current_username="henryz2004",
            include_header=False,
        )
        turns = extractor.extract(window)
        assert turns is not None
        assert turns[0].speaker == "You"  # Badge stripped, then relabeled
        assert turns[1].speaker == "Alice"

    # -- Server tag stripping --

    def test_strips_server_tag(self):
        """'MGpai Server Tag: PALM' should become 'MGpai'."""
        extractor = DiscordExtractor()
        window = self._build_discord_tree(
            messages=[("MGpai Server Tag: PALM", "Hi Henry")],
            contact_name="Thread",
            include_header=False,
            include_status_bar=False,
        )
        turns = extractor.extract(window)
        assert turns is not None
        assert turns[0].speaker == "MGpai"

    def test_strips_server_tag_single_word(self):
        """Server tags with single-word tags should be stripped."""
        extractor = DiscordExtractor()
        window = self._build_discord_tree(
            messages=[("Alice Server Tag: VIP", "hello")],
            contact_name="Thread",
            include_header=False,
            include_status_bar=False,
        )
        turns = extractor.extract(window)
        assert turns is not None
        assert turns[0].speaker == "Alice"

    def test_no_server_tag_unchanged(self):
        """Normal names without 'Server Tag:' should be left unchanged."""
        extractor = DiscordExtractor()
        window = self._build_discord_tree(
            messages=[("Charlie", "hello")],
            contact_name="Thread",
            include_header=False,
            include_status_bar=False,
        )
        turns = extractor.extract(window)
        assert turns is not None
        assert turns[0].speaker == "Charlie"

    def test_server_tag_stripped_before_you_relabeling(self):
        """Server tag should be stripped before 'You' relabeling."""
        extractor = DiscordExtractor()
        window = self._build_discord_tree(
            messages=[
                ("henryz2004 Server Tag: DEV", "my message"),
                ("Alice", "reply"),
            ],
            contact_name="Server Channel",
            current_username="henryz2004",
            include_header=False,
        )
        turns = extractor.extract(window)
        assert turns is not None
        assert turns[0].speaker == "You"
        assert turns[1].speaker == "Alice"

    # -- Display name matching --

    def _build_status_bar_with_display_name(
        self, username: str, display_name: str,
    ) -> MagicMock:
        """Build a status bar with both username and display name."""
        return make_ax_element(
            role="AXGroup",
            subrole="AXLandmarkRegion",
            description="User status and settings",
            children=[
                make_ax_element(
                    role="AXGroup",
                    children=[
                        make_ax_element(role="AXStaticText", value=username),
                        make_ax_element(role="AXStaticText", value=display_name),
                    ],
                ),
                make_ax_element(
                    role="AXButton",
                    description="Manage profile and status",
                ),
            ],
        )

    def test_display_name_attribution_in_server_channel(self):
        """Display name should be matched for 'You' attribution in server channels."""
        extractor = DiscordExtractor()
        msg_list = make_ax_element(
            role="AXList",
            subrole="AXContentList",
            description="Messages in Account Issue",
            children=[
                self._build_discord_message("Henry", "my message"),
                self._build_discord_message("Alice", "their message"),
            ],
        )
        status_bar = self._build_status_bar_with_display_name("henryz2004", "Henry")
        window = make_ax_element(role="AXWindow", children=[msg_list, status_bar])

        turns = extractor.extract(window)
        assert turns is not None
        assert len(turns) == 2
        assert turns[0].speaker == "You"  # "Henry" matches display name
        assert turns[1].speaker == "Alice"

    def test_display_name_same_as_username_no_duplicate_match(self):
        """When display name equals username, attribution should still work."""
        extractor = DiscordExtractor()
        msg_list = make_ax_element(
            role="AXList",
            subrole="AXContentList",
            description="Messages in general",
            children=[
                self._build_discord_message("alice123", "my message"),
                self._build_discord_message("Bob", "their message"),
            ],
        )
        # Same name for both — display_name will be empty (dedup in code)
        status_bar = self._build_status_bar_with_display_name("alice123", "alice123")
        window = make_ax_element(role="AXWindow", children=[msg_list, status_bar])

        turns = extractor.extract(window)
        assert turns is not None
        assert turns[0].speaker == "You"
        assert turns[1].speaker == "Bob"

    def test_no_display_name_username_only(self):
        """When no display name is available, username matching still works."""
        extractor = DiscordExtractor()
        window = self._build_discord_tree(
            messages=[
                ("henryz2004", "my message"),
                ("Alice", "their message"),
            ],
            contact_name="general",
            current_username="henryz2004",
            include_header=False,
        )
        turns = extractor.extract(window)
        assert turns is not None
        assert turns[0].speaker == "You"
        assert turns[1].speaker == "Alice"

    # -- Graceful failure --

    def test_graceful_failure_returns_none(self):
        """Exception during extraction should not crash — falls back."""
        extractor = DiscordExtractor()

        bad = MagicMock()
        bad._ax_attrs = {"AXRole": "AXWindow"}

        def raise_on_children(el, attr):
            if attr == "AXChildren":
                raise RuntimeError("AX failure")
            return getattr(el, "_ax_attrs", {}).get(attr)

        with patch(
            "autocompleter.conversation_extractors.ax_get_attribute",
            side_effect=raise_on_children,
        ):
            turns = extractor.extract(bad)
            assert turns is None

    # -- get_extractor registry --

    def test_get_extractor_returns_discord(self):
        """get_extractor('Discord') should return a DiscordExtractor."""
        ext = get_extractor("Discord")
        assert isinstance(ext, DiscordExtractor)
