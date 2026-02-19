"""Tests for conversation_extractors module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from autocompleter.conversation_extractors import (
    ChatGPTExtractor,
    ClaudeDesktopExtractor,
    ConversationExtractor,
    ConversationTurn,
    GeminiExtractor,
    GenericExtractor,
    IMessageExtractor,
    SlackExtractor,
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


@pytest.fixture(autouse=True)
def patch_ax():
    """Patch ax_get_attribute for all tests in this module."""
    with patch(
        "autocompleter.conversation_extractors.ax_get_attribute",
        side_effect=_ax_get_attribute_dispatcher,
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

    def test_returns_imessage_extractor(self):
        ext = get_extractor("Messages")
        assert isinstance(ext, IMessageExtractor)

    def test_returns_generic_for_unknown_app(self):
        ext = get_extractor("SomeRandomApp")
        assert isinstance(ext, GenericExtractor)

    def test_returns_generic_for_empty_string(self):
        ext = get_extractor("")
        assert isinstance(ext, GenericExtractor)

    def test_all_extractors_are_conversation_extractor_subclasses(self):
        for name in ["Google Gemini", "Slack", "ChatGPT", "Claude", "Messages", "Unknown"]:
            ext = get_extractor(name)
            assert isinstance(ext, ConversationExtractor)


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
# Tests for ClaudeDesktopExtractor
# ===========================================================================

class TestClaudeDesktopExtractor:
    def test_relabels_assistant_as_claude(self):
        """Claude Desktop should relabel 'ChatGPT' speaker to 'Claude'."""
        extractor = ClaudeDesktopExtractor()

        articles = [
            make_ax_element(
                role="AXGroup",
                subrole="AXArticle",
                role_description="article",
                children=[
                    make_ax_element(role="AXStaticText", value="Hello Claude"),
                ],
            ),
            make_ax_element(
                role="AXGroup",
                subrole="AXArticle",
                role_description="article",
                children=[
                    make_ax_element(
                        role="AXStaticText",
                        value="Hello! How can I help you today?",
                    ),
                ],
            ),
        ]
        container = make_ax_element(role="AXGroup", children=articles)
        window = make_ax_element(role="AXWindow", children=[container])

        turns = extractor.extract(window)
        assert turns is not None
        assert len(turns) == 2
        assert turns[0].speaker == "User"
        assert turns[1].speaker == "Claude"

    def test_falls_back_to_generic(self):
        """If article pattern fails, ClaudeDesktopExtractor tries GenericExtractor."""
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
    def test_extracts_sent_and_received(self):
        extractor = IMessageExtractor()

        sent = make_ax_element(
            role="AXRow",
            subrole="sent",
            children=[
                make_ax_element(role="AXStaticText", value="Hey, are you free tonight?"),
            ],
        )
        received = make_ax_element(
            role="AXRow",
            subrole="received",
            children=[
                make_ax_element(role="AXStaticText", value="Yeah, what's up?"),
            ],
        )
        table = make_ax_element(role="AXTable", children=[sent, received])
        window = make_ax_element(role="AXWindow", children=[table])

        turns = extractor.extract(window)
        assert turns is not None
        assert len(turns) == 2
        assert turns[0].speaker == "Me"
        assert turns[1].speaker == "Them"

    def test_unknown_direction_uses_unknown(self):
        extractor = IMessageExtractor()

        row = make_ax_element(
            role="AXRow",
            children=[
                make_ax_element(role="AXStaticText", value="A message with no direction"),
            ],
        )
        window = make_ax_element(role="AXWindow", children=[row])

        turns = extractor.extract(window)
        assert turns is not None
        assert turns[0].speaker == "Unknown"

    def test_no_messages_returns_none(self):
        extractor = IMessageExtractor()
        window = make_ax_element(
            role="AXWindow",
            children=[
                make_ax_element(role="AXTable", children=[]),
            ],
        )

        turns = extractor.extract(window)
        assert turns is None

    def test_max_turns_limiting(self):
        extractor = IMessageExtractor()
        rows = [
            make_ax_element(
                role="AXRow",
                subrole="sent" if i % 2 == 0 else "received",
                children=[
                    make_ax_element(role="AXStaticText", value=f"Message number {i} text"),
                ],
            )
            for i in range(10)
        ]
        table = make_ax_element(role="AXTable", children=rows)
        window = make_ax_element(role="AXWindow", children=[table])

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
