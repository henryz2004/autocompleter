"""Tests for WhatsApp conversation extractor."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from autocompleter.conversation_extractors import (
    ConversationTurn,
    WhatsAppExtractor,
    get_extractor,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ax_dispatch(element, attribute):
    """Mock ax_get_attribute using _ax_attrs dict."""
    attrs = getattr(element, "_ax_attrs", None)
    if attrs is not None:
        return attrs.get(attribute)
    return None


def _ax_children_dispatch(element):
    """Mock ax_get_children using _ax_attrs dict."""
    attrs = getattr(element, "_ax_attrs", None)
    if attrs is not None:
        return attrs.get("AXChildren") or []
    return []


def make_ax(role="", subrole="", rdesc="", value=None, title="",
            desc="", children=None):
    """Build a MagicMock AX element with the given attributes."""
    elem = MagicMock()
    elem._ax_attrs = {
        "AXRole": role,
        "AXSubrole": subrole,
        "AXRoleDescription": rdesc,
        "AXValue": value,
        "AXTitle": title,
        "AXDescription": desc,
        "AXChildren": children,
    }
    return elem


def _build_whatsapp_window(chat_name, table_desc, messages, nav_heading_value=None):
    """Build a minimal WhatsApp AX tree for testing.

    Parameters
    ----------
    chat_name : str
        Name shown in the Nav bar heading description.
    table_desc : str
        Description for the message table (e.g., "Messages in chat with X").
    messages : list[dict]
        Each dict has keys: role (default "AXStaticText"), desc, value (default "").
    nav_heading_value : str or None
        Optional AXValue for the nav heading (e.g., group member list).
    """
    # Build message children
    msg_children = []
    for msg in messages:
        msg_children.append(make_ax(
            role=msg.get("role", "AXStaticText"),
            desc=msg.get("desc", ""),
            value=msg.get("value", ""),
            rdesc=msg.get("rdesc", ""),
            children=msg.get("children"),
        ))

    table = make_ax(
        role="AXGroup", rdesc="table", desc=table_desc,
        children=msg_children,
    )

    nav_heading = make_ax(
        role="AXHeading", rdesc="heading", desc=chat_name,
        value=nav_heading_value,
    )
    nav_bar = make_ax(
        role="AXGroup", rdesc="Nav bar",
        children=[nav_heading],
    )

    # Wrap in a basic WhatsApp window structure
    chat_panel = make_ax(
        role="AXGroup", children=[nav_bar, table],
    )
    window = make_ax(
        role="AXWindow", subrole="AXStandardWindow",
        children=[chat_panel],
    )
    return window


# ---------------------------------------------------------------------------
# get_extractor dispatch
# ---------------------------------------------------------------------------

class TestWhatsAppDispatch:
    def test_dispatches_with_ltr_mark(self):
        ext = get_extractor("\u200eWhatsApp")
        assert isinstance(ext, WhatsAppExtractor)

    def test_dispatches_without_ltr_mark(self):
        ext = get_extractor("WhatsApp")
        assert isinstance(ext, WhatsAppExtractor)


# ---------------------------------------------------------------------------
# Message parsing (unit tests for internal methods)
# ---------------------------------------------------------------------------

class TestWhatsAppMessageParsing:
    """Test the message description parsing logic directly."""

    @pytest.fixture
    def ext(self):
        return WhatsAppExtractor()

    # -- sent messages --

    def test_parse_sent_basic(self, ext):
        desc = "Your message, hello there!, March4,at4:10 PM, Sent to John, Red"
        turn = ext._parse_message(desc, "John")
        assert turn is not None
        assert turn.speaker == "Me"
        assert turn.text == "hello there!"

    def test_parse_sent_blue_status(self, ext):
        desc = "Your message, test msg, March4,at4:10 PM, Sent to Jane, Blue"
        turn = ext._parse_message(desc, "Jane")
        assert turn.speaker == "Me"
        assert turn.text == "test msg"

    def test_parse_sent_with_year(self, ext):
        desc = "Your message, old msg, May4,2025at8:14 AM, Sent to Group, Red"
        turn = ext._parse_message(desc, "Group")
        assert turn.text == "old msg"

    def test_parse_sent_to_phone_number(self, ext):
        desc = "Your message, yes, March4,at5:15 PM, Sent to + 5 2,9 9 8,2 2 7,2 5 1 7, Red"
        turn = ext._parse_message(desc, "Hotel")
        assert turn.speaker == "Me"
        assert turn.text == "yes"

    def test_parse_sent_multiline(self, ext):
        desc = (
            "Your message, line one\nline two\nline three, "
            "March4,at10:10 PM, Sent to You, Red"
        )
        turn = ext._parse_message(desc, "You")
        assert "line one" in turn.text
        assert "line three" in turn.text

    # -- received group messages --

    def test_parse_received_group_named_sender(self, ext):
        desc = (
            "Message from Jerry Wu, Looking good!, "
            "March1,at1:49 AM, Received in Wework survivors"
        )
        turn = ext._parse_message(desc, "Wework survivors")
        assert turn.speaker == "Jerry Wu"
        assert turn.text == "Looking good!"

    def test_parse_received_group_phone_sender(self, ext):
        desc = (
            "Message from + 1,8 4 0,2 1 8,1 9 0 0, hey people!, "
            "May17,2025at8:51 AM, Received in Marina Plunge Club"
        )
        turn = ext._parse_message(desc, "Marina Plunge Club")
        assert turn.speaker == "+ 1,8 4 0,2 1 8,1 9 0 0"
        assert turn.text == "hey people!"

    def test_parse_received_group_long_message(self, ext):
        desc = (
            "Message from Maybe Nathan Davenport, "
            "Good morning everyone! It's a beautiful morning for plunging!, "
            "May4,2025at8:14 AM, Received in Marina Plunge Club"
        )
        turn = ext._parse_message(desc, "Marina Plunge Club")
        assert turn.speaker == "Maybe Nathan Davenport"
        assert "Good morning everyone!" in turn.text

    def test_parse_received_group_link_preview(self, ext):
        desc = (
            "Message from Richard Song, Link, here's our clawhub skill: "
            "https://clawhub.ai/byungkyu/api-gateway, API Gateway, "
            "March1,at1:48 AM, Received in Wework survivors"
        )
        turn = ext._parse_message(desc, "Wework survivors")
        assert turn.speaker == "Richard Song"
        assert "clawhub" in turn.text

    # -- received 1:1 messages --

    def test_parse_received_dm(self, ext):
        desc = (
            "message, Sure, please provide your full name, "
            "March4,at4:16 PM, Received from + 5 2,9 9 8,2 2 7,2 5 1 7"
        )
        turn = ext._parse_message(desc, "Bahía Tolok Hotel Boutique")
        assert turn.speaker == "Bahía Tolok Hotel Boutique"
        assert "Sure, please provide your full name" in turn.text

    def test_parse_received_dm_edited(self, ext):
        desc = (
            "message, OK, would that be the days from March 17th to 20th?, "
            "March4,at5:01 PM, Received from + 5 2,9 9 8,2 2 7,2 5 1 7, Edited"
        )
        turn = ext._parse_message(desc, "Hotel")
        assert turn is not None
        assert "would that be the days" in turn.text

    def test_parse_received_dm_short(self, ext):
        desc = "message, Ready, March4,at6:38 PM, Received from + 5 2,9 9 8,2 2 7,2 5 1 7"
        turn = ext._parse_message(desc, "Hotel")
        assert turn.text == "Ready"

    # -- skipped messages --

    def test_skip_photo(self, ext):
        desc = "Photo, March4,at3:11 PM, Received from + 5 2,9 9 8,2 2 7,2 5 1 7"
        # Media messages are skipped in _extract_turns, not _parse_message
        # So _parse_message returns None for unrecognized prefixes
        turn = ext._parse_message(desc, "Hotel")
        assert turn is None

    def test_skip_system_message(self, ext):
        desc = "Use WhatsApp on your phone to see older messages."
        turn = ext._parse_message(desc, "Chat")
        assert turn is None

    # -- U+200E handling --

    def test_ltr_marks_stripped(self, ext):
        # Descriptions as they appear in raw AX tree with U+200E
        desc = (
            "\u200eYour message, hello, "
            "March4,at4:10 PM, \u200eSent to John, \u200eRed"
        )
        # strip U+200E as _extract_turns does
        desc_clean = desc.replace("\u200e", "").strip()
        turn = ext._parse_message(desc_clean, "John")
        assert turn.speaker == "Me"
        assert turn.text == "hello"


# ---------------------------------------------------------------------------
# Full extraction with mock AX tree
# ---------------------------------------------------------------------------

class TestWhatsAppFullExtraction:
    """Integration tests using mock AX trees."""

    @pytest.fixture
    def ext(self):
        return WhatsAppExtractor()

    def test_group_chat_extraction(self, ext):
        messages = [
            {"role": "AXHeading", "desc": ""},  # date separator
            {"desc": "Message from Alice, Hello everyone!, March1,at10:00 AM, Received in Test Group"},
            {"desc": "Message from Bob, Hi Alice!, March1,at10:01 AM, Received in Test Group"},
            {"desc": "Your message, Hey all!, March1,at10:02 AM, Sent to Test Group, Red"},
        ]
        window = _build_whatsapp_window(
            "Test Group",
            "\u200eMessages in chat with Test Group",
            messages,
        )
        with patch(
            "autocompleter.conversation_extractors.ax_get_attribute",
            side_effect=_ax_dispatch,
        ), patch(
            "autocompleter.conversation_extractors.ax_get_children",
            side_effect=_ax_children_dispatch,
        ):
            turns = ext.extract(window)

        assert turns is not None
        assert len(turns) == 3
        assert turns[0] == ConversationTurn(speaker="Alice", text="Hello everyone!")
        assert turns[1] == ConversationTurn(speaker="Bob", text="Hi Alice!")
        assert turns[2] == ConversationTurn(speaker="Me", text="Hey all!")

    def test_dm_extraction(self, ext):
        messages = [
            {"desc": "Your message, hi, March4,at4:10 PM, Sent to + 5 2,1 2 3, Red"},
            {"desc": "message, hey there, March4,at4:11 PM, Received from + 5 2,1 2 3"},
        ]
        window = _build_whatsapp_window(
            "Contact Name",
            "\u200eMessages in chat with Contact Name",
            messages,
        )
        with patch(
            "autocompleter.conversation_extractors.ax_get_attribute",
            side_effect=_ax_dispatch,
        ), patch(
            "autocompleter.conversation_extractors.ax_get_children",
            side_effect=_ax_children_dispatch,
        ):
            turns = ext.extract(window)

        assert turns is not None
        assert len(turns) == 2
        assert turns[0].speaker == "Me"
        assert turns[0].text == "hi"
        assert turns[1].speaker == "Contact Name"
        assert turns[1].text == "hey there"

    def test_skips_date_separators_and_unread_markers(self, ext):
        messages = [
            {"role": "AXHeading", "desc": "May 4, 2025"},
            {"role": "AXButton", "desc": "40 unread messages"},
            {"desc": "Message from X, hello, May4,2025at8:14 AM, Received in G"},
        ]
        window = _build_whatsapp_window("G", "\u200eMessages in chat with G", messages)
        with patch(
            "autocompleter.conversation_extractors.ax_get_attribute",
            side_effect=_ax_dispatch,
        ), patch(
            "autocompleter.conversation_extractors.ax_get_children",
            side_effect=_ax_children_dispatch,
        ):
            turns = ext.extract(window)

        assert turns is not None
        assert len(turns) == 1
        assert turns[0].speaker == "X"

    def test_skips_system_and_media(self, ext):
        messages = [
            # System message inside AXGroup wrapper (skipped by role check)
            {"role": "AXGroup", "desc": "Use WhatsApp on your phone to see older messages."},
            # Photo (skipped as media)
            {"desc": "Photo, March4,at3:11 PM, Received from + 5 2,1 2 3"},
            # Actual message
            {"desc": "Your message, test, March4,at4:10 PM, Sent to Chat, Red"},
        ]
        window = _build_whatsapp_window("Chat", "\u200eMessages in chat with Chat", messages)
        with patch(
            "autocompleter.conversation_extractors.ax_get_attribute",
            side_effect=_ax_dispatch,
        ), patch(
            "autocompleter.conversation_extractors.ax_get_children",
            side_effect=_ax_children_dispatch,
        ):
            turns = ext.extract(window)

        assert turns is not None
        assert len(turns) == 1
        assert turns[0].text == "test"

    def test_max_turns_limit(self, ext):
        messages = [
            {"desc": f"Message from A, msg{i}, March1,at10:0{i} AM, Received in G"}
            for i in range(10)
        ]
        window = _build_whatsapp_window("G", "\u200eMessages in chat with G", messages)
        with patch(
            "autocompleter.conversation_extractors.ax_get_attribute",
            side_effect=_ax_dispatch,
        ), patch(
            "autocompleter.conversation_extractors.ax_get_children",
            side_effect=_ax_children_dispatch,
        ):
            turns = ext.extract(window, max_turns=5)

        assert turns is not None
        assert len(turns) == 5

    def test_no_table_returns_none(self, ext):
        window = make_ax(role="AXWindow", children=[
            make_ax(role="AXGroup", children=[]),
        ])
        with patch(
            "autocompleter.conversation_extractors.ax_get_attribute",
            side_effect=_ax_dispatch,
        ), patch(
            "autocompleter.conversation_extractors.ax_get_children",
            side_effect=_ax_children_dispatch,
        ):
            assert ext.extract(window) is None

    def test_chat_name_from_nav_bar(self, ext):
        """Verify chat name is extracted from the Nav bar heading."""
        messages = [
            {"desc": "message, hi, March4,at4:10 PM, Received from + 1,2 3 4"},
        ]
        window = _build_whatsapp_window(
            "Bahía Tolok Hotel Boutique",
            "\u200eMessages in chat with Bahía Tolok Hotel Boutique",
            messages,
        )
        with patch(
            "autocompleter.conversation_extractors.ax_get_attribute",
            side_effect=_ax_dispatch,
        ), patch(
            "autocompleter.conversation_extractors.ax_get_children",
            side_effect=_ax_children_dispatch,
        ):
            turns = ext.extract(window)

        assert turns is not None
        assert turns[0].speaker == "Bahía Tolok Hotel Boutique"


# ---------------------------------------------------------------------------
# Timestamp regex edge cases
# ---------------------------------------------------------------------------

class TestTimestampStripping:
    @pytest.fixture
    def ext(self):
        return WhatsAppExtractor()

    def test_strip_with_year(self, ext):
        assert ext._strip_timestamp("hello, May4,2025at8:14 AM") == "hello"

    def test_strip_without_year(self, ext):
        assert ext._strip_timestamp("hello, March4,at4:10 PM") == "hello"

    def test_no_timestamp_returns_text(self, ext):
        assert ext._strip_timestamp("no timestamp here") == "no timestamp here"

    def test_empty_after_strip_returns_none(self, ext):
        assert ext._strip_timestamp(", March4,at4:10 PM") is None


# ---------------------------------------------------------------------------
# Phone number splitting
# ---------------------------------------------------------------------------

class TestPhoneNumberSplitting:
    @pytest.fixture
    def ext(self):
        return WhatsAppExtractor()

    def test_named_contact(self, ext):
        sender, text = ext._split_sender_text("Jerry Wu, hello")
        assert sender == "Jerry Wu"
        assert text == "hello"

    def test_phone_number(self, ext):
        sender, text = ext._split_sender_text(
            "+ 1,8 4 0,2 1 8,1 9 0 0, hey people!"
        )
        assert sender == "+ 1,8 4 0,2 1 8,1 9 0 0"
        assert text == "hey people!"

    def test_international_phone(self, ext):
        sender, text = ext._split_sender_text(
            "+ 5 2,9 9 8,2 2 7,2 5 1 7, hello"
        )
        assert sender == "+ 5 2,9 9 8,2 2 7,2 5 1 7"
        assert text == "hello"

    def test_name_with_spaces(self, ext):
        sender, text = ext._split_sender_text(
            "Maybe Nathan Davenport, Good morning"
        )
        assert sender == "Maybe Nathan Davenport"
        assert text == "Good morning"
