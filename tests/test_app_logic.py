"""Tests for pure functions in app.py — _extract_first_segment and _hash_content."""

from __future__ import annotations

import pytest

# Import the class to access its static methods.
# We need to be careful because app.py imports heavy macOS deps, but
# these are guarded by try/except, so the import should succeed.
from autocompleter.app import Autocompleter


# ---------------------------------------------------------------------------
# _extract_first_segment tests
# ---------------------------------------------------------------------------

class TestExtractFirstSegment:
    def test_newline_boundary(self):
        """Text with newline returns only the first line."""
        result = Autocompleter._extract_first_segment("Hello world\nMore text here")
        assert result == "Hello world"

    def test_period_boundary(self):
        """Text with '. ' returns up to and including the period."""
        result = Autocompleter._extract_first_segment("First sentence. Second sentence")
        assert result == "First sentence."

    def test_no_boundary_returns_all(self):
        """Text with no sentence boundary returns the full text."""
        result = Autocompleter._extract_first_segment("Hello world")
        assert result == "Hello world"

    def test_empty_string(self):
        result = Autocompleter._extract_first_segment("")
        assert result == ""

    def test_newline_takes_precedence_over_period(self):
        """When both newline and period exist, newline wins."""
        result = Autocompleter._extract_first_segment("First. And more\nSecond line")
        assert result == "First. And more"


# ---------------------------------------------------------------------------
# _hash_content tests
# ---------------------------------------------------------------------------

class TestHashContent:
    def test_deterministic(self):
        """Same input always produces the same hash."""
        h1 = Autocompleter._hash_content("hello world")
        h2 = Autocompleter._hash_content("hello world")
        assert h1 == h2

    def test_different_inputs_different_hashes(self):
        h1 = Autocompleter._hash_content("hello")
        h2 = Autocompleter._hash_content("world")
        assert h1 != h2

    def test_unicode(self):
        """Unicode text should hash without errors."""
        h = Autocompleter._hash_content("Bonjour le monde")
        assert isinstance(h, str)
        assert len(h) == 32  # MD5 hex digest length
