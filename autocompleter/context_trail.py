"""Cross-app context trail.

Tracks app switches and maintains a rolling window of recent app snapshots
so the suggestion engine can include context from recently visited apps
(e.g. reading docs in Chrome -> writing code in VS Code).
"""

from __future__ import annotations

import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional


# Default cap on how much visible text to store per snapshot.
DEFAULT_TEXT_SUMMARY_CHARS = 500


@dataclass
class AppSnapshot:
    """A point-in-time snapshot of an app's visible content."""

    app_name: str
    window_title: str
    text_summary: str  # First N chars of visible content
    timestamp: float
    source_url: str = ""


class ContextTrail:
    """Maintains a rolling trail of recent app switches.

    Each time the user switches to a different app, a snapshot of the
    *previous* app's visible content is recorded. This allows the
    suggestion engine to include relevant cross-app context when
    generating completions.
    """

    def __init__(
        self,
        maxlen: int = 10,
        text_summary_chars: int = DEFAULT_TEXT_SUMMARY_CHARS,
    ) -> None:
        self._trail: deque[AppSnapshot] = deque(maxlen=maxlen)
        self._current_app: str = ""
        self._current_window: str = ""
        self._current_subtree: str = ""
        self._current_url: str = ""
        self._text_summary_chars = text_summary_chars

    def record(
        self,
        app_name: str,
        window_title: str,
        subtree_xml: str = "",
        url: str = "",
    ) -> None:
        """Record an observation from the observe loop.

        If the app or window has changed since the last call, push a
        snapshot of the *previous* content onto the trail.
        """
        if self._current_app and (
            app_name != self._current_app
            or window_title != self._current_window
        ):
            self._push_snapshot()

        self._current_app = app_name
        self._current_window = window_title
        self._current_subtree = subtree_xml
        self._current_url = url

    def _push_snapshot(self) -> None:
        """Push a snapshot of the current (soon-to-be-previous) app."""
        if not self._current_app:
            return

        text_summary = self._build_text_summary(self._current_subtree)

        if self._trail:
            last = self._trail[-1]
            if (
                last.app_name == self._current_app
                and last.window_title == self._current_window
            ):
                self._trail[-1] = AppSnapshot(
                    app_name=self._current_app,
                    window_title=self._current_window,
                    text_summary=text_summary,
                    timestamp=time.time(),
                    source_url=self._current_url,
                )
                return

        snapshot = AppSnapshot(
            app_name=self._current_app,
            window_title=self._current_window,
            text_summary=text_summary,
            timestamp=time.time(),
            source_url=self._current_url,
        )
        self._trail.append(snapshot)

    def _build_text_summary(self, subtree_xml: str) -> str:
        """Build a text summary by stripping XML tags from subtree context."""
        if not subtree_xml:
            return ""
        # Strip XML tags to get plain text
        plain = re.sub(r"<[^>]+>", " ", subtree_xml)
        plain = " ".join(plain.split())
        if len(plain) > self._text_summary_chars:
            plain = plain[: self._text_summary_chars]
        return plain

    # Apps whose cross-app snapshots are typically notification previews
    # or system-level UI rather than substantive content the user was reading.
    _NOISY_CROSS_APP_SOURCES = frozenset({
        "Notification Center",
        "NotificationCenter",
        "UserNotificationCenter",
        "Control Center",
        "SystemUIServer",
        "Spotlight",
        "loginwindow",
    })

    def get_recent_cross_app_context(
        self,
        current_app: str,
        max_age_seconds: float = 60.0,
        max_entries: int = 3,
    ) -> List[AppSnapshot]:
        """Return recent snapshots from apps other than current_app.

        Returns most-recent-first, limited to max_entries, excluding
        entries older than max_age_seconds and known noisy sources.
        """
        now = time.time()
        cutoff = now - max_age_seconds

        results: List[AppSnapshot] = []
        # Iterate newest-first (deque is ordered oldest-to-newest)
        for snapshot in reversed(self._trail):
            if len(results) >= max_entries:
                break
            if snapshot.timestamp < cutoff:
                continue
            if snapshot.app_name == current_app:
                continue
            # Skip noisy system-level apps
            if snapshot.app_name in self._NOISY_CROSS_APP_SOURCES:
                continue
            # Skip snapshots with no substantive content after filtering
            if not snapshot.text_summary.strip():
                continue
            results.append(snapshot)

        return results

    @staticmethod
    def format_cross_app_context(snapshots: List[AppSnapshot]) -> str:
        """Format snapshots into a readable string for the LLM.

        Example output:
            [Recent activity from other apps]
            - Chrome ("React Hooks Documentation"): useState allows you to...
            - VS Code ("App.tsx"): import { useState } from 'react';...
        """
        if not snapshots:
            return ""

        lines = ["[Recent activity from other apps]"]
        for snap in snapshots:
            title_part = f' ("{snap.window_title}")' if snap.window_title else ""
            # Truncate the summary for display (first line or first 200 chars)
            summary = snap.text_summary.replace("\n", " ").strip()
            if len(summary) > 200:
                summary = summary[:200] + "..."
            lines.append(f"- {snap.app_name}{title_part}: {summary}")

        return "\n".join(lines)

    def clear(self) -> None:
        """Reset the trail."""
        self._trail.clear()
        self._current_app = ""
        self._current_window = ""
        self._current_subtree = ""
        self._current_url = ""
