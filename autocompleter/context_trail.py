"""Cross-app context trail.

Tracks app switches and maintains a rolling window of recent app snapshots
so the suggestion engine can include context from recently visited apps
(e.g. reading docs in Chrome -> writing code in VS Code).
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional

from .input_observer import VisibleContent


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
        self._current_content: Optional[VisibleContent] = None
        self._text_summary_chars = text_summary_chars

    def record(self, visible_content: VisibleContent) -> None:
        """Record an observation from the observe loop.

        If the app has changed since the last call, push a snapshot of
        the *previous* app's content onto the trail.
        """
        app_name = visible_content.app_name
        window_title = visible_content.window_title

        if self._current_app and app_name != self._current_app:
            # App switch detected -- snapshot the previous app
            self._push_snapshot()

        # Update tracking state
        self._current_app = app_name
        self._current_window = window_title
        self._current_content = visible_content

    def _push_snapshot(self) -> None:
        """Push a snapshot of the current (soon-to-be-previous) app."""
        if not self._current_content:
            return

        text_summary = self._build_text_summary(self._current_content)

        # Deduplication: don't record consecutive entries for the same
        # app+window combination.
        if self._trail:
            last = self._trail[-1]
            if (
                last.app_name == self._current_app
                and last.window_title == self._current_window
            ):
                # Update the existing entry's timestamp and summary instead
                # of adding a duplicate.
                self._trail[-1] = AppSnapshot(
                    app_name=self._current_app,
                    window_title=self._current_window,
                    text_summary=text_summary,
                    timestamp=time.time(),
                    source_url=self._current_content.url,
                )
                return

        snapshot = AppSnapshot(
            app_name=self._current_app,
            window_title=self._current_window,
            text_summary=text_summary,
            timestamp=time.time(),
            source_url=self._current_content.url,
        )
        self._trail.append(snapshot)

    def _build_text_summary(self, content: VisibleContent) -> str:
        """Build a truncated text summary from visible content."""
        if not content.text_elements:
            return ""
        combined = "\n".join(content.text_elements)
        if len(combined) > self._text_summary_chars:
            combined = combined[: self._text_summary_chars]
        return combined

    def get_recent_cross_app_context(
        self,
        current_app: str,
        max_age_seconds: float = 60.0,
        max_entries: int = 3,
    ) -> List[AppSnapshot]:
        """Return recent snapshots from apps other than current_app.

        Returns most-recent-first, limited to max_entries, excluding
        entries older than max_age_seconds.
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
        self._current_content = None
