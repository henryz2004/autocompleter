"""Context assembly and suggestion feedback store.

Assembles tiered context for the suggestion engine using subtree XML as the
sole local context source.  Persists suggestion feedback in SQLite for
accept-rate tracking and negative-example filtering.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .shell_parser import ParsedTerminalBuffer

logger = logging.getLogger(__name__)


class ContextStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._opened = False
        self._local = threading.local()

    def open(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._opened = True
        # Ensure tables exist
        conn = self._get_conn()
        self._create_tables(conn)

    def close(self) -> None:
        self._opened = False
        # Close the current thread's connection if any
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    def _get_conn(self) -> sqlite3.Connection:
        """Get or create a connection for the current thread."""
        if not self._opened:
            raise RuntimeError("ContextStore is not open. Call open() first.")
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self.db_path))
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        return conn

    def _create_tables(self, conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS suggestion_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                source_app TEXT NOT NULL,
                mode TEXT NOT NULL,
                suggestion_text TEXT NOT NULL,
                action TEXT NOT NULL,
                suggestion_index INTEGER,
                total_suggestions INTEGER,
                latency_ms REAL
            );

            CREATE INDEX IF NOT EXISTS idx_feedback_timestamp
                ON suggestion_feedback(timestamp DESC);

            CREATE INDEX IF NOT EXISTS idx_feedback_source_app
                ON suggestion_feedback(source_app);

            CREATE INDEX IF NOT EXISTS idx_feedback_action
                ON suggestion_feedback(action);
            """
        )

    def get_continuation_context(
        self,
        before_cursor: str,
        after_cursor: str,
        source_app: str,
        window_title: str = "",
        source_url: str = "",
        cross_app_context: str = "",
        subtree_context: str | None = None,
        memory_context: str = "",
    ) -> str:
        """Build lean context for continuation mode.

        Subtree XML is the sole local context source.  Context tiers:
          1. Metadata: app, window title, URL
          2. Cross-app context from recently visited apps
          3. Long-term memory
          4. Subtree context (nearby content from the focused element)
          5. Cursor state: before/after cursor text (always included, raw)
        """
        parts = []

        # Metadata
        meta_parts = [f"App: {source_app}"]
        if window_title:
            meta_parts.append(f"Window: {window_title}")
        if source_url:
            meta_parts.append(f"URL: {source_url}")
        parts.append(" | ".join(meta_parts))

        # Cross-app
        if cross_app_context:
            parts.append(cross_app_context)

        # Memory
        if memory_context:
            parts.append(memory_context)

        # Subtree
        if subtree_context:
            parts.append(f"Nearby content:\n{subtree_context}")

        # Cursor state
        parts.append(f"Text before cursor:\n{before_cursor}")
        if after_cursor.strip():
            parts.append(f"Text after cursor:\n{after_cursor}")

        return "\n\n".join(parts)

    def get_reply_context(
        self,
        conversation_turns: list[dict[str, str]],
        source_app: str,
        window_title: str = "",
        source_url: str = "",
        draft_text: str = "",
        max_turns: int = 8,
        cross_app_context: str = "",
        subtree_context: str | None = None,
        memory_context: str = "",
    ) -> str:
        """Build context for reply mode.

        Tier 1: Structured recent turns with speaker labels.
        Tier 2: Draft state if user has typed a partial reply.
        Tier 2.5: Cross-app context from recently visited apps.
        Tier 3: Metadata with timestamps (useful for pacing).

        When conversation_turns is empty, falls back to subtree context
        as "Nearby content".
        """
        parts: list[str] = []

        # Tier 3: metadata
        meta_parts = [f"App: {source_app}"]
        if window_title:
            meta_parts.append(f"Channel: {window_title}")
        if source_url:
            meta_parts.append(f"URL: {source_url}")
        parts.append(" | ".join(meta_parts))

        # Tier 2.5a: cross-app context
        if cross_app_context:
            parts.append(cross_app_context)

        # Tier 2.5b: long-term memory
        if memory_context:
            parts.append(memory_context)

        # Tier 1: speaker-labeled conversation turns (primary).
        turns = conversation_turns[-max_turns:]
        if turns:
            turn_lines = []
            for turn in turns:
                speaker = getattr(turn, "speaker", None) or (turn.get("speaker", "Unknown") if isinstance(turn, dict) else "Unknown")
                text = getattr(turn, "text", None) or (turn.get("text", "") if isinstance(turn, dict) else "")
                timestamp = getattr(turn, "timestamp", None) or (turn.get("timestamp", "") if isinstance(turn, dict) else "")
                if timestamp:
                    turn_lines.append(f"- {speaker} [{timestamp}]: {text}")
                else:
                    turn_lines.append(f"- {speaker}: {text}")
            parts.append("Conversation:\n" + "\n".join(turn_lines))

            # If all turns are from the same speaker, supplement with
            # subtree context so the LLM can see what the other side said.
            speakers = {getattr(t, "speaker", None) or (t.get("speaker") if isinstance(t, dict) else None) for t in turns}
            if len(speakers) <= 1 and subtree_context:
                parts.append(
                    "Additional visible content (assistant responses may be here):\n"
                    + subtree_context
                )
                logger.debug(
                    "[CTX] One-sided conversation (%d turns, speaker=%s) — "
                    "supplementing with subtree context",
                    len(turns), speakers.pop() if speakers else "?",
                )
        else:
            # Fallback: subtree context when no structured turns detected
            if subtree_context:
                parts.append(f"Nearby content:\n{subtree_context}")

        # Tier 2: draft state
        if draft_text.strip():
            parts.append(f"Draft so far:\n{draft_text}")

        return "\n\n".join(parts)

    def get_shell_context(
        self,
        parsed: ParsedTerminalBuffer,
        source_app: str,
        window_title: str = "",
        cross_app_context: str = "",
        memory_context: str = "",
    ) -> str:
        """Build context for shell/terminal apps.

        Uses the parsed terminal buffer to assemble structured context
        instead of sending the raw buffer as "Text before cursor."

        Args:
            parsed: ParsedTerminalBuffer from shell_parser.
            source_app: Terminal app name (e.g. "Terminal", "iTerm2").
            window_title: Window/tab title (often shows current dir or process).
            cross_app_context: Cross-app context from recently visited apps.
            memory_context: Long-term memory context (formatted string).

        Returns:
            Assembled context string for the LLM.
        """
        parts: list[str] = []

        # Metadata
        meta_parts = [f"App: {source_app}"]
        if window_title:
            meta_parts.append(f"Window: {window_title}")
        parts.append(" | ".join(meta_parts))

        # Cross-app context
        if cross_app_context:
            parts.append(cross_app_context)

        # Long-term memory
        if memory_context:
            parts.append(memory_context)

        # Recent commands (structured list)
        if parsed.recent_commands:
            cmd_list = "\n".join(f"  {cmd}" for cmd in parsed.recent_commands[-10:])
            parts.append(f"Recent commands:\n{cmd_list}")

        # Recent command output (commands + their output)
        if parsed.recent_output.strip():
            parts.append(
                "Recent terminal session:\n" + parsed.recent_output.strip()
            )

        # Current command state
        if parsed.current_command.strip():
            parts.append(f"Command being typed:\n{parsed.current_command}")
        else:
            parts.append("Command line is empty (at prompt).")

        if parsed.prompt_string:
            parts.append(f"Prompt: {parsed.prompt_string.strip()}")

        return "\n\n".join(parts)

    def record_feedback(
        self,
        source_app: str,
        mode: str,
        suggestion_text: str,
        action: str,
        suggestion_index: int | None = None,
        total_suggestions: int | None = None,
        latency_ms: float | None = None,
    ) -> int:
        """Record feedback on a suggestion (accepted, dismissed, regenerated).

        Returns the feedback entry ID.
        """
        conn = self._get_conn()
        cursor = conn.execute(
            """
            INSERT INTO suggestion_feedback
                (timestamp, source_app, mode, suggestion_text, action,
                 suggestion_index, total_suggestions, latency_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                time.time(),
                source_app,
                mode,
                suggestion_text,
                action,
                suggestion_index,
                total_suggestions,
                latency_ms,
            ),
        )
        conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    def get_feedback_stats(
        self,
        source_app: str | None = None,
        mode: str | None = None,
        hours: int = 24,
    ) -> dict:
        """Get aggregated feedback statistics.

        Returns a dict with:
            total_shown, total_accepted, total_dismissed, accept_rate,
            avg_accepted_index, avg_latency_ms
        """
        conn = self._get_conn()
        cutoff = time.time() - (hours * 3600)

        conditions = ["timestamp > ?"]
        params: list = [cutoff]

        if source_app is not None:
            conditions.append("source_app = ?")
            params.append(source_app)
        if mode is not None:
            conditions.append("mode = ?")
            params.append(mode)

        where = " AND ".join(conditions)

        # Total shown = accepted + dismissed (regenerated doesn't count as
        # a final disposition, but we include it in the total for completeness)
        cursor = conn.execute(
            f"SELECT COUNT(*) FROM suggestion_feedback WHERE {where}",
            params,
        )
        total_shown = cursor.fetchone()[0]

        cursor = conn.execute(
            f"SELECT COUNT(*) FROM suggestion_feedback WHERE {where} AND action = 'accepted'",
            params,
        )
        total_accepted = cursor.fetchone()[0]

        cursor = conn.execute(
            f"SELECT COUNT(*) FROM suggestion_feedback WHERE {where} AND action = 'dismissed'",
            params,
        )
        total_dismissed = cursor.fetchone()[0]

        accept_rate = total_accepted / total_shown if total_shown > 0 else 0.0

        cursor = conn.execute(
            f"SELECT AVG(suggestion_index) FROM suggestion_feedback WHERE {where} AND action = 'accepted' AND suggestion_index IS NOT NULL",
            params,
        )
        row = cursor.fetchone()
        avg_accepted_index = row[0] if row[0] is not None else 0.0

        cursor = conn.execute(
            f"SELECT AVG(latency_ms) FROM suggestion_feedback WHERE {where} AND latency_ms IS NOT NULL",
            params,
        )
        row = cursor.fetchone()
        avg_latency_ms = row[0] if row[0] is not None else 0.0

        return {
            "total_shown": total_shown,
            "total_accepted": total_accepted,
            "total_dismissed": total_dismissed,
            "accept_rate": accept_rate,
            "avg_accepted_index": avg_accepted_index,
            "avg_latency_ms": avg_latency_ms,
        }

    def get_recent_dismissed_patterns(
        self,
        source_app: str | None = None,
        limit: int = 20,
    ) -> list[str]:
        """Return recently dismissed suggestion texts for negative filtering."""
        conn = self._get_conn()
        conditions = ["action = 'dismissed'"]
        params: list = []

        if source_app is not None:
            conditions.append("source_app = ?")
            params.append(source_app)

        where = " AND ".join(conditions)
        params.append(limit)

        cursor = conn.execute(
            f"""
            SELECT DISTINCT suggestion_text FROM suggestion_feedback
            WHERE {where}
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            params,
        )
        # Deduplicate while preserving recency order
        seen: set[str] = set()
        results: list[str] = []
        for (text,) in cursor.fetchall():
            if text not in seen:
                seen.add(text)
                results.append(text)
        return results
