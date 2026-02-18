"""Persistent context store backed by SQLite.

Accumulates text content observed through the accessibility API over time.
Indexes by source (app, URL if available, timestamp). Provides sliced context
to the suggestion engine: recent observations + relevant historical context.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ContextEntry:
    id: int | None
    source_app: str
    source_url: str
    content: str
    timestamp: float
    entry_type: str  # "visible_text", "user_input", "conversation"


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
            CREATE TABLE IF NOT EXISTS context_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_app TEXT NOT NULL,
                source_url TEXT DEFAULT '',
                content TEXT NOT NULL,
                timestamp REAL NOT NULL,
                entry_type TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_timestamp
                ON context_entries(timestamp DESC);

            CREATE INDEX IF NOT EXISTS idx_source_app
                ON context_entries(source_app);

            CREATE INDEX IF NOT EXISTS idx_entry_type
                ON context_entries(entry_type);
            """
        )

    def add_entry(
        self,
        source_app: str,
        content: str,
        entry_type: str,
        source_url: str = "",
        timestamp: float | None = None,
    ) -> int:
        """Store a new context entry. Returns the entry ID."""
        if timestamp is None:
            timestamp = time.time()

        conn = self._get_conn()

        # Deduplicate: skip if identical content was stored in the last 5 seconds
        cursor = conn.execute(
            """
            SELECT id FROM context_entries
            WHERE content = ? AND source_app = ? AND timestamp > ?
            LIMIT 1
            """,
            (content, source_app, timestamp - 5.0),
        )
        if cursor.fetchone():
            return -1

        cursor = conn.execute(
            """
            INSERT INTO context_entries (source_app, source_url, content, timestamp, entry_type)
            VALUES (?, ?, ?, ?, ?)
            """,
            (source_app, source_url, content, timestamp, entry_type),
        )
        conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    def get_recent(self, limit: int = 50) -> list[ContextEntry]:
        """Get the most recent context entries."""
        conn = self._get_conn()
        cursor = conn.execute(
            """
            SELECT id, source_app, source_url, content, timestamp, entry_type
            FROM context_entries
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [ContextEntry(*row) for row in cursor.fetchall()]

    def get_by_source(
        self, source_app: str, limit: int = 30
    ) -> list[ContextEntry]:
        """Get recent entries from a specific app."""
        conn = self._get_conn()
        cursor = conn.execute(
            """
            SELECT id, source_app, source_url, content, timestamp, entry_type
            FROM context_entries
            WHERE source_app = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (source_app, limit),
        )
        return [ContextEntry(*row) for row in cursor.fetchall()]

    def search(self, query: str, limit: int = 20) -> list[ContextEntry]:
        """Simple text search across context entries."""
        conn = self._get_conn()
        cursor = conn.execute(
            """
            SELECT id, source_app, source_url, content, timestamp, entry_type
            FROM context_entries
            WHERE content LIKE ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (f"%{query}%", limit),
        )
        return [ContextEntry(*row) for row in cursor.fetchall()]

    def get_sliced_context(
        self, source_app: str, max_chars: int = 4000
    ) -> str:
        """Build a context string for the LLM from recent + relevant entries.

        Returns a text block combining recent entries from the current app
        with relevant historical entries, trimmed to max_chars.
        """
        parts: list[str] = []
        total_chars = 0

        # Recent entries from the current app (most relevant)
        app_entries = self.get_by_source(source_app, limit=20)
        for entry in app_entries:
            line = f"[{entry.entry_type}] {entry.content}"
            if total_chars + len(line) > max_chars:
                break
            parts.append(line)
            total_chars += len(line)

        # Fill remaining space with recent entries from any source
        if total_chars < max_chars:
            recent = self.get_recent(limit=30)
            for entry in recent:
                if entry.source_app == source_app:
                    continue  # Already included above
                line = f"[{entry.source_app}] {entry.content}"
                if total_chars + len(line) > max_chars:
                    break
                parts.append(line)
                total_chars += len(line)

        return "\n".join(parts)

    def prune(self, max_age_hours: int = 72, max_entries: int = 5000) -> int:
        """Remove old entries to keep the store bounded. Returns count removed."""
        conn = self._get_conn()
        cutoff = time.time() - (max_age_hours * 3600)

        # Remove entries older than max_age_hours
        cursor = conn.execute(
            "DELETE FROM context_entries WHERE timestamp < ?", (cutoff,)
        )
        removed = cursor.rowcount

        # If still over max_entries, remove oldest
        count_cursor = conn.execute(
            "SELECT COUNT(*) FROM context_entries"
        )
        count = count_cursor.fetchone()[0]
        if count > max_entries:
            excess = count - max_entries
            conn.execute(
                """
                DELETE FROM context_entries WHERE id IN (
                    SELECT id FROM context_entries
                    ORDER BY timestamp ASC
                    LIMIT ?
                )
                """,
                (excess,),
            )
            removed += excess

        conn.commit()
        return removed

    def entry_count(self) -> int:
        conn = self._get_conn()
        cursor = conn.execute("SELECT COUNT(*) FROM context_entries")
        return cursor.fetchone()[0]
