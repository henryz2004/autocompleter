"""Persistent context store backed by SQLite.

Accumulates text content observed through the accessibility API over time.
Indexes by source (app, URL if available, timestamp). Provides sliced context
to the suggestion engine: recent observations + relevant historical context.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .embeddings import EmbeddingProvider

logger = logging.getLogger(__name__)


@dataclass
class ContextEntry:
    id: int | None
    source_app: str
    source_url: str
    content: str
    timestamp: float
    entry_type: str  # "visible_text", "user_input", "conversation"
    window_title: str = ""


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
                entry_type TEXT NOT NULL,
                window_title TEXT DEFAULT '',
                embeddings BLOB DEFAULT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_timestamp
                ON context_entries(timestamp DESC);

            CREATE INDEX IF NOT EXISTS idx_source_app
                ON context_entries(source_app);

            CREATE INDEX IF NOT EXISTS idx_entry_type
                ON context_entries(entry_type);

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
        # Migrate: add window_title column if missing (existing DBs)
        try:
            conn.execute("SELECT window_title FROM context_entries LIMIT 1")
        except sqlite3.OperationalError:
            conn.execute(
                "ALTER TABLE context_entries ADD COLUMN window_title TEXT DEFAULT ''"
            )
            conn.commit()
        # Migrate: add embeddings column if missing (existing DBs)
        try:
            conn.execute("SELECT embeddings FROM context_entries LIMIT 1")
        except sqlite3.OperationalError:
            conn.execute(
                "ALTER TABLE context_entries ADD COLUMN embeddings BLOB DEFAULT NULL"
            )
            conn.commit()

    def add_entry(
        self,
        source_app: str,
        content: str,
        entry_type: str,
        source_url: str = "",
        window_title: str = "",
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
            INSERT INTO context_entries
                (source_app, source_url, content, timestamp, entry_type, window_title)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (source_app, source_url, content, timestamp, entry_type, window_title),
        )
        conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    _SELECT_COLS = (
        "id, source_app, source_url, content, timestamp, entry_type, window_title"
    )

    def get_recent(self, limit: int = 50) -> list[ContextEntry]:
        """Get the most recent context entries."""
        conn = self._get_conn()
        cursor = conn.execute(
            f"""
            SELECT {self._SELECT_COLS}
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
            f"""
            SELECT {self._SELECT_COLS}
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
            f"""
            SELECT {self._SELECT_COLS}
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

    def get_continuation_context(
        self,
        before_cursor: str,
        after_cursor: str,
        source_app: str,
        window_title: str = "",
        source_url: str = "",
        max_local_chars: int = 600,
        visible_text: list[str] | None = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        use_semantic_context: bool = False,
        cross_app_context: str = "",
    ) -> str:
        """Build lean context for continuation mode.

        Tier 1 (mandatory): before_cursor and after_cursor -- sent raw.
        Tier 2 (live surroundings): visible text from the current window,
            passed directly rather than pulled from stale DB entries.
            Falls back to recent DB entries if visible_text is not provided.
        Tier 2.5a (cross-app): recent context from other apps the user
            visited, if provided.
        Tier 2.5b (semantic): If embedding_provider is given and
            use_semantic_context is True, append semantically relevant
            historical entries.
        Tier 3 (light metadata): app, window_title, url. No timestamps.
        """
        parts: list[str] = []

        # Tier 3: metadata header (light)
        meta_parts = [f"App: {source_app}"]
        if window_title:
            meta_parts.append(f"Window: {window_title}")
        if source_url:
            meta_parts.append(f"URL: {source_url}")
        parts.append(" | ".join(meta_parts))

        # Tier 2.5: cross-app context (between metadata and visible text)
        if cross_app_context:
            parts.append(cross_app_context)

        # Tier 2: live visible text (preferred) or recent DB entries (fallback)
        local_chars = 0
        local_parts: list[str] = []
        if visible_text:
            for element in visible_text:
                snippet = element.strip()
                if not snippet:
                    continue
                # Skip elements that are just the cursor text itself (avoid duplication)
                if snippet == before_cursor.strip() or snippet == after_cursor.strip():
                    continue
                remaining = max_local_chars - local_chars
                if remaining <= 0:
                    break
                snippet = snippet[:remaining]
                local_parts.append(snippet)
                local_chars += len(snippet)
        else:
            # Fallback: pull from DB
            app_entries = self.get_by_source(source_app, limit=5)
            for entry in app_entries:
                if entry.entry_type == "user_input":
                    continue  # Skip raw input -- we have cursor state
                remaining = max_local_chars - local_chars
                if remaining <= 0:
                    break
                snippet = entry.content[:remaining]
                if snippet.strip():
                    local_parts.append(snippet.strip())
                    local_chars += len(snippet)
        if local_parts:
            parts.append("Visible context:\n" + "\n".join(local_parts))

        # Tier 2.5: semantic context (optional)
        if use_semantic_context and embedding_provider is not None:
            query = before_cursor.strip()
            if query:
                try:
                    semantic_entries = self.get_semantically_relevant(
                        query=query,
                        provider=embedding_provider,
                        top_k=3,
                        max_age_hours=24,
                    )
                    # Deduplicate against already-included visible context
                    existing = set(local_parts)
                    semantic_parts = [
                        s for s in semantic_entries if s not in existing
                    ]
                    if semantic_parts:
                        parts.append(
                            "Related context:\n" + "\n".join(semantic_parts[:3])
                        )
                except Exception:
                    logger.debug(
                        "Semantic context lookup failed in continuation mode",
                        exc_info=True,
                    )

        # Tier 1: cursor state (raw, always included)
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
        visible_text: list[str] | None = None,
        max_age_seconds: float = 300.0,
        embedding_provider: Optional[EmbeddingProvider] = None,
        use_semantic_context: bool = False,
        cross_app_context: str = "",
    ) -> str:
        """Build context for reply mode.

        Tier 1: Structured recent turns with speaker labels.
        Tier 1.5: Semantic context blended with conversation turns.
        Tier 2: Draft state if user has typed a partial reply.
        Tier 2.5: Cross-app context from recently visited apps.
        Tier 3: Metadata with timestamps (useful for pacing).

        When conversation_turns is empty, falls back to:
          1. Live visible_text (if provided)
          2. Recent DB entries (filtered by max_age_seconds, default 5 min)
        """
        parts: list[str] = []

        # Tier 3: metadata
        meta_parts = [f"App: {source_app}"]
        if window_title:
            meta_parts.append(f"Channel: {window_title}")
        if source_url:
            meta_parts.append(f"URL: {source_url}")
        parts.append(" | ".join(meta_parts))

        # Tier 2.5: cross-app context
        if cross_app_context:
            parts.append(cross_app_context)

        # Tier 1: conversation turns
        turns = conversation_turns[-max_turns:]
        if turns:
            turn_lines = []
            for turn in turns:
                speaker = turn.get("speaker", "Unknown")
                text = turn.get("text", "")
                turn_lines.append(f"- {speaker}: {text}")
            parts.append("Conversation:\n" + "\n".join(turn_lines))
        else:
            # Fallback 1: live visible text from the current window
            fallback_parts: list[str] = []
            total = 0
            if visible_text:
                for element in visible_text:
                    snippet = element.strip()
                    if not snippet:
                        continue
                    if total + len(snippet) > 1500:
                        break
                    fallback_parts.append(snippet)
                    total += len(snippet)
            # Fallback 2: recent DB entries (time-filtered)
            if not fallback_parts:
                cutoff = time.time() - max_age_seconds
                app_entries = self.get_by_source(source_app, limit=10)
                for entry in app_entries:
                    if entry.timestamp < cutoff:
                        continue
                    if total + len(entry.content) > 1500:
                        break
                    fallback_parts.append(entry.content)
                    total += len(entry.content)
            if fallback_parts:
                parts.append(
                    "Visible page content (no conversation detected):\n"
                    + "\n".join(fallback_parts)
                )

        # Tier 1.5: semantic context (optional)
        if use_semantic_context and embedding_provider is not None:
            # Build query from last conversation turn or draft
            query = ""
            if turns:
                query = turns[-1].get("text", "")
            elif draft_text.strip():
                query = draft_text.strip()

            if query:
                try:
                    semantic_entries = self.get_semantically_relevant(
                        query=query,
                        provider=embedding_provider,
                        top_k=3,
                        max_age_hours=24,
                    )
                    if semantic_entries:
                        parts.append(
                            "Related context:\n" + "\n".join(semantic_entries[:3])
                        )
                except Exception:
                    logger.debug(
                        "Semantic context lookup failed in reply mode",
                        exc_info=True,
                    )

        # Tier 2: draft state
        if draft_text.strip():
            parts.append(f"Draft so far:\n{draft_text}")

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
            SELECT suggestion_text FROM suggestion_feedback
            WHERE {where}
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            params,
        )
        return [row[0] for row in cursor.fetchall()]

    # ---- Semantic context ----

    def _serialize_embedding(self, vector: list[float]) -> bytes:
        """Serialize an embedding vector to bytes for SQLite storage."""
        return json.dumps(vector).encode("utf-8")

    def _deserialize_embedding(self, blob: bytes) -> list[float]:
        """Deserialize an embedding vector from SQLite bytes."""
        return json.loads(blob.decode("utf-8"))

    def _cache_embedding(self, entry_id: int, vector: list[float]) -> None:
        """Store a computed embedding in the database."""
        conn = self._get_conn()
        conn.execute(
            "UPDATE context_entries SET embeddings = ? WHERE id = ?",
            (self._serialize_embedding(vector), entry_id),
        )
        conn.commit()

    def get_semantically_relevant(
        self,
        query: str,
        provider: EmbeddingProvider,
        top_k: int = 5,
        max_age_hours: float = 24,
    ) -> list[str]:
        """Find the most semantically relevant stored entries for a query.

        Uses cached embeddings when available; computes and caches them lazily
        for entries that don't have them yet.

        Args:
            query: The text to find relevant context for.
            provider: An EmbeddingProvider instance.
            top_k: Number of top results to return.
            max_age_hours: Only consider entries from the last N hours.

        Returns:
            List of content strings sorted by semantic relevance.
        """
        from .embeddings import cosine_similarity

        if not query.strip():
            return []

        conn = self._get_conn()
        cutoff = time.time() - (max_age_hours * 3600)

        cursor = conn.execute(
            """
            SELECT id, content, embeddings
            FROM context_entries
            WHERE timestamp > ?
            ORDER BY timestamp DESC
            LIMIT 200
            """,
            (cutoff,),
        )
        rows = cursor.fetchall()
        if not rows:
            return []

        # Separate entries with and without cached embeddings
        entries_with_cache: list[tuple[int, str, list[float]]] = []
        entries_without_cache: list[tuple[int, str, int]] = []  # (id, content, index)

        for row in rows:
            entry_id, content, emb_blob = row
            if emb_blob is not None:
                vector = self._deserialize_embedding(emb_blob)
                entries_with_cache.append((entry_id, content, vector))
            else:
                entries_without_cache.append(
                    (entry_id, content, len(entries_with_cache) + len(entries_without_cache))
                )

        # Compute missing embeddings
        if entries_without_cache:
            texts_to_embed = [content for _, content, _ in entries_without_cache]
            # Include query so vocabulary is shared (important for TF-IDF)
            all_texts = texts_to_embed + [query]
            all_vectors = provider.embed(all_texts)

            if all_vectors and len(all_vectors) == len(all_texts):
                for i, (entry_id, content, _) in enumerate(entries_without_cache):
                    vector = all_vectors[i]
                    self._cache_embedding(entry_id, vector)
                    entries_with_cache.append((entry_id, content, vector))
                query_vector_from_batch = all_vectors[-1]
            else:
                query_vector_from_batch = None
        else:
            query_vector_from_batch = None

        # Compute query embedding
        if query_vector_from_batch is not None:
            query_vector = query_vector_from_batch
        else:
            # All entries had cached embeddings; embed query alone with corpus
            all_contents = [content for _, content, _ in entries_with_cache]
            all_texts = all_contents + [query]
            all_vectors = provider.embed(all_texts)
            if all_vectors and len(all_vectors) == len(all_texts):
                query_vector = all_vectors[-1]
            else:
                return []

        # Score entries by cosine similarity
        scored: list[tuple[str, float]] = []
        for _, content, vector in entries_with_cache:
            score = cosine_similarity(query_vector, vector)
            scored.append((content, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [content for content, _ in scored[:top_k]]

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
