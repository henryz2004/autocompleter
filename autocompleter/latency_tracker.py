"""Latency tracking for the autocomplete pipeline.

Records per-trigger timing breakdowns across pipeline stages and persists
them to SQLite for trend analysis.

Pipeline stages (in order):
    trigger          — hotkey pressed
    context_ready    — context assembled and ready for LLM
    llm_start        — LLM API call initiated
    first_suggestion — first suggestion text available (streaming)
    llm_done         — all suggestions received from LLM
    displayed        — suggestions shown in overlay
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Stages in pipeline order — used for display and validation
STAGES = (
    "trigger",
    "context_ready",
    "llm_start",
    "first_suggestion",
    "llm_done",
    "displayed",
)


@dataclass
class LatencyRecord:
    """Timing breakdown for a single trigger event."""

    generation_id: int = 0
    timestamp: float = 0.0  # wall-clock time of trigger
    app_name: str = ""
    mode: str = ""
    provider: str = ""
    model: str = ""
    suggestion_count: int = 0
    # Durations in milliseconds (None = stage not reached)
    context_ms: float | None = None       # trigger → context_ready
    llm_ttft_ms: float | None = None      # llm_start → first_suggestion
    llm_total_ms: float | None = None     # llm_start → llm_done
    e2e_first_ms: float | None = None     # trigger → first_suggestion
    e2e_total_ms: float | None = None     # trigger → displayed


class LatencyTracker:
    """Lightweight per-trigger stopwatch.

    Usage::

        tracker = LatencyTracker()
        tracker.start(generation_id=42)
        tracker.mark("trigger")
        # ... assemble context ...
        tracker.mark("context_ready")
        # ... call LLM ...
        tracker.mark("llm_start")
        # ... first result arrives ...
        tracker.mark("first_suggestion")
        # ... all results ...
        tracker.mark("llm_done")
        tracker.mark("displayed")

        record = tracker.finish(app_name="Discord", mode="reply", ...)
    """

    def __init__(self) -> None:
        self._stamps: dict[str, float] = {}
        self._generation_id: int = 0

    def start(self, generation_id: int = 0) -> None:
        """Reset and begin a new timing session."""
        self._stamps = {}
        self._generation_id = generation_id

    def mark(self, stage: str) -> None:
        """Record a timestamp for the given pipeline stage."""
        self._stamps[stage] = time.perf_counter()

    def elapsed_since(self, stage: str) -> float | None:
        """Milliseconds elapsed since *stage* was marked, or None."""
        if stage not in self._stamps:
            return None
        return (time.perf_counter() - self._stamps[stage]) * 1000

    def _delta_ms(self, start: str, end: str) -> float | None:
        """Milliseconds between two stages, or None if either is missing."""
        if start in self._stamps and end in self._stamps:
            return (self._stamps[end] - self._stamps[start]) * 1000
        return None

    def finish(
        self,
        app_name: str = "",
        mode: str = "",
        provider: str = "",
        model: str = "",
        suggestion_count: int = 0,
    ) -> LatencyRecord:
        """Compute the latency record and log a summary line."""
        rec = LatencyRecord(
            generation_id=self._generation_id,
            timestamp=time.time(),
            app_name=app_name,
            mode=mode,
            provider=provider,
            model=model,
            suggestion_count=suggestion_count,
            context_ms=self._delta_ms("trigger", "context_ready"),
            llm_ttft_ms=self._delta_ms("llm_start", "first_suggestion"),
            llm_total_ms=self._delta_ms("llm_start", "llm_done"),
            e2e_first_ms=self._delta_ms("trigger", "first_suggestion"),
            e2e_total_ms=self._delta_ms("trigger", "displayed"),
        )

        # Structured log line for easy grepping
        parts = [f"gen={rec.generation_id}"]
        if rec.context_ms is not None:
            parts.append(f"ctx={rec.context_ms:.0f}ms")
        if rec.llm_ttft_ms is not None:
            parts.append(f"ttft={rec.llm_ttft_ms:.0f}ms")
        if rec.llm_total_ms is not None:
            parts.append(f"llm={rec.llm_total_ms:.0f}ms")
        if rec.e2e_first_ms is not None:
            parts.append(f"e2e_first={rec.e2e_first_ms:.0f}ms")
        if rec.e2e_total_ms is not None:
            parts.append(f"e2e={rec.e2e_total_ms:.0f}ms")
        parts.append(f"n={rec.suggestion_count}")
        parts.append(f"app={rec.app_name}")
        parts.append(f"mode={rec.mode}")

        logger.info("[LATENCY] %s", " | ".join(parts))
        return rec


# ---- SQLite persistence ----

_CREATE_TABLE = """\
CREATE TABLE IF NOT EXISTS latency_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    generation_id INTEGER,
    timestamp REAL,
    app_name TEXT,
    mode TEXT,
    provider TEXT,
    model TEXT,
    suggestion_count INTEGER,
    context_ms REAL,
    llm_ttft_ms REAL,
    llm_total_ms REAL,
    e2e_first_ms REAL,
    e2e_total_ms REAL
)
"""

_INSERT = """\
INSERT INTO latency_metrics
    (generation_id, timestamp, app_name, mode, provider, model,
     suggestion_count, context_ms, llm_ttft_ms, llm_total_ms,
     e2e_first_ms, e2e_total_ms)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


class LatencyStore:
    """Persists latency records to SQLite.

    Thread-safe: uses per-thread connections like ContextStore.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._local = threading.local()

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self._db_path))
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(_CREATE_TABLE)
            conn.commit()
            self._local.conn = conn
        return conn

    def save(self, record: LatencyRecord) -> None:
        """Persist a latency record."""
        conn = self._conn()
        conn.execute(_INSERT, (
            record.generation_id,
            record.timestamp,
            record.app_name,
            record.mode,
            record.provider,
            record.model,
            record.suggestion_count,
            record.context_ms,
            record.llm_ttft_ms,
            record.llm_total_ms,
            record.e2e_first_ms,
            record.e2e_total_ms,
        ))
        conn.commit()

    def get_stats(
        self,
        last_n: int = 0,
        last_hours: float = 0,
    ) -> dict:
        """Compute aggregate latency statistics.

        Args:
            last_n: If > 0, limit to the last N records.
            last_hours: If > 0, limit to records within the last N hours.

        Returns:
            Dict with count, and per-metric: p50, p90, p99, mean, min, max.
        """
        conn = self._conn()
        where_parts = []
        params: list = []

        if last_hours > 0:
            cutoff = time.time() - last_hours * 3600
            where_parts.append("timestamp > ?")
            params.append(cutoff)

        where = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""
        order = "ORDER BY timestamp DESC"
        limit = f"LIMIT {last_n}" if last_n > 0 else ""

        query = f"""
            SELECT context_ms, llm_ttft_ms, llm_total_ms,
                   e2e_first_ms, e2e_total_ms, suggestion_count,
                   app_name, mode, provider, model
            FROM latency_metrics
            {where}
            {order}
            {limit}
        """
        rows = conn.execute(query, params).fetchall()
        if not rows:
            return {"count": 0}

        metrics = {
            "context_ms": [],
            "llm_ttft_ms": [],
            "llm_total_ms": [],
            "e2e_first_ms": [],
            "e2e_total_ms": [],
        }
        for row in rows:
            for i, key in enumerate(metrics):
                if row[i] is not None:
                    metrics[key].append(row[i])

        result: dict = {"count": len(rows)}
        for key, values in metrics.items():
            if not values:
                result[key] = None
                continue
            values.sort()
            n = len(values)
            result[key] = {
                "mean": sum(values) / n,
                "min": values[0],
                "max": values[-1],
                "p50": values[n // 2],
                "p90": values[int(n * 0.9)],
                "p99": values[int(n * 0.99)],
            }

        # Mode and app breakdown
        from collections import Counter
        result["apps"] = dict(Counter(r[6] for r in rows).most_common(10))
        result["modes"] = dict(Counter(r[7] for r in rows).most_common())
        result["providers"] = dict(Counter(r[8] for r in rows if r[8]).most_common())
        result["models"] = dict(Counter(r[9] for r in rows if r[9]).most_common())

        return result


def print_stats(db_path: Path, last_n: int = 50, last_hours: float = 0) -> None:
    """Print a formatted latency report to stdout."""
    store = LatencyStore(db_path)
    stats = store.get_stats(last_n=last_n, last_hours=last_hours)

    if stats["count"] == 0:
        print("No latency data recorded yet.")
        return

    scope = f"last {last_n}" if last_n else "all"
    if last_hours:
        scope = f"last {last_hours:.0f}h"
    print(f"\n{'='*60}")
    print(f"  Latency Report ({scope}, n={stats['count']})")
    print(f"{'='*60}\n")

    metric_labels = {
        "context_ms": "Context assembly",
        "llm_ttft_ms": "LLM TTFT (time to first token)",
        "llm_total_ms": "LLM total",
        "e2e_first_ms": "End-to-end (first suggestion)",
        "e2e_total_ms": "End-to-end (all suggestions)",
    }

    for key, label in metric_labels.items():
        data = stats.get(key)
        if data is None:
            print(f"  {label:40s}  (no data)")
            continue
        print(
            f"  {label:40s}  "
            f"p50={data['p50']:6.0f}ms  "
            f"p90={data['p90']:6.0f}ms  "
            f"mean={data['mean']:6.0f}ms  "
            f"[{data['min']:.0f}-{data['max']:.0f}ms]"
        )

    print()
    if stats.get("apps"):
        print(f"  Apps:      {stats['apps']}")
    if stats.get("modes"):
        print(f"  Modes:     {stats['modes']}")
    if stats.get("providers"):
        print(f"  Providers: {stats['providers']}")
    if stats.get("models"):
        print(f"  Models:    {stats['models']}")
    print()
