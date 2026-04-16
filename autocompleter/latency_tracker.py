"""Latency tracking for the autocomplete pipeline.

Records per-trigger timing breakdowns across pipeline stages and persists
them to SQLite for trend analysis.

Primary stages (in order):
    trigger          — hotkey / auto-trigger / regenerate started
    context_ready    — context assembled and ready for LLM
    llm_start        — LLM API call initiated
    first_suggestion — first suggestion text available (streaming)
    llm_done         — all suggestions received from LLM
    displayed        — final suggestions shown in overlay

The tracker also supports finer-grained ad hoc markers (focused/caret/
subtree/visible/context-build/overlay-first-show) so the app can measure
the current critical path without changing behavior.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from collections import Counter
from dataclasses import dataclass
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
    # Fine-grained measurements
    focus_ms: float | None = None
    caret_ms: float | None = None
    subtree_ms: float | None = None
    visible_fetch_ms: float | None = None
    context_build_ms: float | None = None
    overlay_first_show_ms: float | None = None
    overlay_final_show_ms: float | None = None
    visible_cache_age_ms: float | None = None
    # Trigger dimensions / diagnostics
    trigger_type: str = ""
    visible_source: str = ""
    use_shell: bool = False
    use_tui: bool = False
    has_conversation_turns: bool = False
    used_subtree_context: bool = False
    used_semantic_context: bool = False
    used_memory_context: bool = False
    fallback_used: bool = False
    visible_content_changed: bool | None = None


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

    def profile(self, record: LatencyRecord | None = None) -> dict[str, object]:
        """Return a JSON-safe latency profile for telemetry/debugging."""
        trigger_stamp = self._stamps.get("trigger")
        stage_offsets_ms: dict[str, int] = {}
        if trigger_stamp is not None:
            for stage, stamp in sorted(self._stamps.items(), key=lambda item: item[1]):
                stage_offsets_ms[stage] = int((stamp - trigger_stamp) * 1000)

        durations_ms = {
            "context": self._delta_ms("trigger", "context_ready"),
            "llm_ttft": self._delta_ms("llm_start", "first_suggestion"),
            "llm_total": self._delta_ms("llm_start", "llm_done"),
            "e2e_first": self._delta_ms("trigger", "first_suggestion"),
            "e2e_total": self._delta_ms("trigger", "displayed"),
            "focus": self._delta_ms("trigger", "focused_ready"),
            "caret": self._delta_ms("focused_ready", "caret_ready"),
            "subtree": self._delta_ms("subtree_start", "subtree_ready"),
            "visible_fetch": self._delta_ms("visible_start", "visible_ready"),
            "context_build": self._delta_ms("context_build_start", "context_ready"),
            "overlay_first_show": self._delta_ms("trigger", "overlay_first_show"),
            "overlay_final_show": self._delta_ms("trigger", "displayed"),
        }
        normalized_durations = {
            key: int(value)
            for key, value in durations_ms.items()
            if value is not None
        }
        if record is not None and record.visible_cache_age_ms is not None:
            normalized_durations["visible_cache_age"] = int(record.visible_cache_age_ms)

        return {
            "generation_id": self._generation_id,
            "stage_offsets_ms": stage_offsets_ms,
            "durations_ms": normalized_durations,
        }

    def finish(
        self,
        app_name: str = "",
        mode: str = "",
        provider: str = "",
        model: str = "",
        suggestion_count: int = 0,
        trigger_type: str = "",
        visible_source: str = "",
        use_shell: bool = False,
        use_tui: bool = False,
        has_conversation_turns: bool = False,
        used_subtree_context: bool = False,
        used_semantic_context: bool = False,
        used_memory_context: bool = False,
        fallback_used: bool = False,
        visible_cache_age_ms: float | None = None,
        visible_content_changed: bool | None = None,
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
            focus_ms=self._delta_ms("trigger", "focused_ready"),
            caret_ms=self._delta_ms("focused_ready", "caret_ready"),
            subtree_ms=self._delta_ms("subtree_start", "subtree_ready"),
            visible_fetch_ms=self._delta_ms("visible_start", "visible_ready"),
            context_build_ms=self._delta_ms("context_build_start", "context_ready"),
            overlay_first_show_ms=self._delta_ms("trigger", "overlay_first_show"),
            overlay_final_show_ms=self._delta_ms("trigger", "displayed"),
            visible_cache_age_ms=visible_cache_age_ms,
            trigger_type=trigger_type,
            visible_source=visible_source,
            use_shell=use_shell,
            use_tui=use_tui,
            has_conversation_turns=has_conversation_turns,
            used_subtree_context=used_subtree_context,
            used_semantic_context=used_semantic_context,
            used_memory_context=used_memory_context,
            fallback_used=fallback_used,
            visible_content_changed=visible_content_changed,
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
        if rec.focus_ms is not None:
            parts.append(f"focus={rec.focus_ms:.0f}ms")
        if rec.visible_fetch_ms is not None:
            parts.append(f"visible={rec.visible_fetch_ms:.0f}ms")
        if rec.context_build_ms is not None:
            parts.append(f"build={rec.context_build_ms:.0f}ms")
        parts.append(f"n={rec.suggestion_count}")
        parts.append(f"app={rec.app_name}")
        parts.append(f"mode={rec.mode}")
        if rec.trigger_type:
            parts.append(f"trigger_type={rec.trigger_type}")
        if rec.visible_source:
            parts.append(f"visible_src={rec.visible_source}")
        if rec.fallback_used:
            parts.append("fallback=1")

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
    e2e_total_ms REAL,
    focus_ms REAL,
    caret_ms REAL,
    subtree_ms REAL,
    visible_fetch_ms REAL,
    context_build_ms REAL,
    overlay_first_show_ms REAL,
    overlay_final_show_ms REAL,
    visible_cache_age_ms REAL,
    trigger_type TEXT DEFAULT '',
    visible_source TEXT DEFAULT '',
    use_shell INTEGER DEFAULT 0,
    use_tui INTEGER DEFAULT 0,
    has_conversation_turns INTEGER DEFAULT 0,
    used_subtree_context INTEGER DEFAULT 0,
    used_semantic_context INTEGER DEFAULT 0,
    used_memory_context INTEGER DEFAULT 0,
    fallback_used INTEGER DEFAULT 0,
    visible_content_changed INTEGER
)
"""

_INSERT = """\
INSERT INTO latency_metrics
    (generation_id, timestamp, app_name, mode, provider, model,
     suggestion_count, context_ms, llm_ttft_ms, llm_total_ms,
     e2e_first_ms, e2e_total_ms, focus_ms, caret_ms, subtree_ms,
     visible_fetch_ms, context_build_ms, overlay_first_show_ms,
     overlay_final_show_ms, visible_cache_age_ms, trigger_type,
     visible_source, use_shell, use_tui, has_conversation_turns,
     used_subtree_context, used_semantic_context, used_memory_context,
     fallback_used, visible_content_changed)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            self._migrate(conn)
            conn.commit()
            self._local.conn = conn
        return conn

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        """Add newly introduced columns to existing DBs."""
        columns = {
            "focus_ms": "REAL",
            "caret_ms": "REAL",
            "subtree_ms": "REAL",
            "visible_fetch_ms": "REAL",
            "context_build_ms": "REAL",
            "overlay_first_show_ms": "REAL",
            "overlay_final_show_ms": "REAL",
            "visible_cache_age_ms": "REAL",
            "trigger_type": "TEXT DEFAULT ''",
            "visible_source": "TEXT DEFAULT ''",
            "use_shell": "INTEGER DEFAULT 0",
            "use_tui": "INTEGER DEFAULT 0",
            "has_conversation_turns": "INTEGER DEFAULT 0",
            "used_subtree_context": "INTEGER DEFAULT 0",
            "used_semantic_context": "INTEGER DEFAULT 0",
            "used_memory_context": "INTEGER DEFAULT 0",
            "fallback_used": "INTEGER DEFAULT 0",
            "visible_content_changed": "INTEGER",
        }
        existing = {
            row[1]
            for row in conn.execute("PRAGMA table_info(latency_metrics)").fetchall()
        }
        for name, col_type in columns.items():
            if name not in existing:
                conn.execute(
                    f"ALTER TABLE latency_metrics ADD COLUMN {name} {col_type}"
                )

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
            record.focus_ms,
            record.caret_ms,
            record.subtree_ms,
            record.visible_fetch_ms,
            record.context_build_ms,
            record.overlay_first_show_ms,
            record.overlay_final_show_ms,
            record.visible_cache_age_ms,
            record.trigger_type,
            record.visible_source,
            int(record.use_shell),
            int(record.use_tui),
            int(record.has_conversation_turns),
            int(record.used_subtree_context),
            int(record.used_semantic_context),
            int(record.used_memory_context),
            int(record.fallback_used),
            (
                None
                if record.visible_content_changed is None
                else int(record.visible_content_changed)
            ),
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
                   e2e_first_ms, e2e_total_ms, focus_ms, caret_ms,
                   subtree_ms, visible_fetch_ms, context_build_ms,
                   overlay_first_show_ms, overlay_final_show_ms,
                   visible_cache_age_ms, suggestion_count,
                   app_name, mode, provider, model, trigger_type,
                   visible_source, use_shell, use_tui,
                   has_conversation_turns, used_subtree_context,
                   used_semantic_context, used_memory_context,
                   fallback_used, visible_content_changed
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
            "focus_ms": [],
            "caret_ms": [],
            "subtree_ms": [],
            "visible_fetch_ms": [],
            "context_build_ms": [],
            "overlay_first_show_ms": [],
            "overlay_final_show_ms": [],
            "visible_cache_age_ms": [],
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
        app_idx = len(metrics) + 1
        mode_idx = len(metrics) + 2
        provider_idx = len(metrics) + 3
        model_idx = len(metrics) + 4
        trigger_type_idx = len(metrics) + 5
        visible_source_idx = len(metrics) + 6
        use_shell_idx = len(metrics) + 7
        use_tui_idx = len(metrics) + 8
        has_turns_idx = len(metrics) + 9
        used_subtree_idx = len(metrics) + 10
        used_semantic_idx = len(metrics) + 11
        used_memory_idx = len(metrics) + 12
        fallback_idx = len(metrics) + 13
        visible_changed_idx = len(metrics) + 14

        result["apps"] = dict(Counter(r[app_idx] for r in rows).most_common(10))
        result["modes"] = dict(Counter(r[mode_idx] for r in rows).most_common())
        result["providers"] = dict(Counter(r[provider_idx] for r in rows if r[provider_idx]).most_common())
        result["models"] = dict(Counter(r[model_idx] for r in rows if r[model_idx]).most_common())
        result["trigger_types"] = dict(Counter(r[trigger_type_idx] for r in rows if r[trigger_type_idx]).most_common())
        result["visible_sources"] = dict(Counter(r[visible_source_idx] for r in rows if r[visible_source_idx]).most_common())
        result["shell_count"] = sum(1 for r in rows if r[use_shell_idx])
        result["tui_count"] = sum(1 for r in rows if r[use_tui_idx])
        result["conversation_turn_count"] = sum(1 for r in rows if r[has_turns_idx])
        result["subtree_count"] = sum(1 for r in rows if r[used_subtree_idx])
        result["semantic_count"] = sum(1 for r in rows if r[used_semantic_idx])
        result["memory_count"] = sum(1 for r in rows if r[used_memory_idx])
        result["fallback_count"] = sum(1 for r in rows if r[fallback_idx])
        changed_rows = [r for r in rows if r[visible_changed_idx] is not None]
        result["visible_changed_count"] = sum(1 for r in changed_rows if r[visible_changed_idx])
        result["visible_changed_measured_count"] = len(changed_rows)

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
        "focus_ms": "Focused element capture",
        "caret_ms": "Caret lookup",
        "subtree_ms": "Subtree context extraction",
        "visible_fetch_ms": "Visible-content resolution",
        "context_build_ms": "Context build only",
        "overlay_first_show_ms": "Overlay first show",
        "overlay_final_show_ms": "Overlay final show",
        "visible_cache_age_ms": "Visible cache age",
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
    if stats.get("trigger_types"):
        print(f"  Triggers:  {stats['trigger_types']}")
    if stats.get("visible_sources"):
        print(f"  Visible:   {stats['visible_sources']}")
    if stats.get("providers"):
        print(f"  Providers: {stats['providers']}")
    if stats.get("models"):
        print(f"  Models:    {stats['models']}")
    print(
        "  Flags:     "
        f"shell={stats.get('shell_count', 0)} "
        f"tui={stats.get('tui_count', 0)} "
        f"turns={stats.get('conversation_turn_count', 0)} "
        f"subtree={stats.get('subtree_count', 0)} "
        f"semantic={stats.get('semantic_count', 0)} "
        f"memory={stats.get('memory_count', 0)} "
        f"fallback={stats.get('fallback_count', 0)}"
    )
    measured = stats.get("visible_changed_measured_count", 0)
    if measured:
        print(
            "  Visible changed after fresh fetch: "
            f"{stats.get('visible_changed_count', 0)}/{measured}"
        )
    print()
