"""Tests for the latency_tracker module."""

import sqlite3
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from autocompleter.latency_tracker import (
    STAGES,
    LatencyRecord,
    LatencyStore,
    LatencyTracker,
    print_stats,
)


# ── LatencyTracker ──────────────────────────────────────────────────────────


class TestLatencyTracker:
    """Unit tests for the LatencyTracker stopwatch."""

    def test_mark_and_delta(self):
        tracker = LatencyTracker()
        tracker.start(generation_id=1)
        tracker.mark("trigger")
        time.sleep(0.01)  # 10 ms
        tracker.mark("context_ready")
        rec = tracker.finish()
        assert rec.context_ms is not None
        assert rec.context_ms >= 5  # at least ~10ms, allow slack

    def test_start_resets_stamps(self):
        tracker = LatencyTracker()
        tracker.start(generation_id=1)
        tracker.mark("trigger")
        tracker.start(generation_id=2)  # reset
        rec = tracker.finish()
        # trigger was cleared by start(), so context_ms should be None
        assert rec.context_ms is None
        assert rec.generation_id == 2

    def test_missing_stages_return_none(self):
        tracker = LatencyTracker()
        tracker.start()
        tracker.mark("trigger")
        tracker.mark("context_ready")
        # Never marked llm_start, first_suggestion, llm_done, displayed
        rec = tracker.finish()
        assert rec.context_ms is not None
        assert rec.llm_ttft_ms is None
        assert rec.llm_total_ms is None
        assert rec.e2e_first_ms is None
        assert rec.e2e_total_ms is None

    def test_full_pipeline(self):
        tracker = LatencyTracker()
        tracker.start(generation_id=42)
        for stage in STAGES:
            tracker.mark(stage)
            time.sleep(0.002)  # small delay between stages
        rec = tracker.finish(
            app_name="Discord",
            mode="reply",
            provider="openai",
            model="gpt-4",
            suggestion_count=3,
        )
        assert rec.generation_id == 42
        assert rec.app_name == "Discord"
        assert rec.mode == "reply"
        assert rec.provider == "openai"
        assert rec.model == "gpt-4"
        assert rec.suggestion_count == 3
        assert rec.context_ms is not None and rec.context_ms >= 0
        assert rec.llm_ttft_ms is not None and rec.llm_ttft_ms >= 0
        assert rec.llm_total_ms is not None and rec.llm_total_ms >= 0
        assert rec.e2e_first_ms is not None and rec.e2e_first_ms >= 0
        assert rec.e2e_total_ms is not None and rec.e2e_total_ms >= 0
        # e2e should be >= context_ms
        assert rec.e2e_total_ms >= rec.context_ms

    def test_elapsed_since_returns_none_for_unset(self):
        tracker = LatencyTracker()
        tracker.start()
        assert tracker.elapsed_since("trigger") is None

    def test_elapsed_since_returns_positive(self):
        tracker = LatencyTracker()
        tracker.start()
        tracker.mark("trigger")
        time.sleep(0.005)
        elapsed = tracker.elapsed_since("trigger")
        assert elapsed is not None
        assert elapsed >= 3  # at least ~5ms with slack

    def test_finish_sets_timestamp(self):
        tracker = LatencyTracker()
        tracker.start()
        before = time.time()
        rec = tracker.finish()
        after = time.time()
        assert before <= rec.timestamp <= after

    def test_finish_logs_latency(self, caplog):
        """Verify the structured [LATENCY] log line is emitted."""
        import logging

        with caplog.at_level(logging.INFO, logger="autocompleter.latency_tracker"):
            tracker = LatencyTracker()
            tracker.start(generation_id=7)
            tracker.mark("trigger")
            tracker.mark("context_ready")
            tracker.finish(app_name="Slack", mode="continuation", suggestion_count=2)

        assert any("[LATENCY]" in r.message for r in caplog.records)
        log_msg = [r.message for r in caplog.records if "[LATENCY]" in r.message][0]
        assert "gen=7" in log_msg
        assert "app=Slack" in log_msg
        assert "mode=continuation" in log_msg
        assert "n=2" in log_msg

    def test_profile_returns_stage_offsets_and_durations(self):
        tracker = LatencyTracker()
        tracker.start(generation_id=9)
        tracker.mark("trigger")
        time.sleep(0.002)
        tracker.mark("context_ready")
        profile = tracker.profile()

        assert profile["generation_id"] == 9
        assert "trigger" in profile["stage_offsets_ms"]
        assert profile["durations_ms"]["context"] >= 0


# ── LatencyStore ────────────────────────────────────────────────────────────


class TestLatencyStore:
    """Tests for SQLite persistence."""

    @pytest.fixture()
    def store(self, tmp_path: Path) -> LatencyStore:
        return LatencyStore(tmp_path / "test_latency.db")

    def _make_record(self, **overrides) -> LatencyRecord:
        defaults = dict(
            generation_id=1,
            timestamp=time.time(),
            app_name="Discord",
            mode="reply",
            provider="openai",
            model="gpt-4",
            suggestion_count=3,
            context_ms=50.0,
            llm_ttft_ms=200.0,
            llm_total_ms=800.0,
            e2e_first_ms=250.0,
            e2e_total_ms=850.0,
        )
        defaults.update(overrides)
        return LatencyRecord(**defaults)

    def test_save_and_retrieve(self, store: LatencyStore):
        rec = self._make_record()
        store.save(rec)
        stats = store.get_stats()
        assert stats["count"] == 1
        assert stats["context_ms"]["mean"] == pytest.approx(50.0)
        assert stats["llm_ttft_ms"]["mean"] == pytest.approx(200.0)

    def test_empty_db_returns_zero_count(self, store: LatencyStore):
        stats = store.get_stats()
        assert stats == {"count": 0}

    def test_multiple_records_aggregation(self, store: LatencyStore):
        for i in range(10):
            store.save(self._make_record(
                generation_id=i,
                context_ms=float(10 * (i + 1)),  # 10, 20, ..., 100
                llm_ttft_ms=float(50 * (i + 1)),
                e2e_total_ms=float(100 * (i + 1)),
            ))
        stats = store.get_stats()
        assert stats["count"] == 10
        assert stats["context_ms"]["min"] == pytest.approx(10.0)
        assert stats["context_ms"]["max"] == pytest.approx(100.0)
        assert stats["context_ms"]["mean"] == pytest.approx(55.0)  # avg of 10..100

    def test_last_n_limit(self, store: LatencyStore):
        now = time.time()
        for i in range(20):
            store.save(self._make_record(
                generation_id=i,
                timestamp=now + i,
                context_ms=float(i * 10),
            ))
        stats = store.get_stats(last_n=5)
        assert stats["count"] == 5

    def test_last_hours_filter(self, store: LatencyStore):
        now = time.time()
        # Old record (2 hours ago)
        store.save(self._make_record(timestamp=now - 7200, context_ms=999.0))
        # Recent record (5 min ago)
        store.save(self._make_record(timestamp=now - 300, context_ms=50.0))
        stats = store.get_stats(last_hours=1)
        assert stats["count"] == 1
        assert stats["context_ms"]["mean"] == pytest.approx(50.0)

    def test_none_metrics_excluded(self, store: LatencyStore):
        store.save(self._make_record(
            context_ms=100.0,
            llm_ttft_ms=None,
            llm_total_ms=None,
            e2e_first_ms=None,
            e2e_total_ms=None,
        ))
        stats = store.get_stats()
        assert stats["count"] == 1
        assert stats["context_ms"]["mean"] == pytest.approx(100.0)
        assert stats["llm_ttft_ms"] is None
        assert stats["llm_total_ms"] is None

    def test_app_and_mode_breakdown(self, store: LatencyStore):
        store.save(self._make_record(app_name="Discord", mode="reply"))
        store.save(self._make_record(app_name="Discord", mode="continuation"))
        store.save(self._make_record(app_name="Slack", mode="reply"))
        stats = store.get_stats()
        assert stats["apps"]["Discord"] == 2
        assert stats["apps"]["Slack"] == 1
        assert stats["modes"]["reply"] == 2
        assert stats["modes"]["continuation"] == 1

    def test_provider_model_breakdown(self, store: LatencyStore):
        store.save(self._make_record(provider="openai", model="gpt-4"))
        store.save(self._make_record(provider="anthropic", model="claude-3"))
        store.save(self._make_record(provider="openai", model="gpt-4"))
        stats = store.get_stats()
        assert stats["providers"]["openai"] == 2
        assert stats["providers"]["anthropic"] == 1
        assert stats["models"]["gpt-4"] == 2
        assert stats["models"]["claude-3"] == 1

    def test_percentiles(self, store: LatencyStore):
        """With 100 records, p50 should be near the median value."""
        for i in range(100):
            store.save(self._make_record(
                generation_id=i,
                context_ms=float(i + 1),  # 1..100
            ))
        stats = store.get_stats()
        assert stats["count"] == 100
        # p50 of 1..100 sorted is index 50 → value 51
        assert stats["context_ms"]["p50"] == pytest.approx(51.0)
        # p90 is index 90 → value 91
        assert stats["context_ms"]["p90"] == pytest.approx(91.0)


# ── print_stats ─────────────────────────────────────────────────────────────


class TestPrintStats:
    """Tests for the formatted report output."""

    def test_empty_db(self, tmp_path: Path, capsys):
        print_stats(tmp_path / "empty.db", last_n=50)
        captured = capsys.readouterr()
        assert "No latency data" in captured.out

    def test_report_includes_metrics(self, tmp_path: Path, capsys):
        db_path = tmp_path / "stats.db"
        store = LatencyStore(db_path)
        store.save(LatencyRecord(
            generation_id=1,
            timestamp=time.time(),
            app_name="Discord",
            mode="reply",
            provider="openai",
            model="gpt-4",
            suggestion_count=3,
            context_ms=50.0,
            llm_ttft_ms=200.0,
            llm_total_ms=800.0,
            e2e_first_ms=250.0,
            e2e_total_ms=850.0,
        ))
        print_stats(db_path, last_n=50)
        captured = capsys.readouterr()
        assert "Latency Report" in captured.out
        assert "Context assembly" in captured.out
        assert "LLM TTFT" in captured.out
        assert "Discord" in captured.out

    def test_last_hours_scope(self, tmp_path: Path, capsys):
        db_path = tmp_path / "scope.db"
        store = LatencyStore(db_path)
        store.save(LatencyRecord(
            generation_id=1,
            timestamp=time.time(),
            context_ms=50.0,
        ))
        print_stats(db_path, last_n=0, last_hours=24)
        captured = capsys.readouterr()
        assert "last 24h" in captured.out
