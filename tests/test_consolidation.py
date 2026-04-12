"""Tests for the memory consolidation pipeline."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from autocompleter.config import Config
from autocompleter.consolidation import (
    _build_consolidation_messages,
    _call_llm,
    _purge_memories,
    _should_consolidate,
    _strip_code_fences,
    _update_timestamp,
    run_consolidation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(tmp_path: Path, **overrides) -> Config:
    defaults = dict(
        data_dir=tmp_path,
        memory_enabled=True,
        memory_llm_provider="groq",
        memory_llm_model="qwen/qwen3-32b",
        memory_embedder_provider="openai",
        memory_embedder_model="text-embedding-3-small",
    )
    defaults.update(overrides)
    return Config(**defaults)


def _make_mock_store(memories: list[dict] | None = None) -> MagicMock:
    store = MagicMock()
    store.enabled = True
    store.get_all_with_ids.return_value = memories or []
    store.delete.return_value = True
    return store


# ---------------------------------------------------------------------------
# Timestamp tests
# ---------------------------------------------------------------------------

class TestShouldConsolidate:
    def test_first_run_no_file(self, tmp_path):
        cfg = _make_config(tmp_path)
        assert _should_consolidate(cfg) is True

    def test_recent_timestamp(self, tmp_path):
        cfg = _make_config(tmp_path)
        ts_path = tmp_path / ".last_consolidation"
        ts_path.write_text(datetime.now(timezone.utc).isoformat())
        assert _should_consolidate(cfg) is False

    def test_stale_timestamp(self, tmp_path):
        cfg = _make_config(tmp_path)
        ts_path = tmp_path / ".last_consolidation"
        old = datetime.now(timezone.utc) - timedelta(hours=25)
        ts_path.write_text(old.isoformat())
        assert _should_consolidate(cfg) is True

    def test_force_overrides(self, tmp_path):
        cfg = _make_config(tmp_path)
        ts_path = tmp_path / ".last_consolidation"
        ts_path.write_text(datetime.now(timezone.utc).isoformat())
        assert _should_consolidate(cfg, force=True) is True

    def test_corrupt_timestamp(self, tmp_path):
        cfg = _make_config(tmp_path)
        ts_path = tmp_path / ".last_consolidation"
        ts_path.write_text("not-a-date")
        assert _should_consolidate(cfg) is True


class TestUpdateTimestamp:
    def test_writes_timestamp(self, tmp_path):
        cfg = _make_config(tmp_path)
        _update_timestamp(cfg)
        ts_path = tmp_path / ".last_consolidation"
        assert ts_path.exists()
        dt = datetime.fromisoformat(ts_path.read_text().strip())
        assert (datetime.now(timezone.utc) - dt).total_seconds() < 5


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

class TestBuildPrompt:
    def test_empty_md(self):
        messages = _build_consolidation_messages("", [
            {"id": "a1", "memory": "likes Python", "updated_at": "2026-04-10T00:00:00Z"},
        ])
        assert len(messages) == 2
        assert "first consolidation" in messages[1]["content"]
        assert "[a1]" in messages[1]["content"]
        assert "likes Python" in messages[1]["content"]

    def test_existing_md(self):
        existing = "## Identity\n- Software engineer"
        messages = _build_consolidation_messages(existing, [
            {"id": "b2", "memory": "prefers PostgreSQL", "updated_at": ""},
        ])
        assert "Software engineer" in messages[1]["content"]
        assert "[b2]" in messages[1]["content"]


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

class TestStripCodeFences:
    def test_json_fenced(self):
        text = '```json\n{"key": "value"}\n```'
        assert _strip_code_fences(text) == '{"key": "value"}'

    def test_plain_fenced(self):
        text = '```\n{"key": "value"}\n```'
        assert _strip_code_fences(text) == '{"key": "value"}'

    def test_no_fences(self):
        text = '{"key": "value"}'
        assert _strip_code_fences(text) == '{"key": "value"}'


# ---------------------------------------------------------------------------
# LLM call (mocked)
# ---------------------------------------------------------------------------

class TestCallLLM:
    @patch("openai.OpenAI")
    def test_valid_response(self, mock_openai_cls, tmp_path):
        cfg = _make_config(tmp_path)
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(
                content=json.dumps({
                    "markdown": "## Identity\n- Engineer",
                    "consolidated_ids": ["a1", "a2"],
                })
            ))]
        )
        result = _call_llm(cfg, [{"role": "user", "content": "test"}])
        assert result is not None
        assert "## Identity" in result["markdown"]
        assert result["consolidated_ids"] == ["a1", "a2"]

    @patch("openai.OpenAI")
    def test_fenced_response(self, mock_openai_cls, tmp_path):
        cfg = _make_config(tmp_path)
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(
                content='```json\n' + json.dumps({
                    "markdown": "## Preferences\n- Likes Go",
                    "consolidated_ids": ["x"],
                }) + '\n```'
            ))]
        )
        result = _call_llm(cfg, [{"role": "user", "content": "test"}])
        assert result is not None
        assert "Likes Go" in result["markdown"]

    @patch("openai.OpenAI")
    def test_garbage_response(self, mock_openai_cls, tmp_path):
        cfg = _make_config(tmp_path)
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="not json at all"))]
        )
        result = _call_llm(cfg, [{"role": "user", "content": "test"}])
        assert result is None

    @patch("openai.OpenAI")
    def test_api_error(self, mock_openai_cls, tmp_path):
        cfg = _make_config(tmp_path)
        mock_openai_cls.side_effect = Exception("connection refused")
        result = _call_llm(cfg, [{"role": "user", "content": "test"}])
        assert result is None


# ---------------------------------------------------------------------------
# Purge
# ---------------------------------------------------------------------------

class TestPurgeMemories:
    def test_all_succeed(self):
        store = _make_mock_store()
        deleted = _purge_memories(store, ["a", "b", "c"])
        assert deleted == 3
        assert store.delete.call_count == 3

    def test_partial_failure(self):
        store = _make_mock_store()
        store.delete.side_effect = [True, False, True]
        deleted = _purge_memories(store, ["a", "b", "c"])
        assert deleted == 2


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

class TestRunConsolidation:
    @patch("autocompleter.consolidation._call_llm")
    def test_full_pipeline(self, mock_llm, tmp_path):
        cfg = _make_config(tmp_path)
        memories = [
            {"id": "m1", "memory": "prefers Python", "created_at": "", "updated_at": "2026-04-10T00:00:00Z"},
            {"id": "m2", "memory": "OOO April 14-18", "created_at": "", "updated_at": "2026-04-10T00:00:00Z"},
        ]
        store = _make_mock_store(memories)

        mock_llm.return_value = {
            "markdown": "## Preferences\n- Prefers Python",
            "consolidated_ids": ["m1", "m2"],
        }

        ok = run_consolidation(store, cfg, force=True)
        assert ok is True

        # memory.md written
        md_path = tmp_path / "memory.md"
        assert md_path.exists()
        assert "Prefers Python" in md_path.read_text()

        # FAISS purged
        assert store.delete.call_count == 2

        # Timestamp updated
        assert (tmp_path / ".last_consolidation").exists()

    def test_no_memories(self, tmp_path):
        cfg = _make_config(tmp_path)
        store = _make_mock_store([])
        ok = run_consolidation(store, cfg, force=True)
        assert ok is True
        # Timestamp still updated
        assert (tmp_path / ".last_consolidation").exists()

    @patch("autocompleter.consolidation._call_llm")
    def test_llm_failure_no_data_loss(self, mock_llm, tmp_path):
        cfg = _make_config(tmp_path)
        store = _make_mock_store([
            {"id": "m1", "memory": "important fact", "created_at": "", "updated_at": ""},
        ])
        mock_llm.return_value = None  # LLM failed

        ok = run_consolidation(store, cfg, force=True)
        assert ok is False

        # FAISS not touched
        store.delete.assert_not_called()
        # No memory.md written
        assert not (tmp_path / "memory.md").exists()
        # Timestamp NOT updated (will retry)
        assert not (tmp_path / ".last_consolidation").exists()

    @patch("autocompleter.consolidation._call_llm")
    def test_idempotent_no_new_memories(self, mock_llm, tmp_path):
        cfg = _make_config(tmp_path)
        # First run: has memories
        store = _make_mock_store([
            {"id": "m1", "memory": "a fact", "created_at": "", "updated_at": ""},
        ])
        mock_llm.return_value = {
            "markdown": "## Identity\n- A fact",
            "consolidated_ids": ["m1"],
        }
        run_consolidation(store, cfg, force=True)

        # Second run: no memories left
        store.get_all_with_ids.return_value = []
        mock_llm.reset_mock()
        ok = run_consolidation(store, cfg, force=True)
        assert ok is True
        # LLM should NOT have been called
        mock_llm.assert_not_called()

    def test_disabled_store(self, tmp_path):
        cfg = _make_config(tmp_path)
        store = _make_mock_store()
        store.enabled = False
        ok = run_consolidation(store, cfg, force=True)
        assert ok is False

    @patch("autocompleter.consolidation._call_llm")
    def test_empty_markdown_response(self, mock_llm, tmp_path):
        cfg = _make_config(tmp_path)
        store = _make_mock_store([
            {"id": "m1", "memory": "fact", "created_at": "", "updated_at": ""},
        ])
        mock_llm.return_value = {"markdown": "", "consolidated_ids": ["m1"]}
        ok = run_consolidation(store, cfg, force=True)
        assert ok is False
        store.delete.assert_not_called()
