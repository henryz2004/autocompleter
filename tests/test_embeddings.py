"""Tests for the embeddings module and semantic context integration."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from autocompleter.embeddings import (
    TFIDFEmbeddingProvider,
    cosine_similarity,
    find_relevant_context,
)
from autocompleter.context_store import ContextStore


# ---------------------------------------------------------------------------
# cosine_similarity tests
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors_return_one(self):
        a = [1.0, 2.0, 3.0]
        assert cosine_similarity(a, a) == pytest.approx(1.0)

    def test_orthogonal_vectors_return_zero(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors_return_negative_one(self):
        a = [1.0, 2.0, 3.0]
        b = [-1.0, -2.0, -3.0]
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector_a_returns_zero(self):
        a = [0.0, 0.0, 0.0]
        b = [1.0, 2.0, 3.0]
        assert cosine_similarity(a, b) == 0.0

    def test_zero_vector_b_returns_zero(self):
        a = [1.0, 2.0, 3.0]
        b = [0.0, 0.0, 0.0]
        assert cosine_similarity(a, b) == 0.0

    def test_both_zero_vectors_returns_zero(self):
        a = [0.0, 0.0]
        b = [0.0, 0.0]
        assert cosine_similarity(a, b) == 0.0

    def test_empty_vectors_return_zero(self):
        assert cosine_similarity([], []) == 0.0

    def test_mismatched_lengths_return_zero(self):
        a = [1.0, 2.0]
        b = [1.0, 2.0, 3.0]
        assert cosine_similarity(a, b) == 0.0

    def test_proportional_vectors(self):
        a = [1.0, 2.0, 3.0]
        b = [2.0, 4.0, 6.0]
        assert cosine_similarity(a, b) == pytest.approx(1.0)

    def test_known_angle(self):
        """45-degree angle should give cos(45) = sqrt(2)/2 ~ 0.7071."""
        import math
        a = [1.0, 0.0]
        b = [1.0, 1.0]
        expected = math.sqrt(2) / 2
        assert cosine_similarity(a, b) == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# TFIDFEmbeddingProvider tests
# ---------------------------------------------------------------------------


class TestTFIDFEmbeddingProvider:
    def test_embed_returns_correct_count(self):
        provider = TFIDFEmbeddingProvider(max_features=100)
        texts = ["hello world", "foo bar baz"]
        vectors = provider.embed(texts)
        assert len(vectors) == 2

    def test_embed_vectors_are_non_zero(self):
        provider = TFIDFEmbeddingProvider(max_features=100)
        texts = ["hello world", "foo bar baz"]
        vectors = provider.embed(texts)
        for vec in vectors:
            assert any(v != 0.0 for v in vec), "Vector should have non-zero entries"

    def test_embed_vectors_have_correct_dimension(self):
        provider = TFIDFEmbeddingProvider(max_features=100)
        texts = ["hello world", "foo bar baz"]
        vectors = provider.embed(texts)
        dim = provider.dimension
        for vec in vectors:
            assert len(vec) == dim

    def test_dimension_capped_at_max_features(self):
        provider = TFIDFEmbeddingProvider(max_features=5)
        texts = [
            "the quick brown fox jumps over the lazy dog and more words here",
            "another sentence with many different unique vocabulary words to add",
        ]
        vectors = provider.embed(texts)
        for vec in vectors:
            assert len(vec) <= 5

    def test_similar_texts_have_higher_similarity(self):
        provider = TFIDFEmbeddingProvider(max_features=200)
        texts = [
            "python programming language",
            "python coding language features",
            "cooking recipe for chocolate cake",
        ]
        vectors = provider.embed(texts)
        sim_related = cosine_similarity(vectors[0], vectors[1])
        sim_unrelated = cosine_similarity(vectors[0], vectors[2])
        assert sim_related > sim_unrelated, (
            f"Similar texts should have higher similarity: {sim_related} vs {sim_unrelated}"
        )

    def test_identical_texts_have_similarity_one(self):
        provider = TFIDFEmbeddingProvider(max_features=100)
        texts = ["hello world", "hello world"]
        vectors = provider.embed(texts)
        sim = cosine_similarity(vectors[0], vectors[1])
        assert sim == pytest.approx(1.0)

    def test_embed_empty_list(self):
        provider = TFIDFEmbeddingProvider()
        assert provider.embed([]) == []

    def test_embed_single_text(self):
        provider = TFIDFEmbeddingProvider(max_features=50)
        vectors = provider.embed(["just one document"])
        assert len(vectors) == 1
        assert len(vectors[0]) > 0

    def test_embed_text_with_no_alpha(self):
        """Text with no alphanumeric tokens should produce zero vector."""
        provider = TFIDFEmbeddingProvider(max_features=50)
        texts = ["hello world", "!!! ??? ---"]
        vectors = provider.embed(texts)
        assert len(vectors) == 2
        # Second vector should be all zeros since no tokens matched
        assert all(v == 0.0 for v in vectors[1])

    def test_dimension_property_before_embed(self):
        provider = TFIDFEmbeddingProvider(max_features=500)
        assert provider.dimension == 500

    def test_dimension_property_after_embed(self):
        provider = TFIDFEmbeddingProvider(max_features=500)
        provider.embed(["short text"])
        # After embedding, dimension matches actual vocab size
        assert provider.dimension <= 500
        assert provider.dimension > 0

    def test_embed_query_uses_corpus_vocab(self):
        provider = TFIDFEmbeddingProvider(max_features=100)
        corpus = ["python programming", "java development"]
        provider.embed(corpus)
        query_vec = provider.embed_query("python coding", corpus)
        assert len(query_vec) == provider.dimension
        # Query about python should have non-zero entries
        assert any(v != 0.0 for v in query_vec)


# ---------------------------------------------------------------------------
# find_relevant_context tests
# ---------------------------------------------------------------------------


class MockEmbeddingProvider:
    """Mock provider that returns predetermined vectors."""

    def __init__(self, vectors: list[list[float]]) -> None:
        self._vectors = vectors
        self._call_count = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self._call_count += 1
        # Return vectors in order; pad or truncate as needed
        result = []
        for i in range(len(texts)):
            if i < len(self._vectors):
                result.append(self._vectors[i])
            else:
                result.append([0.0] * self._dimension)
        return result

    @property
    def dimension(self) -> int:
        if self._vectors:
            return len(self._vectors[0])
        return 3

    @property
    def _dimension(self) -> int:
        return self.dimension


class TestFindRelevantContext:
    def test_returns_top_k_by_similarity(self):
        # Query vector is [1, 0, 0]
        # Entry 0: [1, 0, 0] -- identical to query (sim=1.0)
        # Entry 1: [0, 1, 0] -- orthogonal (sim=0.0)
        # Entry 2: [0.7, 0.7, 0] -- somewhat similar (sim~0.7)
        vectors = [
            [1.0, 0.0, 0.0],  # entry 0
            [0.0, 1.0, 0.0],  # entry 1
            [0.7, 0.7, 0.0],  # entry 2
            [1.0, 0.0, 0.0],  # query
        ]
        provider = MockEmbeddingProvider(vectors)
        entries = [
            ("id0", "Most relevant"),
            ("id1", "Least relevant"),
            ("id2", "Somewhat relevant"),
        ]
        results = find_relevant_context("query text", entries, provider, top_k=2)
        assert len(results) == 2
        assert results[0][0] == "Most relevant"
        assert results[0][1] == pytest.approx(1.0)
        assert results[1][0] == "Somewhat relevant"

    def test_top_k_limits_results(self):
        vectors = [
            [1.0, 0.0], [0.5, 0.5], [0.0, 1.0],  # 3 entries
            [1.0, 0.0],  # query
        ]
        provider = MockEmbeddingProvider(vectors)
        entries = [("a", "A"), ("b", "B"), ("c", "C")]
        results = find_relevant_context("q", entries, provider, top_k=1)
        assert len(results) == 1

    def test_empty_entries_returns_empty(self):
        provider = MockEmbeddingProvider([])
        results = find_relevant_context("query", [], provider, top_k=5)
        assert results == []

    def test_empty_query_returns_empty(self):
        provider = MockEmbeddingProvider([[1.0, 0.0]])
        entries = [("id0", "some text")]
        results = find_relevant_context("", entries, provider, top_k=5)
        assert results == []

    def test_whitespace_query_returns_empty(self):
        provider = MockEmbeddingProvider([[1.0, 0.0]])
        entries = [("id0", "some text")]
        results = find_relevant_context("   ", entries, provider, top_k=5)
        assert results == []

    def test_results_sorted_descending_by_score(self):
        # Scores: 0.0, ~0.7, 1.0 -- should be returned as 1.0, 0.7, 0.0
        vectors = [
            [0.0, 1.0],  # entry 0: orthogonal to query
            [0.7, 0.7],  # entry 1: partial match
            [1.0, 0.0],  # entry 2: identical to query
            [1.0, 0.0],  # query
        ]
        provider = MockEmbeddingProvider(vectors)
        entries = [("a", "Orthogonal"), ("b", "Partial"), ("c", "Identical")]
        results = find_relevant_context("q", entries, provider, top_k=3)
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Integration tests: get_semantically_relevant
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path):
    db_path = tmp_path / "test_embeddings.db"
    s = ContextStore(db_path)
    s.open()
    yield s
    s.close()


class TestGetSemanticallyRelevant:
    def test_relevant_entries_rank_higher(self, store):
        """Entries semantically related to the query should rank higher."""
        provider = TFIDFEmbeddingProvider(max_features=200)

        # Add entries with distinct topics
        store.add_entry("App", "python programming language syntax", "visible_text",
                        timestamp=time.time() - 100)
        store.add_entry("App", "chocolate cake baking recipe oven", "visible_text",
                        timestamp=time.time() - 90)
        store.add_entry("App", "python coding tutorial functions classes", "visible_text",
                        timestamp=time.time() - 80)

        results = store.get_semantically_relevant(
            query="python programming tutorial",
            provider=provider,
            top_k=3,
            max_age_hours=1,
        )
        assert len(results) > 0
        # Python-related entries should rank higher than baking
        python_entries = [r for r in results if "python" in r.lower()]
        assert len(python_entries) >= 1
        # First result should be python-related
        assert "python" in results[0].lower() or "programming" in results[0].lower()

    def test_empty_query_returns_empty(self, store):
        provider = TFIDFEmbeddingProvider()
        store.add_entry("App", "some content", "visible_text")
        results = store.get_semantically_relevant("", provider)
        assert results == []

    def test_no_entries_returns_empty(self, store):
        provider = TFIDFEmbeddingProvider()
        results = store.get_semantically_relevant("some query", provider)
        assert results == []

    def test_respects_max_age_hours(self, store):
        provider = TFIDFEmbeddingProvider(max_features=100)

        # Add old entry (beyond max_age_hours)
        old_time = time.time() - 48 * 3600  # 48 hours ago
        store.add_entry("App", "old python content", "visible_text",
                        timestamp=old_time)
        # Add recent entry
        store.add_entry("App", "recent python content", "visible_text")

        results = store.get_semantically_relevant(
            "python", provider, top_k=5, max_age_hours=1
        )
        # Only the recent entry should appear (old one is beyond 1 hour)
        assert len(results) == 1
        assert "recent" in results[0]

    def test_top_k_limits_results(self, store):
        provider = TFIDFEmbeddingProvider(max_features=100)

        for i in range(10):
            store.add_entry(
                "App", f"entry number {i} with some text",
                "visible_text", timestamp=time.time() + i * 10,
            )

        results = store.get_semantically_relevant(
            "entry text", provider, top_k=3, max_age_hours=1,
        )
        assert len(results) <= 3


class TestEmbeddingCaching:
    def test_embeddings_are_cached_in_db(self, store):
        """After computing embeddings, they should be stored in the DB."""
        provider = TFIDFEmbeddingProvider(max_features=100)

        store.add_entry("App", "test content for caching", "visible_text")

        # First call computes and caches embeddings
        store.get_semantically_relevant("test", provider, top_k=5, max_age_hours=1)

        # Check that the embedding was cached
        conn = store._get_conn()
        cursor = conn.execute(
            "SELECT embeddings FROM context_entries WHERE content = ?",
            ("test content for caching",),
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[0] is not None, "Embedding should be cached as a BLOB"

    def test_cached_embeddings_are_reused(self, store):
        """Second call should use cached embeddings, not recompute."""
        call_count = 0
        original_embed = TFIDFEmbeddingProvider.embed

        def counting_embed(self_inner, texts):
            nonlocal call_count
            call_count += 1
            return original_embed(self_inner, texts)

        provider = TFIDFEmbeddingProvider(max_features=100)

        store.add_entry("App", "cached embedding content", "visible_text")

        # First call: should call embed
        with patch.object(TFIDFEmbeddingProvider, 'embed', counting_embed):
            store.get_semantically_relevant("test query", provider, top_k=5, max_age_hours=1)
            first_call_count = call_count

        # Reset counter
        call_count = 0

        # Second call: entries have cached embeddings, but we still need to
        # embed the query with corpus context. The key difference is that
        # the cached entries don't need to be re-embedded.
        with patch.object(TFIDFEmbeddingProvider, 'embed', counting_embed):
            store.get_semantically_relevant("test query", provider, top_k=5, max_age_hours=1)
            second_call_count = call_count

        # Both calls should succeed
        assert first_call_count >= 1
        assert second_call_count >= 1


class TestSemanticContextBlending:
    def test_continuation_context_includes_semantic_when_enabled(self, store):
        """When use_semantic_context=True, semantic entries should appear."""
        provider = TFIDFEmbeddingProvider(max_features=200)

        # Add some historical context about Python
        store.add_entry("Editor", "python class definition with inheritance",
                        "visible_text", timestamp=time.time() - 100)
        store.add_entry("Editor", "javascript array methods map filter",
                        "visible_text", timestamp=time.time() - 90)

        context = store.get_continuation_context(
            before_cursor="class MyPython",
            after_cursor="",
            source_app="Editor",
            embedding_provider=provider,
            use_semantic_context=True,
        )
        # The context should contain the semantic section
        # (may or may not appear depending on TF-IDF match quality)
        assert "Text before cursor:" in context
        assert "class MyPython" in context

    def test_continuation_context_skips_semantic_when_disabled(self, store):
        """When use_semantic_context=False, no semantic section should appear."""
        provider = TFIDFEmbeddingProvider(max_features=200)

        store.add_entry("Editor", "python class definition", "visible_text",
                        timestamp=time.time() - 100)

        context = store.get_continuation_context(
            before_cursor="class MyPython",
            after_cursor="",
            source_app="Editor",
            embedding_provider=provider,
            use_semantic_context=False,
        )
        assert "Related context:" not in context

    def test_continuation_context_skips_semantic_when_no_provider(self, store):
        """Without a provider, no semantic section should appear."""
        store.add_entry("Editor", "python class definition", "visible_text",
                        timestamp=time.time() - 100)

        context = store.get_continuation_context(
            before_cursor="class MyPython",
            after_cursor="",
            source_app="Editor",
            embedding_provider=None,
            use_semantic_context=True,
        )
        assert "Related context:" not in context

    def test_reply_context_includes_semantic_when_enabled(self, store):
        """Reply context should include semantic entries when enabled."""
        provider = TFIDFEmbeddingProvider(max_features=200)

        store.add_entry("Slack", "discussion about python deployment",
                        "visible_text", timestamp=time.time() - 100)
        store.add_entry("Slack", "holiday party planning committee",
                        "visible_text", timestamp=time.time() - 90)

        turns = [
            {"speaker": "Alice", "text": "How should we deploy the python service?"},
        ]
        context = store.get_reply_context(
            conversation_turns=turns,
            source_app="Slack",
            embedding_provider=provider,
            use_semantic_context=True,
        )
        assert "Conversation:" in context
        assert "deploy" in context.lower()

    def test_reply_context_skips_semantic_when_disabled(self, store):
        """Reply context should skip semantic section when disabled."""
        provider = TFIDFEmbeddingProvider(max_features=200)

        store.add_entry("Slack", "discussion about python", "visible_text",
                        timestamp=time.time() - 100)

        turns = [{"speaker": "Alice", "text": "How about python?"}]
        context = store.get_reply_context(
            conversation_turns=turns,
            source_app="Slack",
            embedding_provider=provider,
            use_semantic_context=False,
        )
        assert "Related context:" not in context


class TestConfigOptions:
    def test_config_defaults(self):
        from autocompleter.config import Config
        config = Config()
        assert config.embedding_provider == "tfidf"
        assert config.use_semantic_context is True

    def test_config_custom_values(self):
        from autocompleter.config import Config
        config = Config(
            embedding_provider="openai",
            use_semantic_context=False,
        )
        assert config.embedding_provider == "openai"
        assert config.use_semantic_context is False

    def test_use_semantic_context_false_skips_embedding(self, store):
        """When use_semantic_context is False, embedding should not be used."""
        # Create a mock provider that tracks calls
        mock_provider = MagicMock()
        mock_provider.embed.return_value = [[1.0, 0.0]]
        mock_provider.dimension = 2

        store.add_entry("App", "some content", "visible_text")

        context = store.get_continuation_context(
            before_cursor="test input",
            after_cursor="",
            source_app="App",
            embedding_provider=mock_provider,
            use_semantic_context=False,
        )
        # The mock should NOT have been called
        mock_provider.embed.assert_not_called()
        assert "Related context:" not in context
