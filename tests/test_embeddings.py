"""Tests for the embeddings module."""

from __future__ import annotations

import pytest

from autocompleter.embeddings import (
    TFIDFEmbeddingProvider,
    cosine_similarity,
    find_relevant_context,
)


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

    def test_zero_vector_a_returns_zero(self):
        a = [0.0, 0.0, 0.0]
        b = [1.0, 2.0, 3.0]
        assert cosine_similarity(a, b) == 0.0

    def test_zero_vector_b_returns_zero(self):
        a = [1.0, 2.0, 3.0]
        b = [0.0, 0.0, 0.0]
        assert cosine_similarity(a, b) == 0.0

    def test_empty_vectors_return_zero(self):
        assert cosine_similarity([], []) == 0.0

    def test_mismatched_lengths_return_zero(self):
        a = [1.0, 2.0]
        b = [1.0, 2.0, 3.0]
        assert cosine_similarity(a, b) == 0.0


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


class TestConfigOptions:
    def test_config_defaults(self):
        from autocompleter.config import Config
        config = Config()
        assert not hasattr(config, "embedding_provider")
        assert not hasattr(config, "use_semantic_context")

    def test_config_custom_values(self):
        from autocompleter.config import Config
        config = Config(llm_provider="anthropic")
        assert config.llm_provider == "anthropic"

    def test_removed_embedding_fields_are_rejected(self):
        from autocompleter.config import Config

        with pytest.raises(TypeError):
            Config(embedding_provider="openai")

        with pytest.raises(TypeError):
            Config(use_semantic_context=False)
