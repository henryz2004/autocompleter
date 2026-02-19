"""Embedding providers and semantic similarity utilities.

Provides lightweight embedding computation for finding semantically relevant
historical context entries. Supports API-based providers (Anthropic Voyage,
OpenAI) and a local TF-IDF fallback that requires no external dependencies.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from typing import List, Optional, Protocol, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Returns 0.0 for zero-length vectors or if either vector is all zeros.
    """
    if len(a) != len(b) or len(a) == 0:
        return 0.0

    dot = sum(ai * bi for ai, bi in zip(a, b))
    norm_a = math.sqrt(sum(ai * ai for ai in a))
    norm_b = math.sqrt(sum(bi * bi for bi in b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# EmbeddingProvider protocol
# ---------------------------------------------------------------------------

class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return embedding vectors for each text."""
        ...

    @property
    def dimension(self) -> int:
        """Vector dimension."""
        ...


# ---------------------------------------------------------------------------
# TF-IDF provider (local, no API calls)
# ---------------------------------------------------------------------------

_TOKENIZE_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenization."""
    return _TOKENIZE_RE.findall(text.lower())


class TFIDFEmbeddingProvider:
    """Lightweight local embedding provider using TF-IDF.

    Builds a vocabulary from the provided texts and returns vectors of
    fixed dimension (top-N terms by document frequency).
    """

    def __init__(self, max_features: int = 500) -> None:
        self._max_features = max_features
        self._vocab: Optional[list[str]] = None
        self._idf: Optional[dict[str, float]] = None
        self._dimension = max_features

    @property
    def dimension(self) -> int:
        """Vector dimension (equals min(max_features, vocabulary size))."""
        if self._vocab is not None:
            return len(self._vocab)
        return self._dimension

    def _build_vocab(self, texts: list[str]) -> None:
        """Build vocabulary and IDF from a corpus of texts."""
        n_docs = len(texts)
        if n_docs == 0:
            self._vocab = []
            self._idf = {}
            return

        # Document frequency: how many documents contain each term
        df: Counter[str] = Counter()
        for text in texts:
            unique_terms = set(_tokenize(text))
            for term in unique_terms:
                df[term] += 1

        # Select top terms by document frequency, break ties alphabetically
        top_terms = sorted(df.keys(), key=lambda t: (-df[t], t))
        top_terms = top_terms[: self._max_features]

        # Compute IDF: log(N / df) + 1  (smoothed)
        idf: dict[str, float] = {}
        for term in top_terms:
            idf[term] = math.log(n_docs / df[term]) + 1.0

        self._vocab = top_terms
        self._idf = idf

    def _text_to_vector(self, text: str) -> list[float]:
        """Convert a single text to a TF-IDF vector."""
        if not self._vocab or not self._idf:
            return []

        tokens = _tokenize(text)
        if not tokens:
            return [0.0] * len(self._vocab)

        tf: Counter[str] = Counter(tokens)
        total = len(tokens)

        vector = []
        for term in self._vocab:
            term_freq = tf.get(term, 0) / total
            vector.append(term_freq * self._idf.get(term, 0.0))

        return vector

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Compute TF-IDF embedding vectors for a list of texts.

        Rebuilds the vocabulary from the provided texts each time.
        """
        if not texts:
            return []

        self._build_vocab(texts)
        return [self._text_to_vector(text) for text in texts]

    def embed_query(self, query: str, corpus_texts: list[str]) -> list[float]:
        """Embed a single query against an existing corpus vocabulary.

        If the vocabulary hasn't been built yet, builds it from corpus_texts.
        Returns the query's TF-IDF vector using the corpus vocabulary.
        """
        if self._vocab is None or self._idf is None:
            self._build_vocab(corpus_texts)
        return self._text_to_vector(query)


# ---------------------------------------------------------------------------
# Anthropic (Voyage) embedding provider
# ---------------------------------------------------------------------------

class AnthropicEmbeddingProvider:
    """Embedding provider using Anthropic's Voyage embeddings.

    Falls back to TF-IDF if the voyageai SDK is not installed.
    """

    def __init__(self, api_key: str = "", model: str = "voyage-2") -> None:
        self._api_key = api_key
        self._model = model
        self._client = None
        self._fallback: Optional[TFIDFEmbeddingProvider] = None
        self._dimension_value = 1024  # Voyage default

        # Try to import voyageai
        try:
            import voyageai  # type: ignore[import-untyped]
            self._client = voyageai.Client(api_key=api_key) if api_key else None
        except ImportError:
            logger.info(
                "voyageai SDK not installed; AnthropicEmbeddingProvider "
                "falling back to TF-IDF"
            )
            self._fallback = TFIDFEmbeddingProvider()

    @property
    def dimension(self) -> int:
        if self._fallback is not None:
            return self._fallback.dimension
        return self._dimension_value

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        if self._fallback is not None:
            return self._fallback.embed(texts)

        if self._client is None:
            logger.warning("No Voyage client and no fallback; returning empty")
            return [[0.0] * self._dimension_value for _ in texts]

        try:
            result = self._client.embed(texts, model=self._model)
            return result.embeddings  # type: ignore[return-value]
        except Exception:
            logger.exception("Voyage embedding failed; falling back to TF-IDF")
            self._fallback = TFIDFEmbeddingProvider()
            return self._fallback.embed(texts)


# ---------------------------------------------------------------------------
# OpenAI embedding provider
# ---------------------------------------------------------------------------

class OpenAIEmbeddingProvider:
    """Embedding provider using OpenAI's text-embedding-3-small model."""

    def __init__(
        self, api_key: str = "", model: str = "text-embedding-3-small"
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._client = None
        self._dimension_value = 1536  # text-embedding-3-small default

    @property
    def dimension(self) -> int:
        return self._dimension_value

    def _get_client(self):
        if self._client is None:
            import openai
            self._client = openai.OpenAI(api_key=self._api_key, timeout=10.0)
        return self._client

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        try:
            client = self._get_client()
            response = client.embeddings.create(input=texts, model=self._model)
            # Sort by index to maintain order
            sorted_data = sorted(response.data, key=lambda d: d.index)
            return [d.embedding for d in sorted_data]
        except Exception:
            logger.exception("OpenAI embedding failed")
            return [[0.0] * self._dimension_value for _ in texts]


# ---------------------------------------------------------------------------
# Semantic search utility
# ---------------------------------------------------------------------------

def find_relevant_context(
    query: str,
    entries: list[tuple[str, str]],
    provider: EmbeddingProvider,
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """Find the most semantically relevant context entries for a query.

    Args:
        query: The search query text.
        entries: List of (entry_id_or_key, entry_text) tuples.
        provider: An embedding provider instance.
        top_k: Number of top results to return.

    Returns:
        List of (entry_text, similarity_score) tuples sorted by descending
        similarity.
    """
    if not entries or not query.strip():
        return []

    texts = [text for _, text in entries]

    # Embed all texts (corpus + query together for TF-IDF vocabulary building)
    all_texts = texts + [query]
    all_vectors = provider.embed(all_texts)

    if not all_vectors or len(all_vectors) != len(all_texts):
        return []

    query_vector = all_vectors[-1]
    entry_vectors = all_vectors[:-1]

    # Compute similarities
    scored: list[tuple[str, float]] = []
    for i, (_, text) in enumerate(entries):
        score = cosine_similarity(query_vector, entry_vectors[i])
        scored.append((text, score))

    # Sort by similarity descending
    scored.sort(key=lambda x: x[1], reverse=True)

    return scored[:top_k]
