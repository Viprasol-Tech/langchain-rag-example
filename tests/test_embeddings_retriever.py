"""Offline tests for embeddings and the in-memory retriever."""

from __future__ import annotations

import pytest

from langchain_rag_example import (
    FakeEmbeddings,
    InMemoryVectorRetriever,
    cosine_similarity,
)


def test_fake_embeddings_deterministic() -> None:
    emb = FakeEmbeddings(dimension=32)
    assert emb.embed_query("vector search") == emb.embed_query("vector search")


def test_fake_embeddings_dimension_and_norm() -> None:
    emb = FakeEmbeddings(dimension=16)
    vec = emb.embed_query("retrieval augmented generation")
    assert len(vec) == 16
    norm = sum(v * v for v in vec) ** 0.5
    assert norm == pytest.approx(1.0, abs=1e-9)


def test_fake_embeddings_empty_text_is_zero_vector() -> None:
    emb = FakeEmbeddings(dimension=8)
    assert emb.embed_query("") == [0.0] * 8


def test_fake_embeddings_rejects_bad_dimension() -> None:
    with pytest.raises(ValueError):
        FakeEmbeddings(dimension=0)


def test_cosine_similarity_basic() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_cosine_similarity_length_mismatch() -> None:
    with pytest.raises(ValueError):
        cosine_similarity([1.0], [1.0, 2.0])


def test_retriever_ranks_relevant_first() -> None:
    texts = [
        "Cats are small domesticated feline animals.",
        "Python is a popular programming language for data science.",
        "The stock market rewards patient long-term investors.",
    ]
    retriever = InMemoryVectorRetriever.from_texts(texts, FakeEmbeddings(), k=1)
    top = retriever.get_relevant_documents("programming language python")
    assert "Python" in top[0].page_content


def test_retriever_with_score_sorted_descending() -> None:
    retriever = InMemoryVectorRetriever.from_texts(
        ["alpha beta gamma", "delta epsilon", "alpha gamma omega"],
        FakeEmbeddings(),
        k=3,
    )
    scored = retriever.similarity_search_with_score("alpha gamma")
    scores = [s for _, s in scored]
    assert scores == sorted(scores, reverse=True)


def test_retriever_rejects_bad_k() -> None:
    with pytest.raises(ValueError):
        InMemoryVectorRetriever.from_texts(["x"], FakeEmbeddings(), k=0)
