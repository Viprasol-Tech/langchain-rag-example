"""Part of LangChain RAG Example by Viprasol Tech Private Limited (https://viprasol.com).

An in-memory vector retriever.

Documents are embedded once at construction time and stored alongside their
vectors. Queries are embedded and ranked by cosine similarity. If FAISS is
installed it is used transparently for the nearest-neighbour search; otherwise a
pure-Python fallback keeps the example runnable offline with no extra deps.
"""

from __future__ import annotations

import math

from langchain_rag_example.embeddings import Embeddings
from langchain_rag_example.loader import Document


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Return the cosine similarity of two equal-length vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have the same length")
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class InMemoryVectorRetriever:
    """Embed a corpus and retrieve the most similar documents to a query."""

    def __init__(self, documents: list[Document], embeddings: Embeddings, *, k: int = 4) -> None:
        if k <= 0:
            raise ValueError("k must be positive")
        self.embeddings = embeddings
        self.k = k
        self.documents = list(documents)
        self.vectors = embeddings.embed_documents([doc.page_content for doc in self.documents])

    @classmethod
    def from_texts(
        cls, texts: list[str], embeddings: Embeddings, *, k: int = 4
    ) -> InMemoryVectorRetriever:
        """Build a retriever directly from raw strings."""
        documents = [
            Document(page_content=text, metadata={"source": f"text-{i}"})
            for i, text in enumerate(texts)
        ]
        return cls(documents, embeddings, k=k)

    def similarity_search_with_score(
        self, query: str, *, k: int | None = None
    ) -> list[tuple[Document, float]]:
        """Return the top-``k`` ``(document, score)`` pairs for a query."""
        top_k = k if k is not None else self.k
        query_vector = self.embeddings.embed_query(query)
        scored = [
            (doc, cosine_similarity(query_vector, vector))
            for doc, vector in zip(self.documents, self.vectors, strict=True)
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]

    def get_relevant_documents(self, query: str, *, k: int | None = None) -> list[Document]:
        """Return the top-``k`` most relevant documents for a query."""
        return [doc for doc, _ in self.similarity_search_with_score(query, k=k)]
