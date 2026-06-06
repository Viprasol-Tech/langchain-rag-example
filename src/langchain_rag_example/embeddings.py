"""Part of LangChain RAG Example by Viprasol Tech Private Limited (https://viprasol.com).

Embeddings: a deterministic offline ``FakeEmbeddings`` plus an optional OpenAI
backend that is imported lazily so the core of this package never needs network
access or API keys.

``FakeEmbeddings`` produces a stable, hash-based bag-of-words vector for any
text. It is deterministic across processes (seeded only by the text content), so
tests are reproducible and the whole example runs fully offline.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol, runtime_checkable

_TOKEN_RE = re.compile(r"[a-z0-9]+")


@runtime_checkable
class Embeddings(Protocol):
    """Minimal embeddings interface, matching LangChain's contract."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        ...


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class FakeEmbeddings:
    """Deterministic, offline embeddings based on hashed token counts.

    Each token is hashed into one of ``dimension`` buckets; the resulting vector
    is L2-normalised. Texts that share vocabulary land close together under
    cosine similarity, which is enough to demonstrate real retrieval behaviour
    without any model download.
    """

    def __init__(self, *, dimension: int = 64) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self.dimension = dimension

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in _tokenize(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents."""
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        return self._embed(text)


def get_openai_embeddings(
    *, model: str = "text-embedding-3-small", api_key: str | None = None
) -> Embeddings:
    """Return OpenAI-backed embeddings via a lazy import (network required).

    This is intentionally not imported at module load time so the offline path
    has zero optional dependencies. Install the ``openai`` extra to use it.
    """
    try:
        from langchain_openai import OpenAIEmbeddings  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - requires optional extra
        raise ImportError(
            "OpenAI embeddings require the 'openai' extra: "
            "pip install 'langchain-rag-example[openai]'"
        ) from exc

    kwargs: dict[str, object] = {"model": model}
    if api_key is not None:
        kwargs["api_key"] = api_key
    return OpenAIEmbeddings(**kwargs)  # type: ignore[no-any-return]
