"""Part of LangChain RAG Example by Viprasol Tech Private Limited (https://viprasol.com).

A small, deterministic character text splitter.

It chunks each document's ``page_content`` into overlapping windows so that
context spanning a chunk boundary is not lost during retrieval. Metadata is
copied onto every produced chunk and annotated with a ``chunk`` index.
"""

from __future__ import annotations

from langchain_rag_example.loader import Document


class CharacterTextSplitter:
    """Split documents into fixed-size character chunks with overlap."""

    def __init__(self, *, chunk_size: int = 200, chunk_overlap: int = 40) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> list[str]:
        """Split a single string into overlapping chunks."""
        if not text:
            return []
        step = self.chunk_size - self.chunk_overlap
        chunks: list[str] = []
        start = 0
        while start < len(text):
            chunk = text[start : start + self.chunk_size]
            if chunk:
                chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
            start += step
        return chunks

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """Split each document into chunk-level :class:`Document` objects."""
        out: list[Document] = []
        for doc in documents:
            for index, piece in enumerate(self.split_text(doc.page_content)):
                metadata = dict(doc.metadata)
                metadata["chunk"] = index
                out.append(Document(page_content=piece, metadata=metadata))
        return out
