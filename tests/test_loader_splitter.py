"""Offline tests for the loader and splitter."""

from __future__ import annotations

import pytest

from langchain_rag_example import CharacterTextSplitter, Document, TextLoader, load_text


def test_load_text_wraps_string() -> None:
    docs = load_text("hello world", source="unit")
    assert len(docs) == 1
    assert docs[0].page_content == "hello world"
    assert docs[0].metadata["source"] == "unit"


def test_text_loader_reads_file(tmp_path: object) -> None:
    path = tmp_path / "doc.txt"  # type: ignore[operator]
    path.write_text("file content here", encoding="utf-8")
    docs = TextLoader(path).load()
    assert docs[0].page_content == "file content here"
    assert "doc.txt" in str(docs[0].metadata["source"])


def test_splitter_overlap_and_count() -> None:
    splitter = CharacterTextSplitter(chunk_size=10, chunk_overlap=4)
    chunks = splitter.split_text("abcdefghijklmnopqrstuvwxyz")
    assert all(len(c) <= 10 for c in chunks)
    assert len(chunks) > 1
    # overlap region shared between consecutive chunks
    assert chunks[0][-4:] == chunks[1][:4]


def test_splitter_empty_text() -> None:
    assert CharacterTextSplitter().split_text("") == []


def test_split_documents_adds_chunk_index() -> None:
    docs = [Document(page_content="x" * 50, metadata={"source": "s"})]
    chunks = CharacterTextSplitter(chunk_size=20, chunk_overlap=5).split_documents(docs)
    assert chunks[0].metadata["chunk"] == 0
    assert chunks[1].metadata["chunk"] == 1
    assert chunks[0].metadata["source"] == "s"


def test_splitter_rejects_bad_params() -> None:
    with pytest.raises(ValueError):
        CharacterTextSplitter(chunk_size=0)
    with pytest.raises(ValueError):
        CharacterTextSplitter(chunk_size=10, chunk_overlap=10)
