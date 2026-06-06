"""Offline tests for the LLM and the RetrievalQA chain."""

from __future__ import annotations

from langchain_rag_example import (
    CharacterTextSplitter,
    Document,
    FakeEmbeddings,
    FakeLLM,
    InMemoryVectorRetriever,
    RetrievalQA,
    format_context,
    load_text,
)

KB = (
    "A text splitter breaks long documents into overlapping chunks. "
    "Embeddings turn text into vectors. "
    "A vector retriever ranks chunks by cosine similarity to the question. "
    "Viprasol Tech builds fintech software and algorithmic trading systems."
)


def _chain(k: int = 3) -> RetrievalQA:
    docs = load_text(KB, source="kb")
    chunks = CharacterTextSplitter(chunk_size=80, chunk_overlap=15).split_documents(docs)
    retriever = InMemoryVectorRetriever(chunks, FakeEmbeddings(), k=k)
    return RetrievalQA(retriever, llm=FakeLLM())


def test_format_context_joins_documents() -> None:
    ctx = format_context([Document(page_content="one"), Document(page_content="two")])
    assert ctx == "one\ntwo"


def test_fake_llm_picks_overlapping_sentence() -> None:
    prompt = (
        "Answer the question using only the context below.\n\n"
        "Context:\nCats meow loudly. Dogs bark at strangers.\n\n"
        "Question: What do dogs do?\n"
    )
    assert "bark" in FakeLLM().invoke(prompt).lower()


def test_fake_llm_unknown_when_no_overlap() -> None:
    prompt = (
        "Answer the question using only the context below.\n\n"
        "Context:\nThe sky is blue.\n\n"
        "Question: quantum chromodynamics tensor calculus?\n"
    )
    assert "don't know" in FakeLLM().invoke(prompt).lower()


def test_chain_answers_from_context() -> None:
    result = _chain().invoke("What does Viprasol Tech build?")
    assert "fintech" in result.answer.lower()
    assert len(result.source_documents) >= 1


def test_chain_run_returns_string() -> None:
    answer = _chain().run("What does a text splitter do?")
    assert isinstance(answer, str)
    assert "chunk" in answer.lower()


def test_chain_build_prompt_contains_question() -> None:
    chain = _chain()
    docs = chain.retriever.get_relevant_documents("embeddings")
    prompt = chain.build_prompt("What are embeddings?", docs)
    assert "Question: What are embeddings?" in prompt
    assert "Context:" in prompt


def test_chain_default_llm_is_fake() -> None:
    docs = load_text(KB)
    retriever = InMemoryVectorRetriever(docs, FakeEmbeddings())
    chain = RetrievalQA(retriever)
    assert isinstance(chain.llm, FakeLLM)
