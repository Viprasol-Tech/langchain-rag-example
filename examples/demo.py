"""Part of LangChain RAG Example by Viprasol Tech Private Limited (https://viprasol.com).

Runnable, fully offline demo of the RAG pipeline:
load -> split -> embed -> retrieve -> answer.

Run it with:  python examples/demo.py
"""

from __future__ import annotations

from langchain_rag_example import (
    CharacterTextSplitter,
    FakeEmbeddings,
    FakeLLM,
    InMemoryVectorRetriever,
    RetrievalQA,
    load_text,
)

KNOWLEDGE = """
Viprasol Tech Private Limited builds fintech software and algorithmic trading systems.
Retrieval-Augmented Generation grounds a language model's answers in retrieved documents.
A text splitter breaks long documents into overlapping chunks for better retrieval.
Embeddings turn text into vectors so that similar meaning maps to nearby points.
A vector retriever ranks chunks by cosine similarity to the user's question.
The RetrievalQA chain feeds the top chunks into the LLM as grounded context.
""".strip()


def main() -> None:
    """Build the pipeline and answer a few questions, all offline."""
    documents = load_text(KNOWLEDGE, source="viprasol-knowledge")
    chunks = CharacterTextSplitter(chunk_size=120, chunk_overlap=20).split_documents(documents)

    retriever = InMemoryVectorRetriever(chunks, FakeEmbeddings(), k=3)
    chain = RetrievalQA(retriever, llm=FakeLLM())

    for question in (
        "What does a text splitter do?",
        "How does the retriever rank chunks?",
        "What does Viprasol Tech build?",
    ):
        result = chain.invoke(question)
        print(f"Q: {question}")
        print(f"A: {result.answer}")
        print(f"   ({len(result.source_documents)} source chunks)\n")


if __name__ == "__main__":
    main()
