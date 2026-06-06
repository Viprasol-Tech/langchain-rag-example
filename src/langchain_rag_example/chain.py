"""Part of LangChain RAG Example by Viprasol Tech Private Limited (https://viprasol.com).

A RetrievalQA-style chain.

It ties the pieces together: retrieve relevant documents for a question, format
them into a grounded prompt, and ask an LLM to answer. The default LLM is the
deterministic offline ``FakeLLM`` so the chain runs end-to-end without network.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_rag_example.llm import BaseLLM, FakeLLM
from langchain_rag_example.loader import Document
from langchain_rag_example.retriever import InMemoryVectorRetriever

_PROMPT_TEMPLATE = (
    "Answer the question using only the context below.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n"
)


def format_context(documents: list[Document]) -> str:
    """Join document contents into a single context block."""
    return "\n".join(doc.page_content.strip() for doc in documents)


@dataclass
class RAGResult:
    """The answer plus the source documents that grounded it."""

    answer: str
    source_documents: list[Document]


class RetrievalQA:
    """Retrieve-then-answer chain over an in-memory vector retriever."""

    def __init__(self, retriever: InMemoryVectorRetriever, llm: BaseLLM | None = None) -> None:
        self.retriever = retriever
        self.llm = llm if llm is not None else FakeLLM()

    def build_prompt(self, question: str, documents: list[Document]) -> str:
        """Render the grounded prompt for ``question`` over ``documents``."""
        return _PROMPT_TEMPLATE.format(context=format_context(documents), question=question)

    def invoke(self, question: str, *, k: int | None = None) -> RAGResult:
        """Run the full retrieve-then-answer pipeline for ``question``."""
        documents = self.retriever.get_relevant_documents(question, k=k)
        prompt = self.build_prompt(question, documents)
        answer = self.llm.invoke(prompt)
        return RAGResult(answer=answer, source_documents=documents)

    def run(self, question: str, *, k: int | None = None) -> str:
        """Convenience wrapper returning just the answer string."""
        return self.invoke(question, k=k).answer
