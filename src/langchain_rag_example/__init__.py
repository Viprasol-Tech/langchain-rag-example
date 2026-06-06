"""Part of LangChain RAG Example by Viprasol Tech Private Limited (https://viprasol.com).

A clean, offline-runnable LangChain Retrieval-Augmented Generation (RAG) example:
load documents, split into chunks, embed them, retrieve by similarity, and answer
a question with a deterministic chain.
"""

from __future__ import annotations

from langchain_rag_example.chain import RetrievalQA, format_context
from langchain_rag_example.embeddings import Embeddings, FakeEmbeddings, get_openai_embeddings
from langchain_rag_example.llm import BaseLLM, FakeLLM
from langchain_rag_example.loader import (
    HAS_LANGCHAIN_CORE,
    Document,
    TextLoader,
    load_text,
)
from langchain_rag_example.retriever import InMemoryVectorRetriever, cosine_similarity
from langchain_rag_example.splitter import CharacterTextSplitter

__all__ = [
    "HAS_LANGCHAIN_CORE",
    "BaseLLM",
    "CharacterTextSplitter",
    "Document",
    "Embeddings",
    "FakeEmbeddings",
    "FakeLLM",
    "InMemoryVectorRetriever",
    "RetrievalQA",
    "TextLoader",
    "cosine_similarity",
    "format_context",
    "get_openai_embeddings",
    "load_text",
]

__version__ = "0.1.0"
