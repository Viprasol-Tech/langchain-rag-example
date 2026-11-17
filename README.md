<p align="center"><img src="docs/assets/logo.png" width="120"></p>

# LangChain RAG Example

> A clean, fully offline LangChain Retrieval-Augmented Generation example — load, split, embed, retrieve, and answer.

*Built and maintained by [Viprasol Tech](https://viprasol.com) — Fintech Experts. Full-Stack Builders.*

[![CI](https://img.shields.io/github/actions/workflow/status/Viprasol-Tech/langchain-rag-example/ci.yml?branch=main&style=flat-square)](https://github.com/Viprasol-Tech/langchain-rag-example/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg?style=flat-square)](https://www.python.org/)
[![Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg?style=flat-square)](https://github.com/astral-sh/ruff)
[![Telegram](https://img.shields.io/badge/Telegram-help-26A5E4.svg?style=flat-square&logo=telegram)](https://t.me/viprasol_help)
[![Stars](https://img.shields.io/github/stars/Viprasol-Tech/langchain-rag-example?style=flat-square)](https://github.com/Viprasol-Tech/langchain-rag-example/stargazers)

A small, readable reference implementation of the classic RAG pipeline. It shows
every stage — document loading, text splitting, embeddings, vector retrieval, and
a RetrievalQA-style answer chain — with **real working code that runs and tests
fully offline** (no network, no API keys). Swap in OpenAI embeddings later via a
lazy optional import when you want to go live.

## ✨ Features

- 📄 **Document loader** — load raw strings or UTF-8 text files into documents
- ✂️ **Character text splitter** — overlapping chunks so context is never lost
- 🧮 **Deterministic offline embeddings** — `FakeEmbeddings`, reproducible, zero deps
- 🔌 **Optional OpenAI embeddings** — lazy import, never required for the core path
- 🔎 **In-memory vector retriever** — cosine-similarity ranking (FAISS-ready)
- 🤖 **Deterministic FakeLLM** — extractive, grounded answers with no model download
- 🔗 **RetrievalQA chain** — retrieve-then-answer, returns sources for every answer
- 🧪 **22 offline tests** — fast, reproducible, network-free
- 🧰 **Strict tooling** — ruff, mypy `strict`, pytest, multi-OS CI
- 🐳 **Dockerized** — reproducible runs anywhere
- 🧩 **langchain-core friendly** — interoperates if installed, never requires it
- 📦 **Typed package** — ships `py.typed`

## 🚀 Quickstart

```bash
git clone https://github.com/Viprasol-Tech/langchain-rag-example.git
cd langchain-rag-example
pip install -e ".[dev]"

# Run the offline demo
PYTHONPATH=src python examples/demo.py

# Run the checks
ruff check . && ruff format --check . && mypy src && pytest
```

## 🧑‍💻 Usage

```python
from langchain_rag_example import (
    CharacterTextSplitter,
    FakeEmbeddings,
    InMemoryVectorRetriever,
    RetrievalQA,
    load_text,
)

docs = load_text(
    "Retrieval-Augmented Generation grounds answers in retrieved documents. "
    "Viprasol Tech builds fintech software and algorithmic trading systems."
)
chunks = CharacterTextSplitter(chunk_size=120, chunk_overlap=20).split_documents(docs)
retriever = InMemoryVectorRetriever(chunks, FakeEmbeddings(), k=3)
chain = RetrievalQA(retriever)

result = chain.invoke("What does Viprasol Tech build?")
print(result.answer)  # grounded answer
print(len(result.source_documents))  # the chunks it used
```

To go live, install the OpenAI extra and swap the embeddings:

```python
from langchain_rag_example import get_openai_embeddings

embeddings = get_openai_embeddings()  # reads OPENAI_API_KEY; requires the [openai] extra
```

## 🗺️ Roadmap

- [x] Offline-first pipeline with deterministic embeddings and LLM
- [x] In-memory cosine retriever
- [ ] Optional FAISS-backed retriever path
- [ ] Markdown / PDF loaders
- [ ] Streaming answers
- [ ] Reranking stage

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) and our
[Code of Conduct](CODE_OF_CONDUCT.md) before opening a PR.

## 📬 Contact — Viprasol Tech Private Limited

- 🌐 Website: [viprasol.com](https://viprasol.com)
- ✉️ Email: [support@viprasol.com](mailto:support@viprasol.com)
- 💬 Telegram: [t.me/viprasol_help](https://t.me/viprasol_help) · 📱 WhatsApp: +91 96336 52112
- 🐙 GitHub: [@Viprasol-Tech](https://github.com/Viprasol-Tech) · 💼 [LinkedIn](https://www.linkedin.com/in/viprasol/) · 𝕏 [@viprasol](https://twitter.com/viprasol)

> *Viprasol Tech — fintech software, algorithmic trading systems, MT4/MT5 bots, AI voice agents, and B2B SaaS. Need a custom build? [Get in touch](mailto:support@viprasol.com).*

## 📄 License

Released under the [MIT License](LICENSE) © 2025 Viprasol Tech Private Limited.
