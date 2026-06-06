# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-01-01

### Added
- Document model and `TextLoader` / `load_text` document loaders.
- `CharacterTextSplitter` with configurable overlapping chunks.
- Deterministic offline `FakeEmbeddings` and optional lazy OpenAI embeddings.
- `InMemoryVectorRetriever` with cosine-similarity ranking.
- Deterministic extractive `FakeLLM`.
- `RetrievalQA` retrieve-then-answer chain returning source documents.
- Runnable offline demo, 22 offline tests, CI, Docker, and full docs.

[0.1.0]: https://github.com/Viprasol-Tech/langchain-rag-example/releases/tag/v0.1.0
