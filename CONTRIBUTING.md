# Contributing

Thanks for your interest in improving **LangChain RAG Example** by
[Viprasol Tech](https://viprasol.com)! Contributions of all sizes are welcome.

## Getting started

```bash
git clone https://github.com/Viprasol-Tech/langchain-rag-example.git
cd langchain-rag-example
pip install -e ".[dev]"
```

## Before you open a PR

Everything must be green and offline:

```bash
ruff check .
ruff format --check .
mypy src
pytest
```

- Keep changes focused and add tests for new behaviour.
- The core path must remain runnable with **no network and no API keys**.
- Match the existing style; ruff and mypy `strict` are the source of truth.
- Use clear, conventional commit messages (e.g. `feat:`, `fix:`, `docs:`).

## Reporting issues

Use the issue templates for bug reports and feature requests. For anything
security-related, please follow [SECURITY.md](SECURITY.md) instead of opening a
public issue.

By contributing you agree your work is licensed under the project's
[MIT License](LICENSE).
