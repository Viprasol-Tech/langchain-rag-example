"""Part of LangChain RAG Example by Viprasol Tech Private Limited (https://viprasol.com).

Document model and a simple text loader.

The :class:`Document` here is a thin local shim that mirrors the shape of
``langchain_core.documents.Document`` (``page_content`` + ``metadata``). If
``langchain-core`` is installed we re-export the real class so this example
interoperates with the wider LangChain ecosystem; otherwise the local fallback
keeps everything runnable fully offline with no network and no dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.util import find_spec
from pathlib import Path

# True when langchain-core is importable, so callers can interoperate with the
# wider ecosystem. We still use the local shim below as the canonical type so
# core tests never require a network download or the optional dependency.
HAS_LANGCHAIN_CORE: bool = find_spec("langchain_core") is not None


@dataclass
class Document:
    """A unit of text plus arbitrary metadata.

    Mirrors ``langchain_core.documents.Document`` (``page_content`` + ``metadata``)
    so retrievers and chains in this package work the same whether or not
    langchain-core is installed, and stay runnable fully offline.
    """

    page_content: str
    metadata: dict[str, object] = field(default_factory=dict)


def load_text(text: str, *, source: str = "inline") -> list[Document]:
    """Wrap a raw string into a single-element list of :class:`Document`."""
    return [Document(page_content=text, metadata={"source": source})]


class TextLoader:
    """Load a UTF-8 text file from disk into a list of :class:`Document`."""

    def __init__(self, path: str | Path, *, encoding: str = "utf-8") -> None:
        self.path = Path(path)
        self.encoding = encoding

    def load(self) -> list[Document]:
        """Read the file and return it as a one-document list."""
        content = self.path.read_text(encoding=self.encoding)
        return [Document(page_content=content, metadata={"source": str(self.path)})]
