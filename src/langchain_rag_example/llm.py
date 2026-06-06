"""Part of LangChain RAG Example by Viprasol Tech Private Limited (https://viprasol.com).

Language models used to answer questions.

``FakeLLM`` is a deterministic, offline "model": given a prompt that contains a
question and retrieved context, it returns the most relevant sentence from the
context. This makes the RetrievalQA chain demonstrably grounded in the documents
while requiring no network access or API key.
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]?")
_WORD_RE = re.compile(r"[a-z0-9]+")


@runtime_checkable
class BaseLLM(Protocol):
    """Minimal LLM interface: turn a prompt string into a completion string."""

    def invoke(self, prompt: str) -> str:
        """Generate a completion for ``prompt``."""
        ...


def _question_from_prompt(prompt: str) -> str:
    match = re.search(r"Question:\s*(.+)", prompt)
    return match.group(1).strip() if match else prompt


def _context_from_prompt(prompt: str) -> str:
    match = re.search(r"Context:\s*(.+?)\n\s*Question:", prompt, re.DOTALL)
    return match.group(1).strip() if match else prompt


def _words(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


class FakeLLM:
    """Deterministic extractive 'LLM' for offline, reproducible answers.

    It picks the context sentence with the greatest word overlap with the
    question. Ties break toward the earliest sentence, so output is stable.
    """

    def invoke(self, prompt: str) -> str:
        """Answer the question in ``prompt`` using only its embedded context."""
        question_words = _words(_question_from_prompt(prompt))
        context = _context_from_prompt(prompt)
        sentences: list[str] = [s.strip() for s in _SENTENCE_RE.findall(context) if s.strip()]
        if not sentences:
            return "I don't know based on the provided context."
        best_sentence = sentences[0]
        best_score = -1
        for sentence in sentences:
            score = len(question_words & _words(sentence))
            if score > best_score:
                best_score = score
                best_sentence = sentence
        if best_score <= 0:
            return "I don't know based on the provided context."
        return best_sentence
