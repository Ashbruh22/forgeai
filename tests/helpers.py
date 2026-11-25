"""Test-only helpers. Fakes live here and are never imported by forgeai/ or demo runs."""

from __future__ import annotations

import hashlib

import numpy as np


class FakeEmbedder:
    """Deterministic hash-bag embedder. Unit tests only; demo runs use the real model."""

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), 64), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in text.lower().split():
                out[row, int(hashlib.md5(word.encode()).hexdigest(), 16) % 64] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)


class SchemaRouter:
    """Scripted LLM for unit tests: routes each call to a handler by the schema's class name.

    Used ONLY to unit-test pipeline logic and failure handling. Demo and exit-test runs use the
    real models (tests/test_live_p3.py).
    """

    def __init__(self, handlers: dict) -> None:
        self.handlers = handlers
        self.calls: list[tuple[str, str]] = []  # (schema title, user message)

    def chat(self, *, model, messages, schema, options, think=False):
        import json

        from forgeai.llm import ChatResult

        title = schema.get("title", "")
        user = next(m["content"] for m in reversed(messages) if m["role"] == "user")
        self.calls.append((title, user))
        body = self.handlers[title](user)
        return ChatResult(content=body if isinstance(body, str) else json.dumps(body))

    def count(self, title: str) -> int:
        return sum(1 for t, _ in self.calls if t == title)
