"""Embedding wrapper (sentence-transformers, CPU). Not an LLM call: Ollama is not involved."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray:
        """Return an ``(n, d)`` float32 array of L2-normalised vectors."""
        ...


class SentenceTransformerEmbedder:
    """Lazily loads a sentence-transformers model and embeds on CPU."""

    def __init__(self, model_name: str, revision: str | None = None, batch_size: int = 32) -> None:
        self.model_name = model_name
        self.revision = revision
        self.batch_size = batch_size
        self._model = None

    def encode(self, texts: list[str]) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device="cpu", revision=self.revision)
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vectors.astype(np.float32)
