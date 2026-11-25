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
