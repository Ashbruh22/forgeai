"""Optional contradiction check with a small natural-language-inference cross-encoder (CPU only).

For a claim and the span of the evidence that best matches it, the model gives probabilities of contradiction,
entailment and neutral. A high contradiction probability turns a "supported" verdict into "contradicted". It never
upgrades a claim: support still needs a verified quote. Whether this check is kept is decided by the canaries
(``tests``-free measurement in DECISIONS.md), via ``judge.use_nli`` in forgeai.yaml.
"""

from __future__ import annotations

import numpy as np


class NliChecker:
    def __init__(self, model_name: str, revision: str | None = None) -> None:
        self.model_name, self.revision = model_name, revision
        self._model = None
        self._labels: list[str] = []

    def _load(self) -> None:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, device="cpu", revision=self.revision)  # never shares the GPU with Ollama
            id2label = self._model.config.id2label  # type: ignore[attr-defined]
            self._labels = [str(id2label[i]).lower() for i in range(len(id2label))]

    def probabilities(self, premise: str, hypothesis: str) -> dict[str, float]:
        """``{"contradiction": p, "entailment": p, "neutral": p}`` for ``hypothesis`` given ``premise``."""
        self._load()
        logits = np.asarray(self._model.predict([(premise, hypothesis)], apply_softmax=False))[0]  # type: ignore[union-attr]
        exp = np.exp(logits - logits.max())
        return {label: round(float(p), 4) for label, p in zip(self._labels, exp / exp.sum(), strict=True)}
