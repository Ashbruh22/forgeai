"""Topic map: the set of topics the corpus actually covers (spec §3.1).

Chunk embeddings are clustered with k-means; each cluster is labelled by the LLM
from its most central chunks, with a deterministic keyword fallback.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
from pydantic import BaseModel, Field
from sklearn.cluster import KMeans

from forgeai.config import ForgeConfig
from forgeai.embeddings import Embedder
from forgeai.ingest.models import Chunk
from forgeai.llm import call_structured

SAMPLES_PER_TOPIC = 5
SAMPLE_CHARS = 400
_STOPWORDS = frozenset(
    "the a an and or of to in on for with is are be by as at from that this it its if any "
    "all must may shall will not no per your you we our their they employees employee".split()
)


class TopicLabel(BaseModel):
    label: str
    description: str


class Topic(BaseModel):
    topic_id: str
    label: str
    description: str
    size: int
    docs: list[str]
    chunk_ids: list[str] = Field(default_factory=list)


class TopicMap(BaseModel):
    topics: list[Topic]


def choose_k(n_chunks: int) -> int:
    if n_chunks < 4:
        return 1
    return max(2, min(12, round(math.sqrt(n_chunks / 2))))


def _keyword_label(texts: list[str]) -> TopicLabel:
    words = Counter(
        w for t in texts for w in re.findall(r"[a-z][a-z-]{3,}", t.lower()) if w not in _STOPWORDS
    )
    top = [w for w, _ in words.most_common(4)]
    label = " ".join(top[:3]).title() or "Miscellaneous"
    return TopicLabel(label=label, description=f"Passages mentioning: {', '.join(top)}.")


def build_topic_map(chunks: list[Chunk], embedder: Embedder, config: ForgeConfig) -> TopicMap:
    usable = [c for c in chunks if c.kind != "schema"]
    if not usable:
        return TopicMap(topics=[])
    vectors = embedder.encode([c.retrieval_text for c in usable])
    k = choose_k(len(usable))
    labels = (
        np.zeros(len(usable), dtype=int)
        if k == 1
        else KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(vectors)
    )
    model = config.agent_model
    topics: list[Topic] = []
    for cluster in sorted(set(labels.tolist())):
        members = [i for i, lab in enumerate(labels) if lab == cluster]
        centroid = vectors[members].mean(axis=0)
        order = sorted(members, key=lambda i: -float(vectors[i] @ centroid))
        samples = [usable[i].text[:SAMPLE_CHARS] for i in order[:SAMPLES_PER_TOPIC]]
        label = call_structured(
            role="ingest.topic_label",
            prompt_id="ingest/topic_label",
            variables={"samples": "\n\n".join(f"- {s}" for s in samples)},
            schema=TopicLabel,
            model=model.name,
            temperature=model.temperature,
            seed=model.seed,
            fallback=lambda s=samples: _keyword_label(s),
        )
        topics.append(
            Topic(
                topic_id=f"topic_{cluster}",
                label=label.label,
                description=label.description,
                size=len(members),
                docs=sorted({usable[i].doc for i in members}),
                chunk_ids=[usable[i].chunk_id for i in members],
            )
        )
    return TopicMap(topics=topics)
