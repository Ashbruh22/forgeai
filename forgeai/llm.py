"""The single gateway for every LLM call in ForgeAI.

``call_structured`` is the only place that talks to Ollama. It provides:

* schema-constrained decoding (Ollama ``format=<json-schema>``),
* Pydantic validation of the reply,
* retry with the validation error fed back to the model,
* a deterministic fallback once retries are exhausted (never a silent crash),
* a disk cache keyed by model, prompt id+version, variables, temperature and seed,
* one line per call in ``trace.jsonl`` (tokens, latency, retries, cache hit, and
  model load/swap time, so batching-by-stage can be verified on an 8 GB GPU).

A dead Ollama server is *not* papered over with the fallback: that raises
``LLMUnavailableError``, because fabricated results are worse than a loud failure.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from forgeai.config import ForgeConfig
from forgeai.prompts import PromptTemplate, load_prompt

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

Fallback = BaseModel | Callable[[], BaseModel] | None


class LLMError(Exception):
    """Base class for LLM wrapper failures."""


class LLMUnavailableError(LLMError):
    """Ollama could not be reached or returned a server error."""


class StructuredCallError(LLMError):
    """Retries were exhausted and no deterministic fallback could be built."""


@dataclass(frozen=True)
class ChatResult:
    content: str
    tokens_in: int = 0
    tokens_out: int = 0
    load_ms: float = 0.0


class Transport(Protocol):
    """Minimal chat interface. The real one talks HTTP; tests inject a fake."""

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        options: dict[str, Any],
        think: bool = False,
    ) -> ChatResult: ...


class OllamaHttpTransport:
    """Talks to a local Ollama server over its REST API."""

    def __init__(self, host: str, timeout_s: float) -> None:
        self._client = httpx.Client(base_url=host, timeout=timeout_s)
        self._no_think: set[str] = set()  # models that reject the `think` flag

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        options: dict[str, Any],
        think: bool = False,
    ) -> ChatResult:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "format": schema,
            "options": options,
        }
        if model not in self._no_think:
            payload["think"] = think  # explicit: Qwen3 thinks by default when this is omitted
        try:
            response = self._client.post("/api/chat", json=payload)
            if response.status_code == 400 and "think" in response.text and "think" in payload:
                self._no_think.add(model)
                payload.pop("think")
                response = self._client.post("/api/chat", json=payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMUnavailableError(f"Ollama request failed for {model!r}: {exc}") from exc
        body = response.json()
        return ChatResult(
            content=body.get("message", {}).get("content", ""),
            tokens_in=int(body.get("prompt_eval_count", 0)),
            tokens_out=int(body.get("eval_count", 0)),
            load_ms=float(body.get("load_duration", 0)) / 1e6,
        )


def fetch_model_digests(host: str, names: list[str], timeout_s: float = 10.0) -> dict[str, str | None]:
    """Return ``{model_name: digest}`` from the Ollama server (None if a model is not pulled)."""
    try:
        tags = httpx.get(f"{host}/api/tags", timeout=timeout_s).json()
    except httpx.HTTPError as exc:
        raise LLMUnavailableError(f"Ollama not reachable at {host}: {exc}") from exc
    installed = {m["name"]: m.get("digest") for m in tags.get("models", [])}
    return {name: installed.get(name) for name in names}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _hash(parts: dict[str, Any]) -> str:
    blob = json.dumps(parts, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class LLMClient:
    """Holds config, cache, trace and the last-loaded model (for swap logging)."""

    def __init__(
        self,
        config: ForgeConfig,
        transport: Transport | None = None,
        trace_path: Path | None = None,
    ) -> None:
        self.config = config
        self.transport: Transport = transport or OllamaHttpTransport(
            config.ollama_host, config.ollama_timeout_s
        )
        self.trace_path = trace_path or (config.cache_dir / "trace.jsonl")
        self.cache_dir = config.llm_cache_dir
        self.last_model: str | None = None
        self.swap_count = 0

    # -- public ---------------------------------------------------------------

    def call_structured(
        self,
        role: str,
        prompt_id: str,
        variables: dict[str, Any],
        schema: type[T],
        model: str,
        temperature: float,
        max_retries: int | None = None,
        *,
        seed: int | None = None,
        fallback: Fallback = None,
        prompts_dir: Path | None = None,
    ) -> T:
        retries_allowed = (
            self.config.limits.max_llm_retries if max_retries is None else max_retries
        )
        template = load_prompt(prompt_id, prompts_dir) if prompts_dir else load_prompt(prompt_id)
        system, user = template.render(variables)
        schema_json = schema.model_json_schema()
        key = _hash(
            {
                "model": model,
                "prompt_id": prompt_id,
                "prompt_version": template.version,
                "prompts_dir": str(prompts_dir) if prompts_dir else None,
                "variables": variables,
                "temperature": temperature,
                "seed": seed,
                "schema": schema_json,
            }
        )

        started = time.perf_counter()
        cached = self._cache_read(key, schema)
        if cached is not None:
            self._trace(
                role=role, template=template, model=model, tokens_in=0, tokens_out=0,
                latency_ms=(time.perf_counter() - started) * 1000, retries=0,
                cache_hit=True, fallback_used=False, load_ms=0.0, swap=False,
            )
            return cached

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        options: dict[str, Any] = {"temperature": temperature, "num_ctx": self.config.num_ctx}
        if seed is not None:
            options["seed"] = seed

        swap = self.last_model is not None and self.last_model != model
        if swap:
            self.swap_count += 1
        self.last_model = model

        tokens_in = tokens_out = 0
        load_ms = 0.0
        error = ""
        retries = 0
        result: BaseModel | None = None
        repair = load_prompt("_repair")
        for attempt in range(retries_allowed + 1):
            reply = self.transport.chat(
                model=model, messages=messages, schema=schema_json, options=options,
                think=self.config.think,
            )
            tokens_in += reply.tokens_in
            tokens_out += reply.tokens_out
            load_ms += reply.load_ms
            try:
                result = schema.model_validate_json(reply.content)
                break
            except (ValidationError, ValueError) as exc:
                error = _short_error(exc)
                if attempt == retries_allowed:
                    break
                retries += 1
                messages.append({"role": "assistant", "content": reply.content})
                messages.append(
                    {"role": "user", "content": repair.render({"error": error})[1]}
                )

        latency_ms = (time.perf_counter() - started) * 1000
        if result is not None:
            self._cache_write(key, result)
            self._trace(
                role=role, template=template, model=model, tokens_in=tokens_in,
                tokens_out=tokens_out, latency_ms=latency_ms, retries=retries,
                cache_hit=False, fallback_used=False, load_ms=load_ms, swap=swap,
            )
            return result  # type: ignore[return-value]

        # Retries exhausted: deterministic fallback, loudly recorded, never cached.
        fallback_value = self._build_fallback(fallback, schema)
        log.warning(
            "LLM output for role=%s prompt=%s invalid after %d retries; using fallback (%s)",
            role, prompt_id, retries, error,
        )
        self._trace(
            role=role, template=template, model=model, tokens_in=tokens_in,
            tokens_out=tokens_out, latency_ms=latency_ms, retries=retries,
            cache_hit=False, fallback_used=True, load_ms=load_ms, swap=swap, error=error,
        )
        if fallback_value is None:
            raise StructuredCallError(
                f"role={role} prompt={prompt_id}: output invalid after {retries} retries "
                f"and no fallback available ({error})"
            )
        return fallback_value  # type: ignore[return-value]

    # -- internals ------------------------------------------------------------

    @staticmethod
    def _build_fallback(fallback: Fallback, schema: type[T]) -> T | None:
        if isinstance(fallback, BaseModel):
            return fallback  # type: ignore[return-value]
        if callable(fallback):
            return fallback()  # type: ignore[return-value]
        try:
            return schema()  # works only when every field has a default
        except ValidationError:
            return None

    def _cache_read(self, key: str, schema: type[T]) -> T | None:
        path = self.cache_dir / f"{key}.json"
        if not path.is_file():
            return None
        try:
            return schema.model_validate_json(path.read_text(encoding="utf-8"))
        except (ValidationError, ValueError, OSError):
            return None

    def _cache_write(self, key: str, value: BaseModel) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / f"{key}.json").write_text(value.model_dump_json(), encoding="utf-8")

    def _trace(
        self,
        *,
        role: str,
        template: PromptTemplate,
        model: str,
        tokens_in: int,
        tokens_out: int,
        latency_ms: float,
        retries: int,
        cache_hit: bool,
        fallback_used: bool,
        load_ms: float,
        swap: bool,
        error: str = "",
    ) -> None:
        record: dict[str, Any] = {
            "ts": _now(),
            "role": role,
            "prompt_id": template.prompt_id,
            "prompt_version": template.version,
            "model": model,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "latency_ms": round(latency_ms, 1),
            "retries": retries,
            "cache_hit": cache_hit,
            "fallback_used": fallback_used,
            "load_ms": round(load_ms, 1),
            "swap": swap,
        }
        if error:
            record["error"] = error
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self.trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _short_error(exc: Exception) -> str:
    text = " ".join(str(exc).split())
    return text[:600]


# -- module-level default client ---------------------------------------------

_default_client: LLMClient | None = None


def set_client(client: LLMClient | None) -> None:
    """Install (or clear) the process-wide client used by ``call_structured``."""
    global _default_client
    _default_client = client


def get_client() -> LLMClient:
    if _default_client is None:
        raise LLMError("No LLM client configured. Call forgeai.llm.set_client(LLMClient(config)).")
    return _default_client


def call_structured(
    role: str,
    prompt_id: str,
    variables: dict[str, Any],
    schema: type[T],
    model: str,
    temperature: float,
    max_retries: int | None = None,
    *,
    seed: int | None = None,
    fallback: Fallback = None,
    prompts_dir: Path | None = None,
) -> T:
    """Module-level entry point: delegates to the configured ``LLMClient``."""
    return get_client().call_structured(
        role, prompt_id, variables, schema, model, temperature, max_retries,
        seed=seed, fallback=fallback, prompts_dir=prompts_dir,
    )
