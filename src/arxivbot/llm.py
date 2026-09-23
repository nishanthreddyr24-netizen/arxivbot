"""A thin, provider-agnostic call into a language model.

Two backends cover nearly everything anyone would want to point this at:

``gemini``
    Google's Generative Language API. The default, because its free tier is
    the only one whose context window comfortably fits a whole paper.

``openai``
    Any OpenAI-compatible ``/chat/completions`` endpoint - which is Groq,
    OpenRouter, Cerebras, Together, vLLM and Ollama, among others. One adapter,
    most of the ecosystem.

This replaces an earlier plan to depend on ``litellm``. Two well-documented
HTTP APIs are about a hundred lines against ``httpx``, which is already a
dependency, and keeping the install to two packages matters more for a tool
people are meant to clone and run than covering the long tail of providers.

Every response is cached on disk by content hash. Free tiers are rate limited
and extraction is re-run constantly during development; without this the daily
quota disappears in an afternoon.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from arxivbot.ingest.fetch import cache_dir

GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)

DEFAULT_MODEL = "gemini-3.6-flash"

# Keys the Gemini schema dialect rejects. JSON Schema produced by pydantic
# carries plenty that it will not accept.
_SCHEMA_ALLOWED = {
    "type",
    "format",
    "description",
    "enum",
    "items",
    "properties",
    "required",
    "nullable",
}


class LLMError(RuntimeError):
    """A model call failed in a way retrying will not fix."""


@dataclass(slots=True)
class LLMConfig:
    """Which model to call and how."""

    provider: str = "gemini"
    model: str = DEFAULT_MODEL
    api_key: str = ""
    base_url: str | None = None
    temperature: float = 0.0
    max_output_tokens: int = 16384
    max_retries: int = 4
    timeout: float = 180.0
    use_cache: bool = True

    @classmethod
    def from_env(cls, **overrides: Any) -> LLMConfig:
        """Build a config from environment variables.

        Reads ``ARXIVBOT_PROVIDER``, ``ARXIVBOT_MODEL``, ``ARXIVBOT_BASE_URL``
        and the first key found among ``GEMINI_API_KEY``, ``GOOGLE_API_KEY``
        and ``OPENAI_API_KEY``.
        """
        _load_dotenv()
        provider = overrides.pop("provider", None) or os.environ.get(
            "ARXIVBOT_PROVIDER", "gemini"
        )
        key = ""
        for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"):
            if value := os.environ.get(name):
                key = value
                break
        config = cls(
            provider=provider,
            model=os.environ.get("ARXIVBOT_MODEL", DEFAULT_MODEL),
            api_key=key,
            base_url=os.environ.get("ARXIVBOT_BASE_URL"),
        )
        for name, value in overrides.items():
            setattr(config, name, value)
        return config

    @property
    def identity(self) -> str:
        """How this model is recorded in a spec's provenance."""
        return f"{self.provider}/{self.model}"


def _load_dotenv(path: Path | None = None) -> None:
    """Read a local ``.env`` if present. Existing variables win."""
    env = path or Path.cwd() / ".env"
    if not env.is_file():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        os.environ.setdefault(name.strip(), value.strip().strip("\"'"))


# ---------------------------------------------------------------- schema ----


def _inline_refs(schema: dict, defs: dict | None = None) -> dict:
    """Resolve ``$ref``/``$defs`` into a self-contained schema.

    Pydantic emits references for nested models; Gemini wants them inlined.
    """
    defs = defs if defs is not None else schema.get("$defs", {})

    def walk(node: Any, depth: int = 0) -> Any:
        if depth > 12 or not isinstance(node, dict):
            return [walk(n, depth + 1) for n in node] if isinstance(node, list) else node
        if ref := node.get("$ref"):
            target = defs.get(ref.rsplit("/", 1)[-1], {})
            merged = walk(copy.deepcopy(target), depth + 1)
            extra = {k: v for k, v in node.items() if k != "$ref"}
            return {**merged, **extra} if extra else merged
        # anyOf is how pydantic spells `X | None`; take the non-null branch.
        if options := node.get("anyOf"):
            concrete = [o for o in options if o.get("type") != "null"]
            chosen = walk(concrete[0], depth + 1) if concrete else {"type": "string"}
            if len(concrete) < len(options):
                chosen = {**chosen, "nullable": True}
            rest = {k: v for k, v in node.items() if k not in ("anyOf", "$ref")}
            return {**chosen, **{k: v for k, v in rest.items() if k in _SCHEMA_ALLOWED}}
        out: dict[str, Any] = {}
        for key, value in node.items():
            if key not in _SCHEMA_ALLOWED:
                continue
            if key == "properties" and isinstance(value, dict):
                # The keys here are field names, not schema keywords, so the
                # allowlist must not be applied to them - only to their values.
                out[key] = {name: walk(sub, depth + 1) for name, sub in value.items()}
            else:
                out[key] = walk(value, depth + 1)
        return out

    return walk({k: v for k, v in schema.items() if k != "$defs"})


def gemini_schema(model: type) -> dict:
    """A pydantic model's schema, in the dialect Gemini accepts."""
    return _inline_refs(model.model_json_schema())


# ------------------------------------------------------------- transport ----


def _cache_path(payload: str) -> Path:
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]
    folder = cache_dir() / "llm"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{digest}.json"


def _retryable(status: int) -> bool:
    return status == 429 or status >= 500


def _request(url: str, payload: dict, headers: dict, config: LLMConfig) -> dict:
    """POST with exponential backoff on rate limits and server errors."""
    last = ""
    for attempt in range(config.max_retries):
        try:
            response = httpx.post(
                url, json=payload, headers=headers, timeout=config.timeout
            )
        except httpx.HTTPError as exc:
            last = f"transport: {exc}"
        else:
            if response.status_code == 200:
                return response.json()
            last = f"HTTP {response.status_code}: {response.text[:300]}"
            if not _retryable(response.status_code):
                raise LLMError(last)

        if attempt < config.max_retries - 1:
            # Jittered backoff: free tiers rate limit aggressively, and a
            # synchronised retry storm makes it worse.
            time.sleep(min(2**attempt + random.random(), 30))
    raise LLMError(f"giving up after {config.max_retries} attempts - {last}")


def _call_gemini(prompt: str, system: str | None, schema: dict | None, config: LLMConfig) -> str:
    generation: dict[str, Any] = {
        "temperature": config.temperature,
        "maxOutputTokens": config.max_output_tokens,
    }
    if schema:
        generation["responseMimeType"] = "application/json"
        generation["responseSchema"] = schema

    payload: dict[str, Any] = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": generation,
    }
    if system:
        payload["systemInstruction"] = {"parts": [{"text": system}]}

    data = _request(
        GEMINI_ENDPOINT.format(model=config.model),
        payload,
        {"x-goog-api-key": config.api_key, "Content-Type": "application/json"},
        config,
    )

    candidates = data.get("candidates") or []
    if not candidates:
        blocked = data.get("promptFeedback", {}).get("blockReason")
        raise LLMError(f"no candidates returned{f' ({blocked})' if blocked else ''}")

    parts = candidates[0].get("content", {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts)
    if not text:
        reason = candidates[0].get("finishReason", "unknown")
        raise LLMError(f"empty response (finishReason={reason})")
    return text


def _call_openai(prompt: str, system: str | None, schema: dict | None, config: LLMConfig) -> str:
    base = (config.base_url or "https://api.openai.com/v1").rstrip("/")
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload: dict[str, Any] = {
        "model": config.model,
        "messages": messages,
        "temperature": config.temperature,
        "max_tokens": config.max_output_tokens,
    }
    if schema:
        payload["response_format"] = {"type": "json_object"}

    data = _request(
        f"{base}/chat/completions",
        payload,
        {"Authorization": f"Bearer {config.api_key}", "Content-Type": "application/json"},
        config,
    )
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise LLMError(f"unexpected response shape: {str(data)[:200]}") from exc


def complete(
    prompt: str,
    *,
    system: str | None = None,
    schema: dict | None = None,
    config: LLMConfig | None = None,
) -> str:
    """Call the model and return its raw text.

    With ``schema``, the model is constrained to emit JSON matching it.
    Identical calls are served from disk rather than repeated.
    """
    config = config or LLMConfig.from_env()
    if not config.api_key:
        raise LLMError(
            "no API key found. Set GEMINI_API_KEY in your environment or a .env file "
            "(get a free one at https://aistudio.google.com/apikey)"
        )

    fingerprint = json.dumps(
        {
            "provider": config.provider,
            "model": config.model,
            "prompt": prompt,
            "system": system,
            "schema": schema,
            "temperature": config.temperature,
        },
        sort_keys=True,
    )
    cached = _cache_path(fingerprint)
    if config.use_cache and cached.exists():
        try:
            return json.loads(cached.read_text(encoding="utf-8"))["text"]
        except (ValueError, KeyError, OSError):
            pass

    if config.provider == "gemini":
        text = _call_gemini(prompt, system, schema, config)
    elif config.provider in ("openai", "openai-compatible"):
        text = _call_openai(prompt, system, schema, config)
    else:
        raise LLMError(f"unknown provider {config.provider!r}; expected gemini or openai")

    if config.use_cache:
        try:
            cached.write_text(
                json.dumps({"model": config.identity, "text": text}), encoding="utf-8"
            )
        except OSError:
            pass
    return text


def complete_json(
    prompt: str,
    model_type: type,
    *,
    system: str | None = None,
    config: LLMConfig | None = None,
):
    """Call the model and validate its reply against a pydantic model."""
    schema = gemini_schema(model_type)
    raw = complete(prompt, system=system, schema=schema, config=config)
    try:
        return model_type.model_validate_json(raw)
    except ValueError as exc:
        raise LLMError(
            f"{model_type.__name__} validation failed: {str(exc)[:400]}"
        ) from exc
