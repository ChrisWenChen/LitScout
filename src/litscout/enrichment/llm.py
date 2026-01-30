from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Optional

import httpx

from litscout.core.rate_limit import RateLimiter

try:
    from json_repair import repair_json as _repair_json
except Exception:  # pragma: no cover - optional dependency
    try:
        from jsonrepair import repair_json as _repair_json
    except Exception:  # pragma: no cover - optional dependency
        _repair_json = None


def _extract_json_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def _load_json(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        extracted = _extract_json_object(text)
        candidates = []
        if extracted:
            candidates.append(extracted)
        candidates.append(text)
        last_exc: Exception | None = None
        for candidate in candidates:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError as exc:
                last_exc = exc
                continue
        if _repair_json is not None:
            for candidate in candidates:
                try:
                    repaired = _repair_json(candidate)
                    payload = json.loads(repaired)
                    print("LLM JSON repaired")
                    return payload
                except Exception as exc:
                    last_exc = exc
                    continue
        if last_exc:
            raise last_exc
        raise


@dataclass
class LLMConfig:
    provider: str
    model: str
    temperature: float
    max_tokens: int
    seed: Optional[int] = None
    base_url: str = "https://api.openai.com"


class LLMClient:
    def __init__(self, config: LLMConfig, api_key: str | None, limiter: RateLimiter | None = None) -> None:
        self.config = config
        self.api_key = api_key
        self.limiter = limiter

    async def generate_json(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def prompt_hash(self, prompt: str) -> str:
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


class OpenAIClient(LLMClient):
    async def generate_json(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for provider=openai")
        headers = {"Authorization": f"Bearer {self.api_key}"}

        payload: dict[str, Any] = {
            "model": self.config.model,
            "input": [
                {
                    "role": "system",
                    "content": "You are a strict JSON generator. Output ONLY valid JSON. No extra text.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": self.config.temperature,
            "max_output_tokens": self.config.max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "enrichment",
                    "schema": schema,
                    "strict": True,
                },
            },
        }
        if self.config.seed is not None:
            payload["seed"] = self.config.seed

        if self.limiter:
            await self.limiter.acquire()

        started = time.monotonic()
        print(
            "LLM request start",
            {
                "provider": self.config.provider,
                "model": self.config.model,
                "base_url": self.config.base_url,
                "prompt_chars": len(prompt),
                "schema_keys": list(schema.keys()),
            },
        )
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.config.base_url}/v1/responses",
                json=payload,
                headers=headers,
                timeout=60.0,
            )
            resp.raise_for_status()
            data = resp.json()

        elapsed_ms = int((time.monotonic() - started) * 1000)
        text = None
        if isinstance(data, dict):
            text = data.get("output_text")
            if not text and "output" in data:
                try:
                    text = data["output"][0]["content"][0]["text"]
                except Exception:
                    text = None

        if not text:
            raise RuntimeError("OpenAI response missing output text")

        print(
            "LLM request done",
            {
                "provider": self.config.provider,
                "model": self.config.model,
                "elapsed_ms": elapsed_ms,
                "response_chars": len(text),
            },
        )
        return _load_json(text)


class OpenAICompatChatClient(LLMClient):
    async def generate_json(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError(f"API key required for provider={self.config.provider}")
        headers = {"Authorization": f"Bearer {self.api_key}"}
        if self.config.provider == "mimo":
            headers = {"api-key": self.api_key}

        schema_hint = (
            "Output MUST match this JSON schema: "
            f"{json.dumps(schema, ensure_ascii=True, separators=(',', ':'))}"
        )
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a strict JSON generator. Output ONLY valid JSON. No extra text.",
                },
                {"role": "system", "content": schema_hint},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.config.temperature,
            "max_completion_tokens": self.config.max_tokens,
            "response_format": {"type": "json_object"},
        }
        if self.config.seed is not None:
            payload["seed"] = self.config.seed
        if self.config.provider == "mimo":
            payload["thinking"] = {"type": "disabled"}

        if self.limiter:
            await self.limiter.acquire()

        started = time.monotonic()
        print(
            "LLM request start",
            {
                "provider": self.config.provider,
                "model": self.config.model,
                "base_url": self.config.base_url,
                "prompt_chars": len(prompt),
                "schema_keys": list(schema.keys()),
            },
        )
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.config.base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=60.0,
            )
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                detail = resp.text
                print(f"LLM request failed ({resp.status_code}): {detail}")
                raise RuntimeError(
                    f"LLM request failed ({resp.status_code}): {detail}"
                ) from exc
            data = resp.json()

        try:
            content = data["choices"][0]["message"]["content"]
        except Exception as exc:
            raise RuntimeError("Chat completion response missing content") from exc

        elapsed_ms = int((time.monotonic() - started) * 1000)
        print(
            "LLM request done",
            {
                "provider": self.config.provider,
                "model": self.config.model,
                "elapsed_ms": elapsed_ms,
                "response_chars": len(content),
            },
        )
        return _load_json(content)


def build_client(config: LLMConfig, api_key: str | None, limiter: RateLimiter | None) -> LLMClient:
    if config.provider == "openai":
        return OpenAIClient(config, api_key=api_key, limiter=limiter)
    if config.provider in ("mimo", "deepseek", "grok", "openai_compat"):
        return OpenAICompatChatClient(config, api_key=api_key, limiter=limiter)
    raise ValueError(f"unsupported provider: {config.provider}")
