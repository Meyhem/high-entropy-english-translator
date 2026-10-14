"""Minimal async Ollama /api/chat client."""

from __future__ import annotations

import asyncio
from typing import Protocol

import httpx

from heet.config import TeacherConfig


class TeacherUnavailable(RuntimeError):
    pass


class ChatClient(Protocol):
    async def chat(self, messages: list[dict], temperature: float) -> str: ...


class OllamaClient:
    def __init__(self, cfg: TeacherConfig, transport: httpx.AsyncBaseTransport | None = None):
        self.cfg = cfg
        self._client = httpx.AsyncClient(base_url=cfg.base_url, timeout=cfg.timeout_s, transport=transport)

    async def preflight(self) -> None:
        try:
            r = await self._client.get("/api/tags")
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise TeacherUnavailable(
                f"cannot reach Ollama at {self.cfg.base_url} ({e!r}); start it with `ollama serve`"
            ) from e
        models = r.json().get("models", [])
        names = {m.get("name") for m in models} | {m.get("model") for m in models}
        wanted = self.cfg.model if ":" in self.cfg.model else f"{self.cfg.model}:latest"
        if wanted not in names:
            raise TeacherUnavailable(
                f"model {self.cfg.model!r} not available at {self.cfg.base_url}; run `ollama pull {self.cfg.model}`"
            )

    async def chat(self, messages: list[dict], temperature: float) -> str:
        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {"temperature": temperature},
        }
        delay = 1.0
        for attempt in range(self.cfg.max_retries + 1):
            try:
                r = await self._client.post("/api/chat", json=payload)
                r.raise_for_status()
                return r.json()["message"]["content"]
            except (httpx.HTTPError, KeyError):
                if attempt == self.cfg.max_retries:
                    raise
                await asyncio.sleep(delay)
                delay *= 2
        raise AssertionError("unreachable")

    async def aclose(self) -> None:
        await self._client.aclose()
