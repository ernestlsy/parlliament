from __future__ import annotations

import json
import os
import subprocess
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class LLMError(RuntimeError):
    pass


class LLMClient(ABC):
    """One shared client is injected into every LLM-backed role."""

    @abstractmethod
    def complete_json(self, *, role: str, system: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


def _extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise LLMError("LLM response did not contain a JSON object")
        try:
            value = json.loads(text[start:end + 1])
        except json.JSONDecodeError as exc:
            raise LLMError(f"invalid JSON from LLM: {exc}") from exc
    if not isinstance(value, dict):
        raise LLMError("LLM response must be a JSON object")
    return value


class CommandLLMClient(LLMClient):
    """Adapter for any local LLM command that accepts and returns JSON on stdio."""

    def __init__(self, command: List[str], timeout_seconds: int = 180):
        if not command:
            raise ValueError("command cannot be empty")
        self.command = command
        self.timeout_seconds = timeout_seconds

    def complete_json(self, *, role: str, system: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        request = json.dumps({"role": role, "system": system, "payload": payload})
        try:
            proc = subprocess.run(
                self.command, input=request, text=True, capture_output=True,
                timeout=self.timeout_seconds, check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise LLMError(f"LLM command failed: {exc}") from exc
        if proc.returncode:
            raise LLMError(f"LLM command exited {proc.returncode}: {proc.stderr[-2000:]}")
        return _extract_json(proc.stdout)


class OpenAICompatibleClient(LLMClient):
    """Small dependency-free adapter for OpenAI-compatible chat-completions APIs."""

    def __init__(
        self, model: str, *, base_url: str = "https://api.openai.com/v1",
        api_key: Optional[str] = None, timeout_seconds: int = 180,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.timeout_seconds = timeout_seconds
        if not self.api_key:
            raise ValueError("an API key is required (argument or OPENAI_API_KEY)")

    def complete_json(self, *, role: str, system: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        body = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload)},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
        }).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=body, method="POST",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                result = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, json.JSONDecodeError, KeyError) as exc:
            raise LLMError(f"LLM HTTP request failed: {exc}") from exc
        try:
            content = result["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected LLM response shape: {result}") from exc
        return _extract_json(content)


class ScriptedLLMClient(LLMClient):
    """Deterministic adapter for tests and reproducible dry runs."""

    def __init__(self, responses: List[Dict[str, Any]]):
        self.responses = list(responses)
        self.calls: List[Dict[str, Any]] = []

    def complete_json(self, *, role: str, system: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        self.calls.append({"role": role, "system": system, "payload": payload})
        if not self.responses:
            raise LLMError(f"no scripted response left for {role}")
        return self.responses.pop(0)

