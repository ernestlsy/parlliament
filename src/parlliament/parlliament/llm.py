"""Provide model clients, structured-response parsing, retries, and LLM auditing.

The module supports OpenAI Responses, compatible Chat Completions, local JSON command adapters, and
scripted test clients while preserving provider errors, request metadata, and token usage.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class LLMError(RuntimeError):
    pass


class LLMClient(ABC):
    """One shared client is injected into every LLM-backed role."""

    @abstractmethod
    def complete_json(self, *, role: str, system: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


class AuditedLLMClient(LLMClient):
    """Persist complete LLM requests, responses, and errors at run and attempt scope."""

    def __init__(self, inner: LLMClient, run_log: Path):
        self.inner = inner
        self.run_log = run_log
        self.context: Dict[str, Any] = {}
        self.sequence = 0
        self.run_log.parent.mkdir(parents=True, exist_ok=True)

    def set_context(self, **context: Any) -> None:
        self.context = dict(context)

    @staticmethod
    def _append(path: Path, event: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(event, sort_keys=True, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _record(self, event: Dict[str, Any]) -> None:
        self._append(self.run_log, event)
        attempt_dir = self.context.get("attempt_dir")
        if attempt_dir:
            self._append(Path(attempt_dir) / "llm_events.jsonl", event)

    def complete_json(self, *, role: str, system: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        self.sequence += 1
        event = {
            "sequence": self.sequence,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "context": {key: value for key, value in self.context.items() if key != "attempt_dir"},
            "role": role,
            "system": system,
            "payload": payload,
        }
        try:
            response = self.inner.complete_json(role=role, system=system, payload=payload)
        except Exception as exc:
            event.update({
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
            self._record(event)
            raise
        usage = getattr(self.inner, "last_usage", None)
        if isinstance(usage, dict):
            event["usage"] = dict(usage)
        event.update({"status": "success", "response": response})
        self._record(event)
        return response


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
    """Dependency-free adapter for OpenAI Responses and compatible Chat Completions APIs."""

    def __init__(
        self, model: str, *, base_url: str = "https://api.openai.com/v1",
        api_key: Optional[str] = None, timeout_seconds: int = 300,
        max_retries: int = 2, retry_backoff_seconds: float = 1.0,
        api_mode: str = "auto", json_mode: bool = True,
        web_search: bool = False, web_search_context_size: str = "high",
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_retries < 0 or self.retry_backoff_seconds < 0:
            raise ValueError("retry settings cannot be negative")
        if api_mode not in {"auto", "responses", "chat"}:
            raise ValueError("api_mode must be one of: auto, responses, chat")
        self.api_mode = api_mode
        self.json_mode = json_mode
        self.web_search = web_search
        self.last_usage: Optional[Dict[str, int]] = None
        if web_search_context_size not in {"low", "medium", "high"}:
            raise ValueError("web_search_context_size must be low, medium, or high")
        self.web_search_context_size = web_search_context_size
        if self.web_search and self._resolved_mode() != "responses":
            raise ValueError("hosted web search requires the Responses API")
        if not self.api_key:
            raise ValueError("an API key is required (argument or OPENAI_API_KEY)")

    def _resolved_mode(self) -> str:
        if self.api_mode != "auto":
            return self.api_mode
        hostname = (urllib.parse.urlparse(self.base_url).hostname or "").lower()
        return "responses" if hostname == "api.openai.com" else "chat"

    @staticmethod
    def _normalized_usage(result: Dict[str, Any], mode: str) -> Optional[Dict[str, int]]:
        usage = result.get("usage")
        if not isinstance(usage, dict):
            return None
        if mode == "responses":
            input_tokens = int(usage.get("input_tokens", 0) or 0)
            output_tokens = int(usage.get("output_tokens", 0) or 0)
        else:
            input_tokens = int(usage.get("prompt_tokens", 0) or 0)
            output_tokens = int(usage.get("completion_tokens", 0) or 0)
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": int(
                usage.get("total_tokens", input_tokens + output_tokens) or 0
            ),
        }

    def _post(self, endpoint: str, body: Dict[str, Any]) -> Dict[str, Any]:
        encoded = json.dumps(body).encode("utf-8")
        raw = ""
        retryable_statuses = {408, 409, 429, 500, 502, 503, 504}
        for attempt in range(self.max_retries + 1):
            request = urllib.request.Request(
                f"{self.base_url}/{endpoint}", data=encoded, method="POST",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    raw = response.read().decode("utf-8")
                break
            except urllib.error.HTTPError as exc:
                try:
                    detail = exc.read().decode("utf-8", errors="replace").strip()
                except Exception:
                    detail = ""
                request_id = exc.headers.get("x-request-id", "") if exc.headers else ""
                should_retry = exc.code in retryable_statuses and attempt < self.max_retries
                exc.close()
                if should_retry:
                    time.sleep(self.retry_backoff_seconds * (2 ** attempt))
                    continue
                suffix = f"; request_id={request_id}" if request_id else ""
                if detail:
                    suffix += f"; response={detail[:8000]}"
                raise LLMError(
                    f"LLM HTTP {exc.code} {exc.reason} at {endpoint}{suffix}; "
                    f"attempts={attempt + 1}"
                ) from exc
            except (TimeoutError, urllib.error.URLError) as exc:
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff_seconds * (2 ** attempt))
                    continue
                raise LLMError(
                    f"LLM network request failed at {endpoint} after {attempt + 1} attempts; "
                    f"timeout_seconds={self.timeout_seconds}; {type(exc).__name__}: {exc}"
                ) from exc
        try:
            result = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMError(
                f"LLM returned non-JSON HTTP response at {endpoint}: {raw[:2000]}"
            ) from exc
        if not isinstance(result, dict):
            raise LLMError(f"LLM HTTP response at {endpoint} must be a JSON object")
        return result

    @staticmethod
    def _responses_text(result: Dict[str, Any]) -> str:
        if isinstance(result.get("output_text"), str):
            return result["output_text"]
        texts = []
        for item in result.get("output", []):
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            for content in item.get("content", []):
                if isinstance(content, dict) and content.get("type") == "output_text":
                    texts.append(str(content.get("text", "")))
        if not texts:
            raise LLMError(f"unexpected Responses API response shape: {result}")
        return "".join(texts)

    @staticmethod
    def _responses_web_metadata(result: Dict[str, Any]) -> Dict[str, Any]:
        calls = []
        citations = []
        seen_citations = set()
        for item in result.get("output", []):
            if not isinstance(item, dict):
                continue
            if item.get("type") == "web_search_call":
                action = item.get("action", {})
                calls.append({
                    "id": item.get("id"),
                    "status": item.get("status"),
                    "action": action if isinstance(action, dict) else {},
                })
                continue
            if item.get("type") != "message":
                continue
            for content in item.get("content", []):
                if not isinstance(content, dict) or content.get("type") != "output_text":
                    continue
                for annotation in content.get("annotations", []):
                    if not isinstance(annotation, dict) or annotation.get("type") != "url_citation":
                        continue
                    url = str(annotation.get("url", "")).strip()
                    title = str(annotation.get("title", "")).strip()
                    if not url or url in seen_citations:
                        continue
                    seen_citations.add(url)
                    citations.append({"title": title or url, "url": url})
        return {"calls": calls, "citations": citations}

    def complete_json(self, *, role: str, system: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        mode = self._resolved_mode()
        self.last_usage = None
        user_input = (
            "Return the result as one valid JSON object only.\n\n"
            f"Request payload:\n{json.dumps(payload)}"
        )
        if mode == "responses":
            body: Dict[str, Any] = {
                "model": self.model,
                "instructions": system,
                "input": user_input,
            }
            # The Responses API rejects provider-enforced JSON mode when hosted
            # web search is present. The input still explicitly requests JSON,
            # and _extract_json validates the returned text locally.
            if self.json_mode and not self.web_search:
                body["text"] = {"format": {"type": "json_object"}}
            if self.web_search:
                body.update({
                    "tools": [{
                        "type": "web_search",
                        "search_context_size": self.web_search_context_size,
                    }],
                    "tool_choice": "required",
                    "include": ["web_search_call.action.sources"],
                })
            result = self._post("responses", body)
            self.last_usage = self._normalized_usage(result, mode)
            parsed = _extract_json(self._responses_text(result))
            if self.web_search:
                metadata = self._responses_web_metadata(result)
                if not metadata["calls"]:
                    raise LLMError("web-search-enabled response did not contain a web_search_call")
                parsed["_web_search"] = metadata
            return parsed

        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_input},
            ],
        }
        if self.json_mode:
            body["response_format"] = {"type": "json_object"}
        result = self._post("chat/completions", body)
        self.last_usage = self._normalized_usage(result, mode)
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
