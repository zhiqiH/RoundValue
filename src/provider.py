"""Provider-neutral real model calls with an OpenAI-compatible DeepSeek adapter.

There is intentionally no mock provider.  A missing credential or an HTTP error
is an experiment failure and is written into the trajectory by the DAG runner.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import json
import os
from pathlib import Path
import time
from typing import Any
import uuid

import httpx

from contracts import ModelRequest, ModelResponse, ProviderError, get_nested, json_hash, redact, utc_now


def _credential_value(root: Path, provider_name: str, provider_config: dict[str, Any]) -> str:
    key_spec = provider_config["api_key"]
    environment_name = key_spec["environment_variable"]
    environment_value = os.environ.get(environment_name)
    if environment_value:
        return environment_value.strip()
    file_path = (root / key_spec["file"]).resolve()
    try:
        file_path.relative_to(root.resolve())
    except ValueError as error:
        raise ProviderError("credential file must be inside the project .secret directory") from error
    if not file_path.is_file():
        raise ProviderError(
            f"missing API key: set {environment_name} or create {key_spec['file']} with the documented JSON field"
        )
    try:
        credential_document = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ProviderError(f"invalid credential JSON in {key_spec['file']}: {error.msg}") from error
    if not isinstance(credential_document, dict):
        raise ProviderError(f"credential file must contain a JSON object: {key_spec['file']}")
    value = get_nested(credential_document, key_spec["field"])
    if not isinstance(value, str) or not value.strip():
        raise ProviderError(
            f"missing non-empty credential field {key_spec['field']} in {key_spec['file']} for {provider_name}"
        )
    if "REPLACE_WITH" in value or "在此" in value:
        raise ProviderError(f"replace the placeholder API key in {key_spec['file']}")
    return value.strip()


class ProviderAdapter(ABC):
    """Small interface that future vendors implement without touching Debate code."""

    @abstractmethod
    def generate(self, request: ModelRequest) -> tuple[ModelResponse, list[dict[str, Any]]]:
        raise NotImplementedError

    def close(self) -> None:
        """Adapters that own resources can override this no-op."""


class OpenAICompatibleProvider(ProviderAdapter):
    """A strict chat-completions adapter used by DeepSeek's compatible endpoint."""

    def __init__(
        self,
        *,
        provider_name: str,
        base_url: str,
        api_key: str,
        timeout_seconds: float,
        max_attempts: int,
        retry_backoff_seconds: float,
        request_defaults: dict[str, Any] | None = None,
        supports_thinking_toggle: bool = False,
    ) -> None:
        self.provider_name = provider_name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts
        self.retry_backoff_seconds = retry_backoff_seconds
        self.request_defaults = request_defaults or {}
        self.supports_thinking_toggle = supports_thinking_toggle
        self.client = httpx.Client(timeout=timeout_seconds)

    @property
    def endpoint(self) -> str:
        return self.base_url if self.base_url.endswith("/chat/completions") else f"{self.base_url}/chat/completions"

    def _payload(self, request: ModelRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": request.messages,
            "temperature": request.temperature,
            "max_tokens": request.max_output_tokens,
            "stream": False,
        }
        forbidden_defaults = {
            "model",
            "messages",
            "temperature",
            "max_tokens",
            "thinking",
            "reasoning_effort",
            "stream",
        }
        for key, value in self.request_defaults.items():
            if key not in forbidden_defaults:
                payload[key] = value
        if request.reasoning_enabled:
            raise ProviderError("this experiment forbids reasoning-enabled requests")
        if self.supports_thinking_toggle:
            # DeepSeek V4 defaults to thinking mode. This is an explicit API
            # field, rather than merely a local bookkeeping boolean.
            payload["thinking"] = {"type": "disabled"}
        return payload

    @staticmethod
    def _usage(body: dict[str, Any], key: str) -> int | None:
        usage = body.get("usage")
        value = usage.get(key) if isinstance(usage, dict) else None
        return value if isinstance(value, int) and value >= 0 else None

    @staticmethod
    def _reasoning_tokens(body: dict[str, Any]) -> int | None:
        usage = body.get("usage")
        details = usage.get("completion_tokens_details") if isinstance(usage, dict) else None
        value = details.get("reasoning_tokens") if isinstance(details, dict) else None
        return value if isinstance(value, int) and value >= 0 else None

    @staticmethod
    def _is_retryable_status(status_code: int) -> bool:
        return status_code == 429 or 500 <= status_code <= 599

    def generate(self, request: ModelRequest) -> tuple[ModelResponse, list[dict[str, Any]]]:
        payload = self._payload(request)
        attempts: list[dict[str, Any]] = []
        request_hash = json_hash(payload)
        for attempt_index in range(1, self.max_attempts + 1):
            started = utc_now()
            started_monotonic = time.monotonic()
            record: dict[str, Any] = {
                "attempt_id": uuid.uuid4().hex,
                "attempt_index": attempt_index,
                "started_at": started,
                "status": "started",
                "request_hash": request_hash,
                "request_payload": redact(payload),
                "metadata": request.metadata,
            }
            try:
                response = self.client.post(
                    self.endpoint,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                latency_ms = max(0, round((time.monotonic() - started_monotonic) * 1000))
            except (httpx.TimeoutException, httpx.NetworkError) as error:
                record.update(
                    {
                        "ended_at": utc_now(),
                        "status": "failed",
                        "retryable": True,
                        "latency_ms": max(0, round((time.monotonic() - started_monotonic) * 1000)),
                        "error_type": type(error).__name__,
                        "error_message": f"transport error: {type(error).__name__}",
                    }
                )
                attempts.append(record)
                if attempt_index < self.max_attempts:
                    time.sleep(self.retry_backoff_seconds * (2 ** (attempt_index - 1)))
                    continue
                raise ProviderError(f"{self.provider_name} transport failed after retries", attempts) from error
            except httpx.HTTPError as error:
                record.update(
                    {
                        "ended_at": utc_now(),
                        "status": "failed",
                        "retryable": False,
                        "error_type": type(error).__name__,
                        "error_message": f"HTTP client error: {type(error).__name__}",
                    }
                )
                attempts.append(record)
                raise ProviderError(f"{self.provider_name} HTTP client failed", attempts) from error

            status_code = response.status_code
            if not 200 <= status_code <= 299:
                retryable = self._is_retryable_status(status_code)
                record.update(
                    {
                        "ended_at": utc_now(),
                        "status": "failed",
                        "retryable": retryable,
                        "http_status": status_code,
                        "latency_ms": latency_ms,
                        "error_type": "HTTPStatusError",
                        "error_message": f"provider returned HTTP {status_code}",
                    }
                )
                attempts.append(record)
                if retryable and attempt_index < self.max_attempts:
                    time.sleep(self.retry_backoff_seconds * (2 ** (attempt_index - 1)))
                    continue
                raise ProviderError(f"{self.provider_name} returned HTTP {status_code}", attempts)
            try:
                body = response.json()
                if not isinstance(body, dict):
                    raise TypeError("response is not a JSON object")
                choice = body["choices"][0]
                message = choice["message"]
                text = message["content"]
                if not isinstance(text, str):
                    raise TypeError("response content is not text")
                if (
                    message.get("reasoning_content") not in (None, "")
                    or choice.get("reasoning_content") not in (None, "")
                    or (self._reasoning_tokens(body) or 0) > 0
                ):
                    raise TypeError("provider returned reasoning content despite thinking=disabled")
            except (ValueError, KeyError, IndexError, TypeError) as error:
                record.update(
                    {
                        "ended_at": utc_now(),
                        "status": "failed",
                        "retryable": False,
                        "http_status": status_code,
                        "latency_ms": latency_ms,
                        "error_type": type(error).__name__,
                        "error_message": "provider response does not match chat-completions schema",
                    }
                )
                attempts.append(record)
                raise ProviderError(f"invalid {self.provider_name} response schema", attempts) from error
            record.update(
                {
                    "ended_at": utc_now(),
                    "status": "succeeded",
                    "retryable": False,
                    "http_status": status_code,
                    "latency_ms": latency_ms,
                    "request_id": body.get("id") or response.headers.get("x-request-id"),
                    "response_model": body.get("model"),
                    "input_tokens": self._usage(body, "prompt_tokens"),
                    "output_tokens": self._usage(body, "completion_tokens"),
                    "input_cache_hit_tokens": self._usage(body, "prompt_cache_hit_tokens"),
                    "input_cache_miss_tokens": self._usage(body, "prompt_cache_miss_tokens"),
                }
            )
            attempts.append(record)
            model_response = ModelResponse(
                text=text,
                response_model=body.get("model") if isinstance(body.get("model"), str) else None,
                request_id=record["request_id"] if isinstance(record.get("request_id"), str) else None,
                finish_reason=choice.get("finish_reason") if isinstance(choice.get("finish_reason"), str) else None,
                input_tokens=self._usage(body, "prompt_tokens"),
                output_tokens=self._usage(body, "completion_tokens"),
                input_cache_hit_tokens=self._usage(body, "prompt_cache_hit_tokens"),
                input_cache_miss_tokens=self._usage(body, "prompt_cache_miss_tokens"),
                latency_ms=latency_ms,
                raw_response=redact(body),
            )
            return model_response, attempts
        raise AssertionError("retry loop unexpectedly exhausted")

    def close(self) -> None:
        self.client.close()


def build_provider(experiment: dict[str, Any]) -> ProviderAdapter:
    """Instantiate the selected adapter using a key that never enters a run record."""

    root = Path(experiment["root"])
    provider_config = experiment["provider"]
    model_config = experiment["model"]
    api_key = _credential_value(root, experiment["provider_name"], provider_config)
    adapter = provider_config["adapter"]
    if adapter != "openai_compatible":
        raise ProviderError(f"no installed adapter for {adapter}")
    return OpenAICompatibleProvider(
        provider_name=experiment["provider_name"],
        base_url=provider_config["base_url"],
        api_key=api_key,
        timeout_seconds=float(provider_config.get("timeout_seconds", 120)),
        max_attempts=int(provider_config.get("max_attempts", 1)),
        retry_backoff_seconds=float(provider_config.get("retry_backoff_seconds", 1)),
        request_defaults=dict(model_config.get("request_defaults", {})),
        supports_thinking_toggle=experiment["provider_name"] == "deepseek",
    )
