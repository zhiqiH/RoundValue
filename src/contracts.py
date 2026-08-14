"""Small, dependency-free contracts shared by the RoundValue experiment modules.

The project intentionally stores plain JSON rather than a database or a hidden
framework.  These helpers make every on-disk record deterministic and safe to
hash without introducing a second configuration language.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


class RoundValueError(RuntimeError):
    """Base error whose message is safe to show in the command-line entry point."""


class ConfigurationError(RoundValueError):
    """Raised when a required JSON contract is invalid or internally inconsistent."""


class ProviderError(RoundValueError):
    """Raised after a real model request cannot be completed."""

    def __init__(self, message: str, attempts: list[dict[str, Any]] | None = None):
        super().__init__(message)
        self.attempts = attempts or []


class ProtocolError(RoundValueError):
    """Raised when a fixed-DAG execution cannot produce a valid checkpoint."""


def utc_now() -> str:
    """Return an ISO timestamp with an explicit UTC offset."""

    return datetime.now(UTC).isoformat()


def canonical_json(value: Any) -> str:
    """Serialize JSON in the stable form used for hashes and prompt packets."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def json_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigurationError(f"{label} must be a JSON object")
    return value


def require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ConfigurationError(f"{label} must be a JSON array")
    return value


def require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{label} must be a non-empty string")
    return value


def get_nested(mapping: Mapping[str, Any], dotted_path: str) -> Any:
    """Read a dotted JSON field without interpreting arbitrary expressions."""

    current: Any = mapping
    for segment in dotted_path.split("."):
        if not isinstance(current, Mapping) or segment not in current:
            return None
        current = current[segment]
    return current


def redact(value: Any) -> Any:
    """Remove accidental credential-looking fields before a JSON record is written."""

    secret_markers = ("api_key", "authorization", "password", "secret", "access_token", "refresh_token")
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key).casefold()
            is_secret = key_text in secret_markers or key_text.endswith("_api_key")
            result[str(key)] = "[REDACTED]" if is_secret else redact(item)
        return result
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value


def parse_json_object(text: str) -> dict[str, Any] | None:
    """Parse one strict JSON object; Markdown fences are intentionally rejected."""

    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


@dataclass(frozen=True)
class ModelRequest:
    messages: list[dict[str, str]]
    model: str
    temperature: float
    max_output_tokens: int
    reasoning_enabled: bool
    metadata: dict[str, str] = field(default_factory=dict)

    def log_view(self) -> dict[str, Any]:
        """The request has no credentials, so it is safe to retain in trajectories."""

        return asdict(self)


@dataclass(frozen=True)
class ModelResponse:
    text: str
    response_model: str | None
    request_id: str | None
    finish_reason: str | None
    input_tokens: int | None
    output_tokens: int | None
    input_cache_hit_tokens: int | None
    input_cache_miss_tokens: int | None
    latency_ms: int
    raw_response: dict[str, Any]

    def log_view(self) -> dict[str, Any]:
        return redact(asdict(self))
