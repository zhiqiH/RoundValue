"""Resource accounting shared by the Debate and Single Agent runners.

Every node record carries the API attempts captured by the provider adapter.
Token, latency, and monetary aggregates are derived only from actually
observed values: an unknown counter remains unknown instead of becoming an
implicit zero.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def node_cumulative(nodes: Sequence[Mapping[str, Any]], model: Mapping[str, Any]) -> dict[str, Any]:
    """Aggregate usage and cost for node records from one or more rounds."""

    attempts = [attempt for node in nodes for attempt in node.get("attempts", [])]
    successful = [attempt for attempt in attempts if attempt.get("status") == "succeeded"]
    input_values = [attempt.get("input_tokens") for attempt in successful]
    output_values = [attempt.get("output_tokens") for attempt in successful]
    cache_hit_values = [attempt.get("input_cache_hit_tokens") for attempt in successful]
    cache_miss_values = [attempt.get("input_cache_miss_tokens") for attempt in successful]
    latency_values = [attempt.get("latency_ms") for attempt in attempts]
    input_known = bool(successful) and all(isinstance(value, int) for value in input_values)
    output_known = bool(successful) and all(isinstance(value, int) for value in output_values)
    cache_hit_known = bool(successful) and all(
        isinstance(value, int) for value in cache_hit_values
    )
    cache_miss_known = bool(successful) and all(
        isinstance(value, int) for value in cache_miss_values
    )
    input_tokens = sum(input_values) if input_known else None
    output_tokens = sum(output_values) if output_known else None
    latency_ms = sum(value for value in latency_values if isinstance(value, int))
    pricing = model.get("pricing", {})
    pricing_map = pricing if isinstance(pricing, dict) else {}
    input_cache_hit_price = pricing_map.get("input_cache_hit_per_million")
    input_cache_miss_price = pricing_map.get("input_cache_miss_per_million")
    output_price = pricing_map.get("output_per_million")
    cache_hit_tokens = sum(cache_hit_values) if cache_hit_known else None
    cache_miss_tokens = sum(cache_miss_values) if cache_miss_known else None
    cost_usd: float | None = None
    if (
        input_known
        and output_known
        and cache_hit_tokens is not None
        and cache_miss_tokens is not None
        and isinstance(input_cache_hit_price, (int, float))
        and isinstance(input_cache_miss_price, (int, float))
        and isinstance(output_price, (int, float))
    ):
        cost_usd = (
            cache_hit_tokens * float(input_cache_hit_price)
            + cache_miss_tokens * float(input_cache_miss_price)
            + output_tokens * float(output_price)
        ) / 1_000_000
    return {
        "logical_calls": len(nodes),
        "api_attempts": len(attempts),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency_ms": latency_ms,
        "cost_usd": cost_usd,
        "input_cache_hit_tokens": cache_hit_tokens,
        "input_cache_miss_tokens": cache_miss_tokens,
        "cost_currency": pricing_map.get("currency", "USD"),
        "usage_complete": input_known and output_known,
        "pricing_complete": cache_hit_known and cache_miss_known,
    }
