"""JSON-only summaries, confidence intervals, and Pareto helpers.

Every bootstrap setting is supplied by the caller and should be copied from the
frozen topology configuration into the run artifacts.  Resource aggregates are
reported as ``null`` if any completed task lacks that counter: an absent token,
cost, or latency value is unknown, not zero.
"""

from __future__ import annotations

import math
import random
from statistics import mean
from typing import Any


def _last_score(record: dict[str, Any]) -> dict[str, Any] | None:
    scores = record.get("scores")
    if not isinstance(scores, list) or not scores:
        return None
    ordered = [
        score
        for score in scores
        if isinstance(score, dict) and isinstance(score.get("round_index"), int)
    ]
    return max(ordered, key=lambda score: score["round_index"]) if ordered else None


def _last_checkpoint(record: dict[str, Any]) -> dict[str, Any] | None:
    checkpoints = record.get("trajectory", {}).get("checkpoints", [])
    if not isinstance(checkpoints, list) or not checkpoints:
        return None
    ordered = [
        item
        for item in checkpoints
        if isinstance(item, dict) and isinstance(item.get("round_index"), int)
    ]
    return max(ordered, key=lambda item: item["round_index"]) if ordered else None


def _optional_number(value: Any, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric, not boolean")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def task_level_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Emit one leakage-free aggregate row per collected task."""

    rows: list[dict[str, Any]] = []
    for record in records:
        task = record.get("task", {})
        trajectory = record.get("trajectory", {})
        checkpoint = _last_checkpoint(record)
        score = _last_score(record)
        cumulative = checkpoint.get("cumulative", {}) if checkpoint else {}
        rows.append(
            {
                "task_id": task.get("task_id"),
                "split": record.get("split"),
                "domain": task.get("domain"),
                "trajectory_status": trajectory.get("status"),
                "completed_rounds": len(trajectory.get("checkpoints", []))
                if isinstance(trajectory, dict)
                else 0,
                "final_quality": score.get("quality") if score else None,
                "input_tokens": cumulative.get("input_tokens"),
                "output_tokens": cumulative.get("output_tokens"),
                "latency_ms": cumulative.get("latency_ms"),
                "wall_clock_ms": cumulative.get("wall_clock_ms"),
                "api_latency_ms": cumulative.get("api_latency_ms"),
                "cost_usd": cumulative.get("cost_usd"),
                "logical_calls": cumulative.get("logical_calls"),
            }
        )
    return rows


def cluster_bootstrap_mean(
    values: list[float], *, seed: int, samples: int
) -> dict[str, float | int | None]:
    """Task-level bootstrap CI with explicitly configured randomness."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("bootstrap seed must be an integer")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 2:
        raise ValueError("bootstrap samples must be an integer of at least 2")
    cleaned = [_optional_number(value, "bootstrap value") for value in values]
    if any(value is None for value in cleaned):
        raise ValueError("bootstrap values must be observed finite numbers")
    numeric_values = [float(value) for value in cleaned if value is not None]
    if not numeric_values:
        return {"mean": None, "ci95_low": None, "ci95_high": None, "n": 0}
    rng = random.Random(seed)
    n = len(numeric_values)
    means = sorted(mean(numeric_values[rng.randrange(n)] for _ in range(n)) for _ in range(samples))
    low_index = max(0, math.floor(0.025 * (samples - 1)))
    high_index = min(samples - 1, math.ceil(0.975 * (samples - 1)))
    return {
        "mean": mean(numeric_values),
        "ci95_low": means[low_index],
        "ci95_high": means[high_index],
        "n": n,
    }


def _complete_aggregate(
    rows: list[dict[str, Any]],
    *,
    key: str,
    seed: int,
    samples: int,
) -> tuple[dict[str, float | int | None] | None, dict[str, Any]]:
    """Aggregate only when every completed task observed the requested field."""

    values = [_optional_number(row.get(key), key) for row in rows]
    observed = [value for value in values if value is not None]
    status = {
        "n_completed_tasks": len(rows),
        "n_observed": len(observed),
        "complete_observation": len(observed) == len(rows),
    }
    if not rows:
        return None, status
    if len(observed) != len(rows):
        return None, status
    return cluster_bootstrap_mean([float(value) for value in observed], seed=seed, samples=samples), status


def _complete_token_aggregate(
    rows: list[dict[str, Any]], *, seed: int, samples: int
) -> tuple[dict[str, float | int | None] | None, dict[str, Any]]:
    """Aggregate input+output tokens only when both counters exist per task."""

    totals: list[float] = []
    observed = 0
    for row in rows:
        input_tokens = _optional_number(row.get("input_tokens"), "input_tokens")
        output_tokens = _optional_number(row.get("output_tokens"), "output_tokens")
        if input_tokens is None or output_tokens is None:
            continue
        observed += 1
        totals.append(input_tokens + output_tokens)
    status = {
        "n_completed_tasks": len(rows),
        "n_observed": observed,
        "complete_observation": observed == len(rows),
    }
    if not rows:
        return None, status
    if observed != len(rows):
        return None, status
    return cluster_bootstrap_mean(totals, seed=seed, samples=samples), status


def summarize_collection(
    records: list[dict[str, Any]], *, bootstrap_seed: int, bootstrap_samples: int
) -> dict[str, Any]:
    """Summarize a collection without filling in unknown resource telemetry.

    ``bootstrap_seed`` and ``bootstrap_samples`` are required so the caller
    controls and records the statistical procedure; no hidden global defaults
    are used.
    """

    rows = task_level_summary(records)
    complete = [row for row in rows if row["trajectory_status"] == "complete"]
    quality, quality_observation = _complete_aggregate(
        complete, key="final_quality", seed=bootstrap_seed, samples=bootstrap_samples
    )
    tokens, token_observation = _complete_token_aggregate(
        complete, seed=bootstrap_seed, samples=bootstrap_samples
    )
    latency, latency_observation = _complete_aggregate(
        complete, key="latency_ms", seed=bootstrap_seed, samples=bootstrap_samples
    )
    wall_clock, wall_clock_observation = _complete_aggregate(
        complete, key="wall_clock_ms", seed=bootstrap_seed, samples=bootstrap_samples
    )
    api_latency, api_latency_observation = _complete_aggregate(
        complete, key="api_latency_ms", seed=bootstrap_seed, samples=bootstrap_samples
    )
    cost, cost_observation = _complete_aggregate(
        complete, key="cost_usd", seed=bootstrap_seed, samples=bootstrap_samples
    )
    input_tokens, input_observation = _complete_aggregate(
        complete, key="input_tokens", seed=bootstrap_seed, samples=bootstrap_samples
    )
    output_tokens, output_observation = _complete_aggregate(
        complete, key="output_tokens", seed=bootstrap_seed, samples=bootstrap_samples
    )
    return {
        "schema_version": "1.0",
        "bootstrap": {"seed": bootstrap_seed, "samples": bootstrap_samples},
        "tasks_total": len(rows),
        "tasks_complete": len(complete),
        "tasks_failed": len(rows) - len(complete),
        "quality": quality,
        "tokens": tokens,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency_ms": latency,
        "wall_clock_ms": wall_clock,
        "api_latency_ms": api_latency,
        "cost_usd": cost,
        "resource_observation": {
            "quality": quality_observation,
            "tokens": token_observation,
            "input_tokens": input_observation,
            "output_tokens": output_observation,
            "latency_ms": latency_observation,
            "wall_clock_ms": wall_clock_observation,
            "api_latency_ms": api_latency_observation,
            "cost_usd": cost_observation,
        },
        "cost_status": "configured"
        if cost_observation["complete_observation"] and complete
        else "unknown_or_incomplete",
    }
