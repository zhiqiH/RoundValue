"""Offline value labels for saved RoundValue trajectories.

``build_labels`` is the only required entry point.  It receives one complete
task record and derives all targets from *already saved* checkpoint scores and
cumulative resource counters.  It never calls a model and never reads a gold
answer itself, which keeps deployment features separate from offline labels.

For a checkpoint at round ``t`` with quality ``Q_t``, cumulative monetary cost
``C_t``, and cumulative wall-clock latency ``L_t`` the module defines:

``ΔQ_t = Q_(t+1) - Q_t``
``V_t  = ΔQ_t - lambda_cost * ΔC_t - mu_latency * ΔL_t``
``G_t  = max_u>=t [Q_u - Q_t - lambda_cost * (C_u-C_t)
                    - mu_latency * (L_u-L_t)]``

The maximum in ``G`` includes the current checkpoint.  Thus ``G=0`` means
that stopping now is at least as good as every available continuation; it is a
finite-horizon oracle target, never an online feature.

``L_t`` is the real elapsed wall-clock time whenever the collecting runner
recorded it; the summed API service time is retained separately as
``api_latency_ms`` and used only as a legacy fallback for older trajectories.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

LABEL_SCHEMA_VERSION = "1.0"
LABEL_VERSION = "roundvalue-value-labels-v1"
_EPSILON = 1e-12


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _number(value: Any, label: str, *, default: float | None = None) -> float:
    """Convert one JSON numeric value, rejecting booleans and nonfinite values."""

    if value is None:
        if default is None:
            raise ValueError(f"{label} is required")
        return float(default)
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric, not boolean")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _round_index(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _task_id(record: Mapping[str, Any]) -> str | None:
    task = _as_mapping(record.get("task")) or _as_mapping(record.get("task_spec")) or record
    value = task.get("task_id", task.get("id", record.get("task_id")))
    return str(value) if value is not None else None


def _trajectory(record: Mapping[str, Any]) -> Mapping[str, Any]:
    return _as_mapping(record.get("trajectory")) or record


def _trajectory_id(record: Mapping[str, Any], trajectory: Mapping[str, Any]) -> str | None:
    value = trajectory.get("trajectory_id", record.get("trajectory_id", trajectory.get("id")))
    return str(value) if value is not None else None


def _ordered_checkpoints(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    trajectory = _trajectory(record)
    raw = trajectory.get("checkpoints", record.get("checkpoints"))
    if not isinstance(raw, Sequence) or isinstance(raw, str | bytes | bytearray):
        raise ValueError("task record requires trajectory.checkpoints as a JSON array")
    checkpoints: list[Mapping[str, Any]] = []
    seen: set[int] = set()
    for index, checkpoint in enumerate(raw):
        item = _as_mapping(checkpoint)
        if item is None:
            raise ValueError(f"checkpoint {index} must be a JSON object")
        round_index = _round_index(
            item.get("round_index", item.get("round")), f"checkpoint {index}.round_index"
        )
        if round_index in seen:
            raise ValueError(f"duplicate checkpoint round_index {round_index}")
        seen.add(round_index)
        checkpoints.append(item)
    return sorted(checkpoints, key=lambda item: int(item.get("round_index", item.get("round"))))


def _quality_from_value(value: Any, label: str) -> float:
    if isinstance(value, Mapping):
        for key in ("quality", "is_correct", "score", "reward"):
            if key in value:
                return _quality_from_value(value[key], f"{label}.{key}")
        raise ValueError(f"{label} has no quality field")
    return _number(value, label)


def _score_by_round(record: Mapping[str, Any]) -> dict[int, float]:
    """Read both canonical score lists and legacy ``{"1": quality}`` maps."""

    raw = record.get("scores")
    if raw is None:
        raw = _trajectory(record).get("scores")
    if raw is None:
        return {}
    result: dict[int, float] = {}
    if isinstance(raw, Mapping):
        iterable: list[tuple[Any, Any]] = list(raw.items())
        for key, item in iterable:
            candidate = _as_mapping(item)
            round_value = candidate.get("round_index", candidate.get("round")) if candidate else key
            try:
                round_index = int(round_value)
            except (TypeError, ValueError) as error:
                raise ValueError("score map keys must be round indices") from error
            quality = _quality_from_value(item, f"scores[{key!r}]")
            if round_index in result and abs(result[round_index] - quality) > _EPSILON:
                raise ValueError(f"conflicting scores for round {round_index}")
            result[round_index] = quality
        return result
    if not isinstance(raw, Sequence) or isinstance(raw, str | bytes | bytearray):
        raise ValueError("scores must be a JSON array or an object indexed by round")
    for index, item in enumerate(raw):
        score = _as_mapping(item)
        if score is None:
            raise ValueError(f"scores[{index}] must be a JSON object")
        round_index = _round_index(
            score.get("round_index", score.get("round")), f"scores[{index}].round_index"
        )
        quality = _quality_from_value(score, f"scores[{index}]")
        if round_index in result and abs(result[round_index] - quality) > _EPSILON:
            raise ValueError(f"conflicting scores for round {round_index}")
        result[round_index] = quality
    return result


def _checkpoint_quality(checkpoint: Mapping[str, Any], scores: Mapping[int, float]) -> float:
    round_index = _round_index(
        checkpoint.get("round_index", checkpoint.get("round")), "checkpoint.round_index"
    )
    if round_index in scores:
        return scores[round_index]
    for key in ("quality", "is_correct", "score", "reward"):
        if key in checkpoint:
            return _quality_from_value(checkpoint[key], f"checkpoint[{round_index}].{key}")
    raise ValueError(
        "missing offline score for checkpoint round "
        f"{round_index}; run scorer.score_trajectory first"
    )


def _first_present(mapping: Mapping[str, Any], keys: Sequence[str]) -> tuple[Any, bool]:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key], True
    return None, False


def _counter(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize cumulative counters while recording whether each measure exists."""

    nested = _as_mapping(checkpoint.get("cumulative")) or {}

    def find(keys: Sequence[str]) -> tuple[float | None, bool]:
        value, present = _first_present(nested, keys)
        if not present:
            value, present = _first_present(checkpoint, keys)
        return (_number(value, "/".join(keys)), True) if present else (None, False)

    cost, cost_available = find(("cost_usd", "api_cost_usd", "cumulative_cost_usd", "cost"))
    # The latency cost is real elapsed wall clock when it was recorded.  Older
    # saved records only carried the summed API service time, which remains
    # the legacy fallback; the explicit API sum is retained alongside it.
    wall_clock, wall_clock_available = find(("wall_clock_ms", "cumulative_wall_clock_ms"))
    legacy_latency, legacy_latency_available = find(
        ("latency_ms", "cumulative_latency_ms", "latency")
    )
    latency = wall_clock if wall_clock_available else legacy_latency
    latency_available = wall_clock_available or legacy_latency_available
    api_latency, api_latency_available = find(
        ("api_latency_ms", "cumulative_api_latency_ms")
    )
    input_tokens, input_available = find(("input_tokens", "cumulative_input_tokens"))
    output_tokens, output_available = find(("output_tokens", "cumulative_output_tokens"))
    logical_calls, calls_available = find(("logical_calls", "cumulative_logical_calls", "calls"))
    return {
        "cost_usd": cost,
        "latency_ms": latency,
        "wall_clock_ms": wall_clock,
        "api_latency_ms": api_latency,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "logical_calls": logical_calls,
        "cost_available": cost_available,
        "latency_available": latency_available,
        "wall_clock_available": wall_clock_available,
        "api_latency_available": api_latency_available,
        "input_tokens_available": input_available,
        "output_tokens_available": output_available,
        "logical_calls_available": calls_available,
    }


def _check_monotonic(rows: Sequence[dict[str, Any]]) -> None:
    for previous, current in zip(rows, rows[1:], strict=False):
        for field in ("cost_usd", "latency_ms", "input_tokens", "output_tokens", "logical_calls"):
            current_value = current["cumulative"][field]
            previous_value = previous["cumulative"][field]
            if current_value is None or previous_value is None:
                continue
            if float(current_value) + _EPSILON < float(previous_value):
                raise ValueError(
                    f"cumulative {field} decreases from round {previous['round_index']} "
                    f"to {current['round_index']}"
                )


def _objective_utility(
    quality: float,
    cost_usd: float | None,
    latency_ms: float | None,
    lambda_cost: float,
    mu_latency: float,
) -> float | None:
    """Return the configured utility without assigning a value to unknown resources."""

    if lambda_cost > 0 and cost_usd is None:
        return None
    if mu_latency > 0 and latency_ms is None:
        return None
    value = float(quality)
    if lambda_cost > 0:
        # The availability check above establishes the narrow type here.
        value -= lambda_cost * float(cost_usd)
    if mu_latency > 0:
        value -= mu_latency * float(latency_ms)
    return float(value)


def _one_step_objective_value(
    delta_quality: float | None,
    delta_cost_usd: float | None,
    delta_latency_ms: float | None,
    lambda_cost: float,
    mu_latency: float,
) -> float | None:
    """Compute ``V`` only when every weighted resource delta is known."""

    if delta_quality is None:
        return None
    if lambda_cost > 0 and delta_cost_usd is None:
        return None
    if mu_latency > 0 and delta_latency_ms is None:
        return None
    value = float(delta_quality)
    if lambda_cost > 0:
        value -= lambda_cost * float(delta_cost_usd)
    if mu_latency > 0:
        value -= mu_latency * float(delta_latency_ms)
    return float(value)


def _transition_label(qualities: Sequence[float], index: int) -> tuple[str, str]:
    """Classify an adjacent quality transition, including post-harm recovery."""

    if index + 1 >= len(qualities):
        return "Terminal", "terminal"
    current = qualities[index]
    following = qualities[index + 1]
    difference = following - current
    if difference < -_EPSILON:
        return "Harm", "harm"
    if difference <= _EPSILON:
        return "Neutral", "neutral"
    # A positive movement after an earlier deterioration is distinct from an
    # initial repair.  This works for binary accuracy and continuous quality.
    prior_peak = max(qualities[: index + 1])
    if prior_peak > current + _EPSILON:
        return "Recovery", "recovery"
    return "Repair", "repair"


def build_labels(
    task_record: Mapping[str, Any],
    lambda_cost: float = 0.0,
    mu_latency: float = 0.0,
) -> list[dict[str, Any]]:
    """Build ΔQ, V, G, and transition labels for one saved task trajectory.

    ``task_record`` follows the canonical JSON contract::

        {"task": {...}, "split": "train", "trajectory": {"checkpoints": [...]},
         "scores": [{"round_index": 1, "quality": 0.0}, ...]}

    Scores may instead live on checkpoints or be an older mapping keyed by
    round.  Cumulative counters may be nested under ``checkpoint.cumulative``
    or be flat. Missing cost/latency remains null. A nonzero cost or latency
    weight is rejected when its resource counter is unknown, preventing an
    unpriced trajectory from being mistaken for a free one.
    """

    if not isinstance(task_record, Mapping):
        raise TypeError("task_record must be a JSON object")
    lambda_value = _number(lambda_cost, "lambda_cost")
    latency_weight = _number(mu_latency, "mu_latency")
    if lambda_value < 0 or latency_weight < 0:
        raise ValueError("lambda_cost and mu_latency must be non-negative")

    checkpoints = _ordered_checkpoints(task_record)
    if not checkpoints:
        return []
    scores = _score_by_round(task_record)
    trajectory = _trajectory(task_record)
    task_id = _task_id(task_record)
    trajectory_id = _trajectory_id(task_record, trajectory)
    split = task_record.get("split", trajectory.get("split"))

    rows: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        round_index = _round_index(
            checkpoint.get("round_index", checkpoint.get("round")), "checkpoint.round_index"
        )
        cumulative = _counter(checkpoint)
        rows.append(
            {
                "round_index": round_index,
                "quality": _checkpoint_quality(checkpoint, scores),
                "cumulative": cumulative,
                "checkpoint": checkpoint,
            }
        )
    _check_monotonic(rows)
    if lambda_value > 0 and not all(row["cumulative"]["cost_available"] for row in rows):
        raise ValueError("cannot construct cost-weighted V/G: cumulative monetary cost is unknown")
    if latency_weight > 0 and not all(row["cumulative"]["latency_available"] for row in rows):
        raise ValueError("cannot construct latency-weighted V/G: cumulative latency is unknown")
    qualities = [float(row["quality"]) for row in rows]

    labels: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        current = row["cumulative"]
        next_row = rows[index + 1] if index + 1 < len(rows) else None
        next_cumulative = next_row["cumulative"] if next_row is not None else None
        delta_q = float(next_row["quality"] - row["quality"]) if next_row is not None else None
        delta_cost = (
            float(next_cumulative["cost_usd"] - current["cost_usd"])
            if next_cumulative is not None
            and next_cumulative["cost_usd"] is not None
            and current["cost_usd"] is not None
            else None
        )
        delta_latency = (
            float(next_cumulative["latency_ms"] - current["latency_ms"])
            if next_cumulative is not None
            and next_cumulative["latency_ms"] is not None
            and current["latency_ms"] is not None
            else None
        )
        one_step_value = _one_step_objective_value(
            delta_q,
            delta_cost,
            delta_latency,
            lambda_value,
            latency_weight,
        )

        current_utility = _objective_utility(
            float(row["quality"]),
            current["cost_usd"],
            current["latency_ms"],
            lambda_value,
            latency_weight,
        )
        best_round: int | None = int(row["round_index"]) if current_utility is not None else None
        best_gain: float | None = 0.0 if current_utility is not None else None
        if current_utility is not None:
            for future in rows[index + 1 :]:
                future_counter = future["cumulative"]
                future_utility = _objective_utility(
                    float(future["quality"]),
                    future_counter["cost_usd"],
                    future_counter["latency_ms"],
                    lambda_value,
                    latency_weight,
                )
                if future_utility is None:
                    # This can only occur for malformed persisted data after
                    # the up-front availability checks.  It must not be
                    # silently treated as a free continuation.
                    best_gain = None
                    best_round = None
                    break
                gain = float(future_utility - current_utility)
                if best_gain is not None and gain > best_gain + _EPSILON:
                    best_gain = gain
                    best_round = int(future["round_index"])

        display_label, machine_label = _transition_label(qualities, index)
        checkpoint_hash = row["checkpoint"].get(
            "checkpoint_hash", row["checkpoint"].get("checkpoint_id")
        )
        label: dict[str, Any] = {
            "schema_version": LABEL_SCHEMA_VERSION,
            "label_version": LABEL_VERSION,
            "task_id": task_id,
            "trajectory_id": trajectory_id,
            "split": str(split) if split is not None else None,
            "round_index": int(row["round_index"]),
            "checkpoint_hash": str(checkpoint_hash) if checkpoint_hash is not None else None,
            "quality": float(row["quality"]),
            "cumulative_cost_usd": float(current["cost_usd"]) if current["cost_usd"] is not None else None,
            "cumulative_latency_ms": float(current["latency_ms"]) if current["latency_ms"] is not None else None,
            "cumulative_wall_clock_ms": float(current["wall_clock_ms"])
            if current["wall_clock_ms"] is not None
            else None,
            "cumulative_api_latency_ms": float(current["api_latency_ms"])
            if current["api_latency_ms"] is not None
            else None,
            "cumulative_input_tokens": float(current["input_tokens"]) if current["input_tokens"] is not None else None,
            "cumulative_output_tokens": float(current["output_tokens"]) if current["output_tokens"] is not None else None,
            "cumulative_logical_calls": float(current["logical_calls"]) if current["logical_calls"] is not None else None,
            "cost_available": bool(current["cost_available"]),
            "latency_available": bool(current["latency_available"]),
            "wall_clock_available": bool(current["wall_clock_available"]),
            "api_latency_available": bool(current["api_latency_available"]),
            "next_round_index": int(next_row["round_index"]) if next_row is not None else None,
            "delta_q": delta_q,
            "delta_quality": delta_q,
            "delta_cost_usd": delta_cost,
            "delta_latency_ms": delta_latency,
            "V": one_step_value,
            "one_step_value": one_step_value,
            "G": float(best_gain) if best_gain is not None else None,
            "finite_horizon_value": float(best_gain) if best_gain is not None else None,
            "best_future_round": best_round,
            "transition_label": display_label,
            "transition": machine_label,
            "label": machine_label,
            "is_terminal": next_row is None,
            "lambda_cost": lambda_value,
            "mu_latency": latency_weight,
        }
        labels.append(label)
    return labels


__all__ = [
    "LABEL_SCHEMA_VERSION",
    "LABEL_VERSION",
    "build_labels",
]
