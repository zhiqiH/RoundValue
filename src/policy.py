"""Leakage-safe replay and lightweight policy fitting for RoundValue.

All functions consume saved JSON records.  In particular,
:func:`replay_policies` never invokes a provider and never derives a deployed
decision from gold answers.  Gold-derived ``G``/``V`` labels are used only by
the two explicitly named oracle analyses or by :func:`fit_linear_policy` on a
separate training split.

The policy model format is JSON serializable and deliberately small.  A model
can be a constant/mean predictor or a ridge-fitted linear predictor; no NumPy,
scikit-learn, or model-serving runtime is required.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from statistics import mean
from typing import Any

try:  # Works when imported as ``src.policy`` and when ``src`` is on sys.path.
    from .labels import build_labels
    from .benchmark_io import public_task
except ImportError:  # pragma: no cover - exercised by the flat user entry point.
    from labels import build_labels
    from benchmark_io import public_task


POLICY_SCHEMA_VERSION = "1.0"
POLICY_VERSION = "roundvalue-replay-v1"
ROUNDVALUE_FEATURES: tuple[str, ...] = (
    "round_index",
    "prompt_length",
    "task_difficulty",
    "answer_length",
    "node_count",
    "node_answer_agreement",
    "answer_stable",
    "cumulative_input_tokens",
    "cumulative_output_tokens",
    "cumulative_cost_usd",
    "cumulative_latency_ms",
    "cumulative_logical_calls",
)
TASK_ONLY_FEATURES: tuple[str, ...] = (
    "round_index",
    "prompt_length",
    "task_difficulty",
)
_EPSILON = 1e-12


def _mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _number(value: Any, default: float = 0.0) -> float:
    if value is None or isinstance(value, bool):
        return float(default)
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _optional_number(value: Any, label: str) -> float | None:
    """Parse a nullable JSON number without fabricating a zero value."""

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


def _required_number(value: Any, label: str) -> float:
    result = _optional_number(value, label)
    if result is None:
        raise ValueError(f"{label} is required")
    return result


def _objective_utility(
    quality: float | None,
    cost_usd: float | None,
    latency_ms: float | None,
    lambda_cost: float,
    mu_latency: float,
) -> float | None:
    """Compute utility only when all weighted resource values are observed."""

    if quality is None:
        return None
    if lambda_cost > 0 and cost_usd is None:
        return None
    if mu_latency > 0 and latency_ms is None:
        return None
    value = quality
    if lambda_cost > 0:
        value -= lambda_cost * cost_usd
    if mu_latency > 0:
        value -= mu_latency * latency_ms
    return float(value)


def _nonnegative_number(value: Any, label: str) -> float:
    result = _required_number(value, label)
    if result < 0:
        raise ValueError(f"{label} must be non-negative")
    return result


def _positive_integer(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return None
    return value


def _trajectory(record: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(record.get("trajectory")) or record


def _task(record: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(record.get("task")) or _mapping(record.get("task_spec")) or record


def _task_id(record: Mapping[str, Any]) -> str:
    task = _task(record)
    value = task.get("task_id", task.get("id", record.get("task_id")))
    return str(value) if value is not None else "unknown_task"


def _trajectory_id(record: Mapping[str, Any], position: int) -> str:
    trajectory = _trajectory(record)
    value = trajectory.get("trajectory_id", record.get("trajectory_id", trajectory.get("id")))
    return str(value) if value is not None else f"trajectory_{position:06d}"


def _record_key(record: Mapping[str, Any], position: int) -> str:
    return f"{_task_id(record)}::{_trajectory_id(record, position)}"


def normalize_records(records: Any) -> list[dict[str, Any]]:
    """Normalize one record, a record list, or an ID-to-record JSON object.

    This is intentionally permissive at the outer boundary only.  Every record
    is copied to a plain ``dict``; structural validation occurs when it is
    replayed or fitted.
    """

    if isinstance(records, Mapping):
        if any(key in records for key in ("trajectory", "checkpoints", "task", "task_spec")):
            return [dict(records)]
        for key in ("records", "task_records", "items"):
            candidate = records.get(key)
            if isinstance(candidate, Sequence) and not isinstance(
                candidate, str | bytes | bytearray
            ):
                return normalize_records(candidate)
        values = list(records.values())
        if values and all(isinstance(value, Mapping) for value in values):
            return [dict(value) for value in values if isinstance(value, Mapping)]
        raise TypeError("records object must be a task record or an object of task records")
    if not isinstance(records, Sequence) or isinstance(records, str | bytes | bytearray):
        raise TypeError("records must be a JSON task record, array, or ID-to-record object")
    result: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError(f"records[{index}] must be a JSON object")
        result.append(dict(record))
    return result


def _ordered_checkpoints(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = _trajectory(record).get("checkpoints", record.get("checkpoints"))
    if not isinstance(raw, Sequence) or isinstance(raw, str | bytes | bytearray):
        raise ValueError("trajectory requires a checkpoints array")
    result: list[Mapping[str, Any]] = []
    rounds: set[int] = set()
    for index, value in enumerate(raw):
        checkpoint = _mapping(value)
        if checkpoint is None:
            raise ValueError(f"checkpoint {index} must be a JSON object")
        round_index = _positive_integer(checkpoint.get("round_index", checkpoint.get("round")))
        if round_index is None:
            raise ValueError(f"checkpoint {index} requires positive integer round_index")
        if round_index in rounds:
            raise ValueError(f"duplicate checkpoint round_index {round_index}")
        rounds.add(round_index)
        result.append(checkpoint)
    return sorted(result, key=lambda item: int(item.get("round_index", item.get("round"))))


def _model_weight(policy_model: Mapping[str, Any] | None, name: str) -> float:
    if policy_model is None:
        return 0.0
    labels = _mapping(policy_model.get("label_parameters")) or {}
    value = labels.get(name, policy_model.get(name))
    if value is None:
        nested_values = [
            candidate[name]
            for candidate in (
                _mapping(policy_model.get("roundvalue")),
                _mapping(policy_model.get("task_only")),
            )
            if candidate is not None and name in candidate
        ]
        if nested_values:
            value = nested_values[0]
            if any(
                abs(
                    _required_number(other, f"policy_model {name}")
                    - _required_number(value, f"policy_model {name}")
                )
                > _EPSILON
                for other in nested_values[1:]
            ):
                raise ValueError(f"nested policy descriptors disagree on {name}")
    if value is None:
        value = 0.0
    result = _required_number(value, f"policy_model {name}")
    if result < 0:
        raise ValueError(f"policy_model {name} must be non-negative")
    return result


def _labels_for_record(
    record: Mapping[str, Any], lambda_cost: float, mu_latency: float
) -> list[dict[str, Any]]:
    """Use persisted labels only when their costs match; otherwise rebuild."""

    persisted = record.get("labels")
    if isinstance(persisted, Sequence) and not isinstance(persisted, str | bytes | bytearray):
        candidate = [dict(item) for item in persisted if isinstance(item, Mapping)]
        if len(candidate) == len(persisted) and candidate:
            same_weights = all(
                _optional_number(item.get("lambda_cost"), "persisted label lambda_cost") is not None
                and _optional_number(item.get("mu_latency"), "persisted label mu_latency") is not None
                and abs(
                    _required_number(item.get("lambda_cost"), "persisted label lambda_cost")
                    - lambda_cost
                )
                <= _EPSILON
                and abs(
                    _required_number(item.get("mu_latency"), "persisted label mu_latency")
                    - mu_latency
                )
                <= _EPSILON
                for item in candidate
            )
            necessary = {"round_index", "quality", "cumulative_cost_usd", "cumulative_latency_ms"}
            resources_available = (
                (lambda_cost <= 0 or all(item.get("cumulative_cost_usd") is not None for item in candidate))
                and (
                    mu_latency <= 0
                    or all(item.get("cumulative_latency_ms") is not None for item in candidate)
                )
            )
            if same_weights and resources_available and all(necessary <= set(item) for item in candidate):
                return sorted(candidate, key=lambda item: int(item["round_index"]))
    return build_labels(record, lambda_cost=lambda_cost, mu_latency=mu_latency)


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, int | float) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, Mapping):
        for key in ("final_answer", "candidate_answer", "answer", "content", "text", "code"):
            if key in value:
                return _text(value[key])
    return ""


def _checkpoint_answer(checkpoint: Mapping[str, Any]) -> str:
    for key in ("final_answer", "answer", "answer_text", "writer_output", "writer", "output"):
        if key in checkpoint:
            answer = _text(checkpoint[key])
            if answer:
                return answer
    return ""


def _node_items(checkpoint: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    nodes = checkpoint.get("nodes", checkpoint.get("messages", []))
    if isinstance(nodes, Mapping):
        values = list(nodes.values())
    elif isinstance(nodes, Sequence) and not isinstance(nodes, str | bytes | bytearray):
        values = list(nodes)
    else:
        return []
    return [item for item in values if isinstance(item, Mapping)]


def _node_answer(node: Mapping[str, Any]) -> str:
    for key in ("candidate_answer", "final_answer", "answer", "output", "content", "text"):
        if key in node:
            value = _text(node[key])
            if value:
                return value
    return ""


def _normalized_answer(value: str) -> str:
    return " ".join(value.casefold().split())


def _node_agreement(checkpoint: Mapping[str, Any]) -> float:
    answers: list[str] = []
    for node in _node_items(checkpoint):
        role = str(node.get("role", node.get("node_id", node.get("id", "")))).casefold()
        if "writer" in role:
            continue
        answer = _normalized_answer(_node_answer(node))
        if answer:
            answers.append(answer)
    if len(answers) < 2:
        return 0.0
    return max(answers.count(value) for value in set(answers)) / len(answers)


def consensus_signal(
    checkpoint: Mapping[str, Any], previous_checkpoint: Mapping[str, Any] | None = None
) -> tuple[bool, str]:
    """Return a public, prefix-only consensus signal and its audit reason.

    The six Planner/Analyst/Critic outputs in a round each carry a
    ``candidate_answer`` under the frozen agent contract, so agreement is the
    largest fraction of those six answers that are identical, requiring at
    least 2/3 for consensus.  Older saved trajectories whose Planner/Critic
    nodes lack that field degrade gracefully to Analyst-only agreement.
    """

    if _node_agreement(checkpoint) >= 2.0 / 3.0:
        return True, "node_answer_agreement"
    if previous_checkpoint is not None:
        current = _normalized_answer(_checkpoint_answer(checkpoint))
        previous = _normalized_answer(_checkpoint_answer(previous_checkpoint))
        if current and current == previous:
            return True, "writer_answer_stable"
    return False, "no_consensus_signal"


def build_policy_features(
    task_record: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    label: Mapping[str, Any],
    previous_checkpoint: Mapping[str, Any] | None = None,
    *,
    task_only: bool = False,
) -> dict[str, float]:
    """Build deployable features from public task/prefix state only.

    The function intentionally does not inspect ``label.quality``, ``V``, or
    ``G``.  The label is used only for counters already present on a checkpoint.
    """

    raw_task = _task(task_record)
    # Never derive an online feature from raw code tests, reference answers,
    # or another offline-only task field. ``public_task`` is exactly the same
    # boundary used for Agent inputs in the debate runner.
    task = public_task(dict(raw_task))
    prompt = _text(task.get("prompt", ""))
    difficulty = _number(
        task.get(
            "difficulty",
            task.get(
                "level",
                task.get("public_metadata", {}).get("difficulty")
                if isinstance(task.get("public_metadata"), Mapping)
                else 0,
            ),
        )
    )
    current_answer = _normalized_answer(_checkpoint_answer(checkpoint))
    previous_answer = (
        _normalized_answer(_checkpoint_answer(previous_checkpoint)) if previous_checkpoint else ""
    )
    features: dict[str, float] = {
        "round_index": float(
            _positive_integer(checkpoint.get("round_index", checkpoint.get("round"))) or 0
        ),
        "prompt_length": float(len(prompt)),
        "task_difficulty": difficulty,
        "answer_length": float(len(_checkpoint_answer(checkpoint))),
        "node_count": float(len(_node_items(checkpoint))),
        "node_answer_agreement": _node_agreement(checkpoint),
        "answer_stable": float(bool(current_answer and current_answer == previous_answer)),
    }
    if task_only:
        return {name: features[name] for name in TASK_ONLY_FEATURES}
    # A RoundValue model is allowed to use resource features only when they
    # were actually observed.  Missing telemetry is not an implicit zero-cost,
    # zero-token, or zero-latency observation.
    features.update(
        {
            "cumulative_input_tokens": _required_number(
                label.get("cumulative_input_tokens"), "cumulative_input_tokens"
            ),
            "cumulative_output_tokens": _required_number(
                label.get("cumulative_output_tokens"), "cumulative_output_tokens"
            ),
            "cumulative_cost_usd": _required_number(
                label.get("cumulative_cost_usd"), "cumulative_cost_usd"
            ),
            "cumulative_latency_ms": _required_number(
                label.get("cumulative_latency_ms"), "cumulative_latency_ms"
            ),
            "cumulative_logical_calls": _required_number(
                label.get("cumulative_logical_calls"), "cumulative_logical_calls"
            ),
        }
    )
    return features


def _policy_spec(
    policy_model: Mapping[str, Any] | None, policy_name: str
) -> Mapping[str, Any] | None:
    if policy_model is None:
        return None
    nested = _mapping(policy_model.get(policy_name))
    if nested is not None:
        return nested
    # A direct saved descriptor is convenient for a single RoundValue policy.
    if policy_name == "roundvalue" and "kind" in policy_model:
        return policy_model
    return None


def predict_policy_value(specification: Mapping[str, Any], features: Mapping[str, float]) -> float:
    """Predict a gain using a JSON constant/mean/linear policy descriptor."""

    kind = str(specification.get("kind", "")).casefold()
    if kind in {"constant", "mean"}:
        return _required_number(
            specification.get("value", specification.get("mean")), "policy descriptor value"
        )
    if kind != "linear":
        raise ValueError("policy descriptor kind must be constant, mean, or linear")
    names_raw = specification.get("feature_names")
    weights_raw = specification.get("weights")
    if not isinstance(names_raw, Sequence) or isinstance(names_raw, str | bytes | bytearray):
        raise ValueError("linear policy requires feature_names array")
    if not isinstance(weights_raw, Sequence) or isinstance(weights_raw, str | bytes | bytearray):
        raise ValueError("linear policy requires weights array")
    names = [str(name) for name in names_raw]
    if len(names) != len(weights_raw):
        raise ValueError("linear policy feature_names and weights length mismatch")
    means_raw = specification.get("means", [0.0] * len(names))
    scales_raw = specification.get("scales", [1.0] * len(names))
    if not isinstance(means_raw, Sequence) or not isinstance(scales_raw, Sequence):
        raise ValueError("linear policy means and scales must be arrays")
    if len(means_raw) != len(names) or len(scales_raw) != len(names):
        raise ValueError("linear policy normalization length mismatch")
    prediction = _required_number(specification.get("intercept"), "linear policy intercept")
    for name, weight, mean, scale in zip(names, weights_raw, means_raw, scales_raw, strict=False):
        scale_value = _required_number(scale, f"linear policy scale for {name}")
        if abs(scale_value) <= _EPSILON:
            scale_value = 1.0
        prediction += _required_number(weight, f"linear policy weight for {name}") * (
            (
                _required_number(features.get(name), f"policy feature {name}")
                - _required_number(mean, f"linear policy mean for {name}")
            )
            / scale_value
        )
    return float(prediction)


def _decision(
    spec: Mapping[str, Any] | None,
    features: Mapping[str, float],
    *,
    policy_name: str,
) -> tuple[bool, float | None, str]:
    if spec is None:
        return True, None, "no_frozen_model_fallback_fixed_3"
    predicted = predict_policy_value(spec, features)
    threshold = _required_number(spec.get("threshold", 0.0), "policy threshold")
    if predicted > threshold:
        return True, predicted, "positive_predicted_value"
    return False, predicted, "nonpositive_predicted_value"


def _label_by_round(labels: Sequence[Mapping[str, Any]]) -> dict[int, Mapping[str, Any]]:
    result: dict[int, Mapping[str, Any]] = {}
    for label in labels:
        round_index = _positive_integer(label.get("round_index"))
        if round_index is None:
            raise ValueError("label requires positive integer round_index")
        if round_index in result:
            raise ValueError(f"duplicate label round_index {round_index}")
        result[round_index] = label
    return result


def _selected_result(
    record: Mapping[str, Any],
    position: int,
    policy: str,
    selected_checkpoint: Mapping[str, Any] | None,
    selected_label: Mapping[str, Any] | None,
    reason: str,
    decision_trace: Sequence[Mapping[str, Any]],
    *,
    available: bool = True,
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "task_id": _task_id(record),
        "trajectory_id": _trajectory_id(record, position),
        "split": str(record.get("split", _trajectory(record).get("split")))
        if record.get("split", _trajectory(record).get("split")) is not None
        else None,
        "policy": policy,
        "available": bool(available),
        "termination_reason": reason,
        "decision_trace": [dict(item) for item in decision_trace],
    }
    if not available or selected_checkpoint is None or selected_label is None:
        base.update(
            {
                "selected_round": None,
                "stop_round": None,
                "quality": None,
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
                "cost_usd": None,
                "latency_ms": None,
                "wall_clock_ms": None,
                "api_latency_ms": None,
                "logical_calls": None,
                "utility": None,
            }
        )
        return base
    input_tokens = _optional_number(
        selected_label.get("cumulative_input_tokens"), "selected cumulative_input_tokens"
    )
    output_tokens = _optional_number(
        selected_label.get("cumulative_output_tokens"), "selected cumulative_output_tokens"
    )
    cost = _optional_number(selected_label.get("cumulative_cost_usd"), "selected cumulative_cost_usd")
    latency = _optional_number(
        selected_label.get("cumulative_latency_ms"), "selected cumulative_latency_ms"
    )
    wall_clock = _optional_number(
        selected_label.get("cumulative_wall_clock_ms"), "selected cumulative_wall_clock_ms"
    )
    api_latency = _optional_number(
        selected_label.get("cumulative_api_latency_ms"), "selected cumulative_api_latency_ms"
    )
    calls = _optional_number(
        selected_label.get("cumulative_logical_calls"), "selected cumulative_logical_calls"
    )
    quality = _optional_number(selected_label.get("quality"), "selected quality")
    lambda_cost = _required_number(selected_label.get("lambda_cost"), "selected lambda_cost")
    mu_latency = _required_number(selected_label.get("mu_latency"), "selected mu_latency")
    round_index = _positive_integer(
        selected_checkpoint.get("round_index", selected_checkpoint.get("round"))
    )
    base.update(
        {
            "selected_round": round_index,
            "stop_round": round_index,
            "quality": quality,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens
            if input_tokens is not None and output_tokens is not None
            else None,
            "cost_usd": cost,
            "latency_ms": latency,
            "wall_clock_ms": wall_clock,
            "api_latency_ms": api_latency,
            "logical_calls": calls,
            "utility": _objective_utility(
                quality,
                cost,
                latency,
                lambda_cost,
                mu_latency,
            ),
            "checkpoint_hash": selected_label.get("checkpoint_hash"),
        }
    )
    return base


def _replay_fixed(
    record: Mapping[str, Any],
    position: int,
    checkpoints: Sequence[Mapping[str, Any]],
    labels: Mapping[int, Mapping[str, Any]],
    stop_round: int,
) -> dict[str, Any]:
    checkpoint = next(
        (
            item
            for item in checkpoints
            if _positive_integer(item.get("round_index", item.get("round"))) == stop_round
        ),
        None,
    )
    if checkpoint is None:
        return _selected_result(
            record,
            position,
            f"fixed_{stop_round}",
            None,
            None,
            "requested_round_unavailable",
            [],
            available=False,
        )
    return _selected_result(
        record,
        position,
        f"fixed_{stop_round}",
        checkpoint,
        labels[stop_round],
        "fixed_round",
        [{"round_index": stop_round, "decision": "STOP", "reason": "fixed_round"}],
    )


def _replay_consensus(
    record: Mapping[str, Any],
    position: int,
    checkpoints: Sequence[Mapping[str, Any]],
    labels: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    trace: list[dict[str, Any]] = []
    selected = checkpoints[-1]
    reason = "max_available_round"
    for index, checkpoint in enumerate(checkpoints[:-1]):
        previous = checkpoints[index - 1] if index else None
        signal, signal_reason = consensus_signal(checkpoint, previous)
        round_index = _positive_integer(checkpoint.get("round_index", checkpoint.get("round")))
        trace.append(
            {
                "round_index": round_index,
                "decision": "STOP" if signal else "CONTINUE",
                "reason": signal_reason,
                "consensus_signal": signal,
            }
        )
        if signal:
            selected = checkpoint
            reason = "consensus_stop"
            break
    else:
        final_round = _positive_integer(selected.get("round_index", selected.get("round")))
        trace.append(
            {"round_index": final_round, "decision": "FORCED_STOP", "reason": "max_available_round"}
        )
    round_index = _positive_integer(selected.get("round_index", selected.get("round")))
    return _selected_result(
        record, position, "consensus", selected, labels[round_index], reason, trace
    )


def _replay_predicted(
    record: Mapping[str, Any],
    position: int,
    checkpoints: Sequence[Mapping[str, Any]],
    labels: Mapping[int, Mapping[str, Any]],
    specification: Mapping[str, Any] | None,
    policy_name: str,
    *,
    task_only: bool,
) -> dict[str, Any]:
    trace: list[dict[str, Any]] = []
    selected = checkpoints[-1]
    reason = "max_available_round"
    for index, checkpoint in enumerate(checkpoints[:-1]):
        round_index = _positive_integer(checkpoint.get("round_index", checkpoint.get("round")))
        previous = checkpoints[index - 1] if index else None
        if specification is None:
            should_continue, prediction, decision_reason = (
                True,
                None,
                "no_frozen_model_fallback_fixed_3",
            )
        else:
            try:
                features = build_policy_features(
                    record, checkpoint, labels[round_index], previous, task_only=task_only
                )
                should_continue, prediction, decision_reason = _decision(
                    specification, features, policy_name=policy_name
                )
            except ValueError as error:
                # The model cannot use a missing resource feature.  Returning
                # an unavailable replay makes the telemetry gap visible instead
                # of silently replacing it with a zero observation.
                return _selected_result(
                    record,
                    position,
                    policy_name,
                    None,
                    None,
                    f"{policy_name}_unavailable_missing_or_invalid_feature",
                    [
                        *trace,
                        {
                            "round_index": round_index,
                            "decision": "UNAVAILABLE",
                            "reason": str(error),
                        },
                    ],
                    available=False,
                )
        trace.append(
            {
                "round_index": round_index,
                "decision": "CONTINUE" if should_continue else "STOP",
                "predicted_value": prediction,
                "reason": decision_reason,
            }
        )
        if not should_continue:
            selected = checkpoint
            reason = f"{policy_name}_stop"
            break
    else:
        final_round = _positive_integer(selected.get("round_index", selected.get("round")))
        trace.append(
            {"round_index": final_round, "decision": "FORCED_STOP", "reason": "max_available_round"}
        )
    selected_round = _positive_integer(selected.get("round_index", selected.get("round")))
    return _selected_result(
        record, position, policy_name, selected, labels[selected_round], reason, trace
    )


def _replay_oracle(
    record: Mapping[str, Any],
    position: int,
    checkpoints: Sequence[Mapping[str, Any]],
    labels: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    def ranking(checkpoint: Mapping[str, Any]) -> tuple[float, int]:
        round_index = _positive_integer(checkpoint.get("round_index", checkpoint.get("round")))
        label = labels[round_index]
        utility = _objective_utility(
            _optional_number(label.get("quality"), "oracle quality"),
            _optional_number(label.get("cumulative_cost_usd"), "oracle cumulative_cost_usd"),
            _optional_number(label.get("cumulative_latency_ms"), "oracle cumulative_latency_ms"),
            _required_number(label.get("lambda_cost"), "oracle lambda_cost"),
            _required_number(label.get("mu_latency"), "oracle mu_latency"),
        )
        if utility is None:
            raise ValueError(
                "oracle utility is unavailable because a weighted cost or latency value is unknown"
            )
        # Do not use unknown monetary cost as a tie-breaker.  Earlier rounds
        # provide a deterministic, resource-neutral tie resolution.
        return utility, -round_index

    selected = max(checkpoints, key=ranking)
    selected_round = _positive_integer(selected.get("round_index", selected.get("round")))
    return _selected_result(
        record,
        position,
        "oracle",
        selected,
        labels[selected_round],
        "trajectory_oracle_max_utility",
        [{"round_index": selected_round, "decision": "STOP", "reason": "trajectory_oracle"}],
    )


def _replay_one_step_oracle(
    record: Mapping[str, Any],
    position: int,
    checkpoints: Sequence[Mapping[str, Any]],
    labels: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    trace: list[dict[str, Any]] = []
    selected = checkpoints[-1]
    reason = "max_available_round"
    for checkpoint in checkpoints[:-1]:
        round_index = _positive_integer(checkpoint.get("round_index", checkpoint.get("round")))
        one_step_value = _optional_number(labels[round_index].get("V"), "oracle one_step_value")
        if one_step_value is None:
            raise ValueError(
                "oracle one-step value is unavailable because a weighted cost or latency delta is unknown"
            )
        should_continue = one_step_value > 0.0
        trace.append(
            {
                "round_index": round_index,
                "decision": "CONTINUE" if should_continue else "STOP",
                "oracle_one_step_value": one_step_value,
                "reason": "positive_one_step_value"
                if should_continue
                else "nonpositive_one_step_value",
            }
        )
        if not should_continue:
            selected = checkpoint
            reason = "oracle_one_step_stop"
            break
    else:
        final_round = _positive_integer(selected.get("round_index", selected.get("round")))
        trace.append(
            {"round_index": final_round, "decision": "FORCED_STOP", "reason": "max_available_round"}
        )
    selected_round = _positive_integer(selected.get("round_index", selected.get("round")))
    return _selected_result(
        record, position, "oracle_one_step", selected, labels[selected_round], reason, trace
    )


def _empty_metrics(total_records: int) -> dict[str, Any]:
    result: dict[str, Any] = {
        "n_records": total_records,
        "n_available": 0,
        "coverage": 0.0 if total_records else None,
        "mean_quality": None,
        "accuracy": None,
        "mean_utility": None,
        "mean_input_tokens": None,
        "mean_output_tokens": None,
        "mean_total_tokens": None,
        "mean_cost_usd": None,
        "mean_latency_ms": None,
        "mean_wall_clock_ms": None,
        "mean_api_latency_ms": None,
        "mean_logical_calls": None,
        "mean_stop_round": None,
        "stop_round_counts": {},
    }
    for name in (
        "quality",
        "utility",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cost_usd",
        "latency_ms",
        "wall_clock_ms",
        "api_latency_ms",
        "logical_calls",
        "stop_round",
    ):
        result[f"n_{name}_observed"] = 0
    return result


def _metrics(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    available = [item for item in results if item.get("available")]
    if not available:
        return _empty_metrics(len(results))

    def complete_mean(key: str) -> tuple[float | None, int]:
        values = [_optional_number(item.get(key), key) for item in available]
        observed = [value for value in values if value is not None]
        # A policy-level resource statistic is unknown when any selected task
        # lacks the underlying counter.  Reporting a partial mean as a complete
        # experiment mean would hide missing telemetry.
        if len(observed) != len(available):
            return None, len(observed)
        return sum(observed) / len(observed), len(observed)

    counts = Counter(str(item.get("stop_round")) for item in available)
    metric_values = {
        key: complete_mean(key)
        for key in (
            "quality",
            "utility",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "cost_usd",
            "latency_ms",
            "wall_clock_ms",
            "api_latency_ms",
            "logical_calls",
            "stop_round",
        )
    }
    return {
        "n_records": len(results),
        "n_available": len(available),
        "coverage": len(available) / len(results) if results else None,
        "mean_quality": metric_values["quality"][0],
        "accuracy": metric_values["quality"][0],
        "mean_utility": metric_values["utility"][0],
        "mean_input_tokens": metric_values["input_tokens"][0],
        "mean_output_tokens": metric_values["output_tokens"][0],
        "mean_total_tokens": metric_values["total_tokens"][0],
        "mean_cost_usd": metric_values["cost_usd"][0],
        "mean_latency_ms": metric_values["latency_ms"][0],
        "mean_wall_clock_ms": metric_values["wall_clock_ms"][0],
        "mean_api_latency_ms": metric_values["api_latency_ms"][0],
        "mean_logical_calls": metric_values["logical_calls"][0],
        "mean_stop_round": metric_values["stop_round"][0],
        **{
            f"n_{name}_observed": count
            for name, (_, count) in metric_values.items()
        },
        "stop_round_counts": {key: int(count) for key, count in sorted(counts.items())},
    }


def _transition_metrics(all_labels: Sequence[Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    for labels in all_labels:
        for label in labels:
            counts[str(label.get("transition", "unknown"))] += 1
    total_decisions = sum(counts[name] for name in ("repair", "neutral", "harm", "recovery"))
    return {
        "n_checkpoints": sum(len(labels) for labels in all_labels),
        "n_decisions": total_decisions,
        "transition_counts": {
            name: int(counts[name])
            for name in ("repair", "neutral", "harm", "recovery", "terminal", "unknown")
        },
        "transition_rates": {
            name: counts[name] / total_decisions if total_decisions else None
            for name in ("repair", "neutral", "harm", "recovery")
        },
    }


def _apply_oracle_regret(
    policy_results: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    oracle = policy_results.get("oracle", [])
    oracle_by_key = {
        f"{item.get('task_id')}::{item.get('trajectory_id')}": item
        for item in oracle
        if item.get("available")
    }
    summaries: dict[str, dict[str, Any]] = {}
    for name, rows in policy_results.items():
        quality_regrets: list[float] = []
        utility_regrets: list[float] = []
        for row in rows:
            key = f"{row.get('task_id')}::{row.get('trajectory_id')}"
            oracle_row = oracle_by_key.get(key)
            if not row.get("available") or oracle_row is None:
                continue
            oracle_quality = _optional_number(oracle_row.get("quality"), "oracle quality")
            row_quality = _optional_number(row.get("quality"), "policy quality")
            if oracle_quality is not None and row_quality is not None:
                quality_regrets.append(oracle_quality - row_quality)
            oracle_utility = _optional_number(oracle_row.get("utility"), "oracle utility")
            row_utility = _optional_number(row.get("utility"), "policy utility")
            if oracle_utility is not None and row_utility is not None:
                utility_regrets.append(oracle_utility - row_utility)
        summaries[name] = {
            "n_oracle_paired": len(quality_regrets),
            "n_oracle_utility_paired": len(utility_regrets),
            "mean_oracle_quality_regret": sum(quality_regrets) / len(quality_regrets)
            if quality_regrets
            else None,
            "mean_oracle_utility_regret": sum(utility_regrets) / len(utility_regrets)
            if utility_regrets
            else None,
        }
    return summaries


def _pairwise_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
    key: str,
    *,
    seed: int,
    samples: int,
) -> dict[str, float | int | None]:
    """Paired per-task difference versus a baseline with bootstrap CI."""

    baseline_by_key = {
        f"{item.get('task_id')}::{item.get('trajectory_id')}": item
        for item in baseline_rows
        if item.get("available")
    }
    paired: list[float] = []
    for row in rows:
        if not row.get("available"):
            continue
        baseline = baseline_by_key.get(f"{row.get('task_id')}::{row.get('trajectory_id')}")
        if baseline is None:
            continue
        value = _optional_number(row.get(key), f"policy {key}")
        baseline_value = _optional_number(baseline.get(key), f"baseline {key}")
        if value is not None and baseline_value is not None:
            paired.append(value - baseline_value)
    if not paired:
        return {
            "n_paired": 0,
            "mean_difference": None,
            "ci95_low": None,
            "ci95_high": None,
        }
    rng = random.Random(seed)
    size = len(paired)
    means = sorted(mean(paired[rng.randrange(size)] for _ in range(size)) for _ in range(samples))
    low_index = max(0, math.floor(0.025 * (samples - 1)))
    high_index = min(samples - 1, math.ceil(0.975 * (samples - 1)))
    return {
        "n_paired": size,
        "mean_difference": mean(paired),
        "ci95_low": means[low_index],
        "ci95_high": means[high_index],
    }


def replay_policies(
    records: Any,
    policy_model: Mapping[str, Any] | None = None,
    *,
    bootstrap_seed: int | None = None,
    bootstrap_samples: int = 2000,
) -> dict[str, Any]:
    """Replay all required stopping baselines on saved trajectory JSON.

    Returned policies are ``fixed_1``, ``fixed_2``, ``fixed_3``, ``consensus``,
    ``task_only``, ``roundvalue``, the full-trajectory ``oracle``, and a
    ``oracle_one_step`` diagnostic.  If no frozen model is supplied, task-only
    and RoundValue explicitly continue to the final saved checkpoint; they do
    *not* fit on the records being evaluated.

    ``policy_model`` may contain ``lambda_cost``, ``mu_latency``, and nested
    ``roundvalue`` / ``task_only`` constant, mean, or linear descriptors.
    Everything returned is built from JSON-native values.

    ``bootstrap_seed`` enables paired per-task bootstrap confidence intervals
    for each policy's quality and total-token difference versus ``fixed_1``;
    without it the replay stays a pure deterministic comparison.
    """

    normalized = normalize_records(records)
    if policy_model is not None and not isinstance(policy_model, Mapping):
        raise TypeError("policy_model must be a JSON object or None")
    if bootstrap_seed is not None and (
        isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, int)
    ):
        raise ValueError("bootstrap_seed must be an integer or None")
    if (
        isinstance(bootstrap_samples, bool)
        or not isinstance(bootstrap_samples, int)
        or bootstrap_samples < 2
    ):
        raise ValueError("bootstrap_samples must be an integer of at least 2")
    model = dict(policy_model) if policy_model is not None else None
    lambda_cost = _model_weight(model, "lambda_cost")
    mu_latency = _model_weight(model, "mu_latency")
    policy_rows: dict[str, list[dict[str, Any]]] = {
        "fixed_1": [],
        "fixed_2": [],
        "fixed_3": [],
        "consensus": [],
        "task_only": [],
        "roundvalue": [],
        "oracle": [],
        "oracle_one_step": [],
    }
    all_labels: list[list[dict[str, Any]]] = []
    for position, record in enumerate(normalized):
        checkpoints = _ordered_checkpoints(record)
        if not checkpoints:
            raise ValueError(f"record {position} has no checkpoints")
        labels = _labels_for_record(record, lambda_cost, mu_latency)
        labels_by_round = _label_by_round(labels)
        checkpoint_rounds = {
            _positive_integer(checkpoint.get("round_index", checkpoint.get("round")))
            for checkpoint in checkpoints
        }
        if checkpoint_rounds != set(labels_by_round):
            raise ValueError(f"record {position} checkpoint and label rounds differ")
        all_labels.append(labels)
        for stop_round in (1, 2, 3):
            policy_rows[f"fixed_{stop_round}"].append(
                _replay_fixed(record, position, checkpoints, labels_by_round, stop_round)
            )
        policy_rows["consensus"].append(
            _replay_consensus(record, position, checkpoints, labels_by_round)
        )
        policy_rows["task_only"].append(
            _replay_predicted(
                record,
                position,
                checkpoints,
                labels_by_round,
                _policy_spec(model, "task_only"),
                "task_only",
                task_only=True,
            )
        )
        policy_rows["roundvalue"].append(
            _replay_predicted(
                record,
                position,
                checkpoints,
                labels_by_round,
                _policy_spec(model, "roundvalue"),
                "roundvalue",
                task_only=False,
            )
        )
        policy_rows["oracle"].append(_replay_oracle(record, position, checkpoints, labels_by_round))
        policy_rows["oracle_one_step"].append(
            _replay_one_step_oracle(record, position, checkpoints, labels_by_round)
        )

    regrets = _apply_oracle_regret(policy_rows)
    policy_metrics: dict[str, dict[str, Any]] = {}
    policies: dict[str, dict[str, Any]] = {}
    for name, rows in policy_rows.items():
        metrics = _metrics(rows)
        metrics.update(regrets[name])
        policy_metrics[name] = metrics
        policies[name] = {"task_results": rows, "metrics": metrics}
    pairwise = None
    if bootstrap_seed is not None:
        pairwise = {
            name: {
                "quality_difference": _pairwise_bootstrap(
                    rows,
                    policy_rows["fixed_1"],
                    "quality",
                    seed=bootstrap_seed,
                    samples=bootstrap_samples,
                ),
                "total_tokens_difference": _pairwise_bootstrap(
                    rows,
                    policy_rows["fixed_1"],
                    "total_tokens",
                    seed=bootstrap_seed,
                    samples=bootstrap_samples,
                ),
            }
            for name, rows in policy_rows.items()
        }
    return {
        "schema_version": POLICY_SCHEMA_VERSION,
        "policy_version": POLICY_VERSION,
        "n_records": len(normalized),
        "replay_mode": "offline_saved_trajectories_only",
        "label_parameters": {"lambda_cost": lambda_cost, "mu_latency": mu_latency},
        "model_status": {
            "provided": model is not None,
            "roundvalue_kind": str(_policy_spec(model, "roundvalue").get("kind"))
            if _policy_spec(model, "roundvalue") is not None
            else None,
            "task_only_kind": str(_policy_spec(model, "task_only").get("kind"))
            if _policy_spec(model, "task_only") is not None
            else None,
        },
        "transition_metrics": _transition_metrics(all_labels),
        "policy_metrics": policy_metrics,
        "policies": policies,
        "pairwise_vs_fixed_1": pairwise,
        "bootstrap": {"seed": bootstrap_seed, "samples": bootstrap_samples}
        if bootstrap_seed is not None
        else None,
    }


def _solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Solve a small dense linear system with deterministic pivoting."""

    size = len(vector)
    augmented = [list(matrix[row]) + [vector[row]] for row in range(size)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) <= _EPSILON:
            raise ValueError("singular linear policy system")
        if pivot != column:
            augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        pivot_value = augmented[column][column]
        augmented[column] = [value / pivot_value for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if abs(factor) <= _EPSILON:
                continue
            augmented[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(augmented[row], augmented[column], strict=False)
            ]
    return [augmented[row][-1] for row in range(size)]


def _training_rows(
    records: Any,
    policy_name: str,
    target: str,
    lambda_cost: float,
    mu_latency: float,
) -> list[tuple[dict[str, float], float]]:
    if policy_name not in {"roundvalue", "task_only"}:
        raise ValueError("policy_name must be roundvalue or task_only")
    valid_targets = {"G", "V", "delta_q", "finite_horizon_value", "one_step_value"}
    if target not in valid_targets:
        raise ValueError(f"target must be one of {sorted(valid_targets)}")
    result: list[tuple[dict[str, float], float]] = []
    for record_index, record in enumerate(normalize_records(records)):
        split = record.get("split", _trajectory(record).get("split"))
        if split != "train":
            raise ValueError(
                "policy fitting accepts frozen training records only; "
                f"records[{record_index}].split is {split!r}"
            )
        checkpoints = _ordered_checkpoints(record)
        labels = _labels_for_record(
            record,
            _nonnegative_number(lambda_cost, "lambda_cost"),
            _nonnegative_number(mu_latency, "mu_latency"),
        )
        label_map = _label_by_round(labels)
        checkpoint_map = {
            _positive_integer(item.get("round_index", item.get("round"))): item
            for item in checkpoints
        }
        for index, checkpoint in enumerate(checkpoints[:-1]):
            round_index = _positive_integer(checkpoint.get("round_index", checkpoint.get("round")))
            label = label_map[round_index]
            value = label.get(target)
            if value is None:
                continue
            previous = checkpoints[index - 1] if index else None
            features = build_policy_features(
                record,
                checkpoint_map[round_index],
                label,
                previous,
                task_only=policy_name == "task_only",
            )
            result.append((features, _required_number(value, f"label target {target}")))
    if not result:
        raise ValueError("no nonterminal labelled rows available for policy fitting")
    return result


def fit_linear_policy(
    records: Any,
    *,
    policy_name: str = "roundvalue",
    target: str = "G",
    lambda_cost: float = 0.0,
    mu_latency: float = 0.0,
    threshold: float = 0.0,
    ridge: float = 1e-6,
) -> dict[str, Any]:
    """Fit a tiny standardized ridge linear policy using only the stdlib.

    Call this on the frozen training split only.  Pass the returned descriptor
    to :func:`replay_policies` for validation or test replay.
    """

    ridge_value = _nonnegative_number(ridge, "ridge")
    lambda_value = _nonnegative_number(lambda_cost, "lambda_cost")
    latency_value = _nonnegative_number(mu_latency, "mu_latency")
    rows = _training_rows(records, policy_name, target, lambda_value, latency_value)
    names = ROUNDVALUE_FEATURES if policy_name == "roundvalue" else TASK_ONLY_FEATURES
    means = [sum(features[name] for features, _ in rows) / len(rows) for name in names]
    scales: list[float] = []
    for feature_index, name in enumerate(names):
        variance = sum((features[name] - means[feature_index]) ** 2 for features, _ in rows) / len(
            rows
        )
        scale = math.sqrt(variance)
        scales.append(scale if scale > _EPSILON else 1.0)
    # First column is the unregularized intercept; all remaining columns are
    # standardized feature values and receive ridge regularization.
    dimensions = len(names) + 1
    normal = [[0.0 for _ in range(dimensions)] for _ in range(dimensions)]
    response = [0.0 for _ in range(dimensions)]
    for features, target_value in rows:
        values = [1.0] + [
            (features[name] - means[index]) / scales[index] for index, name in enumerate(names)
        ]
        for left in range(dimensions):
            response[left] += values[left] * target_value
            for right in range(dimensions):
                normal[left][right] += values[left] * values[right]
    for index in range(1, dimensions):
        normal[index][index] += ridge_value
    coefficients = _solve_linear_system(normal, response)
    return {
        "schema_version": POLICY_SCHEMA_VERSION,
        "policy_version": POLICY_VERSION,
        "kind": "linear",
        "policy_name": policy_name,
        "target": target,
        "feature_names": list(names),
        "weights": [float(value) for value in coefficients[1:]],
        "intercept": float(coefficients[0]),
        "means": [float(value) for value in means],
        "scales": [float(value) for value in scales],
        "threshold": _required_number(threshold, "threshold"),
        "ridge": ridge_value,
        "lambda_cost": lambda_value,
        "mu_latency": latency_value,
        "training_rows": len(rows),
    }


def fit_policy_models(
    records: Any,
    *,
    lambda_cost: float = 0.0,
    mu_latency: float = 0.0,
    target: str = "G",
    ridge: float = 1e-6,
) -> dict[str, Any]:
    """Fit the paired RoundValue and task-only linear descriptors in one JSON object."""

    materialized = normalize_records(records)
    lambda_value = _nonnegative_number(lambda_cost, "lambda_cost")
    latency_value = _nonnegative_number(mu_latency, "mu_latency")
    return {
        "schema_version": POLICY_SCHEMA_VERSION,
        "policy_version": POLICY_VERSION,
        "label_parameters": {
            "lambda_cost": lambda_value,
            "mu_latency": latency_value,
        },
        "roundvalue": fit_linear_policy(
            materialized,
            policy_name="roundvalue",
            target=target,
            lambda_cost=lambda_value,
            mu_latency=latency_value,
            ridge=ridge,
        ),
        "task_only": fit_linear_policy(
            materialized,
            policy_name="task_only",
            target=target,
            lambda_cost=lambda_value,
            mu_latency=latency_value,
            ridge=ridge,
        ),
    }


__all__ = [
    "POLICY_SCHEMA_VERSION",
    "POLICY_VERSION",
    "ROUNDVALUE_FEATURES",
    "TASK_ONLY_FEATURES",
    "build_policy_features",
    "consensus_signal",
    "fit_linear_policy",
    "fit_policy_models",
    "normalize_records",
    "predict_policy_value",
    "replay_policies",
]
