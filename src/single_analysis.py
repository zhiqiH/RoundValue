"""Offline analysis for the automatic Single-Agent baseline observation.

Single-Agent is a baseline condition, not a topology.  Every normal Debate run
collects one independent ``single_agent`` observation per task as a sibling of
the Debate trajectory, and this module derives its aggregate metrics plus the
paired Single-vs-Debate outcome counts.

It deliberately builds none of the multi-round Debate concepts: no Delta-Q, V,
G, RoundValue fit, stopping threshold, continuation decision,
Repair/Harm/Recovery transition, or trajectory Oracle.  Only the saved
``answer`` is scored and ``reasoning_summary`` can never rescue a wrong
option.  All functions read saved JSON only and never call a model provider.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from report import cluster_bootstrap_mean
from scorer import score_single_observation


SINGLE_BOOTSTRAP_SEED = 20260813
SINGLE_BOOTSTRAP_SAMPLES = 2000
SINGLE_BASELINE_ID = "single_agent"
PAIRED_TARGETS = ("fixed_1", "fixed_5", "roundvalue", "oracle")


def _mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return (
        result
        if result == result and result not in (float("inf"), float("-inf"))
        else None
    )


def _observation(record: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(record.get("single_agent")) or {}


def _prediction(record: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(_observation(record).get("prediction")) or {}


def _single_score(record: Mapping[str, Any]) -> Mapping[str, Any]:
    scores = record.get("single_agent_scores")
    if isinstance(scores, list) and scores:
        first = _mapping(scores[0])
        if first is not None:
            return first
    return {}


def _single_quality(record: Mapping[str, Any]) -> float | None:
    score = _single_score(record)
    if score:
        return _number(score.get("quality"))
    try:
        derived = score_single_observation(record)
    except (TypeError, ValueError):
        return None
    return _number(derived[0].get("quality")) if derived else None


def single_observation_rows(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Emit one aggregate row per task for its Single-Agent observation."""

    rows: list[dict[str, Any]] = []
    for record in records:
        task = _mapping(record.get("task")) or {}
        observation = _observation(record)
        prediction = _prediction(record)
        score = _single_score(record)
        solver = _mapping(observation.get("solver")) or {}
        cumulative = _mapping(prediction.get("cumulative")) or {}
        input_tokens = _number(cumulative.get("input_tokens"))
        output_tokens = _number(cumulative.get("output_tokens"))
        quality = _number(score.get("quality"))
        attempts = (
            solver.get("attempts")
            if isinstance(solver.get("attempts"), list)
            else []
        )
        fallback = solver.get("fallback", prediction.get("fallback"))
        rows.append(
            {
                "task_id": task.get("task_id"),
                "split": record.get("split"),
                "domain": task.get("domain"),
                "observation_status": observation.get("status")
                if observation
                else "missing",
                "predicted_answer": prediction.get("answer"),
                "canonical_answer": score.get("predicted_answer"),
                "quality": quality,
                "is_correct": bool(quality == 1.0) if quality is not None else None,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens
                if input_tokens is not None and output_tokens is not None
                else None,
                "wall_clock_ms": _number(
                    cumulative.get("wall_clock_ms", observation.get("wall_clock_ms"))
                ),
                "api_latency_ms": _number(cumulative.get("api_latency_ms")),
                "cost_usd": _number(cumulative.get("cost_usd")),
                "logical_calls": _number(cumulative.get("logical_calls")),
                "api_attempts": len(attempts),
                "transport_retries": sum(
                    1
                    for attempt in attempts
                    if _mapping(attempt) is not None
                    and attempt.get("status") == "failed"
                ),
                "format_repairs": solver.get(
                    "format_repairs", prediction.get("format_repairs", 0)
                ),
                "fallback": bool(fallback is not None),
                "fallback_type": fallback.get("type")
                if _mapping(fallback) is not None
                else None,
                "finish_reason": solver.get(
                    "finish_reason", prediction.get("finish_reason")
                ),
                "truncated": bool(
                    solver.get(
                        "truncation_encountered", prediction.get("truncated", False)
                    )
                ),
                "truncated_attempts": solver.get(
                    "truncated_attempts", prediction.get("truncated_attempts", 0)
                ),
                "failure_reason": observation.get("failure_reason"),
                "error": solver.get("error") or observation.get("error"),
                "scoring_error": record.get("single_agent_scoring_error"),
            }
        )
    return rows


def _resource_aggregate(
    rows: Sequence[dict[str, Any]], key: str
) -> dict[str, Any]:
    values = [_number(row.get(key)) for row in rows]
    observed = [value for value in values if value is not None]
    complete = len(observed) == len(rows) and bool(rows)
    return {
        "mean": sum(observed) / len(observed) if complete else None,
        "total": sum(observed) if complete else None,
        "n_observed": len(observed),
        "n_total": len(rows),
        "complete_observation": complete,
    }


def _quality_aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    values = [_number(row.get("quality")) for row in rows]
    observed = [value for value in values if value is not None]
    if not rows:
        return {
            "accuracy": None,
            "n_observed": 0,
            "n_total": 0,
            "complete_observation": False,
            "bootstrap": None,
        }
    complete = len(observed) == len(rows)
    accuracy = sum(observed) / len(observed) if complete else None
    bootstrap = None
    if complete and observed:
        bootstrap = cluster_bootstrap_mean(
            observed,
            seed=SINGLE_BOOTSTRAP_SEED,
            samples=SINGLE_BOOTSTRAP_SAMPLES,
        )
    return {
        "accuracy": accuracy,
        "n_observed": len(observed),
        "n_total": len(rows),
        "complete_observation": complete,
        "bootstrap": bootstrap,
    }


def summarize_single_baseline(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate the Single-Agent baseline without zero-filling unknowns."""

    rows = single_observation_rows(records)
    defined = any(_observation(record) for record in records)
    complete = [
        row for row in rows if row["observation_status"] == "complete"
    ]
    by_split: dict[str, list[dict[str, Any]]] = {}
    for row in complete:
        split = str(row.get("split") or "unknown")
        by_split.setdefault(split, []).append(row)
    return {
        "schema_version": "1.0",
        "baseline": SINGLE_BASELINE_ID,
        "defined": bool(defined),
        "tasks_total": len(rows),
        "tasks_complete": len(complete),
        "tasks_failed": len(rows) - len(complete),
        "accuracy": _quality_aggregate(complete),
        "accuracy_by_split": {
            split: _quality_aggregate(split_rows)
            for split, split_rows in sorted(by_split.items())
        },
        "resources": {
            key: _resource_aggregate(complete, key)
            for key in (
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "wall_clock_ms",
                "api_latency_ms",
                "cost_usd",
                "logical_calls",
            )
        },
        "api_attempts": sum(
            int(row.get("api_attempts") or 0) for row in complete
        ),
        "transport_retries": sum(
            int(row.get("transport_retries") or 0) for row in complete
        ),
        "format_repairs": sum(
            int(row.get("format_repairs") or 0) for row in complete
        ),
        "fallback_count": sum(1 for row in complete if row.get("fallback")),
        "finish_reason_distribution": {
            str(reason): count
            for reason, count in sorted(
                Counter(
                    str(row.get("finish_reason"))
                    for row in complete
                    if row.get("finish_reason") is not None
                ).items()
            )
        },
        "truncated": {
            "count": sum(1 for row in complete if row.get("truncated")),
            "task_ids": [
                str(row.get("task_id")) for row in complete if row.get("truncated")
            ],
        },
        "task_rows": rows,
    }


def build_single_baseline(
    manifest: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Derive the complete JSON-safe Single-Agent section for one run."""

    rows = single_observation_rows(records)
    summary = summarize_single_baseline(records)
    split_counts: Counter[str] = Counter(
        str(record.get("split")) for record in records
    )
    error_counts: Counter[str] = Counter()
    for row in rows:
        if row["observation_status"] not in (None, "complete", "missing"):
            error = _mapping(row.get("error")) or {}
            error_counts[str(error.get("type") or "unknown")] += 1
    model_selection = _mapping(manifest.get("model_selection")) or {}
    return {
        "schema_version": "1.0",
        "baseline": SINGLE_BASELINE_ID,
        "run_id": manifest["run_id"],
        "generated_at": _now_iso(),
        "model": {
            "model_id": manifest.get("selected_model_id"),
            "provider": model_selection.get("provider"),
            "requested_model": model_selection.get("requested_model"),
            "temperature": model_selection.get("temperature"),
            "max_output_tokens": model_selection.get("max_output_tokens"),
            "reasoning": model_selection.get("reasoning"),
        },
        "defined": summary["defined"],
        "tasks_total": len(rows),
        "tasks_complete": summary["tasks_complete"],
        "tasks_failed": summary["tasks_failed"],
        "split_counts": dict(split_counts),
        "accuracy": summary["accuracy"],
        "accuracy_by_split": summary["accuracy_by_split"],
        "resources": summary["resources"],
        "calls": {
            "logical_calls": summary["resources"]["logical_calls"]["total"],
            "api_attempts": summary["api_attempts"],
            "transport_retries": summary["transport_retries"],
            "format_repairs": summary["format_repairs"],
            "fallback_count": summary["fallback_count"],
        },
        "finish_reason_distribution": summary["finish_reason_distribution"],
        "truncated": summary["truncated"],
        "errors": dict(error_counts),
        "task_rows": rows,
    }


def _debate_round_qualities(
    records: Sequence[Mapping[str, Any]],
) -> dict[int, dict[str, float]]:
    """Map each saved Debate round to task-level checkpoint quality."""

    result: dict[int, dict[str, float]] = {}
    for record in records:
        task = _mapping(record.get("task")) or {}
        task_id = str(task.get("task_id"))
        scores = record.get("scores")
        if not isinstance(scores, list):
            continue
        for item in scores:
            score = _mapping(item)
            if score is None:
                continue
            round_index = score.get("round_index")
            quality = _number(score.get("quality"))
            if (
                isinstance(round_index, int)
                and not isinstance(round_index, bool)
                and quality is not None
            ):
                result.setdefault(int(round_index), {})[task_id] = quality
    return result


def _policy_task_qualities(
    replay: Mapping[str, Any], name: str
) -> dict[str, float]:
    """Read task-level qualities from one saved policy replay condition."""

    policies = replay.get("policies")
    if not isinstance(policies, Mapping):
        return {}
    item = _mapping(policies.get(name))
    if item is None:
        return {}
    rows = item.get("task_results")
    if not isinstance(rows, list):
        return {}
    result: dict[str, float] = {}
    for row in rows:
        entry = _mapping(row)
        if entry is None or not entry.get("available"):
            continue
        task_id = entry.get("task_id")
        quality = _number(entry.get("quality"))
        if task_id is not None and quality is not None:
            result[str(task_id)] = quality
    return result


def _paired_counts(
    single: Mapping[str, float], debate: Mapping[str, float]
) -> dict[str, Any]:
    counts = Counter()
    for task_id, single_quality in single.items():
        debate_quality = debate.get(task_id)
        if debate_quality is None:
            continue
        single_correct = single_quality == 1.0
        debate_correct = debate_quality == 1.0
        if single_correct and debate_correct:
            counts["both_correct"] += 1
        elif single_correct and not debate_correct:
            counts["single_correct_debate_wrong"] += 1
        elif not single_correct and debate_correct:
            counts["single_wrong_debate_correct"] += 1
        else:
            counts["both_wrong"] += 1
    fields = {
        name: int(counts[name])
        for name in (
            "both_correct",
            "single_correct_debate_wrong",
            "single_wrong_debate_correct",
            "both_wrong",
        )
    }
    fields["n_paired"] = sum(fields.values())
    return fields


def paired_single_vs_debate(
    records: Sequence[Mapping[str, Any]],
    replay: Mapping[str, Any],
) -> dict[str, Any]:
    """Compute paired Single-Agent outcome counts over the same frozen tasks.

    The four required comparisons are Single-Agent vs Fixed-1, Fixed-5,
    RoundValue, and Oracle when each quantity is defined.  The field names are
    deliberately distinct from the within-debate Repair/Harm vocabulary.
    """

    single: dict[str, float] = {}
    for record in records:
        task = _mapping(record.get("task")) or {}
        task_id = str(task.get("task_id"))
        quality = _single_quality(record)
        if quality is not None:
            single[task_id] = quality
    rounds = _debate_round_qualities(records)
    targets: dict[str, dict[str, float]] = {}
    for name in ("fixed_1", "fixed_5"):
        round_index = int(name.split("_")[1])
        targets[name] = rounds.get(round_index, {})
    for name in ("roundvalue", "oracle"):
        targets[name] = _policy_task_qualities(replay, name)
    paired: dict[str, Any] = {}
    for name in PAIRED_TARGETS:
        debate = targets.get(name)
        if debate is None or not debate or not single:
            paired[name] = {"defined": False}
            continue
        counts = _paired_counts(single, debate)
        counts["defined"] = bool(debate)
        single_observed = [value for value in single.values()]
        debate_observed = [debate[task_id] for task_id in single if task_id in debate]
        counts["single_accuracy"] = (
            sum(single_observed) / len(single_observed)
            if single_observed
            else None
        )
        counts["debate_accuracy"] = (
            sum(debate_observed) / len(debate_observed)
            if debate_observed
            else None
        )
        paired[name] = counts
    return paired


def baseline_table(
    single_summary: Mapping[str, Any],
    replay: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Merge Single-Agent with the Debate baselines into one ordered table."""

    rows: list[dict[str, Any]] = []
    resources = _mapping(single_summary.get("resources")) or {}
    rows.append(
        {
            "condition": SINGLE_BASELINE_ID,
            "display_name": "Single-Agent",
            "defined": bool(single_summary.get("defined")),
            "accuracy": _number(
                (_mapping(single_summary.get("accuracy")) or {}).get("accuracy")
            ),
            "n_tasks": single_summary.get("tasks_complete"),
            "mean_input_tokens": _number(
                (_mapping(resources.get("input_tokens")) or {}).get("mean")
            ),
            "mean_output_tokens": _number(
                (_mapping(resources.get("output_tokens")) or {}).get("mean")
            ),
            "mean_total_tokens": _number(
                (_mapping(resources.get("total_tokens")) or {}).get("mean")
            ),
            "mean_wall_clock_ms": _number(
                (_mapping(resources.get("wall_clock_ms")) or {}).get("mean")
            ),
            "mean_api_latency_ms": _number(
                (_mapping(resources.get("api_latency_ms")) or {}).get("mean")
            ),
            "mean_cost_usd": _number(
                (_mapping(resources.get("cost_usd")) or {}).get("mean")
            ),
            "mean_logical_calls": _number(
                (_mapping(resources.get("logical_calls")) or {}).get("mean")
            ),
        }
    )
    policy_metrics = replay.get("policy_metrics")
    if not isinstance(policy_metrics, Mapping):
        policy_metrics = {}
    display_names = {
        "fixed_1": "Fixed-1",
        "fixed_2": "Fixed-2",
        "fixed_3": "Fixed-3",
        "fixed_4": "Fixed-4",
        "fixed_5": "Fixed-5",
        "roundvalue": "RoundValue",
        "oracle": "Oracle",
    }
    for name in (
        "fixed_1",
        "fixed_2",
        "fixed_3",
        "fixed_4",
        "fixed_5",
        "roundvalue",
        "oracle",
    ):
        metrics = _mapping(policy_metrics.get(name))
        rows.append(
            {
                "condition": name,
                "display_name": display_names[name],
                "defined": metrics is not None,
                "accuracy": _number(metrics.get("accuracy"))
                if metrics is not None
                else None,
                "n_tasks": metrics.get("n_records")
                if metrics is not None
                else None,
                "mean_input_tokens": _number(metrics.get("mean_input_tokens"))
                if metrics is not None
                else None,
                "mean_output_tokens": _number(metrics.get("mean_output_tokens"))
                if metrics is not None
                else None,
                "mean_total_tokens": _number(metrics.get("mean_total_tokens"))
                if metrics is not None
                else None,
                "mean_wall_clock_ms": _number(metrics.get("mean_wall_clock_ms"))
                if metrics is not None
                else None,
                "mean_api_latency_ms": _number(
                    metrics.get("mean_api_latency_ms")
                )
                if metrics is not None
                else None,
                "mean_cost_usd": _number(metrics.get("mean_cost_usd"))
                if metrics is not None
                else None,
                "mean_logical_calls": _number(metrics.get("mean_logical_calls"))
                if metrics is not None
                else None,
            }
        )
    return rows


def _now_iso() -> str:
    from contracts import utc_now

    return utc_now()


__all__ = [
    "PAIRED_TARGETS",
    "SINGLE_BASELINE_ID",
    "baseline_table",
    "build_single_baseline",
    "paired_single_vs_debate",
    "single_observation_rows",
    "summarize_single_baseline",
]
