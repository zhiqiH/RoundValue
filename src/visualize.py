"""Offline report builder for saved RoundValue runs.

This module reads task records, scores, labels, and (when available) the
frozen policy replay and renders the readable artifacts:

* ``report_data.json`` -- machine-readable tables used by the HTML below,
* ``task_level_results.csv`` -- one row per collected task,
* ``report.html`` -- a self-contained page with the requested summary tables
  and quality-vs-token / quality-vs-latency charts (inline SVG, no CDN),

plus five standalone policy-level PNG charts in ``charts/``:
policy quality-vs-tokens, policy quality-vs-latency, RoundValue-vs-baselines,
adaptive stop-round distribution, and oracle quality regret.  Matplotlib is
imported lazily inside the chart renderer so the rest of the module stays
lightweight.

It never contacts a provider and never rescore checkpoints.
"""

from __future__ import annotations

import csv
import html
import json
import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from labels import build_labels
from storage import read_json, write_json


POLICY_ORDER = (
    "fixed_1",
    "fixed_2",
    "fixed_3",
    "task_only",
    "roundvalue",
    "oracle",
)

POLICY_DISPLAY_NAMES: dict[str, str] = {
    "fixed_1": "Fixed-1",
    "fixed_2": "Fixed-2",
    "fixed_3": "Fixed-3",
    "task_only": "Task-only",
    "roundvalue": "RoundValue",
    "oracle": "Oracle",
}

PAIRED_BASELINE_ORDER = ("fixed_1", "fixed_2", "fixed_3", "task_only")
ADAPTIVE_STOP_ORDER = ("task_only", "roundvalue", "oracle")
ORACLE_REGRET_ORDER = (
    "fixed_1",
    "fixed_2",
    "fixed_3",
    "task_only",
    "roundvalue",
)


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _fmt(value: Any, digits: int = 3) -> str:
    number = _number(value)
    if number is None:
        return "unknown"
    return f"{number:.{digits}f}"


def _get(value: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in value and value[key] is not None:
            return value[key]
    return default


def _scores_by_round(record: Mapping[str, Any]) -> dict[int, float]:
    scores: dict[int, float] = {}
    raw = record.get("scores")
    if not isinstance(raw, list):
        return scores
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        round_index = item.get("round_index")
        quality = _number(item.get("quality"))
        if isinstance(round_index, int) and not isinstance(round_index, bool) and quality is not None:
            scores[int(round_index)] = quality
    return scores


def _checkpoints(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = record.get("trajectory", {}).get("checkpoints", [])
    if not isinstance(raw, list):
        return []
    ordered = [
        item
        for item in raw
        if isinstance(item, Mapping)
        and isinstance(item.get("round_index"), int)
        and not isinstance(item.get("round_index"), bool)
    ]
    return sorted(ordered, key=lambda item: int(item["round_index"]))


def _wall_clock(cumulative: Mapping[str, Any]) -> float | None:
    # Only a measured elapsed time is reported as wall clock.  Older records
    # whose ``latency_ms`` was the summed API time stay "unknown" here rather
    # than being relabelled.
    return _number(_get(cumulative, "wall_clock_ms", "cumulative_wall_clock_ms"))


def _api_latency(cumulative: Mapping[str, Any]) -> float | None:
    value = _number(_get(cumulative, "api_latency_ms", "cumulative_api_latency_ms"))
    if value is None and _wall_clock(cumulative) is None:
        value = _number(_get(cumulative, "latency_ms", "cumulative_latency_ms"))
    return value


def _task_row(
    record: Mapping[str, Any],
    scores: Mapping[int, float],
    checkpoints: Sequence[Mapping[str, Any]],
    transitions: Sequence[str],
    max_rounds: int,
) -> dict[str, Any]:
    task = record.get("task", {})
    trajectory = record.get("trajectory", {})
    last = checkpoints[-1] if checkpoints else {}
    cumulative = last.get("cumulative", {}) if isinstance(last.get("cumulative"), Mapping) else {}
    input_tokens = _number(_get(cumulative, "input_tokens", "cumulative_input_tokens"))
    output_tokens = _number(_get(cumulative, "output_tokens", "cumulative_output_tokens"))
    row: dict[str, Any] = {
        "task_id": task.get("task_id"),
        "split": record.get("split"),
        "domain": task.get("domain"),
        "trajectory_status": trajectory.get("status"),
        "completed_rounds": len(checkpoints),
        "final_quality": scores[max(scores)] if scores else None,
        "cumulative_input_tokens": input_tokens,
        "cumulative_output_tokens": output_tokens,
        "cumulative_total_tokens": input_tokens + output_tokens
        if input_tokens is not None and output_tokens is not None
        else None,
        "cumulative_wall_clock_ms": _wall_clock(cumulative),
        "cumulative_api_latency_ms": _api_latency(cumulative),
        "cumulative_cost_usd": _number(_get(cumulative, "cost_usd", "cumulative_cost_usd")),
        "cumulative_logical_calls": _number(
            _get(cumulative, "logical_calls", "cumulative_logical_calls")
        ),
        "transitions": ">".join(transitions) if transitions else None,
    }
    for round_index in range(1, max_rounds + 1):
        row[f"quality_round_{round_index}"] = scores.get(round_index)
    return row


def _mean_over(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    values = [_number(row.get(key)) for row in rows]
    observed = [value for value in values if value is not None]
    return {
        "mean": sum(observed) / len(observed) if observed else None,
        "n_observed": len(observed),
        "n_total": len(rows),
    }


def _per_round_tables(
    records: Sequence[Mapping[str, Any]], max_rounds: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_round: list[dict[str, Any]] = []
    cumulative: list[dict[str, Any]] = []
    for round_index in range(1, max_rounds + 1):
        with_round: list[dict[str, Any]] = []
        for record in records:
            scores = _scores_by_round(record)
            checkpoint = next(
                (
                    item
                    for item in _checkpoints(record)
                    if int(item["round_index"]) == round_index
                ),
                None,
            )
            if checkpoint is None or round_index not in scores:
                continue
            round_cost = checkpoint.get("round_cost", {})
            if not isinstance(round_cost, Mapping):
                round_cost = {}
            cumulative_counters = checkpoint.get("cumulative", {})
            if not isinstance(cumulative_counters, Mapping):
                cumulative_counters = {}
            input_tokens = _number(_get(round_cost, "input_tokens"))
            output_tokens = _number(_get(round_cost, "output_tokens"))
            row = {
                "round_index": round_index,
                "n_tasks": 1,
                "quality": scores[round_index],
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens
                if input_tokens is not None and output_tokens is not None
                else None,
                "wall_clock_ms": _wall_clock(round_cost) or _wall_clock(cumulative_counters),
                "api_latency_ms": _api_latency(round_cost) or _api_latency(cumulative_counters),
                "cost_usd": _number(_get(round_cost, "cost_usd")),
                "logical_calls": _number(_get(round_cost, "logical_calls")),
                "cumulative_input_tokens": _number(_get(cumulative_counters, "input_tokens")),
                "cumulative_output_tokens": _number(_get(cumulative_counters, "output_tokens")),
                "cumulative_total_tokens": _number(_get(cumulative_counters, "input_tokens"))
                + _number(_get(cumulative_counters, "output_tokens"))
                if _number(_get(cumulative_counters, "input_tokens")) is not None
                and _number(_get(cumulative_counters, "output_tokens")) is not None
                else None,
                "cumulative_wall_clock_ms": _wall_clock(cumulative_counters),
                "cumulative_api_latency_ms": _api_latency(cumulative_counters),
                "cumulative_cost_usd": _number(_get(cumulative_counters, "cost_usd")),
                "cumulative_logical_calls": _number(_get(cumulative_counters, "logical_calls")),
            }
            with_round.append(row)
        if not with_round:
            continue
        per_round.append(
            {
                "round_index": round_index,
                "n_tasks": len(with_round),
                "accuracy": _mean_over(with_round, "quality")["mean"],
                "mean_input_tokens": _mean_over(with_round, "input_tokens")["mean"],
                "mean_output_tokens": _mean_over(with_round, "output_tokens")["mean"],
                "mean_total_tokens": _mean_over(with_round, "total_tokens")["mean"],
                "mean_wall_clock_ms": _mean_over(with_round, "wall_clock_ms")["mean"],
                "mean_api_latency_ms": _mean_over(with_round, "api_latency_ms")["mean"],
                "mean_cost_usd": _mean_over(with_round, "cost_usd")["mean"],
                "mean_logical_calls": _mean_over(with_round, "logical_calls")["mean"],
            }
        )
        cumulative.append(
            {
                "round_index": round_index,
                "n_tasks": len(with_round),
                "accuracy": _mean_over(with_round, "quality")["mean"],
                "mean_input_tokens": _mean_over(with_round, "cumulative_input_tokens")["mean"],
                "mean_output_tokens": _mean_over(with_round, "cumulative_output_tokens")["mean"],
                "mean_total_tokens": _mean_over(with_round, "cumulative_total_tokens")["mean"],
                "mean_wall_clock_ms": _mean_over(with_round, "cumulative_wall_clock_ms")["mean"],
                "mean_api_latency_ms": _mean_over(with_round, "cumulative_api_latency_ms")[
                    "mean"
                ],
                "mean_cost_usd": _mean_over(with_round, "cumulative_cost_usd")["mean"],
                "mean_logical_calls": _mean_over(with_round, "cumulative_logical_calls")["mean"],
            }
        )
    return per_round, cumulative


def _policy_table(replay: dict[str, Any]) -> list[dict[str, Any]]:
    metrics = replay.get("policy_metrics", {})
    rows: list[dict[str, Any]] = []
    for name in POLICY_ORDER:
        if name not in metrics:
            continue
        item = dict(metrics[name])
        rows.append(
            {
                "policy": name,
                "n_available": item.get("n_available"),
                "coverage": item.get("coverage"),
                "accuracy": item.get("accuracy"),
                "mean_utility": item.get("mean_utility"),
                "mean_total_tokens": item.get("mean_total_tokens"),
                "mean_wall_clock_ms": item.get("mean_wall_clock_ms"),
                "mean_api_latency_ms": item.get("mean_api_latency_ms"),
                "mean_cost_usd": item.get("mean_cost_usd"),
                "mean_stop_round": item.get("mean_stop_round"),
                "mean_oracle_quality_regret": item.get("mean_oracle_quality_regret"),
                "stop_round_counts": item.get("stop_round_counts", {}),
            }
        )
    return rows


def _policy_task_rows(replay: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Reduce replay task results to the fields the policy charts need."""

    policies = replay.get("policies", {})
    if not isinstance(policies, Mapping):
        return {}
    rows_by_policy: dict[str, list[dict[str, Any]]] = {}
    for name, item in policies.items():
        if not isinstance(item, Mapping):
            continue
        raw_rows = item.get("task_results")
        if not isinstance(raw_rows, list):
            continue
        rows: list[dict[str, Any]] = []
        for raw in raw_rows:
            if not isinstance(raw, Mapping):
                continue
            rows.append(
                {
                    "task_id": raw.get("task_id"),
                    "trajectory_id": raw.get("trajectory_id"),
                    "quality": _number(raw.get("quality")),
                    "total_tokens": _number(raw.get("total_tokens")),
                    "wall_clock_ms": _number(raw.get("wall_clock_ms")),
                    "stop_round": raw.get("stop_round"),
                    "available": bool(raw.get("available", False)),
                }
            )
        rows_by_policy[str(name)] = rows
    return rows_by_policy


def _paired_differences(
    rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
    key: str,
) -> list[float]:
    """Task-level paired ``row[key] - baseline[key]`` values."""

    baseline_by_key = {
        (row.get("task_id"), row.get("trajectory_id")): row
        for row in baseline_rows
        if row.get("available")
    }
    paired: list[float] = []
    for row in rows:
        if not row.get("available"):
            continue
        baseline = baseline_by_key.get((row.get("task_id"), row.get("trajectory_id")))
        if baseline is None:
            continue
        value = _number(row.get(key))
        baseline_value = _number(baseline.get(key))
        if value is not None and baseline_value is not None:
            paired.append(value - baseline_value)
    return paired


def _bootstrap_summary(
    values: Sequence[float], *, seed: int, samples: int
) -> dict[str, float | int | None]:
    """Mean plus percentile 95% CI for a paired difference sample."""

    if not values:
        return {"n_paired": 0, "mean": None, "ci95_low": None, "ci95_high": None}
    rng = random.Random(seed)
    size = len(values)
    means = sorted(
        sum(values[rng.randrange(size)] for _ in range(size)) / size
        for _ in range(samples)
    )
    low_index = max(0, math.floor(0.025 * (samples - 1)))
    high_index = min(samples - 1, math.ceil(0.975 * (samples - 1)))
    return {
        "n_paired": size,
        "mean": sum(values) / size,
        "ci95_low": means[low_index],
        "ci95_high": means[high_index],
    }


def _build_policy_chart_data(replay: Mapping[str, Any]) -> dict[str, Any]:
    """Derive the policy-level chart bundle persisted on analysis.json.

    Figures 1/2 come from aggregate policy metrics; Figures 3/5 add paired
    task-level bootstrap intervals, so the renderer never has to read the
    replay file or trajectories again.
    """

    bootstrap = replay.get("bootstrap")
    if not isinstance(bootstrap, Mapping):
        bootstrap = {}
    seed = int(_number(bootstrap.get("seed")) or 20260813)
    samples = int(_number(bootstrap.get("samples")) or 2000)
    task_rows = {
        name: rows
        for name, rows in _policy_task_rows(replay).items()
        if name in POLICY_ORDER
    }
    policy_metrics = replay.get("policy_metrics", {})
    if not isinstance(policy_metrics, Mapping):
        policy_metrics = {}

    policy_points: list[dict[str, Any]] = []
    for name in POLICY_ORDER:
        metrics = policy_metrics.get(name)
        if not isinstance(metrics, Mapping):
            continue
        policy_points.append(
            {
                "policy": name,
                "accuracy": _number(metrics.get("accuracy")),
                "mean_total_tokens": _number(metrics.get("mean_total_tokens")),
                "mean_wall_clock_ms": _number(metrics.get("mean_wall_clock_ms")),
                "mean_stop_round": _number(metrics.get("mean_stop_round")),
                "mean_oracle_quality_regret": _number(
                    metrics.get("mean_oracle_quality_regret")
                ),
                "n_available": metrics.get("n_available"),
                "stop_round_counts": metrics.get("stop_round_counts", {}),
            }
        )

    roundvalue_rows = task_rows.get("roundvalue", [])
    oracle_rows = task_rows.get("oracle", [])
    baselines: dict[str, dict[str, Any]] = {}
    for baseline in PAIRED_BASELINE_ORDER:
        baseline_rows = task_rows.get(baseline, [])
        baselines[baseline] = {
            "accuracy_difference": _bootstrap_summary(
                _paired_differences(roundvalue_rows, baseline_rows, "quality"),
                seed=seed,
                samples=samples,
            ),
            "total_tokens_difference": _bootstrap_summary(
                _paired_differences(roundvalue_rows, baseline_rows, "total_tokens"),
                seed=seed,
                samples=samples,
            ),
        }

    regrets: dict[str, dict[str, Any]] = {}
    for name in ORACLE_REGRET_ORDER:
        regrets[name] = _bootstrap_summary(
            _paired_differences(oracle_rows, task_rows.get(name, []), "quality"),
            seed=seed,
            samples=samples,
        )

    stop_distribution: dict[str, dict[str, float]] = {}
    for name in ADAPTIVE_STOP_ORDER:
        counts: Counter[str] = Counter()
        for row in task_rows.get(name, []):
            if row.get("available") and row.get("stop_round") is not None:
                counts[str(row.get("stop_round"))] += 1
        total = sum(counts.values())
        stop_distribution[name] = {
            str(round_index): counts.get(str(round_index), 0) / total * 100.0
            if total
            else 0.0
            for round_index in range(1, 4)
        }

    return {
        "policy_points": policy_points,
        "task_rows": task_rows,
        "roundvalue_vs_baselines": baselines,
        "oracle_regret": regrets,
        "stop_distribution": stop_distribution,
        "bootstrap": {"seed": seed, "samples": samples},
    }


def _scatter_points(records: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    token_points: list[dict[str, Any]] = []
    latency_points: list[dict[str, Any]] = []
    for record in records:
        scores = _scores_by_round(record)
        for checkpoint in _checkpoints(record):
            round_index = int(checkpoint["round_index"])
            quality = scores.get(round_index)
            if quality is None:
                continue
            cumulative = checkpoint.get("cumulative", {})
            if not isinstance(cumulative, Mapping):
                continue
            input_tokens = _number(_get(cumulative, "input_tokens"))
            output_tokens = _number(_get(cumulative, "output_tokens"))
            tokens = input_tokens + output_tokens if input_tokens is not None and output_tokens is not None else None
            latency = _wall_clock(cumulative)
            if tokens is not None:
                token_points.append(
                    {"x": tokens, "y": quality, "round_index": round_index}
                )
            if latency is not None:
                latency_points.append(
                    {"x": latency, "y": quality, "round_index": round_index}
                )
    return token_points, latency_points


def _svg_scatter(
    points: Sequence[Mapping[str, Any]],
    *,
    x_label: str,
    y_label: str,
    title: str,
) -> str:
    width, height = 720, 420
    margin = {"left": 70, "right": 30, "top": 48, "bottom": 64}
    if not points:
        return (
            f'<svg width="{width}" height="{height}" role="img" aria-label="{html.escape(title)}">'
            f'<text x="{width / 2}" y="{height / 2}" text-anchor="middle" '
            'font-size="14" fill="#666">no chart data</text></svg>'
        )
    xs = [float(point["x"]) for point in points]
    ys = [float(point["y"]) for point in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_max - x_min < 1e-9:
        x_min, x_max = x_min - 1, x_max + 1
    if y_max - y_min < 1e-9:
        y_min, y_max = y_min - 0.5, y_max + 0.5
    x_pad = (x_max - x_min) * 0.04
    y_pad = (y_max - y_min) * 0.12
    x_min, x_max = x_min - x_pad, x_max + x_pad
    y_min, y_max = y_min - y_pad, y_max + y_pad
    plot_width = width - margin["left"] - margin["right"]
    plot_height = height - margin["top"] - margin["bottom"]

    def px(value: float) -> float:
        return margin["left"] + (value - x_min) / (x_max - x_min) * plot_width

    def py(value: float) -> float:
        return margin["top"] + (y_max - value) / (y_max - y_min) * plot_height

    colors = {1: "#2563eb", 2: "#dc2626", 3: "#16a34a"}
    parts = [
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        'xmlns="http://www.w3.org/2000/svg" role="img">',
        f'<text x="{width / 2}" y="24" text-anchor="middle" font-size="16" '
        f'font-weight="600">{html.escape(title)}</text>',
    ]
    for tick in range(5):
        fraction = tick / 4
        x_value = x_min + fraction * (x_max - x_min)
        y_value = y_min + fraction * (y_max - y_min)
        x_pos = px(x_value)
        y_pos = py(y_value)
        parts.append(
            f'<line x1="{x_pos}" y1="{margin["top"]}" x2="{x_pos}" '
            f'y2="{margin["top"] + plot_height}" stroke="#e5e7eb"/>'
        )
        parts.append(
            f'<line x1="{margin["left"]}" y1="{y_pos}" x2="{margin["left"] + plot_width}" '
            f'y2="{y_pos}" stroke="#e5e7eb"/>'
        )
        parts.append(
            f'<text x="{x_pos}" y="{margin["top"] + plot_height + 18}" text-anchor="middle" '
            f'font-size="11" fill="#4b5563">{_fmt(x_value, 1)}</text>'
        )
        parts.append(
            f'<text x="{margin["left"] - 8}" y="{y_pos + 4}" text-anchor="end" '
            f'font-size="11" fill="#4b5563">{_fmt(y_value, 2)}</text>'
        )
    for point in points:
        round_index = int(point["round_index"])
        color = colors.get(round_index, "#6b7280")
        parts.append(
            f'<circle cx="{px(float(point["x"]))}" cy="{py(float(point["y"]))}" r="3.2" '
            f'fill="{color}" fill-opacity="0.55"/>'
        )
    parts.append(
        f'<text x="{margin["left"] + plot_width / 2}" y="{height - 14}" text-anchor="middle" '
        f'font-size="13" fill="#111827">{html.escape(x_label)}</text>'
    )
    parts.append(
        f'<text x="18" y="{margin["top"] + plot_height / 2}" text-anchor="middle" '
        f'font-size="13" fill="#111827" transform="rotate(-90 18 '
        f'{margin["top"] + plot_height / 2})">{html.escape(y_label)}</text>'
    )
    legend_x = margin["left"] + plot_width - 150
    for round_index, color in colors.items():
        parts.append(f'<circle cx="{legend_x}" cy="{margin["top"] + 14 + round_index * 18}" r="4" fill="{color}"/>')
        parts.append(
            f'<text x="{legend_x + 10}" y="{margin["top"] + 18 + round_index * 18}" '
            f'font-size="11" fill="#111827">round {round_index}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _table_html(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    head = "".join(f"<th>{html.escape(str(header))}</th>" for header in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return (
        f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead>'
        f"<tbody>{body}</tbody></table></div>"
    )


def _render_html(report: Mapping[str, Any]) -> str:
    run_id = str(report.get("run_id", ""))
    generated = str(report.get("generated_at", ""))
    per_round = report["per_round"]
    cumulative = report["cumulative"]
    transitions = report["transition_counts"]
    policies = report["policy_table"]
    stop_matrix = report["stop_round_matrix"]

    round_headers = [
        "Round",
        "Tasks",
        "Accuracy",
        "Tokens (round)",
        "Wall-clock (round, ms)",
        "API time (round, ms)",
        "Cost (round, USD)",
        "Logical calls (round)",
        "Tokens (cumulative)",
        "Wall-clock (cumulative, ms)",
        "Cost (cumulative, USD)",
        "Logical calls (cumulative)",
    ]
    round_rows: list[list[str]] = []
    for per, cum in zip(per_round, cumulative, strict=False):
        round_rows.append(
            [
                str(per["round_index"]),
                str(per["n_tasks"]),
                _fmt(per["accuracy"]),
                _fmt(per["mean_total_tokens"], 0),
                _fmt(per["mean_wall_clock_ms"], 0),
                _fmt(per["mean_api_latency_ms"], 0),
                _fmt(per["mean_cost_usd"], 6),
                _fmt(per["mean_logical_calls"], 0),
                _fmt(cum["mean_total_tokens"], 0),
                _fmt(cum["mean_wall_clock_ms"], 0),
                _fmt(cum["mean_cost_usd"], 6),
                _fmt(cum["mean_logical_calls"], 0),
            ]
        )

    transition_rows = [
        [
            str(transitions.get(name, 0))
            for name in ("repair", "neutral", "harm", "recovery", "terminal")
        ]
    ]

    policy_rows = []
    for policy in policies:
        policy_rows.append(
            [
                str(policy["policy"]),
                str(policy["n_available"]),
                _fmt(policy["accuracy"]),
                _fmt(policy["mean_total_tokens"], 0),
                _fmt(policy["mean_wall_clock_ms"], 0),
                _fmt(policy["mean_cost_usd"], 6),
                _fmt(policy["mean_stop_round"], 2),
                _fmt(policy["mean_oracle_quality_regret"]),
            ]
        )

    stop_policy_names = [str(policy["policy"]) for policy in policies]
    stop_rows = []
    for round_index in sorted(stop_matrix, key=lambda item: int(item)):
        counts = stop_matrix[round_index]
        stop_rows.append(
            [str(round_index)]
            + [str(counts.get(name, 0)) for name in stop_policy_names]
        )

    token_svg = _svg_scatter(
        report["quality_token_points"],
        x_label="cumulative tokens",
        y_label="quality",
        title=f"Quality vs tokens (run {run_id})",
    )
    latency_svg = _svg_scatter(
        report["quality_latency_points"],
        x_label="cumulative wall-clock (ms)",
        y_label="quality",
        title=f"Quality vs wall-clock latency (run {run_id})",
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RoundValue report {html.escape(run_id)}</title>
<style>
  body {{ font-family: -apple-system, "Segoe UI", Roboto, sans-serif; margin: 0;
         background: #f8fafc; color: #0f172a; }}
  main {{ max-width: 1080px; margin: 0 auto; padding: 24px 20px 60px; }}
  h1 {{ font-size: 24px; margin: 0 0 4px; }}
  h2 {{ font-size: 18px; margin: 32px 0 8px; }}
  .meta {{ color: #475569; font-size: 13px; margin-bottom: 16px; }}
  .table-wrap {{ overflow-x: auto; background: #fff; border: 1px solid #e2e8f0;
                 border-radius: 8px; margin: 12px 0; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
  th, td {{ text-align: right; padding: 7px 10px; border-bottom: 1px solid #e2e8f0;
            white-space: nowrap; }}
  th:first-child, td:first-child {{ text-align: left; }}
  thead th {{ background: #f1f5f9; color: #334155; }}
  .chart {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 8px;
            margin: 12px 0; padding: 8px; }}
  .chart svg {{ display: block; margin: 0 auto; max-width: 100%; height: auto; }}
</style>
</head>
<body>
<main>
  <h1>RoundValue report</h1>
  <div class="meta">run_id: {html.escape(run_id)} &middot; generated: {html.escape(generated)}</div>
  <h2>Overview</h2>
  {_table_html(
      ["Tasks total", "Complete", "Failed", "Split counts"],
      [[
          str(report["tasks_total"]),
          str(report["tasks_complete"]),
          str(report["tasks_failed"]),
          ", ".join(f"{key}={value}" for key, value in sorted(report["split_counts"].items())),
      ]],
  )}
  <h2>Per-round and cumulative resources</h2>
  {_table_html(round_headers, round_rows)}
  <h2>Repair / Neutral / Harm / Recovery</h2>
  {_table_html(
      ["Repair", "Neutral", "Harm", "Recovery", "Terminal"],
      transition_rows,
  )}
  <h2>Policy comparison</h2>
  {_table_html(
      [
          "Policy",
          "Available",
          "Accuracy",
          "Tokens",
          "Wall-clock (ms)",
          "Cost (USD)",
          "Stop round",
          "Oracle regret",
      ],
      policy_rows,
  )}
  <h2>Stop-round distribution</h2>
  {_table_html(["Round", *stop_policy_names], stop_rows)}
  <h2>Charts</h2>
  <div class="chart">{token_svg}</div>
  <div class="chart">{latency_svg}</div>
</main>
</body>
</html>
"""


def build_analysis(
    manifest: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    *,
    label_parameters: tuple[float, float],
    replay: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive the complete, JSON-safe analysis bundle from scored records.

    This is the shared aggregation used by step3_analyze.  It only reads its
    arguments; writing and rendering happen in separate functions so step4 can
    be a strict consumer of the saved ``analysis.json``.
    """

    lambda_cost, mu_latency = label_parameters
    usable_records = [record for record in records if _checkpoints(record)]
    max_rounds = max(
        (int(item["round_index"]) for record in usable_records for item in _checkpoints(record)),
        default=1,
    )
    scores_by_record = {id(record): _scores_by_round(record) for record in usable_records}
    transitions: Counter[str] = Counter()
    task_rows: list[dict[str, Any]] = []
    for record in usable_records:
        checkpoint_list = _checkpoints(record)
        transition_list: list[str] = []
        try:
            labels = build_labels(record, lambda_cost=lambda_cost, mu_latency=mu_latency)
            transition_list = [
                str(label.get("transition", "unknown"))
                for label in labels
                if label.get("transition") not in (None, "terminal")
            ]
            for label in labels:
                transitions[str(label.get("transition", "unknown"))] += 1
        except (TypeError, ValueError, KeyError):
            pass
        task_rows.append(
            _task_row(
                record,
                scores_by_record[id(record)],
                checkpoint_list,
                transition_list,
                max_rounds,
            )
        )

    failed_rows = [
        {
            "task_id": record.get("task", {}).get("task_id"),
            "split": record.get("split"),
            "domain": record.get("task", {}).get("domain"),
            "trajectory_status": record.get("trajectory", {}).get("status"),
            "completed_rounds": 0,
            "final_quality": None,
            "failure_reason": record.get("trajectory", {}).get("failure_reason"),
            "scoring_error": record.get("scoring_error"),
            "transitions": None,
        }
        for record in records
        if not _checkpoints(record)
    ]
    task_rows.extend(failed_rows)

    per_round, cumulative = _per_round_tables(usable_records, max_rounds)
    token_points, latency_points = _scatter_points(usable_records)
    policy_table = _policy_table(replay)
    stop_round_matrix: dict[str, dict[str, int]] = {}
    for policy in policy_table:
        for round_index, count in policy.get("stop_round_counts", {}).items():
            stop_round_matrix.setdefault(str(round_index), {})[
                str(policy["policy"])
            ] = int(count)

    split_counts: Counter[str] = Counter(str(record.get("split")) for record in records)
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "run_id": manifest["run_id"],
        "generated_at": _now_iso(),
        "max_rounds": max_rounds,
        "tasks_total": len(records),
        "tasks_complete": sum(
            record.get("trajectory", {}).get("status") == "complete" for record in records
        ),
        "tasks_failed": sum(
            record.get("trajectory", {}).get("status") != "complete" for record in records
        ),
        "split_counts": dict(split_counts),
        "per_round": per_round,
        "cumulative": cumulative,
        "transition_counts": {
            name: int(transitions[name])
            for name in ("repair", "neutral", "harm", "recovery", "terminal", "unknown")
        },
        "policy_table": policy_table,
        "stop_round_matrix": stop_round_matrix,
        "pairwise_vs_fixed_1": replay.get("pairwise_vs_fixed_1"),
        "policy_charts": _build_policy_chart_data(replay),
        "quality_token_points": token_points,
        "quality_latency_points": latency_points,
        "task_rows": task_rows,
    }
    return report


def write_analysis(manifest: Mapping[str, Any], analysis: Mapping[str, Any]) -> Path:
    """Persist the analysis bundle into the run's results directory."""

    path = Path(manifest["result_dir"]) / "analysis.json"
    write_json(path, dict(analysis))
    return path


def render_analysis(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Render a previously saved ``analysis.json`` without reading trajectories."""

    result_dir = Path(manifest["result_dir"])
    analysis = read_json(result_dir / "analysis.json")
    max_rounds = int(analysis.get("max_rounds", 3))
    csv_path = result_dir / "task_level_results.csv"
    html_path = result_dir / "report.html"
    summary_path = result_dir / "summary_conclusion.txt"
    _write_task_csv(csv_path, analysis.get("task_rows", []), max_rounds)
    html_path.write_text(_render_html(analysis), encoding="utf-8")
    summary_path.write_text(_render_conclusion(analysis), encoding="utf-8")
    _remove_legacy_charts(result_dir)
    chart_paths = _render_png_charts(result_dir, analysis)
    return {
        "json": str(result_dir / "analysis.json"),
        "csv": str(csv_path),
        "html": str(html_path),
        "summary": str(summary_path),
        "charts": chart_paths,
    }


def _remove_legacy_charts(result_dir: Path) -> None:
    """Delete the five round/checkpoint-level PNGs replaced by policy charts."""

    for name in (
        "chart_accuracy_by_round.png",
        "chart_policy_comparison.png",
        "chart_quality_vs_tokens.png",
        "chart_quality_vs_latency.png",
        "chart_stop_round_distribution.png",
    ):
        path = result_dir / name
        if path.exists():
            path.unlink()


def _render_png_charts(
    result_dir: Path, analysis: Mapping[str, Any]
) -> list[str]:
    """Write the five policy-level PNG charts into ``result_dir/charts/``.

    Figures 1/2 are one-point-per-policy quality-vs-resource scatters,
    Figure 3 is the paired RoundValue-minus-baseline point-range plot,
    Figure 4 is the 100% stacked stop-round distribution, and Figure 5 is
    oracle quality regret.
    """

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
    except ImportError as error:
        raise RuntimeError(
            "PNG charts require matplotlib; install it with 'pip install matplotlib'"
        ) from error

    charts_dir = result_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    def save(fig: Any, name: str) -> None:
        path = charts_dir / name
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        paths.append(str(path))

    chart_data = analysis.get("policy_charts")
    if not isinstance(chart_data, Mapping):
        chart_data = {}
    points = [
        point
        for point in chart_data.get("policy_points", [])
        if isinstance(point, Mapping) and point.get("policy")
    ]
    points_by_name = {str(point["policy"]): point for point in points}

    fixed_color = "#6b7280"
    adaptive_color = "#9ca3af"
    oracle_edge = "#4b5563"
    accent_color = "#d97706"
    round_colors = {1: "#0072B2", 2: "#E69F00", 3: "#009E73"}

    def resource_value(point: Mapping[str, Any], key: str) -> float | None:
        value = _number(point.get(key))
        if value is None or key != "mean_wall_clock_ms":
            return value
        return value / 1000.0

    def policy_quality_scatter(
        key: str, x_label: str, title: str, filename: str
    ) -> None:
        if not points:
            return
        fig, ax = plt.subplots(figsize=(8.6, 5.0))

        fixed_points: list[tuple[float, float]] = []
        for name in ("fixed_1", "fixed_2", "fixed_3"):
            point = points_by_name.get(name)
            if point is None:
                continue
            x = resource_value(point, key)
            y = _number(point.get("accuracy"))
            if x is None or y is None:
                continue
            fixed_points.append((x, y))
            ax.scatter(
                [x],
                [y],
                s=72,
                marker="o",
                facecolor=fixed_color,
                edgecolor=fixed_color,
                linewidths=0,
                zorder=3,
            )
        if len(fixed_points) >= 2:
            ax.plot(
                [x for x, _ in fixed_points],
                [y for _, y in fixed_points],
                color=fixed_color,
                linewidth=1.2,
                zorder=2,
                alpha=0.75,
            )

        for name, marker in (("task_only", "s"),):
            point = points_by_name.get(name)
            if point is None:
                continue
            x = resource_value(point, key)
            y = _number(point.get("accuracy"))
            if x is None or y is None:
                continue
            ax.scatter(
                [x],
                [y],
                s=64,
                marker=marker,
                facecolor=adaptive_color,
                edgecolor=adaptive_color,
                linewidths=0,
                zorder=3,
            )

        roundvalue = points_by_name.get("roundvalue")
        if roundvalue is not None:
            x = resource_value(roundvalue, key)
            y = _number(roundvalue.get("accuracy"))
            if x is not None and y is not None:
                ax.scatter(
                    [x],
                    [y],
                    s=230,
                    marker="*",
                    facecolor=accent_color,
                    edgecolor=accent_color,
                    linewidths=0.8,
                    zorder=4,
                )
                ax.annotate(
                    "RoundValue",
                    (x, y),
                    textcoords="offset points",
                    xytext=(13, 12),
                    color=accent_color,
                    fontsize=11,
                    fontweight="bold",
                    zorder=5,
                )

        for name, marker in (("oracle", "*"),):
            point = points_by_name.get(name)
            if point is None:
                continue
            x = resource_value(point, key)
            y = _number(point.get("accuracy"))
            if x is None or y is None:
                continue
            ax.scatter(
                [x],
                [y],
                s=130,
                marker=marker,
                facecolor="white",
                edgecolor=oracle_edge,
                linewidths=1.1,
                zorder=3,
            )

        ax.set_xlabel(x_label)
        ax.set_ylabel("Accuracy")
        ax.set_title(title)
        ax.grid(True, color="#e5e7eb", linewidth=0.8, alpha=0.7)
        ax.set_axisbelow(True)

        accuracies = [
            value
            for point in points
            if (value := _number(point.get("accuracy"))) is not None
        ]
        if accuracies:
            low = max(0.0, min(accuracies) - 0.03)
            high = min(1.0, max(accuracies) + 0.03)
            if high - low < 0.04:
                center = (low + high) / 2
                low = max(0.0, center - 0.02)
                high = min(1.0, low + 0.04)
            ax.set_ylim(low, high)
        ax.margins(x=0.07)

        # Faint Pareto frontier over the non-dominated policies, when it has
        # more than one distinct point.
        unique: dict[tuple[float, float], None] = {}
        for point in points:
            x = resource_value(point, key)
            y = _number(point.get("accuracy"))
            if x is not None and y is not None:
                unique[(round(x, 6), round(y, 6))] = None
        candidates = list(unique)
        frontier = sorted(
            (x, y)
            for x, y in candidates
            if not any(
                (other_y >= y and other_x <= x) and (other_y > y or other_x < x)
                for other_x, other_y in candidates
            )
        )
        if len(frontier) >= 2:
            ax.plot(
                [x for x, _ in frontier],
                [y for _, y in frontier],
                color="#94a3b8",
                linewidth=1.0,
                linestyle=(0, (5, 4)),
                zorder=1,
                alpha=0.9,
            )

        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color=fixed_color,
                linewidth=1.2,
                markersize=7,
                markerfacecolor=fixed_color,
                markeredgecolor=fixed_color,
                label="Fixed budget curve",
            ),
            Line2D(
                [0],
                [0],
                marker="s",
                color="none",
                markersize=7,
                markerfacecolor=adaptive_color,
                markeredgecolor=adaptive_color,
                label="Task-only",
            ),
            Line2D(
                [0],
                [0],
                marker="*",
                color="none",
                markersize=12,
                markerfacecolor=accent_color,
                markeredgecolor=accent_color,
                label="RoundValue",
            ),
            Line2D(
                [0],
                [0],
                marker="*",
                color="none",
                markersize=12,
                markerfacecolor="white",
                markeredgecolor=oracle_edge,
                label="Oracle",
            ),
        ]
        ax.legend(handles=handles, loc="best", frameon=True, fontsize=9)
        fig.tight_layout()
        save(fig, filename)

    def roundvalue_vs_baselines(chart_data: Mapping[str, Any]) -> None:
        baselines = chart_data.get("roundvalue_vs_baselines")
        if not isinstance(baselines, Mapping) or not baselines:
            return
        names = [name for name in PAIRED_BASELINE_ORDER if name in baselines]
        if not names:
            return
        fig, (ax_accuracy, ax_tokens) = plt.subplots(1, 2, figsize=(11.8, 4.8))
        y_positions = list(range(len(names) - 1, -1, -1))
        display_names = [POLICY_DISPLAY_NAMES.get(name, name) for name in names]

        def point_range(ax: Any, *, key: str, fmt: str) -> tuple[list[float], list[float]]:
            lows: list[float] = []
            highs: list[float] = []
            for name, y_position in zip(names, y_positions, strict=False):
                stat = baselines[name].get(key)
                if not isinstance(stat, Mapping):
                    continue
                mean_value = _number(stat.get("mean"))
                if mean_value is None:
                    continue
                low = _number(stat.get("ci95_low"))
                high = _number(stat.get("ci95_high"))
                xerr = None
                if low is not None and high is not None and low <= mean_value <= high:
                    xerr = [[mean_value - low], [high - mean_value]]
                significant = low is not None and high is not None and (low > 0 or high < 0)
                color = accent_color if significant else fixed_color
                ax.errorbar(
                    [mean_value],
                    [y_position],
                    xerr=xerr,
                    fmt="o",
                    markersize=6,
                    color=color,
                    ecolor=color,
                    elinewidth=1.4,
                    capsize=4,
                    zorder=3,
                )
                ax.annotate(
                    format(mean_value, fmt),
                    (mean_value, y_position),
                    textcoords="offset points",
                    xytext=(6, 0),
                    va="center",
                    fontsize=9,
                    color=color,
                )
                if low is not None:
                    lows.append(low)
                if high is not None:
                    highs.append(high)
            return lows, highs

        accuracy_lows, accuracy_highs = point_range(
            ax_accuracy, key="accuracy_difference", fmt="+.3f"
        )
        token_lows, token_highs = point_range(
            ax_tokens, key="total_tokens_difference", fmt="+,.0f"
        )

        def set_limits(ax: Any, lows: list[float], highs: list[float]) -> None:
            if not lows or not highs:
                ax.set_xlim(-0.08, 0.08)
                return
            low = min(lows + [0.0])
            high = max(highs + [0.0])
            if high - low < 1e-9:
                ax.set_xlim(-0.08, 0.08)
                return
            padding = (high - low) * 0.18
            ax.set_xlim(low - padding, high + padding)

        set_limits(ax_accuracy, accuracy_lows, accuracy_highs)
        set_limits(ax_tokens, token_lows, token_highs)

        for ax in (ax_accuracy, ax_tokens):
            ax.axvline(0.0, color="#111827", linewidth=1.0, zorder=2)
            ax.grid(True, axis="x", color="#e5e7eb", linewidth=0.8, alpha=0.7)
            ax.set_axisbelow(True)
            ax.set_yticks(y_positions)
            ax.set_yticklabels(display_names)
            ax.tick_params(axis="y", length=0)

        ax_accuracy.set_title("Δ Accuracy (RoundValue − Baseline)")
        ax_accuracy.set_xlabel("Δ accuracy (positive = RoundValue higher)")
        ax_tokens.set_title("Δ Tokens (RoundValue − Baseline)")
        ax_tokens.set_xlabel("Δ tokens (negative = RoundValue cheaper)")
        fig.suptitle("RoundValue vs Baselines", fontsize=13, fontweight="bold")
        fig.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="none",
                    markersize=7,
                    markerfacecolor=accent_color,
                    markeredgecolor=accent_color,
                    label="95% CI excludes 0",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="none",
                    markersize=7,
                    markerfacecolor=fixed_color,
                    markeredgecolor=fixed_color,
                    label="95% CI spans 0",
                ),
            ],
            loc="lower center",
            bbox_to_anchor=(0.5, -0.03),
            ncol=2,
            frameon=False,
        )
        fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.94))
        save(fig, "chart_roundvalue_vs_baselines.png")

    def adaptive_stop_distribution(chart_data: Mapping[str, Any]) -> None:
        stop = chart_data.get("stop_distribution")
        if not isinstance(stop, Mapping) or not stop:
            return
        names = [name for name in ADAPTIVE_STOP_ORDER if name in stop]
        if not names:
            return
        fig, ax = plt.subplots(figsize=(9.2, 4.8))
        display_names = [
            POLICY_DISPLAY_NAMES.get(name, name) for name in reversed(names)
        ]
        for y_position, name in enumerate(reversed(names)):
            values = stop[name]
            if not isinstance(values, Mapping):
                continue
            left = 0.0
            for round_index in (1, 2, 3):
                percent = _number(values.get(str(round_index))) or 0.0
                if percent <= 0:
                    continue
                color = round_colors.get(round_index, "#6b7280")
                ax.barh(y_position, percent, left=left, color=color, height=0.6, zorder=2)
                if percent >= 8:
                    text_color = "#111827" if round_index == 2 else "white"
                    ax.text(
                        left + percent / 2,
                        y_position,
                        f"{percent:.0f}%",
                        ha="center",
                        va="center",
                        fontsize=9,
                        color=text_color,
                        zorder=3,
                    )
                left += percent
        ax.set_yticks(range(len(display_names)))
        ax.set_yticklabels(display_names)
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(0.0, 100.0)
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        ax.set_xticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
        ax.set_xlabel("Percentage of Tasks")
        ax.set_title("Adaptive Stop-Round Distribution")
        ax.legend(
            handles=[
                Patch(
                    facecolor=round_colors[round_index],
                    label=f"Stop after Round {round_index}",
                )
                for round_index in (1, 2, 3)
            ],
            loc="lower right",
            fontsize=9,
        )
        fig.tight_layout()
        save(fig, "chart_adaptive_stop_distribution.png")

    def oracle_regret(chart_data: Mapping[str, Any]) -> None:
        regrets = chart_data.get("oracle_regret")
        if not isinstance(regrets, Mapping) or not regrets:
            return
        rows: list[tuple[str, float]] = []
        for name in ORACLE_REGRET_ORDER:
            stat = regrets.get(name)
            if not isinstance(stat, Mapping):
                continue
            mean_value = _number(stat.get("mean"))
            if mean_value is not None:
                rows.append((name, mean_value))
        rows.sort(key=lambda item: item[1], reverse=True)
        if not rows:
            return

        fig, ax = plt.subplots(figsize=(8.8, 4.8))
        y_positions = list(range(len(rows) - 1, -1, -1))
        span_lows: list[float] = []
        span_highs: list[float] = []
        for (name, mean_value), y_position in zip(rows, y_positions, strict=False):
            stat = regrets[name]
            low = _number(stat.get("ci95_low"))
            high = _number(stat.get("ci95_high"))
            xerr = None
            if low is not None and high is not None and low <= mean_value <= high:
                xerr = [[mean_value - low], [high - mean_value]]
            color = accent_color if name == "roundvalue" else fixed_color
            marker = "*" if name == "roundvalue" else "o"
            ax.errorbar(
                [mean_value],
                [y_position],
                xerr=xerr,
                fmt=marker,
                markersize=8,
                color=color,
                ecolor="#9ca3af",
                elinewidth=1.4,
                capsize=4,
                zorder=3,
            )
            ax.annotate(
                f"{mean_value:.3f}",
                (mean_value, y_position),
                textcoords="offset points",
                xytext=(6, 0),
                va="center",
                fontsize=9,
                color=color,
            )
            if low is not None:
                span_lows.append(low)
            if high is not None:
                span_highs.append(high)

        ax.axvline(0.0, color="#111827", linewidth=1.0, zorder=2)
        if span_lows and span_highs:
            low = min(span_lows + [0.0])
            high = max(span_highs + [0.0])
            if high - low < 1e-9:
                ax.set_xlim(-0.08, 0.08)
            else:
                padding = (high - low) * 0.18
                ax.set_xlim(low - padding, high + padding)
        else:
            ax.set_xlim(-0.08, 0.08)
        ax.grid(True, axis="x", color="#e5e7eb", linewidth=0.8, alpha=0.7)
        ax.set_axisbelow(True)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(
            [POLICY_DISPLAY_NAMES.get(name, name) for name, _ in rows]
        )
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("Mean Oracle Quality Regret (lower is better; 0 = Oracle)")
        ax.set_title("Oracle Quality Regret")
        fig.tight_layout()
        save(fig, "chart_oracle_regret.png")

    policy_quality_scatter(
        "mean_total_tokens",
        "Mean Total Tokens per Task",
        "Policy Quality vs Tokens",
        "chart_policy_quality_vs_tokens.png",
    )
    policy_quality_scatter(
        "mean_wall_clock_ms",
        "Mean Wall-clock Time per Task (s)",
        "Policy Quality vs Latency",
        "chart_policy_quality_vs_latency.png",
    )
    roundvalue_vs_baselines(chart_data)
    adaptive_stop_distribution(chart_data)
    oracle_regret(chart_data)

    return paths


def _render_conclusion(analysis: Mapping[str, Any]) -> str:
    """Return a short plain-text conclusion derived only from saved results."""

    lines: list[str] = [
        f"RoundValue summary for run {analysis.get('run_id')}",
        f"tasks: {analysis.get('tasks_total')} total, "
        f"{analysis.get('tasks_complete')} complete, "
        f"{analysis.get('tasks_failed')} failed",
    ]
    for row in analysis.get("per_round", []):
        lines.append(
            f"round {row.get('round_index')}: accuracy="
            f"{_fmt(row.get('accuracy'))}, tasks={row.get('n_tasks')}, "
            f"tokens={_fmt(row.get('mean_total_tokens'), 0)}, "
            f"wall_clock_ms={_fmt(row.get('mean_wall_clock_ms'), 0)}, "
            f"logical_calls={_fmt(row.get('mean_logical_calls'), 0)}"
        )
    transitions = analysis.get("transition_counts", {})
    lines.append(
        "transitions: repair="
        f"{transitions.get('repair', 0)}, neutral={transitions.get('neutral', 0)}, "
        f"harm={transitions.get('harm', 0)}, recovery={transitions.get('recovery', 0)}"
    )
    roundvalue = next(
        (
            policy
            for policy in analysis.get("policy_table", [])
            if policy.get("policy") == "roundvalue"
        ),
        None,
    )
    if roundvalue is not None:
        lines.append(
            "RoundValue policy: accuracy="
            f"{_fmt(roundvalue.get('accuracy'))}, tokens="
            f"{_fmt(roundvalue.get('mean_total_tokens'), 0)}, "
            f"wall_clock_ms={_fmt(roundvalue.get('mean_wall_clock_ms'), 0)}, "
            f"stop_round={_fmt(roundvalue.get('mean_stop_round'), 2)}"
        )
        pairwise = (analysis.get("pairwise_vs_fixed_1") or {}).get("roundvalue") or {}
        quality_diff = pairwise.get("quality_difference") or {}
        if quality_diff.get("n_paired"):
            lines.append(
                "RoundValue vs fixed-1 paired accuracy difference: "
                f"mean={_fmt(quality_diff.get('mean_difference'))}, "
                f"95% CI=[{_fmt(quality_diff.get('ci95_low'))}, "
                f"{_fmt(quality_diff.get('ci95_high'))}], "
                f"n_paired={quality_diff.get('n_paired')}"
            )
    return "\n".join(lines) + "\n"


def _now_iso() -> str:
    from contracts import utc_now

    return utc_now()


def _write_task_csv(path: Path, rows: Sequence[Mapping[str, Any]], max_rounds: int) -> None:
    fieldnames = [
        "task_id",
        "split",
        "domain",
        "trajectory_status",
        "completed_rounds",
        "final_quality",
        *[f"quality_round_{round_index}" for round_index in range(1, max_rounds + 1)],
        "cumulative_input_tokens",
        "cumulative_output_tokens",
        "cumulative_total_tokens",
        "cumulative_wall_clock_ms",
        "cumulative_api_latency_ms",
        "cumulative_cost_usd",
        "cumulative_logical_calls",
        "transitions",
        "failure_reason",
        "scoring_error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            flattened = dict(row)
            if flattened.get("scoring_error"):
                flattened["scoring_error"] = json.dumps(
                    flattened["scoring_error"], ensure_ascii=False, sort_keys=True
                )
            writer.writerow(flattened)


__all__ = [
    "build_analysis",
    "render_analysis",
    "write_analysis",
]
