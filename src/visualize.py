"""Offline report builder for saved RoundValue runs.

This module reads task records, scores, labels, and (when available) the
frozen policy replay and renders the readable artifacts:

* ``report_data.json`` -- machine-readable tables used by the HTML below,
* ``task_level_results.csv`` -- one row per collected task,
* ``report.html`` -- a self-contained page with the requested summary tables
  and quality-vs-token / quality-vs-latency charts (inline SVG, no CDN),

plus standalone PNG charts for the same summaries.  Matplotlib is imported
lazily inside the chart renderer so the rest of the module stays lightweight.

It never contacts a provider and never rescore checkpoints.
"""

from __future__ import annotations

import csv
import html
import json
import math
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
    "consensus",
    "task_only",
    "roundvalue",
    "oracle_one_step",
    "oracle",
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
    chart_paths = _render_png_charts(result_dir, analysis)
    return {
        "json": str(result_dir / "analysis.json"),
        "csv": str(csv_path),
        "html": str(html_path),
        "summary": str(summary_path),
        "charts": chart_paths,
    }


def _render_png_charts(
    result_dir: Path, analysis: Mapping[str, Any]
) -> list[str]:
    """Write standalone PNG charts from one saved analysis bundle."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "PNG charts require matplotlib; install it with 'pip install matplotlib'"
        ) from error

    paths: list[str] = []
    run_id = str(analysis.get("run_id", ""))
    round_colors = {1: "#2563eb", 2: "#dc2626", 3: "#16a34a"}

    def save(fig: Any, name: str) -> None:
        path = result_dir / name
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        paths.append(str(path))

    per_round = analysis.get("per_round", [])
    if per_round:
        rows = [
            (int(row["round_index"]), _number(row.get("accuracy")))
            for row in per_round
            if isinstance(row, Mapping)
            and isinstance(row.get("round_index"), int)
        ]
        rows = [(round_index, accuracy) for round_index, accuracy in rows if accuracy is not None]
        if rows:
            fig, ax = plt.subplots(figsize=(7.5, 4.2))
            bars = ax.bar(
                [str(round_index) for round_index, _ in rows],
                [accuracy for _, accuracy in rows],
                color=round_colors[1],
                width=0.5,
            )
            ax.bar_label(bars, fmt="%.3f", padding=2)
            ax.set_ylim(0.0, 1.05)
            ax.set_ylabel("accuracy")
            ax.set_xlabel("round")
            ax.set_title(f"Accuracy by round - {run_id}")
            save(fig, "chart_accuracy_by_round.png")

    policies = [
        policy
        for policy in analysis.get("policy_table", [])
        if isinstance(policy, Mapping) and policy.get("policy")
    ]
    if policies:
        names = [str(policy["policy"]) for policy in policies]
        accuracies = [_number(policy.get("accuracy")) for policy in policies]
        stop_rounds = [_number(policy.get("mean_stop_round")) for policy in policies]
        fig, (ax_accuracy, ax_stop) = plt.subplots(
            1, 2, figsize=(12.5, 4.5), sharey=False
        )
        accuracy_rows = [
            (name, value)
            for name, value in zip(names, accuracies, strict=False)
            if value is not None
        ]
        bars = ax_accuracy.bar(
            [name for name, _ in accuracy_rows],
            [value for _, value in accuracy_rows],
            color="#2563eb",
            width=0.6,
        )
        ax_accuracy.bar_label(bars, fmt="%.2f", padding=2)
        ax_accuracy.set_ylim(0.0, 1.05)
        ax_accuracy.set_ylabel("accuracy")
        ax_accuracy.tick_params(axis="x", labelrotation=30)
        ax_accuracy.set_title("Policy accuracy")

        stop_rows = [
            (name, value)
            for name, value in zip(names, stop_rounds, strict=False)
            if value is not None
        ]
        bars = ax_stop.bar(
            [name for name, _ in stop_rows],
            [value for _, value in stop_rows],
            color="#16a34a",
            width=0.6,
        )
        ax_stop.bar_label(bars, fmt="%.2f", padding=2)
        ax_stop.set_ylabel("mean stop round")
        ax_stop.tick_params(axis="x", labelrotation=30)
        ax_stop.set_title("Policy mean stop round")
        fig.suptitle(f"Policy comparison - {run_id}")
        fig.tight_layout()
        save(fig, "chart_policy_comparison.png")

    def scatter(
        points: Any,
        *,
        x_label: str,
        title: str,
        filename: str,
    ) -> None:
        if not points:
            return
        fig, ax = plt.subplots(figsize=(8.2, 5.0))
        rounds = sorted({int(point["round_index"]) for point in points})
        for round_index in rounds:
            xs = [float(point["x"]) for point in points if int(point["round_index"]) == round_index]
            ys = [float(point["y"]) for point in points if int(point["round_index"]) == round_index]
            ax.scatter(
                xs,
                ys,
                s=22,
                alpha=0.55,
                color=round_colors.get(round_index, "#6b7280"),
                label=f"round {round_index}",
            )
        ax.set_xlabel(x_label)
        ax.set_ylabel("quality")
        ax.set_title(f"{title} - {run_id}")
        ax.legend(title="round")
        save(fig, filename)

    scatter(
        analysis.get("quality_token_points", []),
        x_label="cumulative tokens",
        title="Quality vs tokens",
        filename="chart_quality_vs_tokens.png",
    )
    scatter(
        analysis.get("quality_latency_points", []),
        x_label="cumulative wall-clock (ms)",
        title="Quality vs latency",
        filename="chart_quality_vs_latency.png",
    )

    stop_matrix = analysis.get("stop_round_matrix", {})
    if policies and stop_matrix:
        policy_names = names
        rounds = sorted({int(round_index) for round_index in stop_matrix})
        fig, ax = plt.subplots(figsize=(9.0, 4.6))
        bottoms = [0] * len(policy_names)
        for round_index in rounds:
            counts = [
                int(stop_matrix.get(str(round_index), {}).get(name, 0))
                for name in policy_names
            ]
            ax.bar(
                policy_names,
                counts,
                bottom=bottoms,
                color=round_colors.get(round_index, "#6b7280"),
                width=0.6,
                label=f"stop round {round_index}",
            )
            bottoms = [bottom + count for bottom, count in zip(bottoms, counts, strict=False)]
        ax.set_ylabel("tasks")
        ax.tick_params(axis="x", labelrotation=30)
        ax.legend(title="stop round")
        ax.set_title(f"Stop-round distribution - {run_id}")
        save(fig, "chart_stop_round_distribution.png")

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
