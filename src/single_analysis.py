"""Offline analysis and rendering for the ``single`` topology.

The single topology has exactly one prediction per task, so this module
deliberately builds none of the multi-round Debate concepts: no Delta-Q, V,
G, RoundValue fit, stopping threshold, continuation decision,
Repair/Harm/Recovery transition, or trajectory Oracle.  It only reads saved
trajectories and derives aggregate metrics into ``results/<run_id>/``.
"""

from __future__ import annotations

import csv
import html
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from report import cluster_bootstrap_mean
from storage import read_json


SINGLE_BOOTSTRAP_SEED = 20260813
SINGLE_BOOTSTRAP_SAMPLES = 2000


def _mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result and result not in (float("inf"), float("-inf")) else None


def _fmt(value: Any, digits: int = 3) -> str:
    number = _number(value)
    if number is None:
        return "unknown"
    return f"{number:.{digits}f}"


def _prediction(record: Mapping[str, Any]) -> Mapping[str, Any]:
    trajectory = _mapping(record.get("trajectory")) or {}
    prediction = _mapping(trajectory.get("prediction"))
    return prediction if prediction is not None else {}


def _score(record: Mapping[str, Any]) -> Mapping[str, Any]:
    scores = record.get("scores")
    if isinstance(scores, list) and scores:
        first = _mapping(scores[0])
        if first is not None:
            return first
    return {}


def single_task_rows(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Emit one aggregate row per single-solver task record."""

    rows: list[dict[str, Any]] = []
    for record in records:
        task = _mapping(record.get("task")) or {}
        trajectory = _mapping(record.get("trajectory")) or {}
        prediction = _prediction(record)
        score = _score(record)
        solver = _mapping(trajectory.get("solver")) or {}
        cumulative = _mapping(prediction.get("cumulative")) or {}
        input_tokens = _number(cumulative.get("input_tokens"))
        output_tokens = _number(cumulative.get("output_tokens"))
        quality = _number(score.get("quality"))
        attempts = solver.get("attempts") if isinstance(solver.get("attempts"), list) else []
        fallback = solver.get("fallback", prediction.get("fallback"))
        rows.append(
            {
                "task_id": task.get("task_id"),
                "split": record.get("split"),
                "domain": task.get("domain"),
                "trajectory_status": trajectory.get("status"),
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
                    cumulative.get("wall_clock_ms", trajectory.get("wall_clock_ms"))
                ),
                "api_latency_ms": _number(cumulative.get("api_latency_ms")),
                "cost_usd": _number(cumulative.get("cost_usd")),
                "logical_calls": _number(cumulative.get("logical_calls")),
                "api_attempts": len(attempts),
                "transport_retries": sum(
                    1 for attempt in attempts if _mapping(attempt) is not None
                    and attempt.get("status") == "failed"
                ),
                "format_repairs": solver.get("format_repairs", prediction.get("format_repairs", 0)),
                "fallback": bool(fallback is not None),
                "fallback_type": fallback.get("type") if _mapping(fallback) is not None else None,
                "finish_reason": solver.get("finish_reason", prediction.get("finish_reason")),
                "truncated": bool(
                    solver.get("truncation_encountered", prediction.get("truncated", False))
                ),
                "truncated_attempts": solver.get(
                    "truncated_attempts", prediction.get("truncated_attempts", 0)
                ),
                "failure_reason": trajectory.get("failure_reason"),
                "error": solver.get("error") or trajectory.get("error"),
                "scoring_error": record.get("scoring_error"),
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


def _mean_over(rows: Sequence[dict[str, Any]], key: str) -> float | None:
    values = [_number(row.get(key)) for row in rows]
    observed = [value for value in values if value is not None]
    if not observed or len(observed) != len(rows):
        return None
    return sum(observed) / len(observed)


def summarize_single_collection(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate one single-topology collection without zero-filling unknowns."""

    rows = single_task_rows(records)
    complete = [
        row for row in rows if row["trajectory_status"] == "complete"
    ]
    by_split: dict[str, list[dict[str, Any]]] = {}
    for row in complete:
        split = str(row.get("split") or "unknown")
        by_split.setdefault(split, []).append(row)
    return {
        "schema_version": "1.0",
        "topology": "single",
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
        "api_attempts": sum(int(row.get("api_attempts") or 0) for row in complete),
        "transport_retries": sum(
            int(row.get("transport_retries") or 0) for row in complete
        ),
        "format_repairs": sum(int(row.get("format_repairs") or 0) for row in complete),
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
    }


def build_single_analysis(
    manifest: Mapping[str, Any], records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Derive the complete JSON-safe single analysis bundle from saved records."""

    rows = single_task_rows(records)
    summary = summarize_single_collection(records)
    split_counts: Counter[str] = Counter(str(record.get("split")) for record in records)
    error_counts: Counter[str] = Counter()
    for row in rows:
        if row["trajectory_status"] != "complete":
            error = _mapping(row.get("error")) or {}
            error_counts[str(error.get("type") or "unknown")] += 1
    complete = [row for row in rows if row["trajectory_status"] == "complete"]
    model_selection = _mapping(manifest.get("model_selection")) or {}
    return {
        "schema_version": "1.0",
        "topology": "single",
        "run_id": manifest["run_id"],
        "topology_id": manifest.get("selected_topology_id", "single"),
        "topology_hash": manifest.get("topology_hash"),
        "generated_at": _now_iso(),
        "model": {
            "model_id": manifest.get("selected_model_id"),
            "provider": model_selection.get("provider"),
            "requested_model": model_selection.get("requested_model"),
            "temperature": model_selection.get("temperature"),
            "max_output_tokens": model_selection.get("max_output_tokens"),
            "reasoning": model_selection.get("reasoning"),
        },
        "tasks_total": len(rows),
        "tasks_complete": len(complete),
        "tasks_failed": len(rows) - len(complete),
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


def _comparison_sections(manifest: Mapping[str, Any]) -> str:
    """Render any saved comparison JSONs from this run's results directory."""

    from comparison import comparison_html_section

    result_dir = Path(manifest["result_dir"])
    comparisons_dir = result_dir / "comparisons"
    if not comparisons_dir.is_dir():
        return ""
    parts: list[str] = []
    for path in sorted(comparisons_dir.glob("comparison_*.json")):
        try:
            document = read_json(path)
        except (ValueError, FileNotFoundError):
            continue
        parts.append(comparison_html_section(document))
    return "\n".join(parts)


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


def _render_single_html(analysis: Mapping[str, Any], manifest: Mapping[str, Any]) -> str:
    run_id = str(analysis.get("run_id", ""))
    generated = str(analysis.get("generated_at", ""))
    resources = _mapping(analysis.get("resources")) or {}
    calls = _mapping(analysis.get("calls")) or {}
    truncated = _mapping(analysis.get("truncated")) or {}

    accuracy_rows = [["Overall", _fmt((_mapping(analysis.get("accuracy")) or {}).get("accuracy"))]]
    for split, item in sorted((_mapping(analysis.get("accuracy_by_split")) or {}).items()):
        accuracy_rows.append([split, _fmt(item.get("accuracy"))])

    resource_rows = []
    for key, label in (
        ("input_tokens", "Input tokens"),
        ("output_tokens", "Output tokens"),
        ("total_tokens", "Total tokens"),
        ("wall_clock_ms", "Wall-clock (ms)"),
        ("api_latency_ms", "API latency (ms)"),
        ("cost_usd", "Cost (USD)"),
        ("logical_calls", "Logical calls"),
    ):
        stat = _mapping(resources.get(key)) or {}
        resource_rows.append(
            [
                label,
                _fmt(stat.get("mean")),
                _fmt(stat.get("total")),
                str(stat.get("n_observed")),
                str(stat.get("n_total")),
            ]
        )

    finish_rows = [
        [str(reason), str(count)]
        for reason, count in sorted(
            (_mapping(analysis.get("finish_reason_distribution")) or {}).items()
        )
    ]
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RoundValue single report {html.escape(run_id)}</title>
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
</style>
</head>
<body>
<main>
  <h1>RoundValue single-topology report</h1>
  <div class="meta">run_id: {html.escape(run_id)} &middot; topology: single &middot; generated: {html.escape(generated)}</div>
  <h2>Overview</h2>
  {_table_html(
      ["Tasks total", "Complete", "Failed", "Split counts"],
      [[
          str(analysis.get("tasks_total")),
          str(analysis.get("tasks_complete")),
          str(analysis.get("tasks_failed")),
          ", ".join(f"{key}={value}" for key, value in sorted((_mapping(analysis.get("split_counts")) or {}).items())),
      ]],
  )}
  <h2>Accuracy</h2>
  {_table_html(["Condition", "Accuracy"], accuracy_rows)}
  <h2>Resources (complete tasks)</h2>
  {_table_html(["Resource", "Mean", "Total", "Observed", "Tasks"], resource_rows)}
  <h2>Calls and repairs</h2>
  {_table_html(
      ["Logical calls", "API attempts", "Transport retries", "Format repairs", "Answer-only fallbacks"],
      [[
          _fmt(calls.get("logical_calls"), 0),
          str(calls.get("api_attempts")),
          str(calls.get("transport_retries")),
          str(calls.get("format_repairs")),
          str(calls.get("fallback_count")),
      ]],
  )}
  <h2>Finish reasons</h2>
  {_table_html(["Finish reason", "Tasks"], finish_rows)}
  <h2>Truncation / incomplete responses</h2>
  {_table_html(
      ["Flagged", "Task IDs"],
      [[str(truncated.get("count")), ", ".join(map(str, truncated.get("task_ids") or []))]],
  )}
  {_comparison_sections(manifest)}
</main>
</body>
</html>
"""


def _render_single_conclusion(analysis: Mapping[str, Any]) -> str:
    lines: list[str] = [
        f"RoundValue single-topology summary for run {analysis.get('run_id')}",
        f"tasks: {analysis.get('tasks_total')} total, "
        f"{analysis.get('tasks_complete')} complete, "
        f"{analysis.get('tasks_failed')} failed",
    ]
    accuracy = _mapping(analysis.get("accuracy")) or {}
    lines.append(f"accuracy: {_fmt(accuracy.get('accuracy'))}")
    for split, item in sorted((_mapping(analysis.get("accuracy_by_split")) or {}).items()):
        lines.append(f"accuracy {split}: {_fmt(item.get('accuracy'))}")
    resources = _mapping(analysis.get("resources")) or {}
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        stat = _mapping(resources.get(key)) or {}
        lines.append(
            f"{key}: mean={_fmt(stat.get('mean'))}, total={_fmt(stat.get('total'))}"
        )
    for key in ("wall_clock_ms", "api_latency_ms"):
        stat = _mapping(resources.get(key)) or {}
        lines.append(
            f"{key}: mean={_fmt(stat.get('mean'))}, total={_fmt(stat.get('total'))}"
        )
    cost = _mapping(resources.get("cost_usd")) or {}
    lines.append(
        f"cost_usd: mean={_fmt(cost.get('mean'), 6)}, total={_fmt(cost.get('total'), 6)}"
    )
    calls = _mapping(analysis.get("calls")) or {}
    lines.append(
        "calls: logical="
        f"{_fmt(calls.get('logical_calls'), 0)}, api_attempts={calls.get('api_attempts')}, "
        f"format_repairs={calls.get('format_repairs')}, "
        f"fallbacks={calls.get('fallback_count')}"
    )
    return "\n".join(lines) + "\n"


def _write_single_task_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = [
        "task_id",
        "split",
        "domain",
        "trajectory_status",
        "predicted_answer",
        "canonical_answer",
        "quality",
        "is_correct",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "wall_clock_ms",
        "api_latency_ms",
        "cost_usd",
        "logical_calls",
        "api_attempts",
        "transport_retries",
        "format_repairs",
        "fallback",
        "fallback_type",
        "finish_reason",
        "truncated",
        "truncated_attempts",
        "failure_reason",
        "scoring_error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            flattened = dict(row)
            for key in ("error", "scoring_error"):
                if flattened.get(key):
                    flattened[key] = json.dumps(
                        flattened[key], ensure_ascii=False, sort_keys=True
                    )
            writer.writerow(flattened)


def render_single_analysis(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Render a saved single ``analysis.json`` without reading trajectories."""

    result_dir = Path(manifest["result_dir"])
    analysis = read_json(result_dir / "analysis.json")
    csv_path = result_dir / "task_level_results.csv"
    html_path = result_dir / "report.html"
    summary_path = result_dir / "summary_conclusion.txt"
    _write_single_task_csv(csv_path, analysis.get("task_rows", []))
    html_path.write_text(_render_single_html(analysis, manifest), encoding="utf-8")
    summary_path.write_text(_render_single_conclusion(analysis), encoding="utf-8")
    return {
        "json": str(result_dir / "analysis.json"),
        "csv": str(csv_path),
        "html": str(html_path),
        "summary": str(summary_path),
        "charts": [],
    }


def _now_iso() -> str:
    from contracts import utc_now

    return utc_now()


__all__ = [
    "build_single_analysis",
    "render_single_analysis",
    "single_task_rows",
    "summarize_single_collection",
]
