"""Fully offline ``single`` vs ``debate`` comparison.

The comparison reads only saved run artifacts (manifests, benchmark
snapshots, frozen splits, raw trajectories, and derived scores).  It never
calls a model API and never rewrites historical artifacts.  Before comparing,
it verifies that both runs used the same benchmark file hash, the same frozen
task IDs, and the same split assignment; mismatches are refused instead of
silently compared.  A cross-model comparison is allowed but is explicitly
labeled as cross-model + cross-topology rather than a same-model topology
comparison.
"""

from __future__ import annotations

import html
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scorer import score_single_record, score_trajectory
from single_analysis import single_task_rows
from storage import read_json, write_json


class ComparisonError(ValueError):
    """Raised when two saved runs cannot be compared safely."""


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


def load_run_manifest(root: Path, run_id: str) -> dict[str, Any]:
    """Load a run manifest from the results side, mirroring open_run."""

    root = root.resolve()
    for relative in (
        f"results/{run_id}/manifest.json",
        f"trajectories/{run_id}/run.json",
    ):
        path = root / relative
        if path.is_file():
            manifest = read_json(path)
            if manifest.get("run_id") == run_id:
                return manifest
    raise ComparisonError(f"run {run_id!r} has no readable manifest")


def run_topology_id(manifest: Mapping[str, Any]) -> str:
    """Historical manifests carry no selector and are treated as Debate runs."""

    topology_id = manifest.get("selected_topology_id")
    return "debate" if topology_id is None else str(topology_id)


def _benchmark_snapshot(root: Path, run_id: str) -> dict[str, Any] | None:
    path = root.resolve() / "trajectories" / run_id / "benchmark_snapshot.json"
    if not path.is_file():
        return None
    try:
        return read_json(path)
    except ValueError:
        return None


def _frozen_splits(root: Path, run_id: str) -> dict[str, Any] | None:
    path = root.resolve() / "trajectories" / run_id / "frozen_splits.json"
    if not path.is_file():
        return None
    try:
        document = read_json(path)
    except ValueError:
        return None
    splits = document.get("splits")
    return splits if isinstance(splits, dict) else None


def _task_records(root: Path, run_id: str) -> list[dict[str, Any]]:
    directory = root.resolve() / "trajectories" / run_id
    paths = sorted(directory.glob("task_*.json"))
    if not paths:
        raise ComparisonError(f"run {run_id!r} has no saved task records")
    return [read_json(path) for path in paths]


def _scores_document(root: Path, run_id: str) -> dict[str, Any] | None:
    path = root.resolve() / "results" / run_id / "scores.json"
    if not path.is_file():
        return None
    try:
        return read_json(path)
    except ValueError:
        return None


def _debate_replay(root: Path, run_id: str) -> dict[str, Any] | None:
    path = root.resolve() / "results" / run_id / "test_policy_replay.json"
    if not path.is_file():
        return None
    try:
        return read_json(path)
    except ValueError:
        return None


def _score_by_task(
    root: Path, run_id: str, records: Sequence[Mapping[str, Any]], topology_id: str
) -> dict[str, list[Mapping[str, Any]]]:
    document = _scores_document(root, run_id)
    by_task_raw = (
        document.get("scores_by_task") if document is not None else None
    )
    if isinstance(by_task_raw, dict) and by_task_raw:
        return {
            str(task_id): [dict(item) for item in items]
            for task_id, items in by_task_raw.items()
            if isinstance(items, list)
        }
    result: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        task = _mapping(record.get("task")) or {}
        task_id = str(task.get("task_id"))
        try:
            scores = (
                score_single_record(record)
                if topology_id == "single"
                else score_trajectory(record)
            )
        except (TypeError, ValueError):
            scores = []
        if scores:
            result[task_id] = scores
    return result


def _mean(resources: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [_number(item.get(key)) for item in resources]
    observed = [value for value in values if value is not None]
    if not resources or not observed or len(observed) != len(resources):
        return None
    return sum(observed) / len(observed)


def _resource_bundle(resources: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    bundle: dict[str, Any] = {}
    for key in ("input_tokens", "output_tokens", "logical_calls"):
        bundle[key] = _mean(resources, key)
    input_mean = bundle.get("input_tokens")
    output_mean = bundle.get("output_tokens")
    bundle["total_tokens"] = (
        input_mean + output_mean
        if input_mean is not None and output_mean is not None
        else None
    )
    bundle["wall_clock_ms"] = _mean(resources, "wall_clock_ms")
    bundle["api_latency_ms"] = _mean(resources, "api_latency_ms")
    bundle["cost_usd"] = _mean(resources, "cost_usd")
    bundle["n_tasks"] = len(resources)
    return bundle


def _condition_metrics(
    accuracies: Sequence[Mapping[str, Any]], resources: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    result = _resource_bundle(resources)
    observed = [
        value for item in accuracies if (value := _number(item.get("quality"))) is not None
    ]
    result["accuracy"] = (
        sum(observed) / len(observed)
        if accuracies and observed and len(observed) == len(accuracies)
        else None
    )
    result["n_scored"] = len(observed)
    result["n_tasks"] = len(accuracies)
    return result


def _paired_counts(
    single_scores: Mapping[str, Mapping[str, Any]],
    debate_scores: Mapping[str, Mapping[str, Any]],
    task_ids: Sequence[str],
) -> dict[str, Any]:
    counts = Counter()
    for task_id in task_ids:
        single = single_scores.get(task_id)
        debate = debate_scores.get(task_id)
        if single is None or debate is None:
            continue
        single_quality = _number(single.get("quality"))
        debate_quality = _number(debate.get("quality"))
        if single_quality is None or debate_quality is None:
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
    return {
        name: int(counts[name])
        for name in (
            "both_correct",
            "single_correct_debate_wrong",
            "single_wrong_debate_correct",
            "both_wrong",
        )
    }


def _splits_from_records(records: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    return {
        str(_mapping(record.get("task") or {}).get("task_id")): str(record.get("split"))
        for record in records
        if _mapping(record.get("task") or {}).get("task_id") is not None
        and record.get("split") is not None
    }


def _model_compatibility(
    single_manifest: Mapping[str, Any], debate_manifest: Mapping[str, Any]
) -> tuple[str, dict[str, Any]]:
    single_selection = _mapping(single_manifest.get("model_selection")) or {}
    debate_selection = _mapping(debate_manifest.get("model_selection")) or {}

    def field(selection: Mapping[str, Any], key: str) -> Any:
        return selection.get(key)

    comparison_type = "same_model_topology_comparison"
    differences: list[str] = []
    for key, label in (
        ("provider", "provider"),
        ("requested_model", "requested model"),
    ):
        if field(single_selection, key) != field(debate_selection, key):
            comparison_type = "cross_model_topology_comparison"
            differences.append(label)
    single_reasoning = _mapping(single_selection.get("reasoning")) or {}
    debate_reasoning = _mapping(debate_selection.get("reasoning")) or {}
    if single_reasoning != debate_reasoning:
        comparison_type = "cross_model_topology_comparison"
        differences.append("reasoning configuration")
    if field(single_selection, "temperature") != field(debate_selection, "temperature"):
        differences.append("temperature")
    if field(single_selection, "max_output_tokens") != field(
        debate_selection, "max_output_tokens"
    ):
        differences.append("max_output_tokens")
    details = {
        "single": dict(single_selection),
        "debate": dict(debate_selection),
        "comparison_type": comparison_type,
        "differences": differences,
        "note": (
            "same provider and requested model: the accuracy difference is a "
            "topology comparison"
            if comparison_type == "same_model_topology_comparison"
            else (
                "provider, requested model, or reasoning configuration differ: "
                "this is a cross-model + cross-topology comparison and must not "
                "be interpreted as topology-only causality"
            )
        ),
    }
    return comparison_type, details


def build_comparison(root: Path, run_a_id: str, run_b_id: str) -> dict[str, Any]:
    """Compare one saved ``single`` run against one saved ``debate`` run."""

    root = root.resolve()
    manifest_a = load_run_manifest(root, run_a_id)
    manifest_b = load_run_manifest(root, run_b_id)
    topology_a = run_topology_id(manifest_a)
    topology_b = run_topology_id(manifest_b)
    if {topology_a, topology_b} != {"single", "debate"}:
        raise ComparisonError(
            "comparison requires exactly one single run and one debate run; "
            f"got {topology_a!r} and {topology_b!r}"
        )
    if topology_a == "single":
        single_run_id, debate_run_id = run_a_id, run_b_id
        single_manifest, debate_manifest = manifest_a, manifest_b
    else:
        single_run_id, debate_run_id = run_b_id, run_a_id
        single_manifest, debate_manifest = manifest_b, manifest_a

    # Compatibility gates: same dataset, same benchmark file hash, same task
    # IDs, same frozen split assignment.
    if single_manifest.get("dataset") != debate_manifest.get("dataset"):
        raise ComparisonError(
            "comparison refused: benchmark/dataset identity differs "
            f"({single_manifest.get('dataset')!r} vs {debate_manifest.get('dataset')!r})"
        )
    single_snapshot = _benchmark_snapshot(root, single_run_id)
    debate_snapshot = _benchmark_snapshot(root, debate_run_id)
    benchmark_hashes = {
        "single": single_snapshot.get("source_sha256") if single_snapshot else None,
        "debate": debate_snapshot.get("source_sha256") if debate_snapshot else None,
    }
    if (
        benchmark_hashes["single"] is not None
        and benchmark_hashes["debate"] is not None
        and benchmark_hashes["single"] != benchmark_hashes["debate"]
    ):
        raise ComparisonError(
            "comparison refused: benchmark file hash differs "
            f"({benchmark_hashes['single']} vs {benchmark_hashes['debate']})"
        )

    single_records = _task_records(root, single_run_id)
    debate_records = _task_records(root, debate_run_id)
    single_task_ids = {
        str(_mapping(record.get("task") or {}).get("task_id")) for record in single_records
    }
    debate_task_ids = {
        str(_mapping(record.get("task") or {}).get("task_id")) for record in debate_records
    }
    if single_task_ids != debate_task_ids:
        raise ComparisonError(
            "comparison refused: frozen task ID sets differ "
            f"(single-only={sorted(single_task_ids - debate_task_ids)}, "
            f"debate-only={sorted(debate_task_ids - single_task_ids)})"
        )

    single_splits = _frozen_splits(root, single_run_id)
    debate_splits = _frozen_splits(root, debate_run_id)
    if single_splits is None and debate_splits is None:
        single_splits = _splits_from_records(single_records)
        debate_splits = _splits_from_records(debate_records)
    if single_splits is None or debate_splits is None:
        raise ComparisonError(
            "comparison refused: one run has a frozen split assignment and the other does not"
        )
    if dict(single_splits) != dict(debate_splits):
        raise ComparisonError("comparison refused: frozen split assignments differ")

    single_scores = _score_by_task(
        root, single_run_id, single_records, topology_id="single"
    )
    debate_scores = _score_by_task(
        root, debate_run_id, debate_records, topology_id="debate"
    )

    # Pair on the test split when both runs contain one, otherwise use the
    # shared task set (for example smoke-split comparisons).
    test_ids = sorted(
        task_id for task_id, split in dict(single_splits).items() if split == "test"
    )
    paired_ids = test_ids if test_ids else sorted(single_task_ids)
    split_used = "test" if test_ids else "shared_task_set"

    single_rows = {
        str(row.get("task_id")): row for row in single_task_rows(single_records)
    }
    single_paired_rows = [single_rows[task_id] for task_id in paired_ids if task_id in single_rows]
    single_score_map = {
        task_id: single_scores[task_id][0] for task_id in paired_ids if single_scores.get(task_id)
    }
    single_resources = [
        {
            "input_tokens": row.get("input_tokens"),
            "output_tokens": row.get("output_tokens"),
            "wall_clock_ms": row.get("wall_clock_ms"),
            "api_latency_ms": row.get("api_latency_ms"),
            "cost_usd": row.get("cost_usd"),
            "logical_calls": row.get("logical_calls"),
        }
        for row in single_paired_rows
    ]
    single_accuracy_rows = [
        {"quality": single_score_map[task_id].get("quality")}
        for task_id in paired_ids
        if task_id in single_score_map
    ]
    single_metrics = _condition_metrics(single_accuracy_rows, single_resources)

    debate_checkpoints: dict[int, dict[str, Mapping[str, Any]]] = {}
    for record in debate_records:
        task = _mapping(record.get("task")) or {}
        task_id = str(task.get("task_id"))
        trajectory = _mapping(record.get("trajectory")) or {}
        for checkpoint in trajectory.get("checkpoints", []):
            item = _mapping(checkpoint)
            if item is None:
                continue
            round_index = item.get("round_index")
            if not isinstance(round_index, int) or isinstance(round_index, bool):
                continue
            debate_checkpoints.setdefault(int(round_index), {})[task_id] = item

    debate_conditions: dict[str, dict[str, Any]] = {}
    for round_index in sorted(debate_checkpoints):
        per_task = {
            task_id: debate_checkpoints[round_index][task_id]
            for task_id in paired_ids
            if task_id in debate_checkpoints[round_index]
        }
        score_rows = [
            {"quality": _score_for_round(debate_scores, task_id, round_index)}
            for task_id in per_task
        ]
        resource_rows = []
        for checkpoint in per_task.values():
            cumulative = _mapping(checkpoint.get("cumulative")) or {}
            resource_rows.append(
                {
                    "input_tokens": cumulative.get("input_tokens"),
                    "output_tokens": cumulative.get("output_tokens"),
                    "wall_clock_ms": cumulative.get("wall_clock_ms"),
                    "api_latency_ms": cumulative.get("api_latency_ms"),
                    "cost_usd": cumulative.get("cost_usd"),
                    "logical_calls": cumulative.get("logical_calls"),
                }
            )
        debate_conditions[f"round_{round_index}"] = _condition_metrics(
            score_rows, resource_rows
        )

    # RoundValue and Oracle exist only for analyzed Debate runs; they are
    # never fabricated for runs that lack them.
    replay = _debate_replay(root, debate_run_id)
    policy_metrics = (
        _mapping(replay.get("policy_metrics")) if replay is not None else None
    )
    for name in ("roundvalue", "oracle"):
        metrics = policy_metrics.get(name) if policy_metrics is not None else None
        debate_conditions[name] = {
            "accuracy": _number(metrics.get("accuracy")) if metrics else None,
            "mean_total_tokens": _number(metrics.get("mean_total_tokens")) if metrics else None,
            "mean_wall_clock_ms": _number(metrics.get("mean_wall_clock_ms")) if metrics else None,
            "mean_api_latency_ms": _number(metrics.get("mean_api_latency_ms")) if metrics else None,
            "mean_cost_usd": _number(metrics.get("mean_cost_usd")) if metrics else None,
            "mean_logical_calls": _number(metrics.get("mean_logical_calls")) if metrics else None,
            "n_tasks": metrics.get("n_records") if metrics else None,
            "defined": metrics is not None,
        }

    paired: dict[str, Any] = {}
    for round_index in (1, 5):
        if round_index not in debate_checkpoints:
            paired[f"debate_round_{round_index}"] = None
            continue
        debate_map = {
            task_id: {"quality": _score_for_round(debate_scores, task_id, round_index)}
            for task_id in paired_ids
            if task_id in debate_scores
        }
        single_map = {
            task_id: single_score_map[task_id]
            for task_id in paired_ids
            if task_id in single_score_map
        }
        counts = _paired_counts(single_map, debate_map, paired_ids)
        counts["n_paired"] = sum(counts.values())
        single_acc = single_metrics.get("accuracy")
        debate_acc = debate_conditions[f"round_{round_index}"].get("accuracy")
        counts["accuracy_difference"] = (
            single_acc - debate_acc
            if single_acc is not None and debate_acc is not None
            else None
        )
        paired[f"debate_round_{round_index}"] = counts

    comparison_type, model_details = _model_compatibility(
        single_manifest, debate_manifest
    )
    differences: dict[str, Any] = {}
    single_accuracy = single_metrics.get("accuracy")
    for name, metrics in debate_conditions.items():
        debate_accuracy = metrics.get("accuracy")
        differences[name] = (
            single_accuracy - debate_accuracy
            if single_accuracy is not None and debate_accuracy is not None
            else None
        )

    return {
        "schema_version": "1.0",
        "comparison_type": comparison_type,
        "generated_at": _now_iso(),
        "single_run_id": single_run_id,
        "debate_run_id": debate_run_id,
        "topologies": {"single": "single", "debate": run_topology_id(debate_manifest)},
        "compatibility": {
            "dataset": single_manifest.get("dataset"),
            "benchmark_sha256_single": benchmark_hashes["single"],
            "benchmark_sha256_debate": benchmark_hashes["debate"],
            "task_set_match": True,
            "split_assignment_match": True,
            "task_count": len(single_task_ids),
            "split_used": split_used,
            "paired_task_count": len(paired_ids),
            "model": model_details,
        },
        "single": single_metrics,
        "debate": debate_conditions,
        "accuracy_difference_single_minus_debate": differences,
        "paired": paired,
    }


def _score_for_round(
    scores_by_task: Mapping[str, Sequence[Mapping[str, Any]]],
    task_id: str,
    round_index: int,
) -> Any:
    for score in scores_by_task.get(task_id, []):
        item = _mapping(score)
        if item is None:
            continue
        if item.get("round_index") == round_index:
            return item.get("quality")
    return None


def write_comparison(
    manifest: Mapping[str, Any], document: Mapping[str, Any], other_run_id: str
) -> Path:
    """Save a comparison into the visualized run's results directory."""

    result_dir = Path(manifest["result_dir"])
    directory = result_dir / "comparisons"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"comparison_{other_run_id}.json"
    write_json(path, dict(document))
    return path


def comparison_html_section(document: Mapping[str, Any]) -> str:
    """Render one comparison document as a self-contained HTML section."""

    compatibility = _mapping(document.get("compatibility")) or {}
    model = _mapping(compatibility.get("model")) or {}
    single = _mapping(document.get("single")) or {}
    debate = _mapping(document.get("debate")) or {}
    paired = _mapping(document.get("paired")) or {}

    condition_rows: list[list[str]] = []
    for name, label in (
        ("single", "Single"),
        ("round_1", "Debate Round 1"),
        ("round_2", "Debate Fixed-2"),
        ("round_3", "Debate Fixed-3"),
        ("round_4", "Debate Fixed-4"),
        ("round_5", "Debate Fixed-5"),
        ("roundvalue", "RoundValue"),
        ("oracle", "Oracle"),
    ):
        metrics = single if name == "single" else _mapping(debate.get(name)) or {}
        if name != "single" and not metrics:
            continue
        condition_rows.append(
            [
                label,
                _fmt(metrics.get("accuracy")),
                _fmt(metrics.get("total_tokens", metrics.get("mean_total_tokens")), 0),
                _fmt(
                    metrics.get("wall_clock_ms", metrics.get("mean_wall_clock_ms")), 0
                ),
                _fmt(metrics.get("api_latency_ms", metrics.get("mean_api_latency_ms")), 0),
                _fmt(metrics.get("cost_usd", metrics.get("mean_cost_usd")), 6),
                _fmt(metrics.get("logical_calls", metrics.get("mean_logical_calls")), 0),
            ]
        )

    paired_rows: list[list[str]] = []
    for key, label in (("debate_round_1", "Debate Round 1"), ("debate_round_5", "Debate Round 5")):
        counts = _mapping(paired.get(key)) or {}
        if not counts:
            paired_rows.append([label, "unavailable", "", "", "", "", ""])
            continue
        paired_rows.append(
            [
                label,
                str(counts.get("both_correct")),
                str(counts.get("single_correct_debate_wrong")),
                str(counts.get("single_wrong_debate_correct")),
                str(counts.get("both_wrong")),
                str(counts.get("n_paired")),
                _fmt(counts.get("accuracy_difference")),
            ]
        )

    def table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
        head = "".join(f"<th>{html.escape(str(header))}</th>" for header in headers)
        body = "".join(
            "<tr>" + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in row) + "</tr>"
            for row in rows
        )
        return (
            '<div class="table-wrap"><table><thead><tr>'
            f"{head}</tr></thead><tbody>{body}</tbody></table></div>"
        )

    return (
        f"<h2>Single vs Debate comparison "
        f"({html.escape(str(document.get('single_run_id')))} vs "
        f"{html.escape(str(document.get('debate_run_id')))})</h2>"
        '<div class="meta">'
        f"type: {html.escape(str(document.get('comparison_type')))} &middot; "
        f"dataset: {html.escape(str(compatibility.get('dataset')))} &middot; "
        f"paired tasks: {compatibility.get('paired_task_count')} "
        f"({html.escape(str(compatibility.get('split_used')))})</div>"
        + table(
            ["Condition", "Accuracy", "Tokens", "Wall-clock (ms)", "API time (ms)", "Cost (USD)", "Logical calls"],
            condition_rows,
        )
        + f'<div class="meta">model note: {html.escape(str(model.get("note")))}</div>'
        + "<h2>Paired task outcome counts</h2>"
        + table(
            [
                "Comparison",
                "Both correct",
                "Single correct, debate wrong",
                "Single wrong, debate correct",
                "Both wrong",
                "Paired",
                "Δ accuracy",
            ],
            paired_rows,
        )
    )


def append_comparison_sections(manifest: Mapping[str, Any]) -> None:
    """Inject saved comparison sections into a rendered Debate report."""

    result_dir = Path(manifest["result_dir"])
    comparisons_dir = result_dir / "comparisons"
    if not comparisons_dir.is_dir():
        return
    html_path = result_dir / "report.html"
    if not html_path.is_file():
        return
    sections: list[str] = []
    for path in sorted(comparisons_dir.glob("comparison_*.json")):
        try:
            document = read_json(path)
        except ValueError:
            continue
        sections.append(comparison_html_section(document))
    if not sections:
        return
    content = html_path.read_text(encoding="utf-8")
    marker = "</body>"
    if marker not in content:
        return
    injected = content.replace(marker, "".join(sections) + marker, 1)
    html_path.write_text(injected, encoding="utf-8")


def _now_iso() -> str:
    from contracts import utc_now

    return utc_now()


__all__ = [
    "ComparisonError",
    "append_comparison_sections",
    "build_comparison",
    "comparison_html_section",
    "load_run_manifest",
    "run_topology_id",
    "write_comparison",
]
