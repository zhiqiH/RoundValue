"""The sole user-facing entry point for RoundValue experiments.

This script deliberately owns orchestration only.  The implementation lives
in ``src/`` and all durable state is JSON.  In particular, ``fit``,
``evaluate``, and ``reproduce`` never construct a provider: they operate only
on trajectories already saved by ``smoke`` or ``collect``.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import sys
from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from benchmark_io import benchmark_provenance, freeze_splits, load_benchmark  # noqa: E402
from config_loader import config_snapshot, load_experiment_config, select_topology  # noqa: E402
from contracts import utc_now  # noqa: E402
from debate_runner import FixedDebateRunner  # noqa: E402
from labels import build_labels, label_summary  # noqa: E402
from policy import fit_policy_models, replay_policies  # noqa: E402
from provider import build_provider  # noqa: E402
from report import summarize_collection  # noqa: E402
from scorer import score_trajectory  # noqa: E402
from storage import (  # noqa: E402
    create_run,
    open_run,
    read_json,
    read_task_records,
    reproducibility_index,
    snapshot_benchmark,
    update_run_status,
    write_json,
    write_result,
    write_task_record,
)

DEFAULT_BENCHMARK = "benchmark/test/smoke_tasks.json"
VALID_SPLITS = frozenset({"train", "validation", "test"})
SPLIT_SEED = 20260813
BOOTSTRAP_SEED = 20260813
BOOTSTRAP_SAMPLES = 2000
POLICY_SELECTION = {
    "lambda_cost": 0.0,
    "mu_latency": 0.0,
    "target": "G",
    "ridge": 1e-6,
    "threshold_candidates": [-0.05, 0.0, 0.05],
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run or replay the fixed, JSON-configured RoundValue experiment."
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=("smoke", "collect", "fit", "evaluate", "reproduce"),
        help="smoke/collect make real API calls; all other modes are offline only.",
    )
    parser.add_argument(
        "--benchmark",
        default=None,
        help=(
            "Project-relative JSON benchmark manifest. Smoke defaults to its visible fixture; "
            "collect requires a frozen train/validation/test benchmark."
        ),
    )
    parser.add_argument(
        "--run-id",
        help="Existing run for offline modes, or an optional explicit new ID for smoke/collect.",
    )
    parser.add_argument(
        "--model-id",
        help="Model profile ID from configs/model_config.json (smoke/collect only).",
    )
    parser.add_argument(
        "--topology-id",
        help="Topology ID from configs/topology.json (smoke/collect only).",
    )
    parser.add_argument(
        "--allow-local-code-evaluation",
        action="store_true",
        help=(
            "Allow local execution of code-benchmark candidates during collection. "
            "Use only in an isolated environment; this is not an OS security sandbox."
        ),
    )
    return parser


def _emit(value: Mapping[str, Any]) -> None:
    """Print a compact JSON status line suitable for a terminal or a runner."""

    print(json.dumps(dict(value), ensure_ascii=False, sort_keys=True))


def _safe_message(error: BaseException) -> str:
    """Keep terminal failures useful without serializing tracebacks or credentials."""

    message = str(error).replace("\r", " ").replace("\n", " ").strip()
    return message[:500] if message else type(error).__name__


def _command_line() -> list[str]:
    return ["roundvalue", *sys.argv[1:]]


def _create_run(
    args: argparse.Namespace,
    experiment: Mapping[str, Any],
    state: dict[str, Any],
) -> dict[str, Any]:
    manifest = create_run(
        PROJECT_ROOT,
        command=_command_line(),
        config_snapshot=config_snapshot(PROJECT_ROOT),
        run_id=args.run_id,
    )
    state["manifest"] = manifest
    manifest = update_run_status(
        manifest,
        "running",
        mode=args.mode,
        selected_model_id=experiment["model_id"],
        selected_topology_id=experiment["topology_id"],
    )
    state["manifest"] = manifest
    return manifest


def _write_benchmark_snapshot(
    manifest: Mapping[str, Any],
    benchmark_path: Path,
    benchmark_document: Mapping[str, Any],
) -> None:
    """Freeze both the manifest content and every referenced benchmark hash."""

    snapshot_benchmark(dict(manifest), benchmark_path)
    provenance = benchmark_provenance(
        PROJECT_ROOT, benchmark_path, dict(benchmark_document)
    )
    write_json(
        Path(manifest["trajectory_dir"]) / "benchmark_provenance.json", provenance
    )


def _score_record(
    record: Mapping[str, Any],
    *,
    allow_local_code_evaluation: bool,
) -> list[dict[str, Any]]:
    """Call the scorer while supporting its explicit safety-gate spelling.

    The scorer is the authority for code execution.  This small adapter lets
    this user entry point stay compatible with a deliberately flat scorer
    interface while ensuring that a code task never silently ignores the
    user's explicit authorization flag.
    """

    task = record.get("task")
    is_code = (
        isinstance(task, Mapping) and str(task.get("domain", "")).casefold() == "code"
    )
    parameters = inspect.signature(score_trajectory).parameters
    keyword: dict[str, Any] = {}
    for name in (
        "allow_local_code_evaluation",
        "allow_local_code_execution",
        "allow_local_execution",
    ):
        if name in parameters:
            keyword[name] = allow_local_code_evaluation
            break
    else:
        if is_code and not allow_local_code_evaluation:
            raise RuntimeError(
                "code scoring requires --allow-local-code-evaluation; "
                "the installed scorer has no explicit local-execution gate"
            )
    return score_trajectory(record, **keyword)


def _task_record(
    task: Mapping[str, Any],
    split: str,
    trajectory: Mapping[str, Any],
    *,
    allow_local_code_evaluation: bool,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema_version": "1.0",
        "created_at": utc_now(),
        "task": dict(task),
        "split": split,
        "trajectory": dict(trajectory),
        "scoring": {
            "mode": "offline_deterministic",
            "allow_local_code_evaluation": allow_local_code_evaluation,
        },
    }
    try:
        scores = _score_record(
            record, allow_local_code_evaluation=allow_local_code_evaluation
        )
        if trajectory.get("status") == "complete" and any(
            isinstance(score.get("quality"), bool)
            or not isinstance(score.get("quality"), int | float)
            or not math.isfinite(float(score["quality"]))
            for score in scores
        ):
            raise ValueError(
                "offline scorer did not produce numeric quality for every checkpoint"
            )
        record["scores"] = scores
    except Exception as error:
        record["scores"] = []
        record["scoring_error"] = {
            "type": type(error).__name__,
            "message": _safe_message(error),
        }
    return record


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Use versioned code defaults; topology JSON contains communication only."""

    parameters = inspect.signature(summarize_collection).parameters
    keyword: dict[str, Any] = {}
    if "bootstrap_seed" in parameters:
        keyword["bootstrap_seed"] = BOOTSTRAP_SEED
    if "bootstrap_samples" in parameters:
        keyword["bootstrap_samples"] = BOOTSTRAP_SAMPLES
    return summarize_collection(records, **keyword)


def _collect_task(
    runner: FixedDebateRunner,
    *,
    task: Mapping[str, Any],
    split: str,
    run_id: str,
    max_rounds: int,
    allow_local_code_evaluation: bool,
) -> dict[str, Any]:
    trajectory = runner.run_trajectory(
        task=dict(task), run_id=run_id, max_rounds=max_rounds
    )
    return _task_record(
        task,
        split,
        trajectory,
        allow_local_code_evaluation=allow_local_code_evaluation,
    )


def _run_smoke(
    args: argparse.Namespace,
    experiment: Mapping[str, Any],
    state: dict[str, Any],
) -> int:
    """Run the independent repository acceptance tasks with real API calls."""

    benchmark = args.benchmark or DEFAULT_BENCHMARK
    benchmark_path, benchmark_document, tasks = load_benchmark(PROJECT_ROOT, benchmark)
    selected_tasks = [task for task in tasks if task.get("domain") == "math"]
    skipped_code_task_ids = [
        str(task["task_id"]) for task in tasks if task.get("domain") == "code"
    ]
    if args.allow_local_code_evaluation:
        selected_tasks.extend(task for task in tasks if task.get("domain") == "code")
        skipped_code_task_ids = []
    if not selected_tasks:
        raise ValueError("smoke benchmark has no runnable math task")
    manifest = _create_run(args, experiment, state)
    _write_benchmark_snapshot(manifest, benchmark_path, benchmark_document)
    provider = build_provider(dict(experiment))
    try:
        runner = FixedDebateRunner(dict(experiment), provider)
        records = [
            _collect_task(
                runner,
                task=task,
                split="smoke",
                run_id=str(manifest["run_id"]),
                max_rounds=1,
                allow_local_code_evaluation=args.allow_local_code_evaluation,
            )
            for task in selected_tasks
        ]
    finally:
        provider.close()
    for record in records:
        write_task_record(manifest, record)
    summary = _summary(records)
    write_result(manifest, "collection_summary", summary)
    write_result(
        manifest,
        "smoke_status",
        {
            "schema_version": "1.0",
            "mode": "smoke",
            "task_ids": [task["task_id"] for task in selected_tasks],
            "skipped_code_task_ids": skipped_code_task_ids,
            "expected_logical_calls": 7 * len(selected_tasks),
            "max_rounds": 1,
            "requires_quality_one": True,
        },
    )
    succeeded = all(
        record["trajectory"].get("status") == "complete"
        and not record.get("scoring_error")
        and len(record.get("scores", [])) == 1
        and record["scores"][0].get("quality") == 1
        for record in records
    )
    status = "smoke_complete" if succeeded else "smoke_failed"
    updated = update_run_status(
        manifest,
        status,
        task_count=len(records),
        complete_task_count=sum(
            record["trajectory"].get("status") == "complete" for record in records
        ),
    )
    state["manifest"] = updated
    _emit(
        {
            "status": status,
            "mode": "smoke",
            "run_id": updated["run_id"],
            "trajectory_dir": updated["trajectory_dir"],
            "result_dir": updated["result_dir"],
        }
    )
    return 0 if succeeded else 1


def _run_collect(
    args: argparse.Namespace,
    experiment: Mapping[str, Any],
    state: dict[str, Any],
) -> int:
    """Collect complete trajectories only; online stopping never changes collection."""

    if not args.benchmark:
        raise ValueError(
            "collect requires --benchmark <project-relative-benchmark.json>"
        )
    benchmark_path, benchmark_document, tasks = load_benchmark(
        PROJECT_ROOT, args.benchmark
    )
    split_seed = SPLIT_SEED
    split_by_task = freeze_splits(tasks, seed=split_seed)
    split_counts = {
        split: sum(value == split for value in split_by_task.values())
        for split in VALID_SPLITS
    }
    absent_splits = [split for split, count in split_counts.items() if count == 0]
    if absent_splits:
        raise ValueError(
            "frozen benchmark has no "
            + ", ".join(sorted(absent_splits))
            + " tasks; collect requires train, validation, and test coverage"
        )
    if not args.allow_local_code_evaluation and any(
        task.get("domain") == "code" for task in tasks
    ):
        raise ValueError(
            "benchmark includes code tasks; pass --allow-local-code-evaluation "
            "only after reviewing the local-execution risk"
        )
    manifest = _create_run(args, experiment, state)
    _write_benchmark_snapshot(manifest, benchmark_path, benchmark_document)
    write_json(
        Path(manifest["trajectory_dir"]) / "frozen_splits.json",
        {
            "schema_version": "1.0",
            "split_seed": split_seed,
            "splits": split_by_task,
        },
    )
    provider = build_provider(dict(experiment))
    records: list[dict[str, Any]] = []
    try:
        runner = FixedDebateRunner(dict(experiment), provider)
        max_rounds = int(experiment["topology"]["max_rounds"])
        for task in tasks:
            split = split_by_task[task["task_id"]]
            record = _collect_task(
                runner,
                task=task,
                split=split,
                run_id=str(manifest["run_id"]),
                max_rounds=max_rounds,
                allow_local_code_evaluation=args.allow_local_code_evaluation,
            )
            records.append(record)
            write_task_record(manifest, record)
    finally:
        provider.close()
    summary = _summary(records)
    write_result(manifest, "collection_summary", summary)
    failed = [
        record
        for record in records
        if record["trajectory"].get("status") != "complete"
        or record.get("scoring_error")
    ]
    status = "collect_complete" if not failed else "collect_failed"
    updated = update_run_status(
        manifest,
        status,
        task_count=len(records),
        complete_task_count=len(records) - len(failed),
        failed_task_ids=[record["task"].get("task_id") for record in failed],
    )
    state["manifest"] = updated
    _emit(
        {
            "status": status,
            "mode": "collect",
            "run_id": updated["run_id"],
            "task_count": len(records),
            "failed_task_count": len(failed),
            "trajectory_dir": updated["trajectory_dir"],
            "result_dir": updated["result_dir"],
        }
    )
    return 0 if not failed else 1


def _require_existing_run(
    args: argparse.Namespace, state: dict[str, Any]
) -> dict[str, Any]:
    if not args.run_id:
        raise ValueError(f"--run-id is required for --mode {args.mode}")
    cached = state.get("manifest")
    if isinstance(cached, Mapping) and cached.get("run_id") == args.run_id:
        return dict(cached)
    manifest = open_run(PROJECT_ROOT, args.run_id)
    state["manifest"] = manifest
    return manifest


def _frozen_experiment(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Use a run's immutable selected topology, never a later config file."""

    snapshots = manifest.get("configs")
    if not isinstance(snapshots, Mapping):
        raise ValueError("run manifest has no frozen JSON configuration snapshot")
    topology_snapshot = snapshots.get("topology.json")
    if not isinstance(topology_snapshot, Mapping):
        raise ValueError("run manifest has no frozen topology.json snapshot")
    topology_document = topology_snapshot.get("content")
    if not isinstance(topology_document, Mapping):
        raise ValueError("run manifest has malformed frozen topology.json content")
    selected_id = manifest.get("selected_topology_id")
    if selected_id is not None and not isinstance(selected_id, str):
        raise ValueError("run manifest has malformed selected_topology_id")
    topology_id, topology = select_topology(dict(topology_document), selected_id)
    return {
        "model_id": manifest.get("selected_model_id"),
        "topology_id": topology_id,
        "topology": topology,
    }


def _records_for_split(
    records: list[dict[str, Any]], split: str
) -> list[dict[str, Any]]:
    selected = [record for record in records if record.get("split") == split]
    if not selected:
        raise ValueError(f"run has no {split} records")
    if any(
        record.get("trajectory", {}).get("status") != "complete" for record in selected
    ):
        raise ValueError(
            f"run has incomplete {split} trajectories; recollect before offline analysis"
        )
    if any(record.get("scoring_error") for record in selected):
        raise ValueError(
            f"run has unscored {split} trajectories; resolve scorer failures before analysis"
        )
    return selected


def _selection_config() -> dict[str, Any]:
    """Return versioned policy defaults, kept separate from communication topology."""

    config = dict(POLICY_SELECTION)
    for key in ("lambda_cost", "mu_latency", "ridge"):
        value = config[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
        ):
            raise ValueError(f"policy selection {key} must be a finite number")
        if value < 0:
            raise ValueError(f"policy selection {key} must be non-negative")
    if not isinstance(config["target"], str) or not config["target"]:
        raise ValueError("policy selection target must be a non-empty string")
    candidates = config["threshold_candidates"]
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(
            "policy selection threshold_candidates must be a non-empty JSON array"
        )
    normalized: list[float] = []
    for value in candidates:
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
        ):
            raise ValueError(
                "policy selection threshold_candidates must be finite numbers"
            )
        normalized.append(float(value))
    config["threshold_candidates"] = sorted(set(normalized))
    return config


def _rank_threshold(
    metrics: Mapping[str, Any], threshold: float
) -> tuple[float, float, float, float]:
    """Maximize validation utility, then quality, then minimize resource use."""

    utility = metrics.get("mean_utility")
    quality = metrics.get("mean_quality")
    tokens = metrics.get("mean_total_tokens")
    if not isinstance(utility, int | float) or not math.isfinite(utility):
        return (-math.inf, -math.inf, -math.inf, -threshold)
    quality_value = float(quality) if isinstance(quality, int | float) else -math.inf
    token_value = float(tokens) if isinstance(tokens, int | float) else math.inf
    return (float(utility), quality_value, -token_value, -threshold)


def _freeze_thresholds(
    models: Mapping[str, Any],
    validation_records: list[dict[str, Any]],
    candidates: list[float],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Tune each learned policy on validation only and return its audit trail."""

    frozen = deepcopy(dict(models))
    audit: dict[str, Any] = {"selection_split": "validation", "policies": {}}
    for policy_name in ("roundvalue", "task_only"):
        if not isinstance(frozen.get(policy_name), dict):
            raise ValueError(f"fitted policy missing {policy_name} descriptor")
        best_threshold: float | None = None
        best_rank: tuple[float, float, float, float] | None = None
        candidate_rows: list[dict[str, Any]] = []
        for threshold in candidates:
            trial = deepcopy(frozen)
            trial[policy_name]["threshold"] = threshold
            replay = replay_policies(validation_records, trial)
            metrics = replay["policy_metrics"][policy_name]
            rank = _rank_threshold(metrics, threshold)
            candidate_rows.append(
                {
                    "threshold": threshold,
                    "metrics": metrics,
                    "selection_rank": list(rank),
                }
            )
            if best_rank is None or rank > best_rank:
                best_rank = rank
                best_threshold = threshold
        if best_threshold is None:
            raise ValueError(
                f"could not select a validation threshold for {policy_name}"
            )
        frozen[policy_name]["threshold"] = best_threshold
        audit["policies"][policy_name] = {
            "selected_threshold": best_threshold,
            "candidates": candidate_rows,
        }
    return frozen, audit


def _run_fit(
    args: argparse.Namespace,
    experiment: Mapping[str, Any],
    state: dict[str, Any],
) -> int:
    """Fit only on train records and freeze thresholds only on validation records."""

    manifest = _require_existing_run(args, state)
    records = read_task_records(manifest)
    train_records = _records_for_split(records, "train")
    validation_records = _records_for_split(records, "validation")
    if any(record.get("split") != "train" for record in train_records):
        raise ValueError("policy fitting accepts train records only")
    selection = _selection_config()
    lambda_cost = float(selection["lambda_cost"])
    mu_latency = float(selection["mu_latency"])
    for record in [*train_records, *validation_records]:
        record["labels"] = build_labels(
            record, lambda_cost=lambda_cost, mu_latency=mu_latency
        )
        if not record["labels"]:
            raise ValueError(
                f"no labels available for task {record['task'].get('task_id')}"
            )
        write_task_record(manifest, record)
    fitted = fit_policy_models(
        train_records,
        lambda_cost=lambda_cost,
        mu_latency=mu_latency,
        target=str(selection["target"]),
        ridge=float(selection["ridge"]),
    )
    frozen, threshold_audit = _freeze_thresholds(
        fitted,
        validation_records,
        list(selection["threshold_candidates"]),
    )
    policy_document = {
        **frozen,
        "fitted_at": utc_now(),
        "fitted_from_run_id": manifest["run_id"],
        "fitting_split": "train",
        "threshold_selection": threshold_audit,
        "selection_parameters": selection,
        "training_task_ids": [
            record["task"].get("task_id") for record in train_records
        ],
        "validation_task_ids": [
            record["task"].get("task_id") for record in validation_records
        ],
    }
    write_json(Path(manifest["trajectory_dir"]) / "policy.json", policy_document)
    write_result(manifest, "policy", policy_document)
    write_result(
        manifest,
        "fit_summary",
        {
            "schema_version": "1.0",
            "run_id": manifest["run_id"],
            "training_records": len(train_records),
            "validation_records": len(validation_records),
            "label_parameters": policy_document["label_parameters"],
            "thresholds": {
                name: policy_document[name]["threshold"]
                for name in ("roundvalue", "task_only")
            },
        },
    )
    updated = update_run_status(manifest, "fit_complete")
    state["manifest"] = updated
    _emit(
        {
            "status": "fit_complete",
            "mode": "fit",
            "run_id": updated["run_id"],
            "result_dir": updated["result_dir"],
        }
    )
    return 0


def _read_frozen_policy(manifest: Mapping[str, Any]) -> dict[str, Any]:
    result_path = Path(manifest["result_dir"]) / "policy.json"
    trajectory_path = Path(manifest["trajectory_dir"]) / "policy.json"
    if result_path.is_file():
        return read_json(result_path)
    if trajectory_path.is_file():
        return read_json(trajectory_path)
    raise FileNotFoundError(
        "frozen policy not found; run --mode fit --run-id <RUN_ID> first"
    )


def _run_evaluate(
    args: argparse.Namespace,
    experiment: Mapping[str, Any],
    state: dict[str, Any],
) -> int:
    """Replay the frozen policy on test trajectories only; no provider exists here."""

    del experiment  # This mode must not construct a provider or alter model settings.
    manifest = _require_existing_run(args, state)
    policy_document = _read_frozen_policy(manifest)
    test_records = _records_for_split(read_task_records(manifest), "test")
    replay = replay_policies(test_records, policy_document)
    replay.update(
        {
            "evaluated_at": utc_now(),
            "run_id": manifest["run_id"],
            "evaluation_split": "test",
        }
    )
    write_result(manifest, "test_policy_replay", replay)
    write_result(
        manifest,
        "evaluation_summary",
        {
            "schema_version": "1.0",
            "run_id": manifest["run_id"],
            "evaluation_split": "test",
            "n_records": replay["n_records"],
            "policy_metrics": replay["policy_metrics"],
        },
    )
    updated = update_run_status(manifest, "evaluate_complete")
    state["manifest"] = updated
    _emit(
        {
            "status": "evaluate_complete",
            "mode": "evaluate",
            "run_id": updated["run_id"],
            "result_dir": updated["result_dir"],
        }
    )
    return 0


def _label_parameters_from_policy_or_defaults(
    manifest: Mapping[str, Any]
) -> tuple[float, float]:
    try:
        policy_document = _read_frozen_policy(manifest)
    except FileNotFoundError:
        selection = _selection_config()
        return float(selection["lambda_cost"]), float(selection["mu_latency"])
    parameters = policy_document.get("label_parameters")
    if not isinstance(parameters, Mapping):
        raise ValueError("frozen policy has no JSON label_parameters object")
    cost = parameters.get("lambda_cost")
    latency = parameters.get("mu_latency")
    if (
        isinstance(cost, bool)
        or not isinstance(cost, int | float)
        or isinstance(latency, bool)
        or not isinstance(latency, int | float)
    ):
        raise ValueError("frozen policy has invalid label parameters")
    return float(cost), float(latency)


def _run_reproduce(
    args: argparse.Namespace,
    experiment: Mapping[str, Any],
    state: dict[str, Any],
) -> int:
    """Rebuild labels and summaries from saved JSON only; no score/API calls occur."""

    manifest = _require_existing_run(args, state)
    records = read_task_records(manifest)
    if not records:
        raise ValueError("run has no saved task records")
    lambda_cost, mu_latency = _label_parameters_from_policy_or_defaults(manifest)
    labels_by_task: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, Any] = {}
    for record in records:
        if not isinstance(record.get("scores"), list):
            task_id = record.get("task", {}).get("task_id")
            raise ValueError(
                f"task {task_id} has no saved scores; reproduce does not rescore"
            )
        labels = build_labels(record, lambda_cost=lambda_cost, mu_latency=mu_latency)
        task_id = str(record.get("task", {}).get("task_id"))
        labels_by_task[task_id] = labels
        summaries[task_id] = label_summary(labels)
    report = _summary(records)
    report.update(
        {
            "reproduced_at": utc_now(),
            "run_id": manifest["run_id"],
            "label_parameters": {"lambda_cost": lambda_cost, "mu_latency": mu_latency},
        }
    )
    write_result(manifest, "reproduced_collection_summary", report)
    write_result(
        manifest,
        "reproduced_labels",
        {
            "schema_version": "1.0",
            "run_id": manifest["run_id"],
            "labels_by_task": labels_by_task,
            "label_summaries_by_task": summaries,
        },
    )
    updated = update_run_status(manifest, "reproduce_complete")
    state["manifest"] = updated
    write_result(updated, "reproducibility_index", reproducibility_index(updated))
    _emit(
        {
            "status": "reproduce_complete",
            "mode": "reproduce",
            "run_id": updated["run_id"],
            "result_dir": updated["result_dir"],
        }
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    state: dict[str, Any] = {"manifest": None}
    try:
        if args.mode in {"smoke", "collect"}:
            experiment = load_experiment_config(
                PROJECT_ROOT,
                model_id=args.model_id,
                topology_id=args.topology_id,
            )
        else:
            if args.model_id or args.topology_id:
                raise ValueError(
                    "--model-id and --topology-id are only valid for smoke or collect; "
                    "offline modes use frozen outputs"
                )
            # Offline work uses the config snapshot captured at collection. It
            # must not silently depend on a subsequently edited topology.json.
            manifest = _require_existing_run(args, state)
            experiment = _frozen_experiment(manifest)
        handlers: dict[
            str, Callable[[argparse.Namespace, Mapping[str, Any], dict[str, Any]], int]
        ] = {
            "smoke": _run_smoke,
            "collect": _run_collect,
            "fit": _run_fit,
            "evaluate": _run_evaluate,
            "reproduce": _run_reproduce,
        }
        return handlers[args.mode](args, experiment, state)
    except KeyboardInterrupt:
        manifest = state.get("manifest")
        if isinstance(manifest, Mapping):
            state["manifest"] = update_run_status(dict(manifest), "interrupted")
        _emit({"status": "interrupted", "mode": args.mode})
        return 130
    except Exception as error:
        manifest = state.get("manifest")
        if isinstance(manifest, Mapping):
            failure_status = f"{args.mode}_failed"
            state["manifest"] = update_run_status(
                dict(manifest),
                failure_status,
                failure={"type": type(error).__name__, "message": _safe_message(error)},
            )
        _emit(
            {
                "status": "failed",
                "mode": args.mode,
                "error": {
                    "type": type(error).__name__,
                    "message": _safe_message(error),
                },
                "run_id": manifest.get("run_id")
                if isinstance(manifest, Mapping)
                else None,
            }
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
