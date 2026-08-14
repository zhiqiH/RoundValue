"""Shared orchestration behind the four sequential step entry points.

The four ``scripts/step*_*.py`` files are the user-facing interface; this
module owns orchestration only.  The implementation lives in ``src/`` and all
durable state is JSON.  Offline stages never construct a provider.
"""

from __future__ import annotations

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
from config_loader import config_snapshot  # noqa: E402
from contracts import file_hash, json_hash, utc_now  # noqa: E402
from debate_runner import FixedDebateRunner  # noqa: E402
from labels import build_labels  # noqa: E402
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
from visualize import build_analysis, render_analysis, write_analysis  # noqa: E402

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


def _emit(value: Mapping[str, Any]) -> None:
    """Print a compact JSON status line suitable for a terminal or a runner."""

    print(json.dumps(dict(value), ensure_ascii=False, sort_keys=True))


def _safe_message(error: BaseException) -> str:
    """Keep terminal failures useful without serializing tracebacks or credentials."""

    message = str(error).replace("\r", " ").replace("\n", " ").strip()
    return message[:500] if message else type(error).__name__


def entrypoint(mode: str, invoke: Callable[[], int], state: dict[str, Any]) -> int:
    """Shared terminal error handling for the four sequential step scripts."""

    try:
        return invoke()
    except KeyboardInterrupt:
        manifest = state.get("manifest")
        if isinstance(manifest, Mapping):
            state["manifest"] = update_run_status(dict(manifest), "interrupted")
        _emit({"status": "interrupted", "mode": mode})
        return 130
    except Exception as error:
        manifest = state.get("manifest")
        if isinstance(manifest, Mapping):
            state["manifest"] = update_run_status(
                dict(manifest),
                f"{mode}_failed",
                failure={"type": type(error).__name__, "message": _safe_message(error)},
            )
        _emit(
            {
                "status": "failed",
                "mode": mode,
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


def _command_line() -> list[str]:
    program = Path(sys.argv[0]).name if sys.argv and sys.argv[0] else "roundvalue"
    return [program, *sys.argv[1:]]


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
    score: bool = True,
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
    if not score:
        # Collection keeps trajectories raw; step3_analyze derives scores
        # offline into results/ and never writes them back here.
        return record
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
    score: bool = True,
) -> dict[str, Any]:
    trajectory = runner.run_trajectory(
        task=dict(task), run_id=run_id, max_rounds=max_rounds
    )
    return _task_record(
        task,
        split,
        trajectory,
        allow_local_code_evaluation=allow_local_code_evaluation,
        score=score,
    )


def _write_frozen_splits(
    manifest: Mapping[str, Any], split_by_task: Mapping[str, str]
) -> None:
    write_json(
        Path(manifest["trajectory_dir"]) / "frozen_splits.json",
        {
            "schema_version": "1.0",
            "split_seed": SPLIT_SEED,
            "splits": split_by_task,
        },
    )


def _record_is_complete(record: Mapping[str, Any]) -> bool:
    """Return true only for a fully collected, scored trajectory.

    A failed provider call, malformed Writer JSON, or missing scorer output
    all make a record eligible for recollection during a resume.
    """

    trajectory = record.get("trajectory")
    if not isinstance(trajectory, Mapping) or trajectory.get("status") != "complete":
        return False
    if record.get("scoring_error"):
        return False
    scores = record.get("scores")
    if "scores" not in record:
        # A raw trajectory collected without scoring is complete when its
        # trajectory is complete; step3_analyze derives scores offline.
        return True
    if not isinstance(scores, list) or not scores:
        return False
    for score in scores:
        quality = score.get("quality") if isinstance(score, Mapping) else None
        if (
            isinstance(quality, bool)
            or not isinstance(quality, int | float)
            or not math.isfinite(float(quality))
        ):
            return False
    return True


def _open_resumable_run(
    args: argparse.Namespace, mode: str
) -> dict[str, Any] | None:
    """Return an existing run eligible for resume, or ``None`` for a new run."""

    if not args.run_id:
        return None
    try:
        manifest = open_run(PROJECT_ROOT, args.run_id)
    except FileNotFoundError:
        return None
    if manifest.get("mode") != mode:
        raise ValueError(
            f"run {args.run_id} was started in mode {manifest.get('mode')!r}; "
            f"resuming mode {mode!r} requires a {mode!r} run"
        )
    return manifest


def _validate_resume_consistency(
    manifest: Mapping[str, Any],
    split_by_task: Mapping[str, str],
) -> None:
    """Refuse to resume against a changed benchmark or split assignment."""

    trajectory_dir = Path(manifest["trajectory_dir"])
    frozen = read_json(trajectory_dir / "frozen_splits.json")
    if frozen.get("splits") != dict(split_by_task):
        raise ValueError(
            "resume aborted: the frozen split assignment no longer matches the benchmark"
        )
    provenance = read_json(trajectory_dir / "benchmark_provenance.json")
    for relative, expected_hash in provenance.get("files", {}).items():
        if file_hash(PROJECT_ROOT / relative) != expected_hash:
            raise ValueError(
                f"resume aborted: benchmark source changed since the run started: {relative}"
            )


def _verify_smoke_gate(
    smoke_run_id: str | None, experiment: Mapping[str, Any]
) -> dict[str, Any]:
    """Require a passing smoke run before formal collection starts."""

    if not smoke_run_id:
        raise ValueError("--smoke-run-id is required; run step1_smoke.py first")
    smoke_manifest = open_run(PROJECT_ROOT, smoke_run_id)
    if smoke_manifest.get("mode") != "smoke":
        raise ValueError(
            f"run {smoke_run_id} is not a smoke run "
            f"(mode={smoke_manifest.get('mode')!r}); rerun step1_smoke.py"
        )
    task_count = smoke_manifest.get("task_count")
    complete_count = smoke_manifest.get("complete_task_count")
    failed_ids = smoke_manifest.get("failed_task_ids") or []
    if (
        not isinstance(task_count, int)
        or task_count < 1
        or complete_count != task_count
        or failed_ids
    ):
        raise ValueError(
            f"smoke run {smoke_run_id} did not pass every task; rerun step1_smoke.py"
        )
    if smoke_manifest.get("config_hash") != json_hash(config_snapshot(PROJECT_ROOT)):
        raise ValueError(
            "configs changed since the smoke run; rerun step1_smoke.py first"
        )
    if smoke_manifest.get("selected_model_id") != experiment.get("model_id"):
        raise ValueError(
            "smoke used a different model profile; rerun step1_smoke.py with the same profile"
        )
    if smoke_manifest.get("selected_topology_id") != experiment.get("topology_id"):
        raise ValueError(
            "smoke used a different topology; rerun step1_smoke.py with the same topology"
        )
    return smoke_manifest


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
    failed_task_ids = [
        record["task"]["task_id"]
        for record in records
        if record["trajectory"].get("status") != "complete"
        or record.get("scoring_error")
        or len(record.get("scores", [])) != 1
        or (record.get("scores") or [{}])[0].get("quality") != 1
    ]
    updated = update_run_status(
        manifest,
        status,
        task_count=len(records),
        complete_task_count=sum(
            record["trajectory"].get("status") == "complete" for record in records
        ),
        failed_task_ids=failed_task_ids,
    )
    state["manifest"] = updated
    _emit(
        {
            "status": status,
            "mode": "smoke",
            "run_id": updated["run_id"],
            "failed_task_ids": failed_task_ids,
            "trajectory_dir": updated["trajectory_dir"],
            "result_dir": updated["result_dir"],
        }
    )
    return 0 if succeeded else 1


def _run_collect(
    args: argparse.Namespace,
    experiment: Mapping[str, Any],
    state: dict[str, Any],
    *,
    score: bool = True,
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
    resuming_manifest = _open_resumable_run(args, "collect")
    if resuming_manifest is not None:
        manifest = resuming_manifest
        state["manifest"] = manifest
        _validate_resume_consistency(manifest, split_by_task)
        existing_records = read_task_records(manifest)
    else:
        manifest = _create_run(args, experiment, state)
        _write_benchmark_snapshot(manifest, benchmark_path, benchmark_document)
        _write_frozen_splits(manifest, split_by_task)
        existing_records = []
    complete_by_task = {
        str(record.get("task", {}).get("task_id")): record
        for record in existing_records
        if _record_is_complete(record)
    }
    provider = build_provider(dict(experiment))
    records: list[dict[str, Any]] = []
    resumed_task_ids: list[str] = []
    try:
        runner = FixedDebateRunner(dict(experiment), provider)
        max_rounds = int(experiment["topology"]["max_rounds"])
        for task in tasks:
            split = split_by_task[task["task_id"]]
            prior = complete_by_task.get(str(task["task_id"]))
            if prior is not None:
                records.append(prior)
                resumed_task_ids.append(str(task["task_id"]))
                continue
            record = _collect_task(
                runner,
                task=task,
                split=split,
                run_id=str(manifest["run_id"]),
                max_rounds=max_rounds,
                allow_local_code_evaluation=args.allow_local_code_evaluation,
                score=score,
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
    failure_details = [
        {
            "task_id": str(record.get("task", {}).get("task_id")),
            "split": record.get("split"),
            "trajectory_status": record.get("trajectory", {}).get("status"),
            "failure_reason": record.get("trajectory", {}).get("failure_reason"),
            "scoring_error": record.get("scoring_error"),
        }
        for record in failed
    ]
    write_result(
        manifest,
        "failure_details",
        {"schema_version": "1.0", "failures": failure_details},
    )
    status = "collect_complete" if not failed else "collect_failed"
    updated = update_run_status(
        manifest,
        status,
        task_count=len(records),
        complete_task_count=len(records) - len(failed),
        failed_task_ids=[record["task"].get("task_id") for record in failed],
        resumed_task_ids=resumed_task_ids,
        resumed=bool(resuming_manifest),
    )
    state["manifest"] = updated
    _emit(
        {
            "status": status,
            "mode": "collect",
            "run_id": updated["run_id"],
            "task_count": len(records),
            "failed_task_count": len(failed),
            "failed_task_ids": [record["task"].get("task_id") for record in failed],
            "resumed_task_count": len(resumed_task_ids),
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


def _run_analyze(
    args: argparse.Namespace,
    experiment: Mapping[str, Any],
    state: dict[str, Any],
) -> int:
    """Offline step3: score, label, fit, select thresholds, and evaluate.

    This function reads trajectories only, derives every result into
    ``results/<run_id>/``, and never writes back to ``trajectories/``.  It
    makes no provider calls.
    """

    del experiment  # Analysis is fully offline and uses the frozen run inputs.
    manifest = _require_existing_run(args, state)
    if manifest.get("mode") != "collect":
        raise ValueError(
            "step3_analyze requires a run collected by step2_collect "
            f"(mode={manifest.get('mode')!r})"
        )
    records = read_task_records(manifest)
    if not records:
        raise ValueError("run has no saved task records")
    frozen = read_json(Path(manifest["trajectory_dir"]) / "frozen_splits.json")
    expected_splits = frozen.get("splits")
    if not isinstance(expected_splits, Mapping):
        raise ValueError("run has no frozen split assignment")
    collected_ids = {
        str(record.get("task", {}).get("task_id")) for record in records
    }
    missing = sorted(set(expected_splits) - collected_ids)
    extra = sorted(collected_ids - set(expected_splits))
    if missing or extra:
        raise ValueError(
            "trajectory coverage mismatch: "
            f"missing={missing}, extra={extra}; resume with step2_collect.py"
        )
    incomplete = [
        str(record.get("task", {}).get("task_id"))
        for record in records
        if record.get("trajectory", {}).get("status") != "complete"
        or record.get("scoring_error")
    ]
    if incomplete:
        raise ValueError(
            f"incomplete trajectories: {incomplete}; resume with step2_collect.py"
        )
    has_code = any(
        str(record.get("task", {}).get("domain", "")).casefold() == "code"
        for record in records
    )
    if has_code and not args.allow_local_code_evaluation:
        raise ValueError(
            "run contains code tasks; pass --allow-local-code-evaluation "
            "for offline deterministic scoring"
        )

    selection = _selection_config()
    lambda_cost = float(selection["lambda_cost"])
    mu_latency = float(selection["mu_latency"])
    scored_records: list[dict[str, Any]] = []
    labels_by_task: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        task_id = str(record.get("task", {}).get("task_id"))
        scored = dict(record)
        scores = _score_record(
            scored, allow_local_code_evaluation=args.allow_local_code_evaluation
        )
        for score in scores:
            quality = score.get("quality")
            if (
                isinstance(quality, bool)
                or not isinstance(quality, int | float)
                or not math.isfinite(float(quality))
            ):
                raise ValueError(
                    f"task {task_id} has an unscored or invalid checkpoint"
                )
        scored["scores"] = scores
        labels = build_labels(
            scored, lambda_cost=lambda_cost, mu_latency=mu_latency
        )
        scored["labels"] = labels
        labels_by_task[task_id] = labels
        scored_records.append(scored)

    train_records = [
        record for record in scored_records if record.get("split") == "train"
    ]
    validation_records = [
        record for record in scored_records if record.get("split") == "validation"
    ]
    test_records = [
        record for record in scored_records if record.get("split") == "test"
    ]
    if not train_records or not validation_records or not test_records:
        raise ValueError("run must contain train, validation, and test records")

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
    replay = replay_policies(test_records, frozen)
    replay.update(
        {
            "evaluated_at": utc_now(),
            "run_id": manifest["run_id"],
            "evaluation_split": "test",
        }
    )
    analysis = build_analysis(
        dict(manifest),
        scored_records,
        label_parameters=(lambda_cost, mu_latency),
        replay=replay,
    )

    write_result(
        manifest,
        "scores",
        {
            "schema_version": "1.0",
            "run_id": manifest["run_id"],
            "scorer_note": "deterministic offline scores derived from saved trajectories",
            "scores_by_task": {
                str(record["task"]["task_id"]): [dict(score) for score in record["scores"]]
                for record in scored_records
            },
        },
    )
    write_result(
        manifest,
        "labels",
        {
            "schema_version": "1.0",
            "run_id": manifest["run_id"],
            "label_parameters": {"lambda_cost": lambda_cost, "mu_latency": mu_latency},
            "labels_by_task": labels_by_task,
        },
    )
    write_result(manifest, "policy", policy_document)
    write_result(manifest, "test_policy_replay", replay)
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
    write_analysis(dict(manifest), analysis)
    updated = update_run_status(
        manifest,
        "analyze_complete",
        analyzed_task_count=len(scored_records),
        training_records=len(train_records),
        validation_records=len(validation_records),
        test_records=len(test_records),
    )
    state["manifest"] = updated
    write_result(updated, "reproducibility_index", reproducibility_index(updated))
    _emit(
        {
            "status": "analyze_complete",
            "mode": "analyze",
            "run_id": updated["run_id"],
            "analyzed_task_count": len(scored_records),
            "result_dir": updated["result_dir"],
        }
    )
    return 0


def _run_visualize(
    args: argparse.Namespace,
    experiment: Mapping[str, Any],
    state: dict[str, Any],
) -> int:
    """Offline step4: render saved results into CSV, HTML, and a conclusion."""

    del experiment  # Visualization reads results only, never trajectories.
    manifest = _require_existing_run(args, state)
    paths = render_analysis(dict(manifest))
    updated = update_run_status(manifest, "visualize_complete")
    state["manifest"] = updated
    _emit(
        {
            "status": "visualize_complete",
            "mode": "visualize",
            "run_id": updated["run_id"],
            "result_dir": updated["result_dir"],
            "html_report": paths["html"],
            "task_csv": paths["csv"],
            "summary": paths["summary"],
        }
    )
    return 0
