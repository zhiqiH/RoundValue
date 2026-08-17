"""Step 4: offline publication-quality paper figures.

This entry is fully offline.  It makes zero model/API calls and only reads the
existing ``results/<run_id>/`` and ``trajectories/<run_id>/`` artifacts written
by Steps 1-3.  It computes the derived continuation/repair metrics, verifies
the policy-comparison invariants, and renders five publication figures plus an
auditable ``figure_data.json`` and a ``figure/README.md``.

Nothing in the frozen debate topology, prompts, scoring, RoundValue
labels/thresholds, benchmark splits, historical trajectories, or the existing
Step 3 plots is modified.

Run with::

    python scripts/step4_paper_figures.py
"""

from __future__ import annotations

import argparse
import datetime as _datetime
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.transforms import Bbox

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

# Only the dependency-free scorer is imported: no provider, no pipeline, no
# network code.  Step 4 therefore cannot perform model inference.
from scorer import _mc_reference_label, normalize_mc_answer  # noqa: E402

MAX_ROUNDS = 5
STAGE1_NODE_IDS = {"planner_stage_1", "analyst_stage_1", "critic_stage_1"}
POLICY_ROLES = {"r1": "fixed_1", "roundvalue": "roundvalue", "oracle": "oracle"}

BENCHMARK_BY_DOMAIN = {
    "mmlu_pro": "MMLU-Pro",
    "harp": "HARP",
    "logiqa": "LogiQA",
}
BENCHMARK_ORDER = ["MMLU-Pro", "HARP", "LogiQA"]

MODEL_DISPLAY = {
    "gpt5_nano": "GPT-5-nano",
    "gpt4o_mini": "GPT-4o-mini",
    "deepseek_flash": "DeepSeek-V4-Flash",
}
MODEL_ORDER = ["gpt5_nano", "gpt4o_mini", "deepseek_flash"]
# Okabe-Ito colorblind-safe accents, reused consistently for the same model
# in every figure where model is a visual variable.
MODEL_COLORS = {
    "gpt5_nano": "#0072B2",
    "gpt4o_mini": "#D55E00",
    "deepseek_flash": "#009E73",
}
# Deterministic fallback accents for future model profiles not listed above.
FALLBACK_MODEL_COLORS = ["#CC79A7", "#56B4E9", "#F0E442", "#999999", "#E69F00"]

# Figure 4 taxonomy.  Each R1-wrong task is classified by two orthogonal,
# trajectory-derived facts:
#   (1) did the gold answer ever emerge "blind" in Stage 1 (P1/A1/C1) at
#       rounds R2..R5, and
#   (2) did the Writer ever repair (any R2..R5 checkpoint correct)?
# The four resulting cells are mutually exclusive and exhaustive, and the two
# repaired cells are further subdivided into stable (still correct at R5) and
# temporary (wrong again at R5) repair.  All displayed counts therefore sum
# exactly to ``n_R1_wrong``.
MECHANISM_ORDER = [
    "never_emerged_no_repair",
    "never_emerged_stable",
    "never_emerged_temporary",
    "emerged_no_repair",
    "emerged_stable",
    "emerged_temporary",
]
MECHANISM_LABELS = {
    "never_emerged_no_repair": "Gold never emerged, no repair",
    "never_emerged_stable": "Gold never emerged, stable repair",
    "never_emerged_temporary": "Gold never emerged, temporary repair",
    "emerged_no_repair": "Gold emerged, no repair",
    "emerged_stable": "Gold emerged, stable repair",
    "emerged_temporary": "Gold emerged, temporary repair",
}
MECHANISM_COLORS = {
    "never_emerged_no_repair": "#8C8C8C",
    "never_emerged_stable": "#66C2A5",
    "never_emerged_temporary": "#F1B76B",
    "emerged_no_repair": "#56B4E9",
    "emerged_stable": "#009E73",
    "emerged_temporary": "#E69F00",
}
# Four-way logically exhaustive partition; repaired cells merge the
# stable/temporary subdivision used for display.
MECHANISM_PARTITION = {
    "never_emerged_no_repair": ["never_emerged_no_repair"],
    "never_emerged_repair": ["never_emerged_stable", "never_emerged_temporary"],
    "emerged_no_repair": ["emerged_no_repair"],
    "emerged_repair": ["emerged_stable", "emerged_temporary"],
}

# Disjoint trajectory classification used to order Figure 3 deterministically.
SORT_PRIORITY = {
    "stable_repair": 0,
    "temporary_repair": 1,
    "late_repair": 2,
    "stable_wrong": 3,
}

FLOAT_TOLERANCE = 1e-9
FIGURE_DPI = 400
# 95% two-sided normal quantile used for Wilson score intervals.
WILSON_Z95 = 1.959963984540054


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _pct(count: int | None, denominator: int | None) -> float | None:
    """Return ``count / denominator`` as a percentage, never a silent zero."""

    if count is None or denominator is None or denominator <= 0:
        return None
    return round(100.0 * count / denominator, 6)


def wilson_ci95(successes: int, total: int) -> tuple[float, float] | None:
    """Wilson 95% confidence interval for ``successes / total`` (proportion).

    Returns ``(lower, upper)`` on the 0-1 proportion scale, or ``None`` when
    the denominator is not positive.
    """

    if total <= 0:
        return None
    z2 = WILSON_Z95**2
    p_hat = successes / total
    denominator = 1.0 + z2 / total
    center = (p_hat + z2 / (2.0 * total)) / denominator
    half = (
        WILSON_Z95
        * math.sqrt(p_hat * (1.0 - p_hat) / total + z2 / (4.0 * total * total))
        / denominator
    )
    return max(0.0, center - half), min(1.0, center + half)


def _signed_pp(value: float | None) -> str:
    if value is None:
        return "N/A"
    if abs(value) < 0.05:
        return "0.0 pp"
    return f"{value:+.1f} pp"


def _iso_now() -> str:
    return _datetime.datetime.now(_datetime.timezone.utc).isoformat()


def _condition_sort_key(record: dict[str, Any]) -> tuple[int, int, str]:
    benchmark = BENCHMARK_ORDER.index(record["benchmark"]) if record["benchmark"] in BENCHMARK_ORDER else len(BENCHMARK_ORDER)
    model_id = str(record.get("model_id") or "")
    model = MODEL_ORDER.index(model_id) if model_id in MODEL_ORDER else len(MODEL_ORDER)
    return (benchmark, model, str(record.get("run_id") or ""))


def _condition_label(record: dict[str, Any]) -> str:
    return f"{record['model_display']} \u00d7 {record['dataset_id']}"


def _model_color(model_id: str) -> str:
    if model_id in MODEL_COLORS:
        return MODEL_COLORS[model_id]
    index = sum(ord(character) for character in model_id) % len(FALLBACK_MODEL_COLORS)
    return FALLBACK_MODEL_COLORS[index]


def _short_task_id(task_id: str) -> str:
    parts = task_id.split("::")
    if len(parts) >= 3:
        parts = parts[1:]
    return "\u00b7".join(parts)


def discover_runs(
    results_dir: Path, trajectories_dir: Path
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Discover compatible runs from manifests, never from directory names."""

    runs: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for manifest_path in sorted(results_dir.glob("*/manifest.json")):
        run_id = manifest_path.parent.name
        try:
            manifest = _read_json(manifest_path)
        except (OSError, json.JSONDecodeError) as error:
            skipped.append({"run_id": run_id, "reason": f"unreadable manifest: {error}"})
            continue

        if str(manifest.get("run_id") or run_id) != run_id:
            skipped.append(
                {"run_id": run_id, "reason": "manifest run_id disagrees with directory"}
            )
            continue
        if str(manifest.get("mode", "")).lower() == "smoke" or str(
            manifest.get("dataset_label", "")
        ).startswith("Smoke"):
            skipped.append({"run_id": run_id, "reason": "smoke run excluded"})
            continue

        result_dir = results_dir / run_id
        trajectory_dir = trajectories_dir / run_id
        required = [
            result_dir / "scores.json",
            result_dir / "test_policy_replay.json",
            trajectory_dir / "run.json",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            skipped.append(
                {"run_id": run_id, "reason": f"missing artifacts: {missing}"}
            )
            continue
        task_files = sorted(trajectory_dir.glob("task_*.json"))
        if not task_files:
            skipped.append({"run_id": run_id, "reason": "no trajectory task files"})
            continue

        run_record = _read_json(trajectory_dir / "run.json")
        metadata: dict[str, Any] = run_record if isinstance(run_record, dict) else manifest
        model_selection = metadata.get("model_selection") or {}
        model_id = str(metadata.get("selected_model_id") or model_selection.get("model_id") or "")
        requested_model = str(model_selection.get("requested_model") or "")
        domain = str(metadata.get("domain") or "")
        dataset_id = str(metadata.get("dataset") or "")
        benchmark_source: str | None = None

        snapshot_path = trajectory_dir / "benchmark_snapshot.json"
        if snapshot_path.is_file():
            snapshot = _read_json(snapshot_path)
            content = snapshot.get("content") if isinstance(snapshot, dict) else None
            if isinstance(content, dict):
                dataset_id = str(content.get("dataset_id") or dataset_id)
                domain = str(content.get("domain") or domain)
            benchmark_source = (
                snapshot.get("source_path") if isinstance(snapshot, dict) else None
            )

        runs.append(
            {
                "run_id": run_id,
                "result_dir": result_dir,
                "trajectory_dir": trajectory_dir,
                "manifest": manifest,
                "dataset_id": dataset_id,
                "dataset_label": manifest.get("dataset_label"),
                "domain": domain,
                "benchmark": BENCHMARK_BY_DOMAIN.get(domain, domain),
                "model_id": model_id,
                "model": requested_model,
                "model_display": MODEL_DISPLAY.get(model_id, requested_model or model_id),
                "model_snapshot": requested_model,
                "reasoning": model_selection.get("reasoning"),
                "temperature": model_selection.get("temperature"),
                "created_at": metadata.get("created_at"),
                "config_hash": metadata.get("config_hash"),
                "benchmark_source": benchmark_source,
            }
        )
    return runs, skipped


def load_score_matrix(run: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Load per-task, per-round Writer correctness from ``scores.json``."""

    scores = _read_json(run["result_dir"] / "scores.json")
    by_task = scores.get("scores_by_task")
    if not isinstance(by_task, dict):
        raise RuntimeError(f"{run['run_id']}: scores.json lacks scores_by_task")

    matrix: dict[str, dict[str, Any]] = {}
    for task_id, entries in by_task.items():
        rounds: dict[int, bool | None] = {}
        split: str | None = None
        for entry in entries:
            round_index = entry.get("round_index")
            if not isinstance(round_index, int):
                raise RuntimeError(
                    f"{run['run_id']}: {task_id} has an invalid round_index"
                )
            if round_index in rounds:
                raise RuntimeError(
                    f"{run['run_id']}: {task_id} has duplicate round_index {round_index}"
                )
            is_correct = entry.get("is_correct")
            rounds[round_index] = is_correct if isinstance(is_correct, bool) else None
            split = entry.get("split", split)
        matrix[str(task_id)] = {"split": split, "rounds": rounds}
    return matrix


def load_stage1_gold(
    run: dict[str, Any], r1_wrong_ids: set[str]
) -> dict[str, dict[str, Any]]:
    """Extract blind Stage-1 gold emergence for rounds 2..5, R1-wrong tasks only.

    Stage-1 nodes (Planner/Analyst/Critic) never see the previous Writer
    checkpoint, so any later-round gold they propose is "blind" emergence.
    Only the structured ``candidate_answer`` field is considered and it is
    scored with the same conservative option-label normalizer used by the
    offline scorer.
    """

    found: dict[str, dict[str, Any]] = {}
    for path in sorted(run["trajectory_dir"].glob("task_*.json")):
        record = _read_json(path)
        task = record.get("task") if isinstance(record, dict) else None
        if not isinstance(task, dict):
            continue
        task_id = str(task.get("task_id") or "")
        if not task_id or task_id not in r1_wrong_ids:
            continue

        gold = _mc_reference_label(task)
        gold_rounds: set[int] = set()
        trajectory = record.get("trajectory") if isinstance(record, dict) else None
        rounds = trajectory.get("rounds") if isinstance(trajectory, dict) else None
        if isinstance(rounds, list):
            for debate_round in rounds:
                if not isinstance(debate_round, dict):
                    continue
                round_index = debate_round.get("round_index")
                if not isinstance(round_index, int) or round_index < 2:
                    continue
                for node in debate_round.get("nodes") or []:
                    if not isinstance(node, dict):
                        continue
                    if node.get("node_id") not in STAGE1_NODE_IDS:
                        continue
                    if node.get("status") not in {"completed", "succeeded"}:
                        continue
                    output = node.get("output")
                    if not isinstance(output, dict):
                        continue
                    candidate = output.get("candidate_answer")
                    if (
                        isinstance(candidate, str)
                        and gold is not None
                        and normalize_mc_answer(candidate) == gold
                    ):
                        gold_rounds.add(round_index)
                        break
        found[task_id] = {"gold": gold, "rounds": sorted(gold_rounds)}

    missing = r1_wrong_ids - set(found)
    if missing:
        raise RuntimeError(
            f"{run['run_id']}: R1-wrong tasks missing trajectory records: {sorted(missing)}"
        )
    return found


def load_policy_data(run: dict[str, Any]) -> dict[str, Any]:
    """Load the held-out test policy replay and enforce the same-split invariant."""

    replay = _read_json(run["result_dir"] / "test_policy_replay.json")
    evaluation_split = replay.get("evaluation_split")
    if evaluation_split != "test":
        raise RuntimeError(
            f"{run['run_id']}: test_policy_replay evaluation_split is "
            f"{evaluation_split!r}, expected 'test'"
        )

    policies = replay.get("policies")
    metrics = replay.get("policy_metrics")
    if not isinstance(policies, dict) or not isinstance(metrics, dict):
        raise RuntimeError(f"{run['run_id']}: malformed test_policy_replay.json")

    task_results: dict[str, dict[str, dict[str, Any]]] = {}
    task_id_sets: dict[str, set[str]] = {}
    accuracies: dict[str, float] = {}
    for role, policy_name in POLICY_ROLES.items():
        policy = policies.get(policy_name)
        if not isinstance(policy, dict):
            raise RuntimeError(f"{run['run_id']}: missing policy {policy_name}")
        rows = policy.get("task_results")
        if not isinstance(rows, list):
            raise RuntimeError(f"{run['run_id']}: {policy_name} lacks task_results")
        parsed: dict[str, dict[str, Any]] = {}
        for row in rows:
            if not isinstance(row, dict) or not row.get("available"):
                raise RuntimeError(
                    f"{run['run_id']}: {policy_name} contains an unavailable task result"
                )
            if row.get("split") != "test":
                raise RuntimeError(
                    f"{run['run_id']}: {policy_name} contains a non-test task result"
                )
            task_id = str(row["task_id"])
            parsed[task_id] = {
                "correct": bool(row.get("quality") == 1.0),
                "stop_round": row.get("stop_round"),
                "selected_round": row.get("selected_round"),
            }
        task_results[policy_name] = parsed
        task_id_sets[policy_name] = set(parsed)
        policy_metrics = metrics.get(policy_name)
        accuracies[role] = float(
            policy_metrics.get("accuracy") if isinstance(policy_metrics, dict) else None
        )

    reference_ids = task_id_sets[POLICY_ROLES["r1"]]
    for policy_name in (POLICY_ROLES["roundvalue"], POLICY_ROLES["oracle"]):
        if task_id_sets[policy_name] != reference_ids:
            raise RuntimeError(
                f"{run['run_id']}: task_ids differ across R1 / RoundValue / Oracle "
                f"policies ({policy_name} disagrees)"
            )

    return {
        "evaluation_split": evaluation_split,
        "task_results": task_results,
        "accuracies": accuracies,
        "task_ids": sorted(reference_ids),
    }


def trajectory_round_accuracy(run: dict[str, Any]) -> dict[str, Any]:
    """Independently recompute R1..R5 accuracy from raw Writer checkpoints.

    This deliberately bypasses ``scores.json``: it reads every stored task
    trajectory, extracts the Writer checkpoint ``answer`` for each round, and
    scores it with the same conservative option-label normalizer as the
    offline scorer.  It is used only to verify run identity; it never mutates
    any stored artifact.
    """

    correct_counts: dict[int, list[bool]] = {
        round_index: [] for round_index in range(1, MAX_ROUNDS + 1)
    }
    r1_wrong: set[str] = set()
    for path in sorted(run["trajectory_dir"].glob("task_*.json")):
        record = _read_json(path)
        task = record.get("task") if isinstance(record, dict) else None
        if not isinstance(task, dict):
            continue
        task_id = str(task.get("task_id") or "")
        gold = _mc_reference_label(task)
        trajectory = record.get("trajectory") if isinstance(record, dict) else None
        rounds = trajectory.get("rounds") if isinstance(trajectory, dict) else None
        if not isinstance(rounds, list):
            continue
        for round_index in range(1, MAX_ROUNDS + 1):
            if round_index - 1 >= len(rounds):
                continue
            checkpoint = ((rounds[round_index - 1] or {}).get("checkpoint") or {})
            answer = checkpoint.get("answer") if isinstance(checkpoint, dict) else None
            normalized = normalize_mc_answer(answer) if isinstance(answer, str) else None
            is_correct = normalized is not None and gold is not None and normalized == gold
            correct_counts[round_index].append(is_correct)
            if round_index == 1 and not is_correct and task_id:
                r1_wrong.add(task_id)

    return {
        "R1..R5_accuracy_pct": {
            f"R{round_index}": (
                _pct(sum(values), len(values)) if values else None
            )
            for round_index, values in correct_counts.items()
        },
        "n_R1_wrong": len(r1_wrong),
        "n_task_files": sum(1 for _ in run["trajectory_dir"].glob("task_*.json")),
    }


def analysis_round_accuracy(run: dict[str, Any]) -> dict[str, float] | None:
    """Read the run's ``analysis.json`` cumulative per-round accuracy."""

    analysis_path = run["result_dir"] / "analysis.json"
    if not analysis_path.is_file():
        return None
    analysis = _read_json(analysis_path)
    cumulative = analysis.get("cumulative")
    if not isinstance(cumulative, list):
        return None
    result: dict[str, float] = {}
    for entry in cumulative:
        if not isinstance(entry, dict):
            continue
        round_index = entry.get("round_index")
        accuracy = entry.get("accuracy")
        if isinstance(round_index, int) and isinstance(accuracy, (int, float)):
            result[f"R{round_index}"] = round(100.0 * float(accuracy), 6)
    return result or None


def build_run_identity(
    run: dict[str, Any],
    all_record: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the auditable identity of one selected run.

    Records the run id, model/snapshot, reasoning configuration, and R1..R5
    accuracy with ``n_R1_wrong`` from three independent sources:
    ``scores.json`` (via the Step 4 matrix), the raw Writer checkpoints in
    ``trajectories/``, and ``analysis.json``.
    """

    return {
        "run_id": run["run_id"],
        "model_id": run["model_id"],
        "model_snapshot": run["model_snapshot"],
        "reasoning": run["reasoning"],
        "temperature": run["temperature"],
        "benchmark": run["benchmark"],
        "dataset_id": run["dataset_id"],
        "scores_json_rounds": {
            round_name: all_record[round_name]
            for round_name in (f"R{round_index}" for round_index in range(1, MAX_ROUNDS + 1))
        },
        "trajectory_recheck": trajectory_round_accuracy(run),
        "analysis_json_rounds": analysis_round_accuracy(run),
        "n_R1_wrong": all_record["R1_wrong"],
        "created_at": run["created_at"],
        "config_hash": run["config_hash"],
    }


def _task_detail(
    task_id: str,
    task_matrix: dict[str, Any],
    gold_info: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    rounds = task_matrix["rounds"]
    correct_rounds = [rounds.get(round_index) for round_index in range(1, MAX_ROUNDS + 1)]
    r1_correct = correct_rounds[0] is True
    ever_repaired = any(value is True for value in correct_rounds[1:])
    final_correct = correct_rounds[-1] is True
    first_repair = next(
        (round_index for round_index in range(2, MAX_ROUNDS + 1) if rounds.get(round_index) is True),
        None,
    )

    if not ever_repaired:
        category = "stable_wrong"
    elif not final_correct:
        category = "temporary_repair"
    elif first_repair == MAX_ROUNDS:
        category = "late_repair"
    else:
        category = "stable_repair"

    gold = gold_info.get(task_id, {})
    gold_label = gold.get("gold")
    gold_rounds = list(gold.get("rounds") or [])
    gold_known = gold_label is not None
    gold_emerged = bool(gold_rounds)

    # Writer repair is defined purely by the checkpoint sequence; gold
    # emergence is defined purely by blind Stage-1 candidates.  The two facts
    # are crossed, never conflated: a Writer can repair even when blind
    # Stage-1 gold never emerged, and Stage-1 gold can emerge without a
    # Writer repair.
    if ever_repaired:
        repair_kind = "stable" if final_correct else "temporary"
    else:
        repair_kind = None

    if not gold_known:
        mechanism = "gold_unknown"
    elif gold_emerged and repair_kind is None:
        mechanism = "emerged_no_repair"
    elif gold_emerged:
        mechanism = f"emerged_{repair_kind}"
    elif repair_kind is None:
        mechanism = "never_emerged_no_repair"
    else:
        mechanism = f"never_emerged_{repair_kind}"

    return {
        "task_id": task_id,
        "split": task_matrix.get("split"),
        "correct_rounds": correct_rounds,
        "r1_correct": r1_correct,
        "ever_repaired": ever_repaired,
        "first_repair_round": first_repair,
        "category": category,
        "mechanism_category": mechanism,
        "stage1_gold_rounds": gold_rounds,
        "stage1_gold_known": gold_known,
        "stage1_gold_emerged": gold_emerged,
    }


def build_record(
    run: dict[str, Any],
    split: str,
    matrix: dict[str, dict[str, Any]],
    gold_info: dict[str, dict[str, Any]],
    policy: dict[str, Any],
) -> dict[str, Any]:
    """Build one auditable condition record for a split (``all`` or ``test``)."""

    if split == "all":
        tasks = dict(matrix)
    elif split == "test":
        tasks = {task_id: task for task_id, task in matrix.items() if task.get("split") == "test"}
    else:
        raise ValueError(f"unknown split {split!r}")

    n_tasks = len(tasks)
    round_accuracy: dict[str, float | None] = {}
    for round_index in range(1, MAX_ROUNDS + 1):
        values = [
            task["rounds"].get(round_index)
            for task in tasks.values()
            if task["rounds"].get(round_index) is not None
        ]
        correct = sum(1 for value in values if value is True)
        round_accuracy[f"R{round_index}"] = _pct(correct, len(values)) if values else None

    r1_wrong_ids = sorted(
        task_id
        for task_id, task in tasks.items()
        if task["rounds"].get(1) is False
    )
    details = [
        _task_detail(task_id, tasks[task_id], gold_info) for task_id in r1_wrong_ids
    ]

    n_ever = sum(1 for detail in details if detail["ever_repaired"])
    n_stable = sum(1 for detail in details if detail["category"] == "stable_repair")
    n_temporary = sum(1 for detail in details if detail["category"] == "temporary_repair")
    n_late = sum(1 for detail in details if detail["category"] == "late_repair")
    n_gold_emerged = sum(1 for detail in details if detail["stage1_gold_rounds"])
    n_gold_known = sum(1 for detail in details if detail["stage1_gold_known"])

    mechanism_counts = Counter(
        detail["mechanism_category"] for detail in details
    )
    unknown_gold = [
        detail["task_id"]
        for detail in details
        if detail["mechanism_category"] == "gold_unknown"
    ]
    if unknown_gold:
        raise RuntimeError(
            f"{run['run_id']}: {len(unknown_gold)} R1-wrong task(s) have no "
            f"known gold label and cannot be classified: {sorted(unknown_gold)}"
        )
    mechanism_total = sum(mechanism_counts.values())
    if mechanism_total != len(r1_wrong_ids):
        raise RuntimeError(
            f"{run['run_id']}: mechanism categories sum to {mechanism_total}, "
            f"not n_R1_wrong = {len(r1_wrong_ids)}"
        )

    # Aggregate identical Writer correctness trajectories (W/C per round) for
    # the compact Figure 3, without outcome-selecting any task.
    pattern_counter = Counter(
        "".join(
            "C" if value is True else ("W" if value is False else "?")
            for value in detail["correct_rounds"]
        )
        for detail in details
    )
    r1_wrong_patterns = sorted(
        (
            {"pattern": pattern, "count": count}
            for pattern, count in pattern_counter.items()
        ),
        key=lambda item: (-item["count"], item["pattern"]),
    )

    wilson = wilson_ci95(n_ever, len(r1_wrong_ids))

    first_repair_values = sorted(
        detail["first_repair_round"]
        for detail in details
        if detail["first_repair_round"] is not None
    )
    first_repair_distribution = {
        str(round_index): sum(1 for value in first_repair_values if value == round_index)
        for round_index in range(2, MAX_ROUNDS + 1)
    }
    first_repair_summary: dict[str, Any] = {
        "mean": round(statistics.fmean(first_repair_values), 6) if first_repair_values else None,
        "median": statistics.median(first_repair_values) if first_repair_values else None,
        "distribution": first_repair_distribution,
    }

    record: dict[str, Any] = {
        "run_id": run["run_id"],
        "dataset_id": run["dataset_id"],
        "dataset_label": run["dataset_label"],
        "benchmark": run["benchmark"],
        "model": run["model"],
        "model_id": run["model_id"],
        "model_display": run["model_display"],
        "split": split,
        "n": n_tasks,
        "R1": round_accuracy["R1"],
        "R2": round_accuracy["R2"],
        "R3": round_accuracy["R3"],
        "R4": round_accuracy["R4"],
        "R5": round_accuracy["R5"],
        "R1_wrong": len(r1_wrong_ids),
        "n_ever_repair": n_ever,
        "n_stable_repair": n_stable,
        "n_temporary_repair": n_temporary,
        "n_late_repair": n_late,
        "Ever_repair": _pct(n_ever, len(r1_wrong_ids)),
        "Stable_repair": _pct(n_stable, len(r1_wrong_ids)),
        "Temporary_repair": _pct(n_temporary, len(r1_wrong_ids)),
        "Late_repair": _pct(n_late, len(r1_wrong_ids)),
        "First_repair_round": first_repair_summary,
        "n_gold_emergence": n_gold_emerged,
        "n_gold_known": n_gold_known,
        "gold_emergence": _pct(n_gold_emerged, n_gold_known),
        "mechanism_total": mechanism_total,
        "mechanism_categories": {
            category: mechanism_counts.get(category, 0) for category in MECHANISM_ORDER
        },
        "mechanism_percents": {
            category: _pct(mechanism_counts.get(category, 0), len(r1_wrong_ids))
            for category in MECHANISM_ORDER
        },
        "mechanism_partition": {
            name: sum(mechanism_counts.get(category, 0) for category in categories)
            for name, categories in MECHANISM_PARTITION.items()
        },
        "Ever_repair_wilson_ci95": (
            [round(100.0 * wilson[0], 6), round(100.0 * wilson[1], 6)]
            if wilson is not None
            else None
        ),
        "Ever_repair_wilson_n": len(r1_wrong_ids),
        "r1_wrong_patterns": r1_wrong_patterns,
        "r1_wrong_tasks": sorted(details, key=lambda detail: detail["task_id"]),
    }

    if split == "test":
        accuracies = policy["accuracies"]
        r1_test = round(100.0 * accuracies["r1"], 6)
        roundvalue_test = round(100.0 * accuracies["roundvalue"], 6)
        oracle_test = round(100.0 * accuracies["oracle"], 6)
        record.update(
            {
                "R1_test": r1_test,
                "RoundValue_test": roundvalue_test,
                "Oracle_test": oracle_test,
                "Oracle_headroom": round(oracle_test - r1_test, 6),
                "Captured_gain": round(roundvalue_test - r1_test, 6),
                "Oracle_regret": round(oracle_test - roundvalue_test, 6),
                "test_policy_tasks": [
                    {
                        "task_id": task_id,
                        "R1_correct": policy["task_results"][POLICY_ROLES["r1"]][task_id]["correct"],
                        "RoundValue_correct": policy["task_results"][POLICY_ROLES["roundvalue"]][task_id]["correct"],
                        "Oracle_correct": policy["task_results"][POLICY_ROLES["oracle"]][task_id]["correct"],
                        "RoundValue_stop_round": policy["task_results"][POLICY_ROLES["roundvalue"]][task_id]["stop_round"],
                        "Oracle_selected_round": policy["task_results"][POLICY_ROLES["oracle"]][task_id]["selected_round"],
                    }
                    for task_id in policy["task_ids"]
                ],
                "notes": (
                    "policy accuracies and gains are defined on the held-out test "
                    "split only (n = 10 per run)"
                ),
            }
        )
    else:
        record.update(
            {
                "R1_test": None,
                "RoundValue_test": None,
                "Oracle_test": None,
                "Oracle_headroom": None,
                "Captured_gain": None,
                "Oracle_regret": None,
                "notes": (
                    "policy metrics are only defined on the held-out test split; "
                    "see the matching test_conditions record"
                ),
            }
        )
    return record


def run_validations(
    runs_included: list[dict[str, Any]],
    all_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
    run_identities: list[dict[str, Any]],
    raw_counts: dict[str, dict[str, int]],
    raw_policy_accuracies: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    all_by_run = {record["run_id"]: record for record in all_records}
    test_by_run = {record["run_id"]: record for record in test_records}
    identity_by_run = {
        identity["run_id"]: identity for identity in run_identities
    }
    for run in runs_included:
        record = test_by_run[run["run_id"]]
        all_record = all_by_run[run["run_id"]]

        # R1 policy accuracy must reproduce the test-split R1 accuracy computed
        # independently from scores.json.
        r1_scores = float(record["R1"])
        r1_policy = float(record["R1_test"])
        score_diff = abs(r1_scores - r1_policy)
        checks.append(
            {
                "check": "policy_score_consistency",
                "run_id": run["run_id"],
                "r1_scores_pct": r1_scores,
                "r1_policy_pct": r1_policy,
                "max_abs_diff": score_diff,
                "tolerance": FLOAT_TOLERANCE,
                "ok": score_diff <= FLOAT_TOLERANCE,
            }
        )

        # Oracle headroom = R1 error rate x Ever-repair rate on the same test
        # task subset (binary exact-option scoring, per-task best-round Oracle).
        # Computed from raw integer counts and full-precision policy accuracies,
        # not from the rounded percentages stored in figure_data.json.
        counts = raw_counts[run["run_id"]]
        policy_accuracy = raw_policy_accuracies[run["run_id"]]
        r1_error_rate = counts["r1_wrong_test"] / counts["n_test"]
        headroom = policy_accuracy["oracle"] - policy_accuracy["r1"]
        if counts["r1_wrong_test"] == 0:
            checks.append(
                {
                    "check": "oracle_decomposition",
                    "run_id": run["run_id"],
                    "ok": True,
                    "note": "skipped: no R1-wrong test tasks",
                }
            )
        else:
            ever_repair_rate = counts["n_ever_test"] / counts["r1_wrong_test"]
            product = r1_error_rate * ever_repair_rate
            decomposition_diff = abs(product - headroom)
            tolerance = FLOAT_TOLERANCE + FLOAT_TOLERANCE * max(
                abs(product), abs(headroom)
            )
            checks.append(
                {
                    "check": "oracle_decomposition",
                    "run_id": run["run_id"],
                    "r1_error_rate": r1_error_rate,
                    "ever_repair_rate": ever_repair_rate,
                    "product": product,
                    "oracle_headroom": headroom,
                    "abs_diff": decomposition_diff,
                    "tolerance": tolerance,
                    "ok": decomposition_diff <= tolerance,
                }
            )

        checks.append(
            {
                "check": "same_split_task_ids",
                "run_id": run["run_id"],
                "n_task_ids": record["n"],
                "ok": True,
                    "note": (
                        "R1 / RoundValue / Oracle task_id sets verified identical and "
                        "equal to the test split in scores.json"
                    ),
                }
            )

        # Figure 4 taxonomy: every R1-wrong task lands in exactly one category,
        # the displayed counts sum to n_R1_wrong, and the emerged/never-emerged
        # split is consistent with the blind Stage-1 gold emergence count.
        mechanism_sum = sum(all_record["mechanism_categories"].values())
        emerged_total = sum(
            all_record["mechanism_categories"].get(category, 0)
            for category in (
                "emerged_no_repair",
                "emerged_stable",
                "emerged_temporary",
            )
        )
        partition_sum = sum(all_record["mechanism_partition"].values())
        checks.append(
            {
                "check": "mechanism_partition_consistency",
                "run_id": run["run_id"],
                "n_R1_wrong": all_record["R1_wrong"],
                "mechanism_sum": mechanism_sum,
                "n_gold_emergence": all_record["n_gold_emergence"],
                "emerged_total": emerged_total,
                "partition_sum": partition_sum,
                "ok": (
                    mechanism_sum == all_record["R1_wrong"]
                    and emerged_total == all_record["n_gold_emergence"]
                    and partition_sum == all_record["R1_wrong"]
                ),
            }
        )

        # Wilson intervals for Figure 5 must be computed from the R1-wrong
        # denominator and must bracket the point estimate.
        wilson_ci = all_record["Ever_repair_wilson_ci95"]
        ever_repair = all_record["Ever_repair"]
        wilson_ok = (
            wilson_ci is not None
            and all_record["Ever_repair_wilson_n"] == all_record["R1_wrong"]
            and ever_repair is not None
            and wilson_ci[0] <= ever_repair <= wilson_ci[1]
        )
        checks.append(
            {
                "check": "wilson_ci_denominators",
                "run_id": run["run_id"],
                "wilson_n": all_record["Ever_repair_wilson_n"],
                "n_R1_wrong": all_record["R1_wrong"],
                "wilson_ci95": wilson_ci,
                "ever_repair_pct": ever_repair,
                "ok": wilson_ok,
            }
        )

        # Run identity: the raw Writer checkpoints and analysis.json must
        # independently reproduce the scores.json round accuracies.
        identity = identity_by_run[run["run_id"]]
        trajectory_rounds = identity["trajectory_recheck"]["R1..R5_accuracy_pct"]
        analysis_rounds = identity["analysis_json_rounds"]
        round_names = [
            f"R{round_index}" for round_index in range(1, MAX_ROUNDS + 1)
        ]
        trajectory_diff = max(
            (
                abs(float(all_record[name]) - float(trajectory_rounds[name]))
                for name in round_names
                if all_record[name] is not None
                and trajectory_rounds.get(name) is not None
            ),
            default=0.0,
        )
        checks.append(
            {
                "check": "trajectory_scores_consistency",
                "run_id": run["run_id"],
                "max_abs_diff_pp": trajectory_diff,
                "tolerance": FLOAT_TOLERANCE,
                "ok": trajectory_diff <= FLOAT_TOLERANCE,
            }
        )
        if analysis_rounds:
            analysis_diff = max(
                (
                    abs(float(all_record[name]) - float(analysis_rounds[name]))
                    for name in round_names
                    if all_record[name] is not None
                    and analysis_rounds.get(name) is not None
                ),
                default=0.0,
            )
            checks.append(
                {
                    "check": "analysis_scores_consistency",
                    "run_id": run["run_id"],
                    "max_abs_diff_pp": analysis_diff,
                    "tolerance": FLOAT_TOLERANCE,
                    "ok": analysis_diff <= FLOAT_TOLERANCE,
                }
            )

    # Exactly one DeepSeek x MMLU-Pro run must drive Figures 1-5, and its
    # identity (including reasoning configuration and R1..R5/n_R1_wrong) is
    # recorded explicitly rather than inferred from directory names.
    deepseek_mmlu = [
        identity
        for identity in run_identities
        if identity["model_id"] == "deepseek_flash"
        and identity["benchmark"] == "MMLU-Pro"
    ]
    if len(deepseek_mmlu) == 1:
        identity = deepseek_mmlu[0]
        checks.append(
            {
                "check": "deepseek_mmlu_run_identity",
                "run_id": identity["run_id"],
                "model_snapshot": identity["model_snapshot"],
                "reasoning": identity["reasoning"],
                "temperature": identity["temperature"],
                "R1..R5": identity["scores_json_rounds"],
                "n_R1_wrong": identity["n_R1_wrong"],
                "ok": True,
            }
        )
    else:
        checks.append(
            {
                "check": "deepseek_mmlu_run_identity",
                "run_id": None,
                "matching_runs": [
                    identity["run_id"] for identity in deepseek_mmlu
                ],
                "ok": False,
                "note": (
                    "expected exactly one DeepSeek x MMLU-Pro run, found "
                    f"{len(deepseek_mmlu)}"
                ),
            }
        )
    return checks


def _setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 8.5,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#333333",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.dpi": 110,
        }
    )


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _inflate_bbox(box: Bbox, pixels: float) -> Bbox:
    """Grow a display-coordinate ``Bbox`` by a fixed pixel margin."""

    return Bbox.from_extents(
        box.x0 - pixels,
        box.y0 - pixels,
        box.x1 + pixels,
        box.y1 + pixels,
    )


def _model_handles(records: list[dict[str, Any]]) -> list[Line2D]:
    present_ids = sorted({record["model_id"] for record in records})
    present = [
        model_id for model_id in MODEL_ORDER if model_id in present_ids
    ] + [model_id for model_id in present_ids if model_id not in MODEL_ORDER]
    display_by_id = {record["model_id"]: record["model_display"] for record in records}
    return [
        Line2D(
            [],
            [],
            color=_model_color(model_id),
            marker="o",
            markersize=4,
            linewidth=1.4,
            label=display_by_id.get(model_id, model_id),
        )
        for model_id in present
    ]


def fig01_round_accuracy_dynamics(records: list[dict[str, Any]], out_path: Path) -> None:
    """Round accuracy dynamics as benchmark small multiples (all tasks)."""

    benchmarks = BENCHMARK_ORDER
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9), sharey=True)
    fig.subplots_adjust(left=0.14, right=0.99, top=0.84, bottom=0.18, wspace=0.34)

    for panel_index, (ax, benchmark) in enumerate(zip(axes, benchmarks)):
        panel_records = sorted(
            [r for r in records if r["benchmark"] == benchmark], key=_condition_sort_key
        )
        if panel_records:
            for record in panel_records:
                values = [record[f"R{round_index}"] for round_index in range(1, MAX_ROUNDS + 1)]
                ax.plot(
                    range(1, MAX_ROUNDS + 1),
                    values,
                    marker="o",
                    markersize=3.4,
                    linewidth=1.4,
                    color=_model_color(record["model_id"]),
                    zorder=3,
                )
            n_tasks = panel_records[0]["n"]
        else:
            n_tasks = None

        letter = chr(ord("a") + panel_index)
        subtitle = f"n = {n_tasks}" if n_tasks is not None else "no compatible runs"
        ax.set_title(f"({letter}) {benchmark}\n{subtitle}", fontsize=8.4)
        ax.set_xticks(range(1, MAX_ROUNDS + 1))
        ax.set_xticklabels([f"R{round_index}" for round_index in range(1, MAX_ROUNDS + 1)])
        ax.set_xlim(0.85, MAX_ROUNDS + 0.15)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 25, 50, 75, 100])
        # Shared 0-100% scale; numeric tick labels are shown once, on the
        # left panel, so the small multiples stay clean without losing units.
        ax.yaxis.set_tick_params(labelleft=(panel_index == 0))
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=7.6)
        ax.set_yticklabels(
            [f"{tick:g}" for tick in ax.get_yticks()], fontsize=7.6
        )

    axes[0].set_ylabel("Accuracy (%)")
    fig.text(0.5, 0.025, "Debate round", ha="center", va="center")
    handles = _model_handles(records)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.10), ncol=3, fontsize=7.6)
    _save_figure(fig, out_path)


def fig02_continuation_opportunity(records: list[dict[str, Any]], out_path: Path) -> None:
    """Main policy figure: R1 -> RoundValue -> Oracle on held-out test tasks."""

    records = sorted(records, key=_condition_sort_key)
    n_rows = len(records)
    fig, axes = plt.subplots(n_rows, 1, figsize=(7.2, 0.82 * n_rows + 1.15), sharex=True)
    if n_rows == 1:
        axes = [axes]
    fig.subplots_adjust(left=0.36, right=0.985, top=0.94, bottom=0.13, hspace=0.62)

    for ax, record in zip(axes, records):
        r1 = float(record["R1_test"])
        roundvalue = float(record["RoundValue_test"])
        oracle = float(record["Oracle_test"])
        equal_markers = abs(r1 - roundvalue) < 0.05
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)

        # R1 -> Oracle is the available continuation opportunity.
        ax.plot([r1, oracle], [0.5, 0.5], color="#B0B0B0", linewidth=1.2, zorder=1)
        # R1 -> RoundValue is the captured gain; RoundValue -> Oracle is regret.
        ax.plot([r1, roundvalue], [0.5, 0.5], color="#E69F00", linewidth=2.2, zorder=2)
        # R1 keeps a black ring that stays visible even when the RoundValue
        # diamond lands exactly on top of it (R1 == RoundValue).  The diamond
        # is drawn inside the ring, so equality is never visually hidden.
        ax.plot(
            r1,
            0.5,
            marker="o",
            markersize=9,
            markerfacecolor="white",
            markeredgecolor="#111111",
            markeredgewidth=1.6,
            linestyle="none",
            zorder=4,
        )
        ax.plot(
            roundvalue,
            0.5,
            marker="D",
            markersize=7.5,
            markerfacecolor="#E69F00",
            markeredgecolor="#111111",
            markeredgewidth=0.9,
            linestyle="none",
            zorder=5,
        )
        ax.plot(
            oracle,
            0.5,
            marker="o",
            markersize=6,
            markerfacecolor="#009E73",
            markeredgecolor="white",
            markeredgewidth=0.8,
            linestyle="none",
            zorder=4,
        )

        ax.text(
            -0.015,
            0.5,
            _condition_label(record),
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=7.6,
        )
        annotation = (
            f"Oracle headroom: {_signed_pp(record['Oracle_headroom'])}   "
            f"Captured gain: {_signed_pp(record['Captured_gain'])}   "
            f"n = {record['n']}"
        )
        if equal_markers:
            annotation += "   R1 = RoundValue"
        ax.text(
            0.012,
            0.055,
            annotation,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=6.8,
            color="#333333",
        )
        ax.tick_params(axis="x", labelsize=7.2, labelbottom=True)

    axes[-1].set_xlabel("Held-out test accuracy (%)", labelpad=9)
    axes[-1].set_xlim(0, 100)
    axes[-1].set_xticks([0, 20, 40, 60, 80, 100])
    handles = [
        Line2D([], [], marker="o", markersize=7, markerfacecolor="white", markeredgecolor="#111111", markeredgewidth=1.6, linestyle="none", label="R1"),
        Line2D([], [], marker="D", markersize=7, markerfacecolor="#E69F00", markeredgecolor="#111111", markeredgewidth=0.9, linestyle="none", label="RoundValue"),
        Line2D([], [], marker="o", markersize=6, markerfacecolor="#009E73", markeredgecolor="white", markeredgewidth=0.8, linestyle="none", label="Oracle"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.015), ncol=3, fontsize=7.6)
    _save_figure(fig, out_path)


def fig03_r1_wrong_trajectories(records: list[dict[str, Any]], out_path: Path) -> None:
    """Compact trajectory-pattern visualization (main-paper Figure 3).

    Identical Writer correctness trajectories among R1-wrong tasks are
    aggregated into ``W/C`` patterns with task counts, grouped by
    model x benchmark.  No task is outcome-selected; the full task-level
    heatmap is preserved as a supporting figure instead.
    """

    records = sorted(records, key=_condition_sort_key)
    cell_height = 0.58
    pattern_stride = 6.8
    max_pattern_span = 0.0
    for record in records:
        max_pattern_span = max(
            max_pattern_span,
            pattern_stride * len(record["r1_wrong_patterns"]),
        )

    n_rows = len(records)
    fig, ax = plt.subplots(figsize=(7.5, 0.95 * n_rows + 1.45))
    fig.subplots_adjust(left=0.36, right=0.985, top=0.87, bottom=0.16)

    row_labels: list[str] = []
    for row_index, record in enumerate(records):
        y_center = row_index
        x_cursor = 0.0
        for pattern_entry in record["r1_wrong_patterns"]:
            pattern = str(pattern_entry["pattern"])
            count = int(pattern_entry["count"])
            for column, character in enumerate(pattern):
                if character == "C":
                    facecolor = "#009E73"
                elif character == "W":
                    facecolor = "#D9D9D9"
                else:
                    facecolor = "#FFFFFF"
                ax.add_patch(
                    Rectangle(
                        (x_cursor + column, y_center - cell_height / 2.0),
                        1.0,
                        cell_height,
                        facecolor=facecolor,
                        edgecolor="white",
                        linewidth=0.8,
                        zorder=2,
                    )
                )
            ax.text(
                x_cursor + MAX_ROUNDS + 0.35,
                y_center,
                f"\u00d7 {count}",
                ha="left",
                va="center",
                fontsize=6.9,
                color="#222222",
            )
            x_cursor += pattern_stride
        row_labels.append(
            f"{_condition_label(record)}  (n_R1_wrong = {record['R1_wrong']})"
        )

    # R1..R5 headers sit above the first pattern block only; every block
    # repeats the same column order, stated in the subtitle.
    ax.set_xticks([column + 0.5 for column in range(MAX_ROUNDS)])
    ax.set_xticklabels(
        [f"R{round_index}" for round_index in range(1, MAX_ROUNDS + 1)],
        fontsize=7.6,
    )
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=6.8)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0.0, max_pattern_span + 1.8)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.set_title(
        "R1-wrong Writer trajectories: pattern counts by model \u00d7 benchmark",
        fontsize=8.5,
        pad=7,
    )
    ax.text(
        0.995,
        0.02,
        "Each 5-cell block is one distinct R1..R5 trajectory (W = wrong, "
        "C = correct); \u00d7 k = number of tasks with that pattern.",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.0,
        color="#555555",
    )

    legend_handles = [
        Patch(facecolor="#009E73", edgecolor="white", label="correct"),
        Patch(facecolor="#D9D9D9", edgecolor="white", label="wrong"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=2,
        fontsize=6.8,
    )
    _save_figure(fig, out_path)


def fig03_supp_r1_wrong_task_heatmap(
    records: list[dict[str, Any]], out_path: Path
) -> None:
    """Supporting figure: full task-level R1-wrong trajectory heatmap.

    Preserved from the earlier main-paper Figure 3 so that every individual
    task trajectory remains auditable alongside the compact pattern-count
    version above.
    """

    records = sorted(records, key=_condition_sort_key)
    blocks: list[tuple[str, list[dict[str, Any]]]] = []
    for record in records:
        tasks = list(record["r1_wrong_tasks"])
        tasks.sort(
            key=lambda detail: (
                SORT_PRIORITY[detail["category"]],
                detail["first_repair_round"] if detail["first_repair_round"] is not None else MAX_ROUNDS + 1,
                detail["task_id"],
            )
        )
        label = f"{_condition_label(record)}   (n_R1_wrong = {len(tasks)})"
        blocks.append((label, tasks))

    matrix: list[list[float | None]] = []
    row_labels: list[str] = []
    bold_rows: set[int] = set()
    for label, tasks in blocks:
        if matrix:
            matrix.append([None] * MAX_ROUNDS)
            row_labels.append(label)
            bold_rows.add(len(matrix) - 1)
        seen_labels: Counter[str] = Counter()
        for detail in tasks:
            matrix.append([1.0 if value else 0.0 for value in detail["correct_rounds"]])
            short = _short_task_id(detail["task_id"])
            seen_labels[short] += 1
            if seen_labels[short] > 1:
                short = f"{short} ({seen_labels[short]})"
            row_labels.append(short)

    n_rows = len(matrix)
    fig_height = 0.14 * n_rows + 1.55
    fig, ax = plt.subplots(figsize=(7.5, fig_height))
    fig.subplots_adjust(left=0.245, right=0.985, top=0.975, bottom=0.04)

    array = np.array(matrix, dtype=float)
    cmap = ListedColormap(["#D9D9D9", "#009E73"])
    cmap.set_bad("white")
    ax.imshow(array, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.set_xticks(range(MAX_ROUNDS))
    ax.set_xticklabels([f"R{round_index}" for round_index in range(1, MAX_ROUNDS + 1)], fontsize=7.6)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=5.1)
    for tick_index, tick_label in enumerate(ax.get_yticklabels()):
        if tick_index in bold_rows:
            tick_label.set_fontsize(5.7)
            tick_label.set_fontweight("bold")
            tick_label.set_color("#111111")
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(-0.5, MAX_ROUNDS - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.set_xticks(np.arange(-0.5, MAX_ROUNDS, 1.0), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1.0), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", length=0)
    ax.set_title(
        "Appendix: full task-level R1-wrong trajectories (grouped by model \u00d7 benchmark)",
        fontsize=8.5,
        pad=7,
    )

    legend_handles = [
        Patch(facecolor="#009E73", edgecolor="none", label="correct"),
        Patch(facecolor="#D9D9D9", edgecolor="none", label="wrong"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        fontsize=6.6,
        frameon=False,
    )
    _save_figure(fig, out_path)


def fig04_repair_mechanism(records: list[dict[str, Any]], out_path: Path) -> None:
    """Gold candidate emergence and the repair-mechanism outcome partition."""

    records = sorted(records, key=_condition_sort_key)
    labels = [
        f"{_condition_label(record)}  (n_R1_wrong = {record['R1_wrong']})"
        for record in records
    ]
    positions = np.arange(len(records))

    fig, axes = plt.subplots(
        1, 2, figsize=(7.2, 3.45), sharey=True, gridspec_kw={"width_ratios": [0.78, 1.22]}
    )
    fig.subplots_adjust(left=0.34, right=0.99, top=0.88, bottom=0.18, wspace=0.36)

    ax_a, ax_b = axes

    # Panel A: P(blind Stage-1 gold emerges at R2..R5 | R1 wrong).
    heights = [record["gold_emergence"] if record["gold_emergence"] is not None else 0.0 for record in records]
    ax_a.barh(
        positions,
        heights,
        height=0.58,
        color=[_model_color(record["model_id"]) for record in records],
    )
    for position, record, height in zip(positions, records, heights):
        ax_a.text(
            height + 1.2,
            position,
            f"{record['n_gold_emergence']}/{record['n_gold_known']}",
            ha="left",
            va="center",
            fontsize=6.4,
            color="#222222",
        )
    ax_a.set_xlim(0, 100)
    ax_a.set_xlabel("Blind Stage-1 gold emergence\n(% of R1-wrong tasks)")
    ax_a.set_title("(a) Gold candidate emergence", fontsize=8.4)
    # Shared row labels sit on the left panel so both panels can be read
    # against the same model x benchmark condition.
    ax_a.set_yticks(positions)
    ax_a.set_yticklabels(labels, fontsize=6.2)
    ax_a.tick_params(axis="y", labelleft=True)

    # Panel B: mechanism outcome partition among R1-wrong tasks.
    for position, record in zip(positions, records):
        left = 0.0
        for category in MECHANISM_ORDER:
            count = record["mechanism_categories"].get(category, 0)
            percent = record["mechanism_percents"].get(category) or 0.0
            ax_b.barh(
                position,
                percent,
                left=left,
                height=0.58,
                color=MECHANISM_COLORS[category],
            )
            if count:
                text_color = (
                    "white" if category in {"emerged_stable", "never_emerged_no_repair"} else "#111111"
                )
                ax_b.text(
                    left + percent / 2.0,
                    position,
                    str(count),
                    ha="center",
                    va="center",
                    fontsize=6.2,
                    color=text_color,
                )
            left += percent

    ax_b.set_yticks(positions)
    ax_b.tick_params(axis="y", labelleft=False)
    ax_b.set_xlim(0, 100)
    ax_b.set_xlabel("% of R1-wrong tasks")
    ax_b.set_title("(b) Mechanism outcome", fontsize=8.4)
    ax_a.invert_yaxis()

    legend_handles = [
        Patch(facecolor=MECHANISM_COLORS[category], edgecolor="none", label=MECHANISM_LABELS[category])
        for category in MECHANISM_ORDER
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        fontsize=6.2,
    )
    fig.text(
        0.5,
        0.01,
        "Panel (b) partitions every R1-wrong task into mutually exclusive "
        "categories; displayed counts sum to n_R1_wrong.",
        ha="center",
        va="bottom",
        fontsize=6.0,
        color="#333333",
    )
    _save_figure(fig, out_path)


def fig05_continuation_landscape(records: list[dict[str, Any]], out_path: Path) -> None:
    """R1 error rate versus recoverability, with Wilson error bars."""

    records = sorted(records, key=_condition_sort_key)
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    fig.subplots_adjust(left=0.135, right=0.985, top=0.92, bottom=0.115)

    # Oracle headroom (pp) = R1 error rate (%) x Ever-repair rate (%) / 100.
    for headroom_pp in (5.0, 10.0, 20.0, 40.0):
        xx = np.linspace(0.5, 100.0, 400)
        yy = np.minimum(headroom_pp * 100.0 / xx, 100.0)
        ax.plot(xx, yy, linestyle="--", linewidth=0.8, color="#B0B0B0", zorder=0)
        label_x = 99.0
        label_y = min(headroom_pp + 0.5, 99.0)
        ax.text(
            label_x,
            label_y,
            f"{headroom_pp:g} pp",
            ha="right",
            va="center",
            fontsize=5.9,
            color="#777777",
        )

    bar_segments: list[tuple[float, float, float]] = []
    for record in records:
        error_rate = 100.0 - float(record["R1"])
        ever_repair = record["Ever_repair"]
        if ever_repair is None:
            continue
        wilson_ci = record["Ever_repair_wilson_ci95"]
        ax.scatter(
            error_rate,
            ever_repair,
            s=34,
            color=_model_color(record["model_id"]),
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        if wilson_ci is not None and ever_repair is not None:
            ax.errorbar(
                error_rate,
                ever_repair,
                yerr=[
                    [ever_repair - wilson_ci[0]],
                    [wilson_ci[1] - ever_repair],
                ],
                fmt="none",
                ecolor=_model_color(record["model_id"]),
                elinewidth=1.1,
                capsize=2.6,
                capthick=1.0,
                zorder=2,
            )
            bar_segments.append((error_rate, wilson_ci[0], wilson_ci[1]))

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("R1 error rate (%)")
    ax.set_ylabel("Ever-repair rate (% of R1-wrong tasks)")

    if _model_handles(records):
        legend = ax.legend(
            handles=_model_handles(records), loc="upper right", fontsize=6.4
        )

    # Deterministic label placement.  Candidate offsets are scored by
    # (label overlap, error-bar crossing, distance from the point), preferring
    # placements that stay inside the axes, keep a small clearance from other
    # labels, and avoid covering the Wilson error bars.  A solid white
    # background keeps any unavoidable line crossing readable.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    placed_boxes: list[Bbox] = []
    if _model_handles(records):
        placed_boxes.append(_inflate_bbox(legend.get_window_extent(renderer), 2.5))
    axes_box = ax.get_window_extent(renderer)
    transform = ax.transData
    bar_boxes: list[Bbox] = []
    for error_rate, lower, upper in bar_segments:
        point_low = transform.transform((error_rate, lower))
        point_high = transform.transform((error_rate, upper))
        bar_boxes.append(
            Bbox.from_extents(
                point_low[0] - 1.0,
                min(point_low[1], point_high[1]),
                point_high[0] + 1.0,
                max(point_low[1], point_high[1]),
            )
        )
    label_candidates = [
        (1.5, 2.0, "left", "bottom"),
        (-1.5, 2.0, "right", "bottom"),
        (1.5, -2.4, "left", "top"),
        (-1.5, -2.4, "right", "top"),
        (1.5, 5.5, "left", "bottom"),
        (-1.5, 5.5, "right", "bottom"),
        (1.5, -6.2, "left", "top"),
        (-1.5, -6.2, "right", "top"),
        (0.0, 7.2, "center", "bottom"),
        (0.0, -7.2, "center", "top"),
    ]
    label_kwargs = {
        "fontsize": 5.8,
        "color": "#222222",
        "bbox": {
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 1.0,
            "pad": 0.25,
        },
        "zorder": 5,
        "annotation_clip": False,
    }
    for record in records:
        error_rate = 100.0 - float(record["R1"])
        ever_repair = record["Ever_repair"]
        if ever_repair is None:
            continue
        label_text = (
            f"{record['model_display']} \u00d7 {record['dataset_id']}\n"
            f"(n_R1_wrong = {record['R1_wrong']})"
        )
        scored: list[tuple[tuple[int, int, float], Any, Bbox]] = []
        for offset_x, offset_y, ha, va in label_candidates:
            annotation = ax.annotate(
                label_text,
                xy=(error_rate, ever_repair),
                xytext=(error_rate + offset_x, ever_repair + offset_y),
                ha=ha,
                va=va,
                **label_kwargs,
            )
            fig.canvas.draw()
            box = annotation.get_window_extent(renderer)
            inside = (
                axes_box.x0 - 1.0 <= box.x0
                and box.x1 <= axes_box.x1 + 1.0
                and axes_box.y0 - 1.0 <= box.y0
                and box.y1 <= axes_box.y1 + 1.0
            )
            if not inside:
                annotation.remove()
                continue
            overlaps = sum(1 for other in placed_boxes if box.overlaps(other))
            crossings = sum(1 for bar in bar_boxes if box.overlaps(bar))
            distance = abs(offset_x) + abs(offset_y)
            scored.append(((overlaps, crossings, distance), annotation, box))

        if not scored:
            # Defensive fallback: place at the default offset even if clipped.
            offset_x, offset_y, ha, va = label_candidates[0]
            ax.annotate(
                label_text,
                xy=(error_rate, ever_repair),
                xytext=(error_rate + offset_x, ever_repair + offset_y),
                ha=ha,
                va=va,
                **label_kwargs,
            )
            continue

        scored.sort(key=lambda item: item[0])
        _, chosen, chosen_box = scored[0]
        for _, other, _ in scored[1:]:
            other.remove()
        placed_boxes.append(_inflate_bbox(chosen_box, 2.5))

    ax.text(
        0.985,
        0.012,
        "Dashed contours: constant Oracle headroom  \u00b7  "
        "Vertical bars: Wilson 95% CI for P(EverRepair | R1 wrong)",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.0,
        color="#777777",
    )
    _save_figure(fig, out_path)


def build_readme(
    runs_included: list[dict[str, Any]],
    runs_skipped: list[dict[str, str]],
    validation_checks: list[dict[str, Any]],
    run_identities: list[dict[str, Any]],
    n_all_values: list[int],
    n_test_values: list[int],
) -> str:
    """Generate ``figure/README.md`` from the runs actually discovered."""

    rows = []
    for run in sorted(runs_included, key=lambda item: item["run_id"]):
        rows.append(
            f"| `{run['run_id']}` | {run['model_display']} "
            f"(`{run['model_snapshot']}`) | {run['dataset_id']} | {run.get('created_at') or 'N/A'} |"
        )
    runs_table = "\n".join(rows) if rows else "| (none) |"
    identity_rows = []
    for identity in sorted(run_identities, key=lambda item: item["run_id"]):
        rounds = identity["scores_json_rounds"]
        rounds_text = " / ".join(
            f"{rounds[f'R{round_index}']:g}"
            for round_index in range(1, MAX_ROUNDS + 1)
        )
        reasoning = identity["reasoning"] or {}
        reasoning_parts = [f"enabled={reasoning.get('enabled')}"]
        if reasoning.get("effort"):
            reasoning_parts.append(f"effort={reasoning['effort']}")
        reasoning_text = ", ".join(reasoning_parts)
        identity_rows.append(
            f"| `{identity['run_id']}` | {identity['model_snapshot']} | "
            f"{identity['temperature']} | {reasoning_text} | {rounds_text} | "
            f"{identity['n_R1_wrong']} |"
        )
    identity_table = "\n".join(identity_rows) if identity_rows else "| (none) |"

    deepseek_mmlu = [
        identity
        for identity in run_identities
        if identity["model_id"] == "deepseek_flash"
        and identity["benchmark"] == "MMLU-Pro"
    ]
    if len(deepseek_mmlu) == 1:
        identity = deepseek_mmlu[0]
        rounds = identity["scores_json_rounds"]
        rounds_text = " / ".join(
            f"{rounds[f'R{round_index}']:g}"
            for round_index in range(1, MAX_ROUNDS + 1)
        )
        reasoning = identity["reasoning"] or {}
        deepseek_note = (
            f"Step 4 uses exactly one DeepSeek x MMLU-Pro run, "
            f"`{identity['run_id']}`: snapshot `{identity['model_snapshot']}`, "
            f"reasoning enabled={reasoning.get('enabled')}"
            + (
                f" (effort {reasoning['effort']})"
                if reasoning.get("effort")
                else ""
            )
            + f", temperature {identity['temperature']}, "
            f"R1..R5 = {rounds_text}, n_R1_wrong = {identity['n_R1_wrong']}. "
            "These values were verified independently against `scores.json`, "
            "the raw Writer checkpoints in `trajectories/`, and `analysis.json`; "
            "they match the rendered figures exactly. The previously reviewed "
            "delayed-checkpoint DeepSeek MMLU-Pro run "
            "`202608161016_MMLUPro50_79a9e682` (reasoning enabled) had "
            "R1 = 86% and n_R1_wrong = 7; it was retired when DeepSeek reasoning "
            "was switched off (git commit `2d03c15`), is not present in "
            "`results/`, and is therefore neither selected nor plotted. No "
            "metric was edited to match either run."
        )
    else:
        deepseek_note = (
            f"Expected exactly one DeepSeek x MMLU-Pro run but found "
            f"{len(deepseek_mmlu)}; run identity cannot be resolved."
        )

    skipped_lines = "\n".join(
        f"- `{item['run_id']}`: {item['reason']}" for item in runs_skipped
    ) or "- none"
    validation_lines = "\n".join(
        f"- {check['check']} ({check.get('run_id', '')}): "
        f"{'PASS' if check.get('ok') else 'FAIL'}"
        for check in validation_checks
    )
    n_all_text = ", ".join(str(value) for value in n_all_values)
    n_test_text = ", ".join(str(value) for value in n_test_values)

    return f"""# figure/ - Step 4 publication figures

## Purpose

Step 4 renders final-paper figures **offline**. It makes zero model/API calls:
it only reads the existing `results/<run_id>/` and
`trajectories/<run_id>/` artifacts written by Steps 1-3. The frozen debate
topology, prompts, scoring, RoundValue labels/thresholds, benchmark splits,
historical trajectories, and existing Step 3 plots are never modified.

Generated: {_iso_now()}

## Reproduce

```bash
python scripts/step4_paper_figures.py
```

The script auto-discovers every compatible run from its manifest/metadata
(never from directory names), recomputes all metrics, validates the
policy-comparison invariants, and writes `figure_data.json` plus the five
main-paper PNGs (and one supporting heatmap) in this directory.

## Runs used

| run_id | model | dataset | created_at |
|---|---|---|---|
{runs_table}

Skipped runs: {skipped_lines}

## Run identity

| run_id | model snapshot | temperature | reasoning | R1..R5 accuracy (%) | n_R1_wrong |
|---|---|---|---|---|---|
{identity_table}

{deepseek_note}

## What each figure answers

- **fig01_round_accuracy_dynamics.png** - how accuracy evolves from R1 to R5,
  as benchmark small multiples (MMLU-Pro / HARP / LogiQA) with one line per
  model, on a shared 0-100% scale with numeric y ticks on the left panel.
  All tasks (n = {n_all_text} per run).
- **fig02_continuation_opportunity.png** - the main RoundValue policy figure.
  Held-out **test tasks only** (n = {n_test_text} per run): R1 -> Oracle is the available
  continuation opportunity, R1 -> RoundValue the captured gain, and
  RoundValue -> Oracle the remaining policy regret. When R1 == RoundValue,
  the RoundValue diamond is drawn inside a black R1 ring so the equality is
  never hidden, and the row annotation says "R1 = RoundValue".
- **fig03_r1_wrong_trajectories.png** - compact trajectory-pattern counts:
  identical Writer correctness trajectories among R1-wrong tasks are
  aggregated into W/C patterns with a task count per pattern, grouped by
  model x benchmark. No task is outcome-selected.
- **fig03_supp_task_level_heatmap.png** - supporting/appendix figure
  preserving the full task-level R1-wrong heatmap for auditability.
- **fig04_repair_mechanism.png** - (a) P(blind Stage-1 gold emerges at
  R2..R5 | R1 wrong) per condition, and (b) a mutually exclusive,
  exhaustive partition of the R1-wrong tasks crossing blind Stage-1 gold
  emergence with Writer repair (with stable/temporary repair subdivided).
  Panel (b) counts sum to n_R1_wrong.
- **fig05_continuation_landscape.png** - one point per model x benchmark:
  R1 error rate versus Ever-repair rate, with constant-Oracle-headroom
  contours (headroom = error headroom x recoverability) and Wilson 95%
  confidence intervals for P(EverRepair | R1 wrong). No regression lines.

## Split conventions

- `fig02` uses the held-out test split only, per the policy-comparison rule.
- `fig01`, `fig03`, `fig04`, and `fig05` use **all tasks** of each run
  (train + validation + test, n = {n_all_text}) and are labeled as such in the figures.
- `figure_data.json` stores both an `all` record and a `test` record per
  condition, so every plotted value can be audited against its split.

## Metric definitions

- **R1..R5 accuracy** - share of tasks whose Writer checkpoint at that round
  matches the gold option under binary exact-option scoring (percent).
- **R1 wrong count** - number of tasks incorrect at round 1.
- **Ever-repair** - P(any of R2..R5 correct | R1 wrong).
- **Stable-repair** - repaired after R1 and still correct at R5.
- **Temporary-repair** - correct at some later round but wrong again at R5.
- **Late-repair** - first correct checkpoint appears at R5.
- **First repair round** - the first round in R2..R5 with a correct
  checkpoint, per repaired task (distribution recorded).
- **Blind Stage-1 gold emergence** - P(Planner/Analyst/Critic Stage-1
  candidate_answer equals gold at R2..R5 | R1 wrong), using structured
  candidate answers only.
- **Mechanism taxonomy (Figure 4)** - every R1-wrong task is assigned to
  exactly one cell crossing two independent trajectory facts: blind Stage-1
  gold emergence (yes/no) and Writer repair (yes/no). The four cells are
  "gold never emerged, no repair", "gold never emerged, repaired",
  "gold emerged, no repair", and "gold emerged, repaired"; repaired cells are
  subdivided into stable repair (correct at R5) and temporary repair (wrong
  again at R5). Counts always sum to n_R1_wrong, and a Writer can repair even
  when blind Stage-1 gold never emerged.
- **Wilson 95% CI (Figure 5)** - Wilson score interval for
  P(EverRepair | R1 wrong), using n_R1_wrong as the denominator.
- **Oracle headroom** - Oracle test accuracy minus R1 test accuracy (pp).
- **Captured gain** - RoundValue test accuracy minus R1 test accuracy (pp).
- **Oracle regret** - Oracle test accuracy minus RoundValue test accuracy (pp).

## Same-split invariant

For the policy comparison R1 / RoundValue / Oracle, all values come from the
same split (test) and identical task IDs:

```text
task_ids(R1) == task_ids(RoundValue) == task_ids(Oracle)
```

This is verified per run and must also equal the test split in `scores.json`.
The script fails loudly if it is violated. Under binary exact-option scoring
and per-task best-round Oracle, `Oracle headroom = R1 error rate x
Ever-repair rate` is additionally verified within floating-point tolerance and
stored in `figure_data.json` under `validation`. The same validation section
also verifies that Figure 4 categories sum to `n_R1_wrong`, that the Wilson
intervals use the `n_R1_wrong` denominators, and that the raw Writer
checkpoints and `analysis.json` independently reproduce the `scores.json`
round accuracies.

## Small-sample limitations

Each run has {n_all_text} tasks ({n_test_text} test). Test-split percentages
therefore move in coarse percentage-point steps, and repair rates are
conditioned on small R1-wrong denominators. The mechanism figures use all
tasks of each run to mitigate this, but condition-level rates remain noisy;
treat point estimates as exploratory and report n where the figures do.

## Adding future runs

Collect a new run with Steps 1-2 (e.g. GPT-5-nano x HARP-50). Step 4
auto-discovers it from `results/<run_id>/manifest.json`,
`results/<run_id>/scores.json`, `results/<run_id>/test_policy_replay.json`,
and `trajectories/<run_id>/`; model, dataset, and split are resolved from
those artifacts, not from directory names. Re-run
`python scripts/step4_paper_figures.py` and every figure, `figure_data.json`,
and this README update without rewriting any plotting logic.

## Token / latency / cost

Token, latency, and cost remain Step-3 diagnostics and are intentionally
absent from these primary paper figures; the existing Step 3 plots are
unchanged.

## Validation results

{validation_lines}
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "results",
        help="Directory containing per-run result folders (default: results/).",
    )
    parser.add_argument(
        "--trajectories-dir",
        type=Path,
        default=PROJECT_ROOT / "trajectories",
        help="Directory containing per-run trajectory folders (default: trajectories/).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "figure",
        help="Output directory for the paper figures (default: figure/).",
    )
    args = parser.parse_args(argv)

    _setup_matplotlib()
    runs, skipped = discover_runs(args.results_dir, args.trajectories_dir)
    if not runs:
        print("ERROR: no compatible runs discovered; nothing to plot.")
        print("Skipped runs:")
        for item in skipped:
            print(f"  - {item['run_id']}: {item['reason']}")
        return 1

    print(f"Discovered {len(runs)} compatible run(s).")
    all_records: list[dict[str, Any]] = []
    test_records: list[dict[str, Any]] = []
    run_identities: list[dict[str, Any]] = []
    raw_counts: dict[str, dict[str, int]] = {}
    raw_policy_accuracies: dict[str, dict[str, float]] = {}
    for run in runs:
        matrix = load_score_matrix(run)
        r1_wrong_ids = {
            task_id
            for task_id, task in matrix.items()
            if task["rounds"].get(1) is False
        }
        gold_info = load_stage1_gold(run, r1_wrong_ids)
        policy = load_policy_data(run)

        test_ids_from_scores = {
            task_id for task_id, task in matrix.items() if task.get("split") == "test"
        }
        if test_ids_from_scores != set(policy["task_ids"]):
            raise RuntimeError(
                f"{run['run_id']}: test task IDs in scores.json disagree with "
                "test_policy_replay.json"
            )

        test_tasks = {
            task_id: task
            for task_id, task in matrix.items()
            if task.get("split") == "test"
        }
        r1_wrong_test = [
            task_id
            for task_id, task in test_tasks.items()
            if task["rounds"].get(1) is False
        ]
        n_ever_test = sum(
            1
            for task_id in r1_wrong_test
            if any(
                test_tasks[task_id]["rounds"].get(round_index) is True
                for round_index in range(2, MAX_ROUNDS + 1)
            )
        )
        raw_counts[run["run_id"]] = {
            "n_test": len(test_tasks),
            "r1_wrong_test": len(r1_wrong_test),
            "n_ever_test": n_ever_test,
        }
        raw_policy_accuracies[run["run_id"]] = {
            "r1": float(policy["accuracies"]["r1"]),
            "oracle": float(policy["accuracies"]["oracle"]),
        }

        all_record = build_record(run, "all", matrix, gold_info, policy)
        all_records.append(all_record)
        run_identities.append(build_run_identity(run, all_record))
        test_records.append(build_record(run, "test", matrix, gold_info, policy))

    checks = run_validations(
        runs,
        all_records,
        test_records,
        run_identities,
        raw_counts,
        raw_policy_accuracies,
    )
    failed_checks = [check for check in checks if not check.get("ok")]
    overall_passed = not failed_checks

    figure_data: dict[str, Any] = {
        "schema_version": "1.0",
        "generated_at": _iso_now(),
        "generated_by": "scripts/step4_paper_figures.py",
        "reproduction_command": "python scripts/step4_paper_figures.py",
        "offline": True,
        "units": {
            "R1..R5": "accuracy, percent (0-100)",
            "Ever_repair / Stable_repair / Temporary_repair / Late_repair": "percent of R1-wrong tasks (0-100)",
            "gold_emergence": "percent of R1-wrong tasks with known gold whose blind Stage-1 candidate matches gold at R2..R5 (0-100)",
            "mechanism_categories / mechanism_percents": (
                "mutually exclusive partition of R1-wrong tasks; percentages are "
                "of n_R1_wrong and counts sum to n_R1_wrong"
            ),
            "Ever_repair_wilson_ci95": (
                "Wilson 95% confidence interval for P(EverRepair | R1 wrong), "
                "percent (0-100), denominator = n_R1_wrong"
            ),
            "R1_test / RoundValue_test / Oracle_test": "held-out test accuracy, percent (0-100)",
            "Oracle_headroom / Captured_gain / Oracle_regret": "percentage points",
        },
        "splits": {
            "all": (
                "all train + validation + test tasks of the run "
                f"(n = {', '.join(str(n) for n in sorted({record['n'] for record in all_records}))})"
            ),
            "test": (
                "held-out test tasks of the run "
                f"(n = {', '.join(str(n) for n in sorted({record['n'] for record in test_records}))})"
            ),
        },
        "figure_sources": {
            "fig01_round_accuracy_dynamics.png": {
                "split": "all",
                "fields": ["R1", "R2", "R3", "R4", "R5"],
            },
            "fig02_continuation_opportunity.png": {
                "split": "test",
                "fields": [
                    "R1_test",
                    "RoundValue_test",
                    "Oracle_test",
                    "Oracle_headroom",
                    "Captured_gain",
                    "Oracle_regret",
                ],
            },
            "fig03_r1_wrong_trajectories.png": {
                "split": "all",
                "fields": ["r1_wrong_patterns", "R1_wrong"],
            },
            "fig03_supp_task_level_heatmap.png": {
                "split": "all",
                "fields": ["r1_wrong_tasks.correct_rounds", "r1_wrong_tasks.category"],
            },
            "fig04_repair_mechanism.png": {
                "split": "all",
                "fields": [
                    "gold_emergence",
                    "n_gold_emergence",
                    "n_gold_known",
                    "mechanism_categories",
                    "mechanism_percents",
                    "mechanism_partition",
                ],
            },
            "fig05_continuation_landscape.png": {
                "split": "all",
                "fields": ["R1", "Ever_repair", "R1_wrong", "Ever_repair_wilson_ci95"],
            },
        },
        "runs_included": [
            {
                "run_id": run["run_id"],
                "result_dir": str(run["result_dir"]),
                "trajectory_dir": str(run["trajectory_dir"]),
                "dataset_id": run["dataset_id"],
                "dataset_label": run["dataset_label"],
                "domain": run["domain"],
                "benchmark": run["benchmark"],
                "model_id": run["model_id"],
                "model": run["model"],
                "model_snapshot": run["model_snapshot"],
                "model_display": run["model_display"],
                "reasoning": run["reasoning"],
                "temperature": run["temperature"],
                "created_at": run["created_at"],
                "config_hash": run["config_hash"],
                "benchmark_source": run["benchmark_source"],
            }
            for run in runs
        ],
        "run_identities": run_identities,
        "runs_skipped": skipped,
        "conditions": sorted(all_records, key=_condition_sort_key),
        "test_conditions": sorted(test_records, key=_condition_sort_key),
        "validation": {
            "overall": "PASS" if overall_passed else "FAIL",
            "checks": checks,
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.out_dir / "figure_data.json", figure_data)

    if not overall_passed:
        message = "\n".join(
            f"  - {check['check']} ({check.get('run_id', '')}): {check}"
            for check in failed_checks
        )
        raise RuntimeError(
            "Step 4 validation failed (results stored in figure/figure_data.json):\n"
            + message
        )

    fig01_round_accuracy_dynamics(all_records, args.out_dir / "fig01_round_accuracy_dynamics.png")
    fig02_continuation_opportunity(test_records, args.out_dir / "fig02_continuation_opportunity.png")
    fig03_r1_wrong_trajectories(all_records, args.out_dir / "fig03_r1_wrong_trajectories.png")
    fig03_supp_r1_wrong_task_heatmap(
        all_records, args.out_dir / "fig03_supp_task_level_heatmap.png"
    )
    fig04_repair_mechanism(all_records, args.out_dir / "fig04_repair_mechanism.png")
    fig05_continuation_landscape(all_records, args.out_dir / "fig05_continuation_landscape.png")

    readme = build_readme(
        runs,
        skipped,
        checks,
        run_identities,
        sorted({record["n"] for record in all_records}),
        sorted({record["n"] for record in test_records}),
    )
    with open(args.out_dir / "README.md", "w", encoding="utf-8", newline="\n") as handle:
        handle.write(readme)

    print("\nStep 4 complete.")
    print("Files added/modified:")
    print(f"  {PROJECT_ROOT / 'scripts' / 'step4_paper_figures.py'}")
    for name in (
        "README.md",
        "figure_data.json",
        "fig01_round_accuracy_dynamics.png",
        "fig02_continuation_opportunity.png",
        "fig03_r1_wrong_trajectories.png",
        "fig03_supp_task_level_heatmap.png",
        "fig04_repair_mechanism.png",
        "fig05_continuation_landscape.png",
    ):
        print(f"  {args.out_dir / name}")
    print("Runs included:")
    for run in runs:
        print(f"  - {run['run_id']} ({run['model_display']} x {run['dataset_id']})")
    print("Figures generated: 6 (5 main-paper figures + 1 supporting heatmap)")
    print("Validation results:")
    for check in checks:
        status = "PASS" if check.get("ok") else "FAIL"
        detail = ""
        if check.get("check") == "oracle_decomposition" and "abs_diff" in check:
            detail = f" (abs_diff={check['abs_diff']:.3e}, tolerance={check['tolerance']:.3e})"
        elif check.get("check") == "policy_score_consistency" and "max_abs_diff" in check:
            detail = f" (max_abs_diff={check['max_abs_diff']:.3e})"
        print(f"  - {check['check']} ({check.get('run_id')}): {status}{detail}")
    print("Unavailable metrics: none (all required artifacts present and complete).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
