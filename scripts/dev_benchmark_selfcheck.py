"""Offline self-check for the MMLU-Pro benchmark, scorer, and analysis path.

This script never contacts a model provider.  It exercises the exact
benchmark-specific boundaries added for MMLU-Pro:

* conservative multiple-choice answer normalization and deterministic
  correct / incorrect / ambiguous / malformed scoring;
* the public-task privacy boundary for multiple-choice tasks;
* deterministic MMLU-Pro-500 / MMLU-Pro-50 construction and their
  300/100/100 and 30/10/10 partitions (via ``verify_real_benchmarks``);
* a synthetic trajectory -> score -> label -> policy fit -> threshold
  selection -> replay -> analysis path built entirely from MMLU-Pro-shaped
  JSON records.
"""

from __future__ import annotations

from copy import deepcopy
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from benchmark_io import load_benchmark, public_task  # noqa: E402
from labels import build_labels  # noqa: E402
from policy import fit_policy_models, replay_policies  # noqa: E402
from scorer import normalize_mc_answer, score_task, score_trajectory  # noqa: E402
from visualize import build_analysis  # noqa: E402
import verify_real_benchmarks  # noqa: E402


def check(condition: bool, label: str) -> None:
    if not condition:
        raise AssertionError(label)
    print(f"PASS {label}")


def _mc_task(task_id: str) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "domain": "mmlu_pro",
        "split": "unassigned",
        "prompt": (
            "Which value equals 2 + 2?\n\nChoices:\n"
            "(A) 2\n(B) 4\n(C) 6\n(D) 8\n\n"
            "Return only the letter of the correct choice (A-D)."
        ),
        "options": ["2", "4", "6", "8"],
        "answer_index": 1,
        "reference_answer": "B",
        "public_metadata": {
            "source_dataset": "MMLU-Pro",
            "source_task_id": "selfcheck",
            "subject": "math",
            "src": "selfcheck",
            "option_count": 4,
        },
    }


def _check_normalization() -> None:
    task = _mc_task("dev::normalization")

    cases = (
        ("B", "B", "bare label"),
        ("b", "B", "lowercase label"),
        ("Answer: B", "B", "answer prefix"),
        ("(B)", "B", "parenthesized label"),
        ("[B]", "B", "bracketed label"),
        ("B.", "B", "trailing period"),
    )
    for text, expected, label in cases:
        check(normalize_mc_answer(text) == expected, f"normalize {label}")

    score = score_task(task, {"final_answer": "B"})
    check(score["quality"] == 1.0 and score["reason"] == "exact_option_match", "correct option scores 1")
    score = score_task(task, {"final_answer": "Answer: (B)"})
    check(score["quality"] == 1.0, "formatted correct option scores 1")
    score = score_task(task, {"final_answer": "A"})
    check(score["quality"] == 0.0 and score["reason"] == "option_mismatch", "wrong option scores 0")
    for malformed in ("B and C", "5", "none of the above", "1.0"):
        score = score_task(task, {"final_answer": malformed})
        check(
            score["quality"] == 0.0
            and score["reason"] == "ambiguous_or_missing_option",
            f"malformed answer rejected conservatively: {malformed!r}",
        )
    score = score_task(task, {"final_answer": ""})
    check(score["quality"] == 0.0 and score["reason"] == "missing_final_answer", "empty answer rejected")
    score = score_task(task, "\\boxed{B}")
    check(score["quality"] == 1.0, "boxed canonical label accepted")

    visible = public_task(task)
    for private in ("reference_answer", "answer_index", "options"):
        check(private not in visible, f"private field {private} stays out of public task")
    check(
        visible.get("public_metadata", {}).get("source_task_id") is None,
        "source_task_id stays out of public task metadata",
    )
    check("(B) 4" in visible["prompt"], "choices remain embedded in the public prompt")


def _check_manifests() -> None:
    verified = verify_real_benchmarks.verify()
    check(verified["status"] == "verified", "manifest verification passes")
    root = PROJECT_ROOT.resolve()

    _, full_document, full_tasks = load_benchmark(
        root, "benchmark/mmlu_pro/MMLU-Pro-500.json"
    )
    _, small_document, small_tasks = load_benchmark(
        root, "benchmark/mmlu_pro/MMLU-Pro-50.json"
    )
    check(len(full_tasks) == 500 and len(small_tasks) == 50, "manifest sizes are 500 and 50")
    check(
        len({task["task_id"] for task in full_tasks}) == 500
        and len({task["task_id"] for task in small_tasks}) == 50,
        "no duplicate task IDs inside either manifest",
    )
    check(
        {task["task_id"] for task in small_tasks}
        <= {task["task_id"] for task in full_tasks},
        "every MMLU-Pro-50 task belongs to MMLU-Pro-500",
    )

    def counts(document: dict[str, Any]) -> dict[str, int]:
        return {
            split: sum(task.get("split") == split for task in document["tasks"])
            for split in ("train", "validation", "test")
        }

    check(counts(full_document) == {"train": 300, "validation": 100, "test": 100}, "MMLU-Pro-500 is 300/100/100")
    check(counts(small_document) == {"train": 30, "validation": 10, "test": 10}, "MMLU-Pro-50 is 30/10/10")
    check(
        all(
            len(task["options"]) == task["public_metadata"]["option_count"]
            and 3 <= len(task["options"]) <= 10
            for task in full_tasks
        ),
        "every MMLU-Pro task carries 3-10 options matching its metadata",
    )


def _synthetic_record(
    task: dict[str, Any], split: str, qualities: tuple[int, ...], index: int
) -> dict[str, Any]:
    checkpoints = []
    for round_index, quality in enumerate(qualities, start=1):
        answer = "B" if quality == 1 else "A"
        checkpoints.append(
            {
                "round_index": round_index,
                "final_answer": answer,
                "checkpoint_hash": f"selfcheck-{index}-{round_index}",
                "nodes": [
                    {
                        "node_id": "planner_stage_1",
                        "role": "planner",
                        "round_index": round_index,
                        "output": {"candidate_answer": answer},
                    },
                    {
                        "node_id": "analyst_stage_1",
                        "role": "analyst",
                        "round_index": round_index,
                        "output": {"candidate_answer": answer},
                    },
                    {
                        "node_id": "critic_stage_1",
                        "role": "critic",
                        "round_index": round_index,
                        "output": {"candidate_answer": "A" if quality == 1 else answer},
                    },
                ],
                "cumulative": {
                    "input_tokens": 100 + 40 * round_index,
                    "output_tokens": 20 * round_index,
                    "cost_usd": 0.001 * round_index,
                    "latency_ms": 100.0 * round_index,
                    "wall_clock_ms": 90 * round_index,
                    "api_latency_ms": 110 * round_index,
                    "logical_calls": 7 * round_index,
                },
            }
        )
    record: dict[str, Any] = {
        "schema_version": "1.0",
        "task": dict(task),
        "split": split,
        "trajectory": {
            "schema_version": "1.0",
            "trajectory_id": f"selfcheck-trajectory-{index}",
            "task_id": task["task_id"],
            "domain": task["domain"],
            "status": "complete",
            "max_rounds": 3,
            "rounds": [],
            "checkpoints": checkpoints,
        },
    }
    record["scores"] = score_trajectory(record)
    record["labels"] = build_labels(record, lambda_cost=0.0, mu_latency=0.0)
    return record


def _check_pipeline() -> None:
    patterns = {
        "train": [(0, 1, 1), (1, 1, 1), (0, 0, 0), (1, 0, 1), (0, 0, 1), (1, 1, 0), (0, 1, 0), (1, 0, 0)],
        "validation": [(0, 1, 1), (1, 0, 1), (1, 1, 1), (0, 0, 0)],
        "test": [(1, 0, 1), (0, 0, 1), (1, 1, 0), (0, 1, 0)],
    }
    records: dict[str, list[dict[str, Any]]] = {}
    for split, split_patterns in patterns.items():
        records[split] = [
            _synthetic_record(
                _mc_task(f"dev::pipeline::{split}::{index}"),
                split,
                pattern,
                index,
            )
            for index, pattern in enumerate(split_patterns)
        ]

    fitted = fit_policy_models(
        records["train"],
        lambda_cost=0.0,
        mu_latency=0.0,
        target="G",
        ridge=1e-6,
    )
    check(
        set(fitted) >= {"roundvalue", "task_only"},
        "policy fit returns RoundValue and task-only descriptors",
    )

    candidates = [-0.1, 0.0, 0.1]
    frozen = deepcopy(fitted)
    for policy_name in ("roundvalue", "task_only"):
        best_threshold = None
        best_rank = None
        for threshold in candidates:
            trial = deepcopy(frozen)
            trial[policy_name]["threshold"] = threshold
            replay = replay_policies(records["validation"], trial)
            metrics = replay["policy_metrics"][policy_name]
            rank = (
                metrics["mean_utility"],
                metrics["mean_quality"],
                -metrics["mean_total_tokens"],
                -threshold,
            )
            if best_rank is None or rank > best_rank:
                best_rank = rank
                best_threshold = threshold
        frozen[policy_name]["threshold"] = best_threshold
    check(
        all(
            isinstance(frozen[name]["threshold"], (int, float))
            for name in ("roundvalue", "task_only")
        ),
        "validation threshold selection runs for both learned policies",
    )

    replay = replay_policies(
        records["test"],
        frozen,
        bootstrap_seed=20260813,
        bootstrap_samples=200,
    )
    check(
        set(replay["policy_metrics"])
        == {"fixed_1", "fixed_2", "fixed_3", "task_only", "roundvalue", "oracle"},
        "replay emits exactly the active comparison set",
    )
    transitions = replay["transition_metrics"]["transition_counts"]
    check(
        all(transitions[name] > 0 for name in ("repair", "neutral", "harm", "recovery")),
        "synthetic trajectories exercise repair/neutral/harm/recovery transitions",
    )

    all_records = [record for split_records in records.values() for record in split_records]
    analysis = build_analysis(
        {"run_id": "offline_benchmark_selfcheck"},
        all_records,
        label_parameters=(0.0, 0.0),
        replay=replay,
    )
    check(
        analysis["split_counts"] == {"train": 8, "validation": 4, "test": 4},
        "analysis records train/validation/test sizes",
    )
    check(
        analysis["policy_table"]
        and all(
            isinstance(item.get("accuracy"), (int, float)) or item.get("accuracy") is None
            for item in analysis["policy_table"]
        ),
        "analysis reports numeric policy accuracies",
    )


def main() -> int:
    _check_normalization()
    _check_manifests()
    _check_pipeline()
    print("PASS all MMLU-Pro benchmark self-checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
