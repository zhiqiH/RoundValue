"""Offline self-check for the real benchmark, scorer, and analysis paths.

This script never contacts a model provider.  It exercises the exact
benchmark-specific boundaries added for MMLU-Pro:

* conservative multiple-choice answer normalization and deterministic
  correct / incorrect / ambiguous / malformed scoring;
* the public-task privacy boundary for multiple-choice tasks;
* HARP and LogiQA option-label mapping, strict MC scoring, privacy, and the
  deterministic 500/50 construction (via ``verify_real_benchmarks``);
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
from policy import build_policy_features, fit_policy_models, replay_policies  # noqa: E402
from scorer import (  # noqa: E402
    normalize_mc_answer,
    score_multiple_choice,
    score_mmlu_pro,
    score_task,
    score_trajectory,
)
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

    score = score_task(task, {"answer": "B", "reasoning_summary": "Four equals two plus two, so choose B."})
    check(score["quality"] == 1.0 and score["reason"] == "exact_option_match", "correct option scores 1")
    score = score_task(task, {"answer": "Answer: (B)", "reasoning_summary": "format tolerance"})
    check(score["quality"] == 1.0, "formatted correct option scores 1")
    score = score_task(task, {"answer": "A", "reasoning_summary": "The reasoning claims B is right."})
    check(score["quality"] == 0.0 and score["reason"] == "option_mismatch", "wrong option scores 0")
    score = score_task(
        task,
        {
            "answer": "A",
            "reasoning_summary": "B is the canonical answer and is fully supported by the evidence.",
        },
    )
    check(
        score["quality"] == 0.0,
        "reasoning_summary content cannot rescue an incorrect answer",
    )
    score = score_task(
        task,
        {
            "answer": "B",
            "reasoning_summary": "placeholder reasoning that is not informative",
        },
    )
    check(
        score["quality"] == 1.0,
        "correctness depends only on the canonical answer field",
    )
    score = score_task(task, {"final_answer": "B"})
    check(
        score["quality"] == 1.0,
        "legacy answer-only final_answer records remain scoreable",
    )
    for malformed in ("B and C", "5", "none of the above", "1.0"):
        score = score_task(task, {"answer": malformed, "reasoning_summary": "reason"})
        check(
            score["quality"] == 0.0
            and score["reason"] == "ambiguous_or_missing_option",
            f"malformed answer rejected conservatively: {malformed!r}",
        )
    score = score_task(task, {"answer": "", "reasoning_summary": "reason"})
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

    check(
        counts(full_document) == {"train": 300, "validation": 100, "test": 100},
        "MMLU-Pro-500 is 300/100/100",
    )
    check(
        counts(small_document) == {"train": 30, "validation": 10, "test": 10},
        "MMLU-Pro-50 is 30/10/10",
    )
    check(
        all(
            len(task["options"]) == task["public_metadata"]["option_count"]
            and 3 <= len(task["options"]) <= 10
            for task in full_tasks
        ),
        "every MMLU-Pro task carries 3-10 options matching its metadata",
    )

    _, _, smoke_tasks = load_benchmark(root, "benchmark/test/smoke_tasks.json")
    check(
        len(smoke_tasks) == 2
        and all(task["domain"] == "mmlu_pro" for task in smoke_tasks),
        "the lightweight smoke benchmark loads as MMLU-Pro tasks",
    )


def _harp_task(task_id: str) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "domain": "harp",
        "split": "train",
        "prompt": (
            "If 64 is divided into three parts proportional to 2, 4, and 6, "
            "the smallest part is:\n\nChoices:\n(A) 11\n(B) 5 1/3\n(C) 5\n"
            "(D) 10 2/3\n(E) None of these answers\n\n"
            "Return only the letter of the correct choice (A-E)."
        ),
        "options": ["11", "5 1/3", "5", "10 2/3", "None of these answers"],
        "answer_index": 3,
        "reference_answer": "D",
        "public_metadata": {
            "source_dataset": "HARP",
            "source_task_id": task_id,
            "source_contest": "AHSME",
            "source_year": "1950",
            "source_number": 1,
            "subject": "prealgebra",
            "level": 2,
            "multiple_choice_only": False,
            "option_count": 5,
        },
        "solution_1": "The human solution must never reach an Agent.",
    }


def _logiqa_task(task_id: str) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "domain": "logiqa",
        "split": "validation",
        "prompt": (
            "Passage:\nA short logical passage.\n\nQuestion:\nWhich one follows?\n\n"
            "Choices:\n(A) First\n(B) Second\n(C) Third\n(D) Fourth\n\n"
            "Return only the letter of the correct choice (A-D)."
        ),
        "options": ["First", "Second", "Third", "Fourth"],
        "answer_index": 2,
        "reference_answer": "C",
        "public_metadata": {
            "source_dataset": "LogiQA 2.0 English MRC",
            "source_task_id": "42",
            "source_split": "dev",
            "reasoning_types": [
                "Categorical Reasoning",
                "Necessry Condtional Reasoning",
            ],
            "option_count": 4,
        },
    }


def _check_new_benchmarks() -> None:
    harp = _harp_task("harp::1950::AHSME::1")
    logiqa = _logiqa_task("logiqa2::dev::42")

    harp_score = score_task(harp, {"answer": "D", "reasoning_summary": "proportional parts"})
    check(
        harp_score["quality"] == 1.0
        and harp_score["reason"] == "exact_option_match"
        and harp_score["domain"] == "harp",
        "HARP source answer maps to one canonical option label",
    )
    logiqa_score = score_task(logiqa, {"answer": "C", "reasoning_summary": "follows"})
    check(
        logiqa_score["quality"] == 1.0
        and logiqa_score["domain"] == "logiqa",
        "LogiQA integer answer maps to the canonical A-D label",
    )
    check(
        score_task(harp, {"answer": "B", "reasoning_summary": "wrong"})["quality"] == 0.0,
        "HARP scorer is exact-option only",
    )
    check(
        score_task(logiqa, {"answer": "B and D", "reasoning_summary": "ambiguous"})[
            "quality"
        ]
        == 0.0
        and score_task(logiqa, {"answer": "B and D", "reasoning_summary": "ambiguous"})[
            "reason"
        ]
        == "ambiguous_or_missing_option",
        "LogiQA scorer rejects ambiguous answers",
    )
    mmlu = _mc_task("dev::shared::scorer")
    check(
        score_multiple_choice(mmlu, {"answer": "B"}, domain="mmlu_pro")["quality"]
        == score_mmlu_pro(mmlu, {"answer": "B"})["quality"]
        == 1.0,
        "shared MC scorer preserves historical MMLU-Pro behavior",
    )

    harp_visible = public_task(harp)
    logiqa_visible = public_task(logiqa)
    for visible, private in (
        (harp_visible, {"answer_index", "reference_answer", "solution_1", "solution"}),
        (logiqa_visible, {"answer_index", "reference_answer"}),
    ):
        for field in private:
            check(field not in visible, f"private field {field} stays out of public task")
    for visible in (harp_visible, logiqa_visible):
        check(
            visible.get("task_id") != visible.get("public_metadata", {}).get(
                "source_task_id"
            )
            and "::" not in str(visible.get("task_id", "")),
            "Agent-facing task identifiers are anonymized",
        )
    check(
        harp_visible.get("public_metadata", {}).get("source_contest") is None
        and harp_visible.get("public_metadata", {}).get("source_year") is None
        and harp_visible.get("public_metadata", {}).get("source_number") is None,
        "HARP source identifiers stay out of public metadata",
    )
    check(
        logiqa_visible.get("public_metadata", {}).get("source_task_id") is None,
        "LogiQA source identifier stays out of public metadata",
    )

    # Invalid gold mappings and option counts must fail validation loudly.
    import json as _json
    import tempfile
    from contracts import ConfigurationError
    from benchmark_io import load_benchmark as _load

    cases = (
        (
            "invalid_answer_index",
            {
                "task_id": "dev::invalid::index",
                "domain": "logiqa",
                "split": "test",
                "prompt": "Choices:\n(A) a\n(B) b\n\nReturn only the letter (A-B).",
                "options": ["a", "b"],
                "answer_index": 5,
                "reference_answer": "A",
            },
        ),
        (
            "invalid_option_count",
            {
                "task_id": "dev::invalid::count",
                "domain": "harp",
                "split": "test",
                "prompt": "Return only the letter of the correct choice (A-B).",
                "options": ["only one"],
                "answer_index": 0,
                "reference_answer": "A",
            },
        ),
        (
            "ambiguous_gold",
            {
                "task_id": "dev::ambiguous::gold",
                "domain": "harp",
                "split": "test",
                "prompt": "Choices:\n(A) a\n(B) b\n\nReturn only the letter (A-B).",
                "options": ["a", "b"],
                "answer_index": 1,
                "reference_answer": "A and B",
            },
        ),
    )
    for label, bad_task in cases:
        with tempfile.TemporaryDirectory() as temporary:
            from pathlib import Path as _Path

            bad_path = _Path(temporary) / "bad.json"
            bad_path.write_text(
                _json.dumps({"schema_version": "1.0", "tasks": [bad_task]}),
                encoding="utf-8",
            )
            raised = False
            try:
                _load(PROJECT_ROOT, bad_path)
            except ConfigurationError:
                raised = True
            check(raised, f"validation fails for {label}")

    verified = verify_real_benchmarks.verify()
    check(verified["status"] == "verified", "all real benchmarks verify together")
    check(
        set(verified["split_counts"])
        >= {"HARP-500", "HARP-50", "LogiQA-500", "LogiQA-50"},
        "verifier reports the new benchmark families",
    )
    check(
        verified["split_counts"]["HARP-500"]
        == {"train": 300, "validation": 100, "test": 100}
        and verified["split_counts"]["HARP-50"]
        == {"train": 30, "validation": 10, "test": 10}
        and verified["split_counts"]["LogiQA-500"]
        == {"train": 300, "validation": 100, "test": 100}
        and verified["split_counts"]["LogiQA-50"]
        == {"train": 30, "validation": 10, "test": 10},
        "new benchmarks carry the exact 300/100/100 and 30/10/10 partitions",
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
                "answer": answer,
                "reasoning_summary": (
                    f"selfcheck reasoning for round {round_index}: "
                    "evidence, assumptions, and option distinctions"
                ),
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
            "max_rounds": 5,
            "rounds": [],
            "checkpoints": checkpoints,
        },
    }
    record["scores"] = score_trajectory(record)
    record["labels"] = build_labels(record, lambda_cost=0.0, mu_latency=0.0)
    return record


def _check_legacy_records() -> None:
    """Old three-round answer-only records stay readable but carry no summary."""

    task = _mc_task("dev::legacy::record")
    checkpoints = []
    for round_index, answer in enumerate(("B", "A", "B"), start=1):
        checkpoints.append(
            {
                "round_index": round_index,
                "final_answer": answer,
                "cumulative": {
                    "input_tokens": 100 * round_index,
                    "output_tokens": 10 * round_index,
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
        "split": "test",
        "trajectory": {
            "schema_version": "1.0",
            "trajectory_id": "legacy-trajectory",
            "task_id": task["task_id"],
            "domain": task["domain"],
            "status": "complete",
            "max_rounds": 3,
            "rounds": [],
            "checkpoints": checkpoints,
        },
    }
    scores = score_trajectory(record)
    check(
        [score["quality"] for score in scores] == [1.0, 0.0, 1.0],
        "legacy final_answer checkpoints score without a reasoning summary",
    )
    labels = build_labels(record | {"scores": scores}, lambda_cost=0.0, mu_latency=0.0)
    check(len(labels) == 3 and all(label["G"] is not None for label in labels), "legacy labels build without summaries")
    replay = replay_policies([record | {"scores": scores, "labels": labels}], None)
    check(
        set(replay["policy_metrics"])
        == {"fixed_1", "fixed_2", "fixed_3", "task_only", "roundvalue", "oracle"}
        and replay.get("max_rounds") == 3,
        "legacy three-round records replay with Fixed-1..3 only",
    )


def _check_pipeline() -> None:
    patterns = {
        "train": [
            (0, 1, 1, 1, 1),
            (1, 1, 1, 1, 1),
            (0, 0, 0, 0, 0),
            (1, 0, 1, 0, 1),
            (0, 0, 1, 0, 1),
            (1, 1, 0, 1, 0),
            (0, 1, 0, 1, 0),
            (1, 0, 0, 0, 1),
        ],
        "validation": [
            (0, 1, 1, 0, 1),
            (1, 0, 1, 1, 1),
            (1, 1, 1, 1, 1),
            (0, 0, 0, 0, 0),
        ],
        "test": [
            (1, 0, 1, 0, 1),
            (0, 0, 1, 1, 0),
            (1, 1, 0, 0, 1),
            (0, 1, 0, 1, 1),
        ],
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
    first_checkpoint = records["train"][0]["trajectory"]["checkpoints"][0]
    first_label = next(
        label
        for label in records["train"][0]["labels"]
        if label["round_index"] == 1
    )
    stopping_features = build_policy_features(
        records["train"][0],
        first_checkpoint,
        first_label,
        None,
    )
    check(
        "quality" not in stopping_features
        and "G" not in stopping_features
        and "delta_q" not in stopping_features,
        "online stopping features exclude gold and future-round value targets",
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
        == {
            "fixed_1",
            "fixed_2",
            "fixed_3",
            "fixed_4",
            "fixed_5",
            "task_only",
            "roundvalue",
            "oracle",
        },
        "replay emits Fixed-1..Fixed-5, task-only, RoundValue, and Oracle",
    )
    check(
        replay.get("max_rounds") == 5,
        "replay records the five-round horizon",
    )
    check(
        replay["policy_metrics"]["fixed_4"]["mean_stop_round"] == 4
        and replay["policy_metrics"]["fixed_5"]["mean_stop_round"] == 5,
        "fixed-round baselines stop at rounds 4 and 5",
    )
    oracle_row = replay["policies"]["oracle"]["task_results"][0]
    check(
        oracle_row["stop_round"] == 1,
        "trajectory Oracle selects the best utility across all five checkpoints",
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
    check(
        analysis["max_rounds"] == 5
        and len(analysis["per_round"]) == 5,
        "analysis reports all five rounds",
    )

    # Without a frozen model the learned policy continues after rounds 1-4
    # and is forced to stop after round 5; its decision trace must never
    # contain a future-round decision.
    no_model = replay_policies([records["test"][0]], None)
    trace = no_model["policies"]["task_only"]["task_results"][0]["decision_trace"]
    check(
        [item["round_index"] for item in trace] == [1, 2, 3, 4, 5]
        and [item["decision"] for item in trace]
        == ["CONTINUE", "CONTINUE", "CONTINUE", "CONTINUE", "FORCED_STOP"],
        "continuation is possible after rounds 1-4 but not after round 5",
    )

    late_record = _synthetic_record(
        _mc_task("dev::oracle::late"),
        "test",
        (0, 0, 0, 0, 1),
        999,
    )
    late_replay = replay_policies([late_record], None)
    check(
        late_replay["policies"]["oracle"]["task_results"][0]["stop_round"] == 5,
        "trajectory Oracle selects the fifth checkpoint when it is the best",
    )


def main() -> int:
    _check_normalization()
    _check_manifests()
    _check_new_benchmarks()
    _check_legacy_records()
    _check_pipeline()
    print("PASS all real-benchmark self-checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
