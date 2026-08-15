"""Verify every generated per-dataset benchmark before a formal collection run.

Run from the repository root with ``python src/verify_real_benchmarks.py``.

The verification checks each self-contained dataset document, its split
partition, and the code privacy boundary, then runs canonical source against a
representative set of the private EvalPlus differential inputs.  ``--all-code``
checks every generated canonical source; the one pinned upstream self-check
exception is asserted explicitly rather than silently treated as a passing
reference implementation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRECTORY = PROJECT_ROOT / "src"
if str(SOURCE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIRECTORY))

from benchmark_io import (  # noqa: E402
    PUBLIC_METADATA_HIDDEN_KEYS,
    load_benchmark,
    public_task,
)
from scorer import score_code  # noqa: E402

from build_real_benchmarks import (  # noqa: E402
    DATASET_SPLIT_RATIOS,
    KNOWN_CANONICAL_SELF_CHECK_EXCEPTIONS,
    _split_sizes,
)


PRIVATE_EVALPLUS_FIELDS = {
    "evalplus_dataset",
    "evalplus_reference_code",
    "evalplus_base_inputs",
    "evalplus_plus_inputs",
    "evalplus_atol",
}
SAMPLE_TASK_IDS = {
    "humanevalplus::HumanEval_0",
    "humanevalplus::HumanEval_32",  # explicit upstream self-check exception
    # Cover every MBPP JSON-to-Python input conversion family represented in
    # the released 378-task subset, plus all special-oracle families.
    "mbppplus::Mbpp/2", "mbppplus::Mbpp/63", "mbppplus::Mbpp/75",
    "mbppplus::Mbpp/106", "mbppplus::Mbpp/124", "mbppplus::Mbpp/250",
    "mbppplus::Mbpp/252", "mbppplus::Mbpp/259", "mbppplus::Mbpp/278",
    "mbppplus::Mbpp/558", "mbppplus::Mbpp/580", "mbppplus::Mbpp/581",
    "mbppplus::Mbpp/590", "mbppplus::Mbpp/615", "mbppplus::Mbpp/722",
    "mbppplus::Mbpp/737", "mbppplus::Mbpp/787", "mbppplus::Mbpp/791",
    "mbppplus::Mbpp/794",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_task_documents(root: Path) -> None:
    human = _read_json(root / "benchmark" / "code" / "HumanEvalPlus.json")
    mbpp = _read_json(root / "benchmark" / "code" / "MBPPPlus.json")
    if len(human.get("tasks", [])) != 164 or len(mbpp.get("tasks", [])) != 378:
        raise AssertionError("official EvalPlus task documents must contain 164 HumanEval+ and 378 MBPP+ tasks")
    for document in (human, mbpp):
        if document.get("domain") != "code":
            raise AssertionError(f"{document.get('dataset_id')} must declare domain code")
    for document in (human, mbpp):
        task_ids = [task.get("task_id") for task in document["tasks"]]
        if len(task_ids) != len(set(task_ids)):
            raise AssertionError("EvalPlus task document contains duplicate task IDs")
        for task in document["tasks"]:
            if task.get("code_evaluator") != "evalplus_differential_v1":
                raise AssertionError(f"unexpected code evaluator for {task.get('task_id')}")
            if not PRIVATE_EVALPLUS_FIELDS.issubset(task):
                raise AssertionError(f"missing private EvalPlus fields for {task.get('task_id')}")


def verify(*, all_code: bool) -> dict[str, Any]:
    root = PROJECT_ROOT.resolve()
    specs = (
        ("benchmark/math/MATH-500.json", "math", "MATH-500", 500),
        ("benchmark/code/HumanEvalPlus.json", "code", "HumanEvalPlus", 164),
        ("benchmark/code/MBPPPlus.json", "code", "MBPPPlus", 378),
    )
    tasks_by_dataset: dict[str, list[dict[str, Any]]] = {}
    split_counts: dict[str, dict[str, int]] = {}
    for relative, domain, dataset_id, expected_total in specs:
        _, document, tasks = load_benchmark(root, relative)
        if document.get("dataset_id") != dataset_id:
            raise AssertionError(f"{relative} must declare dataset_id {dataset_id}")
        if document.get("domain") != domain:
            raise AssertionError(f"{relative} must declare domain {domain}")
        if len(tasks) != expected_total:
            raise AssertionError(f"{relative} must contain {expected_total} tasks")
        if any(task.get("domain") != domain for task in tasks):
            raise AssertionError(f"{relative} mixes domains")
        if len({task["task_id"] for task in tasks}) != len(tasks):
            raise AssertionError(f"{relative} contains duplicate task IDs")
        counts = {
            split: sum(task.get("split") == split for task in tasks)
            for split in ("train", "validation", "test")
        }
        if counts != _split_sizes(expected_total, DATASET_SPLIT_RATIOS):
            raise AssertionError(f"unexpected {dataset_id} split counts: {counts}")
        provenance = document.get("provenance")
        if not isinstance(provenance, dict):
            raise AssertionError(f"{relative} must embed a provenance object")
        if provenance.get("dataset_id") != dataset_id:
            raise AssertionError(f"{relative} provenance dataset_id mismatch")
        if provenance.get("domain") != domain:
            raise AssertionError(f"{relative} provenance domain mismatch")
        provenance_counts = provenance.get("split", {}).get("counts")
        if provenance_counts != counts:
            raise AssertionError(f"{relative} provenance split counts mismatch")
        test_task_ids = provenance.get("test_task_ids")
        if not isinstance(test_task_ids, list) or len(test_task_ids) != counts["test"]:
            raise AssertionError(f"{relative} test_task_ids must list exactly {counts['test']} tasks")
        tasks_by_dataset[dataset_id] = tasks
        split_counts[dataset_id] = counts

    math_tasks = tasks_by_dataset["MATH-500"]
    math_prompt_sets = {
        split: {
            " ".join(task["prompt"].split())
            for task in math_tasks
            if task.get("split") == split
        }
        for split in ("train", "validation", "test")
    }
    if (
        math_prompt_sets["train"] & math_prompt_sets["validation"]
        or math_prompt_sets["train"] & math_prompt_sets["test"]
        or math_prompt_sets["validation"] & math_prompt_sets["test"]
    ):
        raise AssertionError("MATH-500 splits must be disjoint partitions")

    code_tasks = tasks_by_dataset["HumanEvalPlus"] + tasks_by_dataset["MBPPPlus"]
    if len(code_tasks) != 542:
        raise AssertionError("code datasets must contain 542 tasks in total")
    if any(task.get("split") not in {"train", "validation", "test"} for task in code_tasks):
        raise AssertionError("every code task must have a train/validation/test split")
    _validate_task_documents(root)
    for task in code_tasks:
        visible = public_task(task)
        leaked = PRIVATE_EVALPLUS_FIELDS & set(visible)
        if leaked:
            raise AssertionError(f"EvalPlus private fields leaked into public task {task['task_id']}: {sorted(leaked)}")
        public_metadata = visible.get("public_metadata") or {}
        leaked_metadata = PUBLIC_METADATA_HIDDEN_KEYS & set(public_metadata)
        if leaked_metadata:
            raise AssertionError(
                f"identifying metadata leaked into public task {task['task_id']}: "
                f"{sorted(leaked_metadata)}"
            )
        visible_task_id = str(visible.get("task_id", ""))
        if visible_task_id == str(task["task_id"]) or "::" in visible_task_id:
            raise AssertionError(
                f"public task {task['task_id']} exposes a recallable task identifier: "
                f"{visible_task_id}"
            )

    known_exceptions = set(KNOWN_CANONICAL_SELF_CHECK_EXCEPTIONS)
    selected = code_tasks if all_code else [
        task for task in code_tasks if task["task_id"] in SAMPLE_TASK_IDS
    ]
    if {task["task_id"] for task in selected} != (set(task["task_id"] for task in code_tasks) if all_code else SAMPLE_TASK_IDS):
        raise AssertionError("canonical verification selection is incomplete")
    failures: list[dict[str, str]] = []
    checked = {"humanevalplus": 0, "mbppplus": 0}
    verified_exceptions: list[str] = []
    for task in selected:
        source = str(task["public_metadata"]["source_dataset"])
        key = "humanevalplus" if source == "EvalPlus HumanEval+" else "mbppplus"
        result = score_code(
            task,
            task["evalplus_reference_code"],
            allow_local_code_execution=True,
        )
        checked[key] += 1
        if task["task_id"] in known_exceptions:
            if result.get("quality") == 0.0 and result.get("reason") == "evalplus_differential_failure":
                verified_exceptions.append(task["task_id"])
            else:
                failures.append({"task_id": task["task_id"], "reason": "known_exception_changed"})
        elif result.get("quality") != 1.0:
            failures.append({"task_id": task["task_id"], "reason": str(result.get("reason"))})
    if failures:
        raise AssertionError(f"canonical EvalPlus verification failed: {failures[:10]}")
    return {
        "status": "verified",
        "split_counts": split_counts,
        "canonical_code_checked": checked,
        "known_upstream_exceptions_verified": verified_exceptions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-code", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(all_code=args.all_code), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
