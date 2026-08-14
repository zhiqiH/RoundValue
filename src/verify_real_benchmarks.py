"""Verify the generated real-data benchmark before a formal collection run.

Run from the repository root with ``python src/verify_real_benchmarks.py``.

The verification checks the manifest topology and privacy boundary, then runs
canonical source against a representative set of the private EvalPlus
differential inputs.  ``--all-code`` checks every generated canonical source;
the one pinned upstream self-check exception is asserted explicitly rather
than silently treated as a passing reference implementation.
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

from build_real_benchmarks import KNOWN_CANONICAL_SELF_CHECK_EXCEPTIONS  # noqa: E402


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
    human = _read_json(root / "benchmark" / "code" / "HumanEvalPlus_tasks.json")
    mbpp = _read_json(root / "benchmark" / "code" / "MBPPPlus_tasks.json")
    if len(human.get("tasks", [])) != 164 or len(mbpp.get("tasks", [])) != 378:
        raise AssertionError("official EvalPlus task documents must contain 164 HumanEval+ and 378 MBPP+ tasks")
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
    _, _, tasks = load_benchmark(root, "benchmark/formal_experiment_v1.json")
    counts = {
        split: sum(task.get("split") == split for task in tasks)
        for split in ("train", "validation", "test")
    }
    if counts != {"train": 140, "validation": 60, "test": 642}:
        raise AssertionError(f"unexpected split counts: {counts}")
    if len({task["task_id"] for task in tasks}) != 842:
        raise AssertionError("formal manifest must contain 842 globally unique tasks")
    math_tasks = [task for task in tasks if task["domain"] == "math"]
    if len(math_tasks) != 300:
        raise AssertionError("formal manifest must contain 300 MATH tasks")
    math_test_prompts = {
        " ".join(task["prompt"].split())
        for task in math_tasks
        if task.get("split") == "test"
    }
    math_development_prompts = {
        " ".join(task["prompt"].split())
        for task in math_tasks
        if task.get("split") in {"train", "validation"}
    }
    if len(math_test_prompts) != 100 or math_test_prompts & math_development_prompts:
        raise AssertionError("MATH development tasks must be disjoint from the MATH-100 test subset")
    code_tasks = [task for task in tasks if task["domain"] == "code"]
    if len(code_tasks) != 542 or any(task.get("split") != "test" for task in code_tasks):
        raise AssertionError("all 542 EvalPlus tasks must remain held-out test tasks")
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
        "split_counts": counts,
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
