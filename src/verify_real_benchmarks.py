"""Verify the two generated math benchmark documents before a formal run.

Run from the repository root with ``python src/verify_real_benchmarks.py``.

The verification checks the self-contained task documents, their
train/validation/test partitions, the public-task privacy boundary, and the
deterministic parent/subset relationship between MATH-500 and MATH-50.
"""

from __future__ import annotations

import argparse
import hashlib
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

from build_real_benchmarks import (  # noqa: E402
    DATASET_SPLIT_RATIOS,
    _split_sizes,
)


def _disjoint_prompts(tasks: list[dict[str, Any]]) -> None:
    prompts = {
        split: {
            " ".join(task["prompt"].split())
            for task in tasks
            if task.get("split") == split
        }
        for split in ("train", "validation", "test")
    }
    if (
        prompts["train"] & prompts["validation"]
        or prompts["train"] & prompts["test"]
        or prompts["validation"] & prompts["test"]
    ):
        raise AssertionError("MATH-500 splits must be disjoint partitions")


def verify() -> dict[str, Any]:
    root = PROJECT_ROOT.resolve()
    specs = (
        ("benchmark/math/MATH-500.json", "MATH-500", 500),
        ("benchmark/math/MATH-50.json", "MATH-50", 50),
    )
    documents: dict[str, dict[str, Any]] = {}
    tasks_by_dataset: dict[str, list[dict[str, Any]]] = {}
    split_counts: dict[str, dict[str, int]] = {}
    for relative, dataset_id, expected_total in specs:
        _, document, tasks = load_benchmark(root, relative)
        if document.get("dataset_id") != dataset_id:
            raise AssertionError(f"{relative} must declare dataset_id {dataset_id}")
        if document.get("domain") != "math":
            raise AssertionError(f"{relative} must declare domain math")
        if len(tasks) != expected_total:
            raise AssertionError(f"{relative} must contain {expected_total} tasks")
        if any(task.get("domain") != "math" for task in tasks):
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
        if provenance.get("domain") != "math":
            raise AssertionError(f"{relative} provenance domain mismatch")
        provenance_counts = provenance.get("split", {}).get("counts")
        if provenance_counts != counts:
            raise AssertionError(f"{relative} provenance split counts mismatch")
        test_task_ids = provenance.get("test_task_ids")
        if not isinstance(test_task_ids, list) or len(test_task_ids) != counts["test"]:
            raise AssertionError(
                f"{relative} test_task_ids must list exactly {counts['test']} tasks"
            )
        documents[dataset_id] = document
        tasks_by_dataset[dataset_id] = tasks
        split_counts[dataset_id] = counts

    math500 = tasks_by_dataset["MATH-500"]
    math50 = tasks_by_dataset["MATH-50"]
    _disjoint_prompts(math500)
    parent_by_id = {task["task_id"]: task for task in math500}
    subset_ids = {task["task_id"] for task in math50}
    if not subset_ids <= set(parent_by_id):
        raise AssertionError("MATH-50 contains tasks that are not in MATH-500")
    for task in math50:
        parent = parent_by_id[task["task_id"]]
        for field in ("prompt", "reference_answer", "difficulty"):
            if task[field] != parent[field]:
                raise AssertionError(
                    f"MATH-50 task {task['task_id']} diverges from MATH-500 on {field}"
                )
    parent_sha = documents["MATH-50"]["provenance"]["parent"].get("sha256")
    math500_bytes = (root / "benchmark" / "math" / "MATH-500.json").read_bytes()
    if parent_sha != hashlib.sha256(math500_bytes).hexdigest():
        raise AssertionError("MATH-50 provenance parent sha256 does not match MATH-500")

    for tasks in (math500, math50):
        for task in tasks:
            visible = public_task(task)
            if "reference_answer" in visible or "solution" in visible or "gold" in visible:
                raise AssertionError(
                    f"private answer fields leaked into public task {task['task_id']}"
                )
            public_metadata = visible.get("public_metadata") or {}
            leaked = PUBLIC_METADATA_HIDDEN_KEYS & set(public_metadata)
            if leaked:
                raise AssertionError(
                    f"identifying metadata leaked into public task {task['task_id']}: "
                    f"{sorted(leaked)}"
                )
            visible_task_id = str(visible.get("task_id", ""))
            if visible_task_id == str(task["task_id"]) or "::" in visible_task_id:
                raise AssertionError(
                    f"public task {task['task_id']} exposes a recallable task identifier"
                )
    return {
        "status": "verified",
        "split_counts": split_counts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(verify(), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
