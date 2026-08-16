"""Verify the generated benchmark documents before a formal run.

Run from the repository root with ``python src/verify_real_benchmarks.py``.

The verification covers every self-contained task document, its
train/validation/test partition, the public-task privacy boundary, and the
deterministic parent/subset relationship for both benchmark families:
MATH-500/MATH-50 (legacy) and MMLU-Pro-500/MMLU-Pro-50 (active).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRECTORY = PROJECT_ROOT / "src"
if str(SOURCE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIRECTORY))

from benchmark_io import (  # noqa: E402
    PRIVATE_TASK_FIELDS,
    PUBLIC_METADATA_HIDDEN_KEYS,
    load_benchmark,
    public_task,
)

from build_real_benchmarks import (  # noqa: E402
    DATASET_SPLIT_RATIOS,
    _split_sizes,
)


SPECS = (
    ("benchmark/math/MATH-500.json", "MATH-500", "math", 500),
    ("benchmark/math/MATH-50.json", "MATH-50", "math", 50),
    ("benchmark/mmlu_pro/MMLU-Pro-500.json", "MMLU-Pro-500", "mmlu_pro", 500),
    ("benchmark/mmlu_pro/MMLU-Pro-50.json", "MMLU-Pro-50", "mmlu_pro", 50),
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
        raise AssertionError("dataset splits must be disjoint partitions")


def _verify_privacy(tasks: list[dict[str, Any]]) -> None:
    for task in tasks:
        visible = public_task(task)
        leaked = PRIVATE_TASK_FIELDS & set(visible)
        if leaked:
            raise AssertionError(
                f"private answer fields leaked into public task {task['task_id']}: "
                f"{sorted(leaked)}"
            )
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
                f"public task {task['task_id']} exposes a recallable task identifier"
            )


def _verify_mmlu_pro_task(task: dict[str, Any]) -> None:
    options = task.get("options")
    if not isinstance(options, list) or not options:
        raise AssertionError(f"{task['task_id']} must carry a non-empty options list")
    answer_index = task.get("answer_index")
    reference = task.get("reference_answer")
    if (
        isinstance(answer_index, bool)
        or not isinstance(answer_index, int)
        or not 0 <= answer_index < len(options)
    ):
        raise AssertionError(f"{task['task_id']} has an invalid answer_index")
    label = chr(ord("A") + answer_index)
    if reference != label or not 1 < len(options) <= 10:
        raise AssertionError(
            f"{task['task_id']} has inconsistent option labels or option count"
        )


def _verify_documents() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, int]]]:
    documents: dict[str, dict[str, Any]] = {}
    tasks_by_dataset: dict[str, list[dict[str, Any]]] = {}
    split_counts: dict[str, dict[str, int]] = {}
    for relative, dataset_id, domain, expected_total in SPECS:
        _, document, tasks = load_benchmark(PROJECT_ROOT, relative)
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
            raise AssertionError(
                f"{relative} test_task_ids must list exactly {counts['test']} tasks"
            )
        if domain == "mmlu_pro":
            for task in tasks:
                _verify_mmlu_pro_task(task)
        _verify_privacy(tasks)
        documents[dataset_id] = document
        tasks_by_dataset[dataset_id] = tasks
        split_counts[dataset_id] = counts
    return documents, split_counts


def _verify_subset(
    documents: dict[str, dict[str, Any]],
    parent_id: str,
    subset_id: str,
    parent_path: str,
    verbatim_fields: tuple[str, ...],
) -> None:
    parent_by_id = {
        task["task_id"]: task for task in documents[parent_id]["tasks"]
    }
    subset_tasks = documents[subset_id]["tasks"]
    subset_ids = {task["task_id"] for task in subset_tasks}
    if not subset_ids <= set(parent_by_id):
        raise AssertionError(f"{subset_id} contains tasks that are not in {parent_id}")
    for task in subset_tasks:
        parent = parent_by_id[task["task_id"]]
        for field in verbatim_fields:
            if task[field] != parent[field]:
                raise AssertionError(
                    f"{subset_id} task {task['task_id']} diverges from {parent_id} on {field}"
                )
    parent_sha = documents[subset_id]["provenance"]["parent"].get("sha256")
    parent_bytes = (PROJECT_ROOT / parent_path).read_bytes()
    if parent_sha != hashlib.sha256(parent_bytes).hexdigest():
        raise AssertionError(f"{subset_id} provenance parent sha256 does not match {parent_id}")


def verify() -> dict[str, Any]:
    root = PROJECT_ROOT.resolve()
    documents, split_counts = _verify_documents()

    _disjoint_prompts(documents["MATH-500"]["tasks"])
    _verify_subset(
        documents,
        "MATH-500",
        "MATH-50",
        "benchmark/math/MATH-500.json",
        ("prompt", "reference_answer", "difficulty"),
    )
    _verify_subset(
        documents,
        "MMLU-Pro-500",
        "MMLU-Pro-50",
        "benchmark/mmlu_pro/MMLU-Pro-500.json",
        (
            "prompt",
            "options",
            "answer_index",
            "reference_answer",
            "public_metadata",
        ),
    )

    # Both small manifests must also reconstruct byte-for-byte from their
    # parent, which proves the subset is deterministic rather than hand-edited.
    from build_math50 import build as build_math50
    from build_mmlu_pro_50 import build as build_mmlu_pro_50

    for builder, parent_relative, subset_relative in (
        (
            build_math50,
            "benchmark/math/MATH-500.json",
            "benchmark/math/MATH-50.json",
        ),
        (
            build_mmlu_pro_50,
            "benchmark/mmlu_pro/MMLU-Pro-500.json",
            "benchmark/mmlu_pro/MMLU-Pro-50.json",
        ),
    ):
        with tempfile.TemporaryDirectory() as temporary:
            rebuilt = Path(temporary) / "subset.json"
            builder(root / parent_relative, rebuilt)
            if rebuilt.read_bytes() != (root / subset_relative).read_bytes():
                raise AssertionError(
                    f"{subset_relative} is not byte-for-byte reproducible from its parent"
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
