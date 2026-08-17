"""Build the deterministic LogiQA-50 subset of the pinned LogiQA-500 asset.

This is a local, offline utility.  It reads
``benchmark/logiqa2/LogiQA-500.json`` and writes
``benchmark/logiqa2/LogiQA-50.json`` without contacting the network.
Selection happens independently inside each parent split (30 train, 10
validation, 10 test) and is stratified by reasoning-type signature.  Every
task field, including the official source split, is copied verbatim from the
parent, so the subset is a strict, byte-for-byte reproducible part of
LogiQA-500 and never crosses an official LogiQA 2.0 split boundary.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from benchmark_build_utils import (
    proportional_counts,
    select_hash_ordered,
    write_json,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATOR_VERSION = "roundvalue-logiqa-v1"
SELECTION_SEED = "roundvalue-logiqa50-selection-v1"
SPLIT_TARGETS = {"train": 30, "validation": 10, "test": 10}
TOTAL = 50


def _signature_key(task: Mapping[str, Any]) -> str:
    metadata = task.get("public_metadata") or {}
    reasoning_types = metadata.get("reasoning_types")
    if not isinstance(reasoning_types, list) or not reasoning_types:
        return "unannotated"
    return "::".join(str(value) for value in reasoning_types)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _sort_tasks(tasks: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    order = {"train": 0, "validation": 1, "test": 2}
    return sorted(
        tasks,
        key=lambda task: (order[str(task["split"])], str(task["task_id"])),
    )


def build(parent_path: Path, output_path: Path) -> dict[str, Any]:
    parent = _read_json(parent_path)
    if parent.get("dataset_id") != "LogiQA-500" or len(parent["tasks"]) != 500:
        raise ValueError(
            "LogiQA-50 must be derived from the 500-task LogiQA-500 asset"
        )

    source_tasks: dict[str, dict[str, Any]] = {
        str(task["task_id"]): dict(task) for task in parent["tasks"]
    }
    if len(source_tasks) != 500:
        raise ValueError("LogiQA-500 contains duplicate task identifiers")

    selected: list[dict[str, Any]] = []
    signature_counts: Counter = Counter()
    for split, count in SPLIT_TARGETS.items():
        split_tasks = [task for task in source_tasks.values() if task["split"] == split]
        by_signature: dict[str, list[dict[str, Any]]] = {}
        for task in split_tasks:
            by_signature.setdefault(_signature_key(task), []).append(task)
        targets = proportional_counts(
            {signature: len(items) for signature, items in by_signature.items()},
            count,
        )
        for signature, signature_count in sorted(targets.items()):
            if signature_count <= 0:
                continue
            chosen = select_hash_ordered(
                by_signature[signature],
                signature_count,
                SELECTION_SEED,
                lambda task: str(task["task_id"]),
            )
            selected.extend(chosen)
            signature_counts[f"{split}::{signature}"] = len(chosen)

    if len(selected) != TOTAL or len({task["task_id"] for task in selected}) != TOTAL:
        raise ValueError(
            f"LogiQA-50 selection must produce exactly {TOTAL} unique tasks"
        )
    selected = _sort_tasks(selected)
    split_counts = Counter(str(task["split"]) for task in selected)
    if split_counts != Counter(SPLIT_TARGETS):
        raise ValueError(f"unexpected LogiQA-50 split counts: {split_counts}")

    parent_hash = hashlib.sha256(parent_path.read_bytes()).hexdigest()
    document: dict[str, Any] = {
        "schema_version": "1.0",
        "dataset_id": "LogiQA-50",
        "benchmark_id": "LogiQA-50-RoundValue-v1",
        "domain": "logiqa",
        "purpose": (
            "A deterministic, reasoning-type stratified 50-task subset of "
            "LogiQA-500, split 30/10/10 and preserving official LogiQA 2.0 "
            "source split boundaries, for fast validation runs."
        ),
        "generator": GENERATOR_VERSION,
        "split_protocol": {
            "seed": SELECTION_SEED,
            "stratification": (
                "reasoning-type-signature stratified selection within each "
                "LogiQA-500 split"
            ),
            "counts": dict(split_counts),
        },
        "tasks": selected,
        "provenance": {
            "schema_version": "1.0",
            "generator": GENERATOR_VERSION,
            "dataset_id": "LogiQA-50",
            "domain": "logiqa",
            "parent": {
                "dataset_id": "LogiQA-500",
                "file": str(parent_path.relative_to(PROJECT_ROOT)),
                "sha256": parent_hash,
            },
            "selection": {
                "method": (
                    "largest-remainder reasoning-type-signature counts within "
                    "each parent split; stable hash ordering within each "
                    "signature"
                ),
                "seed": SELECTION_SEED,
                "total": TOTAL,
                "signature_counts": dict(sorted(signature_counts.items())),
            },
            "split": {
                "method": (
                    "parent LogiQA-500 split retained verbatim; official "
                    "LogiQA 2.0 train/dev/test boundaries preserved"
                ),
                "counts": dict(split_counts),
            },
            "sources": parent.get("provenance", {}).get("sources", {}),
            "raw_source_record_sha256": parent.get("provenance", {}).get(
                "raw_source_record_sha256", {}
            ),
            "test_task_ids": [
                str(task["task_id"]) for task in selected if task["split"] == "test"
            ],
        },
    }
    write_json(output_path, document)
    try:
        relative_output = str(output_path.relative_to(PROJECT_ROOT))
    except ValueError:
        relative_output = str(output_path)
    return {
        "output": relative_output,
        "signature_counts": dict(sorted(signature_counts.items())),
        "split_counts": dict(split_counts),
    }


def main() -> int:
    summary = build(
        PROJECT_ROOT / "benchmark" / "logiqa2" / "LogiQA-500.json",
        PROJECT_ROOT / "benchmark" / "logiqa2" / "LogiQA-50.json",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
