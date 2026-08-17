"""Build the deterministic HARP-50 subset of the pinned HARP-500 asset.

This is a local, offline utility.  It reads ``benchmark/harp/HARP-500.json``
and writes ``benchmark/harp/HARP-50.json`` without contacting the network.
The 30/10/10 selection happens independently inside each parent split and is
stratified by the parent's ``level x subject`` structure.  Every task field
is copied verbatim from the parent, so each HARP-50 problem stays traceable
to HARP-500 and the subset is byte-for-byte reproducible.
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
GENERATOR_VERSION = "roundvalue-harp-v1"
SELECTION_SEED = "roundvalue-harp50-selection-v1"
SPLIT_TARGETS = {"train": 30, "validation": 10, "test": 10}
TOTAL = 50


def _stratum(task: Mapping[str, Any]) -> str:
    metadata = task.get("public_metadata") or {}
    return f"{metadata.get('level')}:{metadata.get('subject')}"


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
    if parent.get("dataset_id") != "HARP-500" or len(parent["tasks"]) != 500:
        raise ValueError("HARP-50 must be derived from the 500-task HARP-500 asset")

    source_tasks: dict[str, dict[str, Any]] = {
        str(task["task_id"]): dict(task) for task in parent["tasks"]
    }
    if len(source_tasks) != 500:
        raise ValueError("HARP-500 contains duplicate task identifiers")

    selected: list[dict[str, Any]] = []
    stratum_counts: Counter = Counter()
    for split, count in SPLIT_TARGETS.items():
        split_tasks = [task for task in source_tasks.values() if task["split"] == split]
        by_stratum: dict[str, list[dict[str, Any]]] = {}
        for task in split_tasks:
            by_stratum.setdefault(_stratum(task), []).append(task)
        targets = proportional_counts(
            {stratum: len(items) for stratum, items in by_stratum.items()},
            count,
        )
        for stratum, stratum_count in sorted(targets.items()):
            if stratum_count <= 0:
                continue
            chosen = select_hash_ordered(
                by_stratum[stratum],
                stratum_count,
                SELECTION_SEED,
                lambda task: str(task["task_id"]),
            )
            selected.extend(chosen)
            stratum_counts[f"{split}::{stratum}"] = len(chosen)

    if len(selected) != TOTAL or len({task["task_id"] for task in selected}) != TOTAL:
        raise ValueError(f"HARP-50 selection must produce exactly {TOTAL} unique tasks")
    selected = _sort_tasks(selected)
    split_counts = Counter(str(task["split"]) for task in selected)
    if split_counts != Counter(SPLIT_TARGETS):
        raise ValueError(f"unexpected HARP-50 split counts: {split_counts}")

    parent_hash = hashlib.sha256(parent_path.read_bytes()).hexdigest()
    document: dict[str, Any] = {
        "schema_version": "1.0",
        "dataset_id": "HARP-50",
        "benchmark_id": "HARP-50-RoundValue-v1",
        "domain": "harp",
        "purpose": (
            "A deterministic, level x subject stratified 50-task subset of "
            "HARP-500, split 30/10/10, for fast validation runs before full "
            "collection."
        ),
        "generator": GENERATOR_VERSION,
        "split_protocol": {
            "seed": SELECTION_SEED,
            "stratification": (
                "level x subject stratified selection within each HARP-500 split"
            ),
            "counts": dict(split_counts),
        },
        "tasks": selected,
        "provenance": {
            "schema_version": "1.0",
            "generator": GENERATOR_VERSION,
            "dataset_id": "HARP-50",
            "domain": "harp",
            "parent": {
                "dataset_id": "HARP-500",
                "file": str(parent_path.relative_to(PROJECT_ROOT)),
                "sha256": parent_hash,
            },
            "selection": {
                "method": (
                    "largest-remainder level x subject counts within each "
                    "parent split; stable hash ordering within each stratum"
                ),
                "seed": SELECTION_SEED,
                "total": TOTAL,
                "stratum_counts": dict(sorted(stratum_counts.items())),
            },
            "split": {
                "method": "parent HARP-500 split retained verbatim",
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
        "stratum_counts": dict(sorted(stratum_counts.items())),
        "split_counts": dict(split_counts),
    }


def main() -> int:
    summary = build(
        PROJECT_ROOT / "benchmark" / "harp" / "HARP-500.json",
        PROJECT_ROOT / "benchmark" / "harp" / "HARP-50.json",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
