"""Build the deterministic MATH-50 subset derived from the pinned MATH-500 asset.

This is a local, offline utility.  It reads ``benchmark/math/MATH-500.json``
and writes ``benchmark/math/MATH-50.json`` without contacting the network.

The subset is representative rather than a random convenience sample:

* subject counts are proportional to MATH-500 using largest-remainder
  rounding, so every subject keeps its relative weight;
* within each subject, tasks are ordered by difficulty and sampled evenly
  across that ordering, so easy/hard problems stay represented;
* the 60/20/20 split is assigned by a deterministic task-id hash, and the
  seed is chosen to minimize distribution drift (subject and difficulty)
  between each split and the subset as a whole.

Task IDs, prompts, reference answers, and public metadata are copied verbatim
from the parent so every MATH-50 problem remains traceable to MATH-500.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATOR_VERSION = "roundvalue-real-benchmarks-v3"
SELECTION_SEED = "roundvalue-math50-selection-v1"
SPLIT_SEED_BASE = "roundvalue-math50-splits-v1"
SPLIT_RATIOS = {"train": 0.6, "validation": 0.2, "test": 0.2}
TOTAL = 50


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _subject(task: Mapping[str, Any]) -> str:
    return str(task["public_metadata"]["subject"])


def _proportional_counts(groups: Mapping[str, int], total: int) -> dict[str, int]:
    denominator = sum(groups.values())
    counts: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    allocated = 0
    for key, count in groups.items():
        exact = total * count / denominator
        counts[key] = math.floor(exact)
        allocated += counts[key]
        remainders.append((exact - counts[key], key))
    for _, key in sorted(remainders, key=lambda item: (-item[0], item[1])):
        if allocated >= total:
            break
        counts[key] += 1
        allocated += 1
    return counts


def _even_spaced(items: Sequence[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    if count >= len(items):
        return list(items)
    if count == 1:
        return [items[len(items) // 2]]
    step = (len(items) - 1) / (count - 1)
    indices = sorted({min(len(items) - 1, round(index * step)) for index in range(count)})
    return [items[index] for index in indices]


def _assign_splits(
    tasks: Sequence[dict[str, Any]], seed: str, sizes: Mapping[str, int]
) -> list[dict[str, Any]]:
    ordered = sorted(tasks, key=lambda task: _sha256_text(f"{seed}:{task['task_id']}"))
    cursor = 0
    for split in ("train", "validation", "test"):
        stop = cursor + sizes[split]
        for task in ordered[cursor:stop]:
            task["split"] = split
        cursor = stop
    return list(ordered)


def _distribution(tasks: Sequence[Mapping[str, Any]], key: Any) -> Counter:
    return Counter(key(task) for task in tasks)


def _tvd(
    actual: Mapping[Any, int],
    expected: Mapping[Any, int],
    actual_total: int,
) -> float:
    expected_total = sum(expected.values())
    keys = set(actual) | set(expected)
    return 0.5 * sum(
        abs(actual.get(key, 0) / actual_total - expected.get(key, 0) / expected_total)
        for key in keys
    )


def _split_drift(
    selected: Sequence[Mapping[str, Any]],
    candidate_seed: str,
    sizes: Mapping[str, int],
) -> float:
    copied = [dict(task) for task in selected]
    _assign_splits(copied, candidate_seed, sizes)
    subject_target = _distribution(selected, _subject)
    difficulty_target = _distribution(selected, lambda task: task["difficulty"])
    total_drift = 0.0
    for split in sizes:
        split_tasks = [task for task in copied if task["split"] == split]
        total_drift += _tvd(
            _distribution(split_tasks, _subject), subject_target, len(split_tasks)
        )
        total_drift += 0.5 * _tvd(
            _distribution(split_tasks, lambda task: task["difficulty"]),
            difficulty_target,
            len(split_tasks),
        )
    return total_drift


def _split_sizes(total: int, ratios: Mapping[str, float]) -> dict[str, int]:
    train = math.floor(total * ratios["train"])
    validation = math.floor(total * ratios["validation"])
    return {"train": train, "validation": validation, "test": total - train - validation}


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build(parent_path: Path, output_path: Path) -> dict[str, Any]:
    parent = _read_json(parent_path)
    if parent.get("dataset_id") != "MATH-500" or len(parent["tasks"]) != 500:
        raise ValueError("MATH-50 must be derived from the 500-task MATH-500 asset")

    source_tasks: dict[str, dict[str, Any]] = {
        str(task["task_id"]): dict(task) for task in parent["tasks"]
    }
    if len(source_tasks) != 500:
        raise ValueError("MATH-500 contains duplicate task identifiers")

    by_subject: dict[str, list[dict[str, Any]]] = {}
    for task in source_tasks.values():
        by_subject.setdefault(_subject(task), []).append(task)
    subject_counts = {subject: len(tasks) for subject, tasks in by_subject.items()}
    targets = _proportional_counts(subject_counts, TOTAL)
    if sum(targets.values()) != TOTAL:
        raise ValueError(f"subject targets must sum to {TOTAL}: {targets}")

    selected: list[dict[str, Any]] = []
    for subject, count in sorted(targets.items()):
        ordered = sorted(
            by_subject[subject],
            key=lambda task: (
                int(task["difficulty"]) if isinstance(task["difficulty"], int) else 0,
                str(task["task_id"]),
            ),
        )
        selected.extend(_even_spaced(ordered, count))
    if len(selected) != TOTAL or len({task["task_id"] for task in selected}) != TOTAL:
        raise ValueError(f"selection must produce exactly {TOTAL} unique tasks")

    sizes = _split_sizes(TOTAL, SPLIT_RATIOS)
    candidates = [SPLIT_SEED_BASE] + [
        f"{SPLIT_SEED_BASE}:{index}" for index in range(200)
    ]
    split_seed = min(
        candidates,
        key=lambda seed: (_split_drift(selected, seed, sizes), seed),
    )
    assigned = [dict(task) for task in selected]
    _assign_splits(assigned, split_seed, sizes)
    split_counts = Counter(task["split"] for task in assigned)
    if split_counts != Counter(sizes):
        raise ValueError(f"unexpected split counts: {split_counts}")

    parent_hash = hashlib.sha256(parent_path.read_bytes()).hexdigest()
    document: dict[str, Any] = {
        "schema_version": "1.0",
        "dataset_id": "MATH-50",
        "benchmark_id": "MATH-50-RoundValue-v1",
        "domain": "math",
        "purpose": (
            "A deterministic, subject-stratified 50-task subset of MATH-500, "
            "split 60/20/20, for fast validation runs before full collection."
        ),
        "generator": GENERATOR_VERSION,
        "split_protocol": {
            "seed": split_seed,
            "ratios": dict(SPLIT_RATIOS),
            "stratification": "subject-stratified subset of MATH-500",
            "counts": dict(split_counts),
        },
        "tasks": assigned,
        "provenance": {
            "schema_version": "1.0",
            "generator": GENERATOR_VERSION,
            "dataset_id": "MATH-50",
            "domain": "math",
            "parent": {
                "dataset_id": "MATH-500",
                "file": str(parent_path.relative_to(PROJECT_ROOT)),
                "sha256": parent_hash,
            },
            "selection": {
                "method": (
                    "largest-remainder subject counts; even-spaced sampling by "
                    "difficulty within each subject"
                ),
                "seed": SELECTION_SEED,
                "total": TOTAL,
                "subject_counts": dict(targets),
            },
            "split": {
                "seed": split_seed,
                "ratios": dict(SPLIT_RATIOS),
                "stratification": "subject-stratified subset of MATH-500",
                "counts": dict(split_counts),
            },
            "sources": parent.get("provenance", {}).get("sources", {}),
            "raw_source_record_sha256": parent.get("provenance", {}).get(
                "raw_source_record_sha256", {}
            ),
            "test_task_ids": [
                task["task_id"] for task in assigned if task["split"] == "test"
            ],
        },
    }
    _write_json(output_path, document)
    return {
        "output": str(output_path.relative_to(PROJECT_ROOT)),
        "subject_counts": dict(targets),
        "split_counts": dict(split_counts),
        "split_seed": split_seed,
    }


def main() -> int:
    summary = build(
        PROJECT_ROOT / "benchmark" / "math" / "MATH-500.json",
        PROJECT_ROOT / "benchmark" / "math" / "MATH-50.json",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
