"""Build the pinned, real-data MATH-500 benchmark asset used by RoundValue.

This is deliberately a build-time utility, not part of the experiment runner.
It downloads the public MATH-500 source and writes one deterministic JSON
document that the normal collect workflow can freeze and hash.  Reference
answers remain private task fields: they are available only to the offline
scorer, never to agents.

Usage (from the repository root)::

    python src/build_real_benchmarks.py

The required ``datasets`` package is an offline build-time dependency only;
the runtime experiment needs nothing beyond ``httpx``, ``numpy``, and
``matplotlib``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
import re
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATOR_VERSION = "roundvalue-real-benchmarks-v3"
DATASET_SPLIT_SEED = "roundvalue-dataset-splits-v1"
DATASET_SPLIT_RATIOS = {"train": 0.6, "validation": 0.2, "test": 0.2}

# Revisions are source commits, not mutable branch names.  The MATH-500 mirror
# reproduces OpenAI's PRM800K held-out split; the provenance document records
# that upstream source and checksum alongside this exact mirror revision.
SOURCES = {
    "math500": {
        "dataset": "HuggingFaceH4/MATH-500",
        "revision": "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
        "split": "test",
        "upstream": "https://github.com/openai/prm800k/tree/main/prm800k/math_splits",
        "upstream_sha256": "35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132",
    },
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _records_hash(records: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(_canonical_json(dict(record)).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _difficulty(value: Any) -> int | str:
    match = re.search(r"(\d+)", str(value))
    return int(match.group(1)) if match else str(value)


def _load_dataset(source: Mapping[str, str], *, cache_dir: Path | None) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "benchmark construction needs the build-time dependency: "
            "python -m pip install 'datasets>=3,<5'"
        ) from error
    dataset = load_dataset(
        str(source["dataset"]),
        revision=str(source["revision"]),
        split=str(source["split"]),
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )
    return [dict(row) for row in dataset]


def _math_task(
    *,
    task_id: str,
    problem: str,
    answer: str,
    subject: str,
    level: Any,
    split: str,
    source_task_id: str,
) -> dict[str, Any]:
    level_text = str(level)
    return {
        "task_id": task_id,
        "domain": "math",
        "split": split,
        "prompt": problem,
        "reference_answer": answer,
        "difficulty": _difficulty(level),
        "public_metadata": {
            "source_dataset": "MATH-500",
            "source_task_id": source_task_id,
            "subject": subject,
            "level": level_text,
        },
    }


def _build_math_tasks(math500: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Build all 500 MATH-500 tasks as one self-contained math dataset."""

    tasks = [
        _math_task(
            task_id=f"math500::{row['unique_id']}",
            problem=str(row["problem"]),
            answer=str(row["answer"]),
            subject=str(row["subject"]),
            level=row["level"],
            split="unassigned",
            source_task_id=str(row["unique_id"]),
        )
        for row in math500
    ]
    if len(tasks) != 500:
        raise ValueError("MATH-500 must produce exactly 500 tasks")
    if len({task["task_id"] for task in tasks}) != len(tasks):
        raise ValueError("duplicate MATH task identifiers")
    return tasks


def _split_sizes(total: int, ratios: Mapping[str, float]) -> dict[str, int]:
    """Turn the shared ratio into exact counts with the remainder in test."""

    train = math.floor(total * ratios["train"])
    validation = math.floor(total * ratios["validation"])
    return {"train": train, "validation": validation, "test": total - train - validation}


def _assign_splits(tasks: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deterministically split MATH-500 into train/validation/test.

    Ordering uses a fixed seed and task-id hash, so rebuilding from the same
    pinned release reproduces the exact partition.
    """

    sizes = _split_sizes(len(tasks), DATASET_SPLIT_RATIOS)
    ordered = sorted(
        tasks,
        key=lambda task: _sha256_text(f"{DATASET_SPLIT_SEED}:{task['task_id']}"),
    )
    cursor = 0
    for split in ("train", "validation", "test"):
        stop = cursor + sizes[split]
        for task in ordered[cursor:stop]:
            task["split"] = split
        cursor = stop
    return list(tasks)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _split_counts(tasks: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts = {"train": 0, "validation": 0, "test": 0}
    for task in tasks:
        split = str(task["split"])
        counts[split] = counts.get(split, 0) + 1
    return counts


def build(*, output_root: Path, cache_dir: Path | None) -> dict[str, Any]:
    raw_math500 = _load_dataset(SOURCES["math500"], cache_dir=cache_dir)
    math_tasks = _build_math_tasks(raw_math500)
    _assign_splits(math_tasks)

    expected_sizes = _split_sizes(len(math_tasks), DATASET_SPLIT_RATIOS)
    actual = _split_counts(math_tasks)
    if actual != expected_sizes:
        raise ValueError(f"unexpected MATH-500 split sizes: {actual}")

    document = {
        "schema_version": "1.0",
        "dataset_id": "MATH-500",
        "benchmark_id": "MATH-500-RoundValue-v1",
        "domain": "math",
        "purpose": (
            "All 500 MATH-500 tasks as one self-contained math dataset with "
            "its own train/validation/test partition."
        ),
        "generator": GENERATOR_VERSION,
        "split_protocol": {
            "seed": DATASET_SPLIT_SEED,
            "ratios": dict(DATASET_SPLIT_RATIOS),
            "stratification": "whole dataset",
            "counts": actual,
        },
        "tasks": math_tasks,
        "provenance": {
            "schema_version": "1.0",
            "generator": GENERATOR_VERSION,
            "dataset_id": "MATH-500",
            "domain": "math",
            "split": {
                "seed": DATASET_SPLIT_SEED,
                "ratios": dict(DATASET_SPLIT_RATIOS),
                "stratification": "whole dataset",
                "counts": actual,
            },
            "sources": {"math500": SOURCES["math500"]},
            "raw_source_record_sha256": {
                "math500": _records_hash(raw_math500),
            },
            "test_task_ids": [
                task["task_id"] for task in math_tasks if task["split"] == "test"
            ],
        },
    }
    _write_json(output_root / "math" / "MATH-500.json", document)
    return {
        "generator": GENERATOR_VERSION,
        "split_seed": DATASET_SPLIT_SEED,
        "split_ratios": dict(DATASET_SPLIT_RATIOS),
        "datasets": {"MATH-500": actual},
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "benchmark",
        help="Benchmark directory to populate (default: repository benchmark/).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional Hugging Face datasets cache directory.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    summary = build(
        output_root=args.output_root.resolve(),
        cache_dir=args.cache_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
