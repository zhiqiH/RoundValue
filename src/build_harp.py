"""Build the pinned HARP-500 multiple-choice benchmark asset for RoundValue.

HARP (Human Annotated Reasoning Problems) is a math-competition benchmark from
the official ``aadityasingh/HARP`` repository.  This builder consumes only the
official multiple-choice asset ``HARP_mcq.jsonl.zip`` and pins one immutable
upstream commit plus the artifact SHA-256, so a rebuild from the same revision
reproduces the committed JSON byte for byte.

The 500 tasks are selected deterministically with a level x subject
stratification (largest-remainder proportional allocation followed by a
stable hash ordering).  Human solutions, gold answers, and source identifiers
never reach Agent-facing task views; they remain offline metadata only.

Usage (from the repository root)::

    python src/build_harp.py
"""

from __future__ import annotations

import argparse
import json
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from benchmark_build_utils import (
    assign_splits,
    choose_split_seed,
    download_if_missing,
    proportional_counts,
    records_hash,
    select_hash_ordered,
    split_sizes,
    write_json,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = PROJECT_ROOT / ".cache" / "upstream" / "sources"

GENERATOR_VERSION = "roundvalue-harp-v1"
SELECTION_SEED = "roundvalue-harp500-selection-v1"
SPLIT_SEED_BASE = "roundvalue-harp500-splits-v1"
SPLIT_RATIOS = {"train": 0.6, "validation": 0.2, "test": 0.2}
TOTAL = 500
POOL_SIZE = 4110
OPTION_LETTERS = ("A", "B", "C", "D", "E")

SOURCE = {
    "project": "aadityasingh/HARP",
    "upstream": "https://github.com/aadityasingh/HARP",
    "revision": "dac2734ff6443bcaf3bbdcb10f13cf21ae9729c2",
    "paper": "arXiv:2412.08819",
    "license": "MIT",
    "asset": "HARP_mcq.jsonl.zip",
    "asset_sha256": "bf1456623321561a1be328042f635046cb408fdbdc5dc1b8d8090053f9ea6824",
    "raw_url": (
        "https://raw.githubusercontent.com/aadityasingh/HARP/"
        "dac2734ff6443bcaf3bbdcb10f13cf21ae9729c2/HARP_mcq.jsonl.zip"
    ),
    "inner_path": "HARP_mcq.jsonl",
}


def _task_id(record: Mapping[str, Any]) -> str:
    return f"harp::{record['year']}::{record['contest']}::{record['number']}"


def _task_stratum(task: Mapping[str, Any]) -> str:
    metadata = task.get("public_metadata") or {}
    return f"{metadata.get('level')}:{metadata.get('subject')}"


def _format_prompt(problem: str, options: Sequence[str]) -> str:
    lines = [problem, "", "Choices:"]
    for label, option in zip(OPTION_LETTERS, options, strict=False):
        lines.append(f"({label}) {option}")
    lines.append("")
    lines.append("Return only the letter of the correct choice (A-E).")
    return "\n".join(lines)


def _load_records(source_root: Path) -> list[dict[str, Any]]:
    asset = source_root / "HARP" / SOURCE["asset"]
    download_if_missing(str(SOURCE["raw_url"]), asset, str(SOURCE["asset_sha256"]))
    with zipfile.ZipFile(asset) as archive:
        names = archive.namelist()
        if names != [SOURCE["inner_path"]]:
            raise ValueError(
                f"unexpected HARP MCQ archive members: {names}"
            )
        lines = archive.read(SOURCE["inner_path"]).decode("utf-8").splitlines()

    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"HARP MCQ line {line_number} is not valid JSON: {error}"
            ) from error
        if not isinstance(record, dict):
            raise ValueError(f"HARP MCQ line {line_number} is not a JSON object")
        _validate_source_record(record, line_number)
        records.append(record)
    if len(records) != POOL_SIZE:
        raise ValueError(
            f"HARP MCQ pool must contain {POOL_SIZE} records, got {len(records)}"
        )
    if len({_task_id(record) for record in records}) != len(records):
        raise ValueError("HARP MCQ pool contains duplicate task identifiers")
    return records


def _validate_source_record(record: dict[str, Any], line_number: int) -> None:
    """Fail loudly whenever the official gold answer is not unambiguous."""

    where = f"HARP MCQ line {line_number}"
    problem = record.get("problem")
    if not isinstance(problem, str) or not problem.strip():
        raise ValueError(f"{where} has no problem text")
    choices = record.get("choices")
    if not isinstance(choices, dict):
        raise ValueError(f"{where} has no choices mapping")
    if list(choices) != list(OPTION_LETTERS):
        raise ValueError(
            f"{where} choices must be exactly {list(OPTION_LETTERS)}, "
            f"got {list(choices)}"
        )
    for label in OPTION_LETTERS:
        if not isinstance(choices[label], str) or not choices[label].strip():
            raise ValueError(f"{where} choice {label} is not non-empty text")
    answer_choice = record.get("answer_choice")
    if answer_choice not in choices:
        raise ValueError(
            f"{where} answer_choice {answer_choice!r} is not exactly one option"
        )
    for field in ("year", "contest", "subject"):
        if not isinstance(record.get(field), str) or not record[field].strip():
            raise ValueError(f"{where} has no valid {field}")
    if isinstance(record.get("number"), bool) or not isinstance(record.get("number"), int):
        raise ValueError(f"{where} has no valid problem number")
    if isinstance(record.get("level"), bool) or not isinstance(record.get("level"), int):
        raise ValueError(f"{where} has no valid difficulty level")


def _build_tasks(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for record in records:
        options = [str(record["choices"][label]) for label in OPTION_LETTERS]
        answer_choice = str(record["answer_choice"])
        task_id = _task_id(record)
        tasks.append(
            {
                "task_id": task_id,
                "domain": "harp",
                "split": "unassigned",
                "prompt": _format_prompt(str(record["problem"]), options),
                "options": options,
                "answer_index": OPTION_LETTERS.index(answer_choice),
                "reference_answer": answer_choice,
                "public_metadata": {
                    "source_dataset": "HARP",
                    "source_task_id": task_id,
                    "source_contest": str(record["contest"]),
                    "source_year": str(record["year"]),
                    "source_number": int(record["number"]),
                    "subject": str(record["subject"]),
                    "level": int(record["level"]),
                    "multiple_choice_only": bool(record.get("multiple_choice_only")),
                    "option_count": len(options),
                },
            }
        )
    return tasks


def _sort_tasks(tasks: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    order = {"train": 0, "validation": 1, "test": 2}
    return sorted(
        tasks,
        key=lambda task: (order[str(task["split"])], str(task["task_id"])),
    )


def build(*, output_root: Path, source_root: Path | None) -> dict[str, Any]:
    root = source_root or DEFAULT_SOURCE_ROOT
    records = _load_records(root)
    tasks = _build_tasks(records)

    by_stratum: dict[str, list[dict[str, Any]]] = {}
    for task in tasks:
        by_stratum.setdefault(_task_stratum(task), []).append(task)
    stratum_counts = {
        stratum: len(items) for stratum, items in by_stratum.items()
    }
    targets = proportional_counts(stratum_counts, TOTAL)

    selected: list[dict[str, Any]] = []
    for stratum, count in sorted(targets.items()):
        selected.extend(
            select_hash_ordered(
                by_stratum[stratum],
                count,
                SELECTION_SEED,
                lambda task: str(task["task_id"]),
            )
        )
    if len(selected) != TOTAL or len({task["task_id"] for task in selected}) != TOTAL:
        raise ValueError(f"selection must produce exactly {TOTAL} unique tasks")

    sizes = split_sizes(TOTAL, SPLIT_RATIOS)
    split_seed = choose_split_seed(
        selected,
        sizes,
        SPLIT_SEED_BASE,
        _task_stratum,
        lambda task: str(task["task_id"]),
    )
    assigned = assign_splits(
        [dict(task) for task in selected],
        split_seed,
        sizes,
        lambda task: str(task["task_id"]),
    )
    actual = Counter(str(task["split"]) for task in assigned)
    if actual != Counter(sizes):
        raise ValueError(f"unexpected HARP-500 split sizes: {actual}")

    assigned = _sort_tasks(assigned)
    stratum_selection = Counter(_task_stratum(task) for task in selected)
    level_subject = Counter(_task_stratum(task) for task in selected)
    document: dict[str, Any] = {
        "schema_version": "1.0",
        "dataset_id": "HARP-500",
        "benchmark_id": "HARP-500-RoundValue-v1",
        "domain": "harp",
        "purpose": (
            "A deterministic, level x subject stratified 500-task subset of the "
            "official HARP multiple-choice pool, with its own 60/20/20 "
            "train/validation/test partition."
        ),
        "generator": GENERATOR_VERSION,
        "split_protocol": {
            "seed": split_seed,
            "ratios": dict(SPLIT_RATIOS),
            "stratification": (
                "level x subject stratified selection from the pinned official "
                "HARP MCQ pool"
            ),
            "counts": dict(actual),
        },
        "tasks": assigned,
        "provenance": {
            "schema_version": "1.0",
            "generator": GENERATOR_VERSION,
            "dataset_id": "HARP-500",
            "domain": "harp",
            "license": {
                "name": "MIT",
                "notice": (
                    "HARP is Copyright (c) 2024 Aaditya Singh, released under "
                    "the MIT License; see the upstream repository LICENSE file."
                ),
            },
            "attribution": (
                "HARP: A challenging human-annotated math reasoning benchmark "
                "(arXiv:2412.08819), official multiple-choice asset "
                "HARP_mcq.jsonl.zip from aadityasingh/HARP."
            ),
            "selection": {
                "method": (
                    "largest-remainder level x subject counts; stable hash "
                    "ordering of selection_seed + task_id within each stratum"
                ),
                "seed": SELECTION_SEED,
                "total": TOTAL,
                "pool_size": POOL_SIZE,
                "stratum_counts": dict(stratum_selection),
                "level_subject_counts": dict(level_subject),
                "subject_counts": dict(
                    Counter(task["public_metadata"]["subject"] for task in selected)
                ),
                "level_counts": dict(
                    Counter(task["public_metadata"]["level"] for task in selected)
                ),
            },
            "split": {
                "seed": split_seed,
                "ratios": dict(SPLIT_RATIOS),
                "stratification": (
                    "drift-minimized level x subject split of the selected pool"
                ),
                "counts": dict(actual),
            },
            "sources": {"harp_mcq": dict(SOURCE)},
            "raw_source_record_sha256": {
                "harp_mcq": records_hash(records),
            },
            "test_task_ids": [
                str(task["task_id"]) for task in assigned if task["split"] == "test"
            ],
        },
    }
    write_json(output_root / "harp" / "HARP-500.json", document)
    return {
        "generator": GENERATOR_VERSION,
        "pool_size": POOL_SIZE,
        "selection_seed": SELECTION_SEED,
        "split_seed": split_seed,
        "stratum_counts": dict(stratum_selection),
        "split_counts": dict(actual),
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
        "--source-root",
        type=Path,
        default=None,
        help=(
            "Directory holding the pinned upstream source files; missing files "
            "are downloaded into .cache/upstream/sources."
        ),
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    summary = build(
        output_root=args.output_root.resolve(),
        source_root=args.source_root.resolve() if args.source_root else None,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
