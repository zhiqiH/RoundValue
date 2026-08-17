"""Build the pinned LogiQA-500 benchmark asset for RoundValue.

The source is the **LogiQA 2.0 English MRC** dataset from the official
``csitfun/LogiQA2.0`` repository (``logiqa/DATA/LOGIQA``).  This is not
LogiQA v1 and not the NLI conversion.  The upstream train/dev/test file
boundaries are preserved verbatim:

* RoundValue train      <- official train (300 selected records)
* RoundValue validation <- official dev   (100 selected records)
* RoundValue test       <- official test  (100 selected records)

Selection is deterministic within each official split: records are stratified
by the exact set of positive reasoning-type annotations (``unannotated`` when
none exist), allocated with largest-remainder proportional counts, and ordered
by a stable hash of the selection seed plus the task identifier.  Upstream
reasoning-type strings, including their spelling, are preserved as data.

Usage (from the repository root)::

    python src/build_logiqa.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from benchmark_build_utils import (
    download_if_missing,
    proportional_counts,
    records_hash,
    select_hash_ordered,
    sha256_text,
    write_json,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = PROJECT_ROOT / ".cache" / "upstream" / "sources"

GENERATOR_VERSION = "roundvalue-logiqa-v1"
SELECTION_SEED = "roundvalue-logiqa500-selection-v1"
TOTAL = 500
SPLIT_TARGETS = {"train": 300, "validation": 100, "test": 100}
OPTION_LETTERS = ("A", "B", "C", "D")
EXPECTED_OPTION_COUNT = 4

SOURCE = {
    "project": "csitfun/LogiQA2.0",
    "upstream": "https://github.com/csitfun/LogiQA2.0",
    "revision": "955e1d3df6c59d9bfb44d9913da1e1a27ec14e18",
    "paper": "IEEE/ACM TASLP 31:2947-2962 (2023), doi:10.1109/TASLP.2023.3293046",
    "license": "CC BY-NC-SA 4.0",
    "subset": "LogiQA 2.0 English MRC",
    "files": {
        "train.txt": {
            "path": "logiqa/DATA/LOGIQA/train.txt",
            "sha256": "98eb412e8ed53b3d65da5ef75b00b7a0bbdea7970c05ad699291a2a0510922de",
        },
        "dev.txt": {
            "path": "logiqa/DATA/LOGIQA/dev.txt",
            "sha256": "bbefb563b7ddc02640ccdc314c1315d5727dba48539d0ecdd126fa351e511b09",
        },
        "test.txt": {
            "path": "logiqa/DATA/LOGIQA/test.txt",
            "sha256": "71940b37ae0184b677c253a148d57ad4e75d6113447b1563c2ca82483e4e4f8d",
        },
    },
}

OFFICIAL_TO_ROUNDVALUE = {"train": "train", "dev": "validation", "test": "test"}


def _raw_url(name: str) -> str:
    info = SOURCE["files"][name]
    return (
        "https://raw.githubusercontent.com/csitfun/LogiQA2.0/"
        f"{SOURCE['revision']}/{info['path']}"
    )


def _type_signature(record: Mapping[str, Any]) -> tuple[str, ...]:
    raw_type = record.get("type")
    if not isinstance(raw_type, dict):
        return ("unannotated",)
    positive = tuple(sorted(str(key) for key, value in raw_type.items() if value))
    return positive or ("unannotated",)


def _canonical_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Preserve the upstream record, including reasoning-type spelling."""

    return dict(record)


def _format_prompt(
    passage: str, question: str, options: Sequence[str]
) -> str:
    lines = ["Passage:", passage, "", "Question:", question, "", "Choices:"]
    for label, option in zip(OPTION_LETTERS, options, strict=False):
        lines.append(f"({label}) {option}")
    lines.append("")
    lines.append("Return only the letter of the correct choice (A-D).")
    return "\n".join(lines)


def _load_records(
    source_root: Path, name: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Parse and validate one official JSON-lines source file."""

    info = SOURCE["files"][name]
    target = source_root / "LogiQA2.0" / name
    download_if_missing(_raw_url(name), target, str(info["sha256"]))
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        target.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{name} line {line_number} is not valid JSON: {error}") from error
        if not isinstance(record, dict):
            raise ValueError(f"{name} line {line_number} is not a JSON object")
        records.append(record)
    return _validate_records(records, name)


def _validate_records(
    records: Sequence[dict[str, Any]], name: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    validated: list[dict[str, Any]] = []
    canonical: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        where = f"{name} record {index}"
        if isinstance(record.get("id"), bool) or not isinstance(record.get("id"), int):
            raise ValueError(f"{where} has no integer id")
        if (
            isinstance(record.get("answer"), bool)
            or not isinstance(record.get("answer"), int)
            or not 0 <= record["answer"] < EXPECTED_OPTION_COUNT
        ):
            raise ValueError(f"{where} has no valid 0-3 answer index")
        options = record.get("options")
        if not isinstance(options, list) or len(options) != EXPECTED_OPTION_COUNT:
            raise ValueError(f"{where} must carry exactly {EXPECTED_OPTION_COUNT} options")
        for option in options:
            if not isinstance(option, str) or not option.strip():
                raise ValueError(f"{where} has an empty option")
        if not isinstance(record.get("text"), str) or not record["text"].strip():
            raise ValueError(f"{where} has no passage text")
        if not isinstance(record.get("question"), str) or not record["question"].strip():
            raise ValueError(f"{where} has no question text")
        validated.append(dict(record))
        canonical.append(_canonical_record(record))
    return validated, canonical


def _canonical_json(record: Mapping[str, Any]) -> str:
    from benchmark_build_utils import canonical_json

    return canonical_json(dict(record))


def _build_tasks(
    records: Sequence[dict[str, Any]], official_split: str, roundvalue_split: str
) -> list[dict[str, Any]]:
    """Render validated upstream records into RoundValue MC tasks.

    Upstream ids are not unique inside one source file (the same id can name
    several different questions).  Ids that are unique keep the readable
    ``logiqa2::<split>::<id>`` form; every id that collides receives a stable
    content-hash suffix, so task identifiers never depend on record order.
    """

    by_id: dict[int, list[Mapping[str, Any]]] = {}
    for record in records:
        by_id.setdefault(int(record["id"]), []).append(record)
    unique_ids = {
        raw_id for raw_id, group in by_id.items() if len(group) == 1
    }

    tasks: list[dict[str, Any]] = []
    used: set[str] = set()
    for record in records:
        raw_id = int(record["id"])
        if raw_id in unique_ids:
            task_id = f"logiqa2::{official_split}::{raw_id}"
        else:
            digest = sha256_text(_canonical_json(record))[:8]
            task_id = f"logiqa2::{official_split}::{raw_id}::{digest}"
        if task_id in used:
            raise ValueError(f"duplicate resolved task id {task_id}")
        used.add(task_id)
        options = [str(option) for option in record["options"]]
        answer_index = int(record["answer"])
        tasks.append(
            {
                "task_id": task_id,
                "domain": "logiqa",
                "split": roundvalue_split,
                "prompt": _format_prompt(
                    str(record["text"]), str(record["question"]), options
                ),
                "options": options,
                "answer_index": answer_index,
                "reference_answer": OPTION_LETTERS[answer_index],
                "public_metadata": {
                    "source_dataset": "LogiQA 2.0 English MRC",
                    "source_task_id": str(raw_id),
                    "source_split": official_split,
                    "reasoning_types": list(_type_signature(record)),
                    "option_count": len(options),
                },
            }
        )
    return tasks


def build(*, output_root: Path, source_root: Path | None) -> dict[str, Any]:
    root = source_root or DEFAULT_SOURCE_ROOT
    observed: dict[str, int] = {}
    raw_hashes: dict[str, str] = {}
    selected_by_split: dict[str, list[dict[str, Any]]] = {}
    signature_selection: Counter = Counter()
    type_distribution: Counter = Counter()

    for official, roundvalue in OFFICIAL_TO_ROUNDVALUE.items():
        name = "train.txt" if official == "train" else f"{official}.txt"
        records, canonical = _load_records(root, name)
        observed[official] = len(records)
        raw_hashes[name] = records_hash(canonical)
        type_distribution.update(_type_signature(record) for record in records)
        tasks = _build_tasks(records, official, roundvalue)
        if len(tasks) != len(records) or len({task["task_id"] for task in tasks}) != len(tasks):
            raise ValueError(f"{name} produced non-unique task identifiers")

        by_signature: dict[tuple[str, ...], list[dict[str, Any]]] = {}
        for task, record in zip(tasks, records, strict=False):
            by_signature.setdefault(_type_signature(record), []).append(task)
        signature_counts = {
            signature: len(items) for signature, items in by_signature.items()
        }
        targets = proportional_counts(
            {_signature_key(signature): count for signature, count in signature_counts.items()},
            SPLIT_TARGETS[roundvalue],
        )
        selected: list[dict[str, Any]] = []
        for signature, count in sorted(
            targets.items(), key=lambda item: (item[0],)
        ):
            if count <= 0:
                continue
            signature_tuple = _signature_from_key(signature)
            chosen = select_hash_ordered(
                by_signature[signature_tuple],
                count,
                SELECTION_SEED,
                lambda task: str(task["task_id"]),
            )
            selected.extend(chosen)
            signature_selection[f"{roundvalue}::{signature}"] = len(chosen)
        if len(selected) != SPLIT_TARGETS[roundvalue]:
            raise ValueError(
                f"{name} selection produced {len(selected)} tasks, "
                f"expected {SPLIT_TARGETS[roundvalue]}"
            )
        selected_by_split[roundvalue] = selected

    tasks = _sort_tasks(
        [task for tasks in selected_by_split.values() for task in tasks]
    )
    split_counts = Counter(str(task["split"]) for task in tasks)
    if split_counts != Counter(SPLIT_TARGETS):
        raise ValueError(f"unexpected LogiQA-500 split counts: {split_counts}")

    document: dict[str, Any] = {
        "schema_version": "1.0",
        "dataset_id": "LogiQA-500",
        "benchmark_id": "LogiQA-500-RoundValue-v1",
        "domain": "logiqa",
        "purpose": (
            "A deterministic 500-task selection from the official LogiQA 2.0 "
            "English MRC train/dev/test files, preserving the upstream split "
            "boundaries as RoundValue train/validation/test."
        ),
        "generator": GENERATOR_VERSION,
        "split_protocol": {
            "ratios": {"train": 0.6, "validation": 0.2, "test": 0.2},
            "stratification": (
                "official LogiQA 2.0 English MRC train/dev/test boundaries; "
                "reasoning-type-signature stratified selection within each"
            ),
            "counts": dict(split_counts),
        },
        "tasks": tasks,
        "provenance": {
            "schema_version": "1.0",
            "generator": GENERATOR_VERSION,
            "dataset_id": "LogiQA-500",
            "domain": "logiqa",
            "license": {
                "name": "CC BY-NC-SA 4.0",
                "notice": (
                    "LogiQA 2.0 is licensed under the Creative Commons "
                    "Attribution-NonCommercial-ShareAlike 4.0 International "
                    "License; see the upstream repository README."
                ),
            },
            "attribution": (
                "LogiQA 2.0 -- An Improved Dataset for Logical Reasoning in "
                "Natural Language Understanding (Liu et al., IEEE/ACM TASLP "
                "2023); English MRC files from csitfun/LogiQA2.0."
            ),
            "selection": {
                "method": (
                    "largest-remainder reasoning-type-signature counts; stable "
                    "hash ordering of selection_seed + task_id within each "
                    "signature and official split"
                ),
                "seed": SELECTION_SEED,
                "total": TOTAL,
                "official_split_counts": dict(observed),
                "selected_split_counts": dict(split_counts),
                "signature_selection_counts": dict(sorted(signature_selection.items())),
                "reasoning_type_distribution": {
                    "::".join(signature): count
                    for signature, count in sorted(type_distribution.items())
                },
            },
            "split": {
                "method": "official LogiQA 2.0 English MRC file boundaries",
                "counts": dict(split_counts),
            },
            "sources": {"logiqa2_english_mrc": dict(SOURCE)},
            "raw_source_record_sha256": dict(raw_hashes),
            "test_task_ids": [
                str(task["task_id"]) for task in tasks if task["split"] == "test"
            ],
        },
    }
    write_json(output_root / "logiqa2" / "LogiQA-500.json", document)
    return {
        "generator": GENERATOR_VERSION,
        "official_split_counts": dict(observed),
        "selected_split_counts": dict(split_counts),
        "signature_selection_counts": dict(sorted(signature_selection.items())),
    }


def _signature_key(signature: tuple[str, ...]) -> str:
    return "::".join(signature)


def _signature_from_key(key: str) -> tuple[str, ...]:
    return tuple(key.split("::"))


def _sort_tasks(tasks: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    order = {"train": 0, "validation": 1, "test": 2}
    return sorted(
        tasks,
        key=lambda task: (order[str(task["split"])], str(task["task_id"])),
    )


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
