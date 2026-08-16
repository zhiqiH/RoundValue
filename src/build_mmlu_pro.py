"""Build the pinned MMLU-Pro-500 benchmark asset used by RoundValue.

This is deliberately a build-time utility, not part of the experiment runner.
It downloads the pinned MMLU-Pro ``test`` split and writes one deterministic
JSON document that the normal collect workflow can freeze and hash.  Gold
option labels and indices remain private task fields: they are available only
to the offline scorer, never to agents.

The 500 tasks are selected deterministically from the 12,032-task pool:
category quotas are proportional to the pool using largest-remainder
rounding, and tasks are sampled evenly across each category's ``src`` tags so
one or two subjects cannot dominate the experiment.  The 60/20/20 partition
uses a fixed seed chosen to minimize category/``src`` drift between each
split and the selected pool.

Usage (from the repository root)::

    python src/build_mmlu_pro.py

The required ``datasets`` package is an offline build-time dependency only;
the runtime experiment needs nothing beyond ``httpx``, ``numpy``, and
``matplotlib``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATOR_VERSION = "roundvalue-mmlu-pro-v1"
SELECTION_SEED = "roundvalue-mmlupro500-selection-v1"
SPLIT_SEED_BASE = "roundvalue-mmlupro500-splits-v1"
SPLIT_RATIOS = {"train": 0.6, "validation": 0.2, "test": 0.2}
TOTAL = 500

# The revision is a source commit, not a mutable branch name.  The
# provenance document records this revision and a hash of every raw row, so
# rebuilding from the same pinned release reproduces the exact experiment.
SOURCE = {
    "dataset": "TIGER-Lab/MMLU-Pro",
    "revision": "b189ec765aa7ed75c8acfea42df31fdae71f97be",
    "split": "test",
    "upstream": "https://github.com/TIGER-AI-Lab/MMLU-Pro",
    "paper": "arXiv:2406.01574",
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


def _format_prompt(question: str, options: Sequence[str]) -> str:
    """Render the verbatim question and choices in the canonical task format.

    MMLU-Pro tasks carry between three and ten choices; the closing
    instruction always names the exact valid label range for that task.
    """

    labels = [chr(ord("A") + index) for index in range(len(options))]
    lines = [question, "", "Choices:"]
    for label, option in zip(labels, options, strict=False):
        lines.append(f"({label}) {option}")
    lines.append("")
    lines.append(
        f"Return only the letter of the correct choice ({labels[0]}-{labels[-1]})."
    )
    return "\n".join(lines)


def _mc_task(
    *,
    question_id: Any,
    question: str,
    options: Sequence[str],
    answer: str,
    answer_index: Any,
    category: str,
    src: str,
    split: str,
) -> dict[str, Any]:
    return {
        "task_id": f"mmlupro::test::{question_id}",
        "domain": "mmlu_pro",
        "split": split,
        "prompt": _format_prompt(question, options),
        "options": [str(option) for option in options],
        "answer_index": int(answer_index),
        "reference_answer": str(answer),
        "public_metadata": {
            "source_dataset": "MMLU-Pro",
            "source_task_id": str(question_id),
            "subject": str(category),
            "src": str(src),
            "option_count": len(options),
        },
    }


def _build_tasks(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Build all MMLU-Pro test tasks in one self-contained candidate pool."""

    tasks = [
        _mc_task(
            question_id=row["question_id"],
            question=str(row["question"]),
            options=list(row["options"]),
            answer=str(row["answer"]),
            answer_index=row["answer_index"],
            category=str(row["category"]),
            src=str(row["src"]),
            split="unassigned",
        )
        for row in rows
    ]
    if len(tasks) != 12032:
        raise ValueError("MMLU-Pro test split must produce exactly 12032 tasks")
    if len({task["task_id"] for task in tasks}) != len(tasks):
        raise ValueError("duplicate MMLU-Pro task identifiers")
    return tasks


def _subject(task: Mapping[str, Any]) -> str:
    return str(task["public_metadata"]["subject"])


def _source_tag(task: Mapping[str, Any]) -> str:
    return str(task["public_metadata"]["src"])


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


def _select_tasks(
    tasks: Sequence[dict[str, Any]], total: int, seed: str
) -> list[dict[str, Any]]:
    """Select a deterministic, category-stratified task pool."""

    by_subject: dict[str, list[dict[str, Any]]] = {}
    for task in tasks:
        by_subject.setdefault(_subject(task), []).append(task)
    subject_counts = {subject: len(items) for subject, items in by_subject.items()}
    targets = _proportional_counts(subject_counts, total)
    if sum(targets.values()) != total:
        raise ValueError(f"subject targets must sum to {total}: {targets}")

    selected: list[dict[str, Any]] = []
    for subject, count in sorted(targets.items()):
        ordered = sorted(
            by_subject[subject],
            key=lambda task: (
                _source_tag(task),
                _sha256_text(f"{seed}:{task['task_id']}"),
            ),
        )
        selected.extend(_even_spaced(ordered, count))
    if len(selected) != total or len({task["task_id"] for task in selected}) != total:
        raise ValueError(f"selection must produce exactly {total} unique tasks")
    return selected


def _split_sizes(total: int, ratios: Mapping[str, float]) -> dict[str, int]:
    train = math.floor(total * ratios["train"])
    validation = math.floor(total * ratios["validation"])
    return {"train": train, "validation": validation, "test": total - train - validation}


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
    src_target = _distribution(selected, _source_tag)
    total_drift = 0.0
    for split in sizes:
        split_tasks = [task for task in copied if task["split"] == split]
        total_drift += _tvd(
            _distribution(split_tasks, _subject), subject_target, len(split_tasks)
        )
        total_drift += 0.5 * _tvd(
            _distribution(split_tasks, _source_tag), src_target, len(split_tasks)
        )
    return total_drift


def _choose_split_seed(
    selected: Sequence[Mapping[str, Any]], sizes: Mapping[str, int]
) -> str:
    candidates = [SPLIT_SEED_BASE] + [
        f"{SPLIT_SEED_BASE}:{index}" for index in range(200)
    ]
    return min(
        candidates,
        key=lambda seed: (_split_drift(selected, seed, sizes), seed),
    )


def _split_counts(tasks: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts = {"train": 0, "validation": 0, "test": 0}
    for task in tasks:
        counts[str(task["split"])] += 1
    return counts


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build(*, output_root: Path, cache_dir: Path | None) -> dict[str, Any]:
    raw_rows = _load_dataset(SOURCE, cache_dir=cache_dir)
    candidate_tasks = _build_tasks(raw_rows)
    selected = _select_tasks(candidate_tasks, TOTAL, SELECTION_SEED)

    sizes = _split_sizes(TOTAL, SPLIT_RATIOS)
    split_seed = _choose_split_seed(selected, sizes)
    assigned = [dict(task) for task in selected]
    _assign_splits(assigned, split_seed, sizes)
    actual = _split_counts(assigned)
    if actual != sizes:
        raise ValueError(f"unexpected MMLU-Pro-500 split sizes: {actual}")

    subject_counts = Counter(_subject(task) for task in selected)
    src_counts = Counter(_source_tag(task) for task in selected)
    document: dict[str, Any] = {
        "schema_version": "1.0",
        "dataset_id": "MMLU-Pro-500",
        "benchmark_id": "MMLU-Pro-500-RoundValue-v1",
        "domain": "mmlu_pro",
        "purpose": (
            "A deterministic, category-stratified 500-task MMLU-Pro subset with "
            "its own 60/20/20 train/validation/test partition."
        ),
        "generator": GENERATOR_VERSION,
        "split_protocol": {
            "seed": split_seed,
            "ratios": dict(SPLIT_RATIOS),
            "stratification": "category- and src-stratified selection from the pinned MMLU-Pro test split",
            "counts": dict(actual),
        },
        "tasks": assigned,
        "provenance": {
            "schema_version": "1.0",
            "generator": GENERATOR_VERSION,
            "dataset_id": "MMLU-Pro-500",
            "domain": "mmlu_pro",
            "selection": {
                "method": (
                    "largest-remainder category counts; even-spaced sampling "
                    "across src tags within each category"
                ),
                "seed": SELECTION_SEED,
                "total": TOTAL,
                "category_counts": dict(subject_counts),
                "src_counts": dict(src_counts),
            },
            "split": {
                "seed": split_seed,
                "ratios": dict(SPLIT_RATIOS),
                "stratification": "category- and src-stratified selection from the pinned MMLU-Pro test split",
                "counts": dict(actual),
            },
            "sources": {"mmlu_pro": dict(SOURCE)},
            "raw_source_record_sha256": {
                "mmlu_pro": _records_hash(raw_rows),
            },
            "test_task_ids": [
                task["task_id"] for task in assigned if task["split"] == "test"
            ],
        },
    }
    _write_json(output_root / "mmlu_pro" / "MMLU-Pro-500.json", document)
    return {
        "generator": GENERATOR_VERSION,
        "selection_seed": SELECTION_SEED,
        "split_seed": split_seed,
        "split_ratios": dict(SPLIT_RATIOS),
        "category_counts": dict(subject_counts),
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
