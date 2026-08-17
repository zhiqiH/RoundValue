"""Small deterministic helpers shared by the RoundValue benchmark builders.

The MMLU-Pro and MATH builders predate this module and keep their own local
implementations so their historical artifacts remain byte-for-byte stable.
The HARP and LogiQA builders reuse these helpers instead of duplicating
sampling and provenance code a third and fourth time.

Nothing in this module contacts a model provider.  Network access happens
only through :func:`download_if_missing`, which is strictly pinned to an
immutable upstream commit and a known artifact SHA-256.
"""

from __future__ import annotations

import hashlib
import json
import math
import urllib.request
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


SPLIT_RATIOS = {"train": 0.6, "validation": 0.2, "test": 0.2}


def canonical_json(value: Any) -> str:
    """Serialize one record with a stable key order for hashing."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def records_hash(records: Iterable[Mapping[str, Any]]) -> str:
    """Hash every canonicalized raw record, one per line, in iteration order."""

    digest = hashlib.sha256()
    for record in records:
        digest.update(canonical_json(dict(record)).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def proportional_counts(groups: Mapping[str, int], total: int) -> dict[str, int]:
    """Largest-remainder proportional allocation that always sums to ``total``."""

    denominator = sum(groups.values())
    if denominator <= 0:
        raise ValueError("proportional allocation requires at least one item")
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
    if sum(counts.values()) != total:
        raise ValueError(f"proportional targets must sum to {total}: {counts}")
    return counts


def split_sizes(total: int, ratios: Mapping[str, float]) -> dict[str, int]:
    train = math.floor(total * ratios["train"])
    validation = math.floor(total * ratios["validation"])
    return {"train": train, "validation": validation, "test": total - train - validation}


def hash_order(
    items: Sequence[Mapping[str, Any]],
    seed: str,
    task_id: Callable[[Mapping[str, Any]], str],
) -> list[Mapping[str, Any]]:
    """Stable order derived only from ``seed`` plus each task identifier."""

    return sorted(items, key=lambda item: sha256_text(f"{seed}:{task_id(item)}"))


def select_hash_ordered(
    items: Sequence[Mapping[str, Any]],
    count: int,
    seed: str,
    task_id: Callable[[Mapping[str, Any]], str],
) -> list[Mapping[str, Any]]:
    if count < 0 or count > len(items):
        raise ValueError(f"cannot select {count} of {len(items)} items")
    return hash_order(items, seed, task_id)[:count]


def distribution(
    items: Iterable[Mapping[str, Any]],
    key: Callable[[Mapping[str, Any]], Any],
) -> Counter:
    return Counter(key(item) for item in items)


def tvd(
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


def assign_splits(
    tasks: Sequence[dict[str, Any]],
    seed: str,
    sizes: Mapping[str, int],
    task_id: Callable[[Mapping[str, Any]], str],
) -> list[dict[str, Any]]:
    """Partition tasks into train/validation/test blocks in stable hash order."""

    ordered = hash_order(tasks, seed, task_id)
    cursor = 0
    for split in ("train", "validation", "test"):
        stop = cursor + sizes[split]
        for task in ordered[cursor:stop]:
            task["split"] = split
        cursor = stop
    return ordered


def choose_split_seed(
    selected: Sequence[Mapping[str, Any]],
    sizes: Mapping[str, int],
    seed_base: str,
    stratify: Callable[[Mapping[str, Any]], Any],
    task_id: Callable[[Mapping[str, Any]], str],
    *,
    candidates: int = 200,
) -> str:
    """Pick the candidate seed whose split composition drifts least."""

    def drift(candidate_seed: str) -> float:
        copied = [dict(task) for task in selected]
        assign_splits(copied, candidate_seed, sizes, task_id)
        target = distribution(selected, stratify)
        total = 0.0
        for split in sizes:
            split_tasks = [task for task in copied if task["split"] == split]
            total += tvd(distribution(split_tasks, stratify), target, len(split_tasks))
        return total

    names = [seed_base] + [f"{seed_base}:{index}" for index in range(candidates)]
    return min(names, key=lambda seed: (drift(seed), seed))


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def download_if_missing(url: str, target: Path, expected_sha256: str) -> None:
    """Fetch one pinned artifact, refusing any content that fails validation."""

    if target.exists() and file_sha256(target) == expected_sha256:
        return
    if target.exists():
        raise ValueError(
            f"cached source {target} does not match the pinned SHA-256; "
            "delete it to force a fresh download"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".download")
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            data = response.read()
        if sha256_bytes(data) != expected_sha256:
            raise ValueError(
                f"downloaded {url} does not match the pinned SHA-256 "
                f"{expected_sha256}"
            )
        temporary.write_bytes(data)
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)
