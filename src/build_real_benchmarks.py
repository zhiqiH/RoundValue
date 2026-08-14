"""Build the pinned, real-data benchmark assets used by RoundValue.

This is deliberately a build-time utility, not part of the experiment runner.
It downloads only public benchmark sources and writes deterministic JSON
documents that the normal ``roundvalue --mode collect`` workflow can freeze
and hash.  EvalPlus canonical solutions and test inputs remain private task
fields: they are available only to the local scorer, never to agents.

Usage (from the repository root)::

    python src/build_real_benchmarks.py

The required ``datasets`` package is an offline build-time dependency only;
the runtime experiment needs nothing beyond ``httpx`` (plus ``numpy`` for
code scoring).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
import re
import unicodedata
from typing import Any
from urllib.request import urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATOR_VERSION = "roundvalue-real-benchmarks-v1"
MATH100_SELECTION_SEED = "roundvalue-math100-v1"
EVALPLUS_REFERENCE_IMPLEMENTATION = "evalplus==0.3.1"

# The official HumanEval+ v0.1.10 canonical implementation for HumanEval/32
# does not satisfy its own property oracle on plus input #119 at atol=1e-4.
# Keep the task because the oracle checks a meaningful mathematical property,
# but record the upstream self-check anomaly so validation does not disguise it.
KNOWN_CANONICAL_SELF_CHECK_EXCEPTIONS = {
    "humanevalplus::HumanEval_32": {
        "source_task_id": "HumanEval/32",
        "reason": "upstream canonical solution misses the find_zero property oracle on plus input 119",
    }
}

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
    "math_development": {
        "dataset": "qwedsacf/competition_math",
        "revision": "e839825f9ec5c6cfa585c654a59610969ec13993",
        "split": "train",
        "upstream": "https://github.com/hendrycks/math",
    },
    "humanevalplus": {
        "version": "v0.1.10",
        "filename": "HumanEvalPlus-v0.1.10.jsonl.gz",
        "url": "https://github.com/evalplus/humanevalplus_release/releases/download/v0.1.10/HumanEvalPlus.jsonl.gz",
        "artifact_sha256": "272720b90ac375502c8ed23cd791c2a93dfb22a911641a494da74a426c09f101",
        "upstream": "https://github.com/evalplus/humanevalplus_release/releases/tag/v0.1.10",
    },
    "mbppplus": {
        "version": "v0.2.0",
        "filename": "MbppPlus-v0.2.0.jsonl.gz",
        "url": "https://github.com/evalplus/mbppplus_release/releases/download/v0.2.0/MbppPlus.jsonl.gz",
        "artifact_sha256": "af43697e8791c4c149bdfd6b489d8b5412507551ac20e28a439f650b8225db63",
        "upstream": "https://github.com/evalplus/mbppplus_release/releases/tag/v0.2.0",
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


def _normalise_problem(problem: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", problem)).strip()


def _balanced_boxed_value(solution: str) -> str | None:
    """Return the final non-empty ``\\boxed{...}`` value in a MATH solution."""

    marker = "\\boxed"
    values: list[str] = []
    cursor = 0
    while True:
        marker_at = solution.find(marker, cursor)
        if marker_at < 0:
            break
        position = marker_at + len(marker)
        while position < len(solution) and solution[position].isspace():
            position += 1
        if position < len(solution) and solution[position] == "{":
            depth = 1
            end = position + 1
            while end < len(solution) and depth:
                if solution[end] == "{" and solution[end - 1] != "\\":
                    depth += 1
                elif solution[end] == "}" and solution[end - 1] != "\\":
                    depth -= 1
                end += 1
            if depth == 0:
                candidate = solution[position + 1 : end - 1].strip()
                if candidate:
                    values.append(candidate)
                cursor = end
                continue
        # A handful of legacy MATH examples use ``\\boxed 2`` without braces.
        bare = re.match(r"\s+([^\s$.,;]+)", solution[position:])
        if bare and bare.group(1).strip():
            values.append(bare.group(1).strip())
        cursor = marker_at + len(marker)
    return values[-1] if values else None


def _difficulty(value: Any) -> int | str:
    match = re.search(r"(\d+)", str(value))
    return int(match.group(1)) if match else str(value)


def _stable_order(items: Iterable[dict[str, Any]], *, namespace: str) -> list[dict[str, Any]]:
    return sorted(
        items,
        key=lambda item: _sha256_text(f"{MATH100_SELECTION_SEED}:{namespace}:{item['task_id']}"),
    )


def _stratified_select(
    items: Iterable[dict[str, Any]], *, count: int, namespace: str
) -> list[dict[str, Any]]:
    """Select a deterministic, round-robin sample over subject/difficulty bins."""

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        metadata = item["public_metadata"]
        groups[(str(metadata["subject"]), str(metadata["level"]))].append(item)
    ordered = {
        key: _stable_order(group, namespace=f"{namespace}:{key[0]}:{key[1]}")
        for key, group in groups.items()
    }
    selected: list[dict[str, Any]] = []
    positions = {key: 0 for key in ordered}
    while len(selected) < count:
        advanced = False
        for key in sorted(ordered):
            position = positions[key]
            if position >= len(ordered[key]):
                continue
            selected.append(ordered[key][position])
            positions[key] += 1
            advanced = True
            if len(selected) == count:
                break
        if not advanced:
            raise ValueError(f"requested {count} items from a pool of only {len(selected)}")
    return selected


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


def _load_evalplus_release(
    source: Mapping[str, str], *, cache_dir: Path
) -> list[dict[str, Any]]:
    """Load a pinned official EvalPlus JSONL release, caching its gzip artifact.

    The Hugging Face mirrors expose pre-expanded Python test programs.  For a
    formal differential evaluation we instead retain the official release's
    base/plus inputs, tolerance, and canonical oracle source.  This avoids
    platform-specific float literals becoming the source of truth.
    """

    cache_dir.mkdir(parents=True, exist_ok=True)
    artifact = cache_dir / str(source["filename"])
    if not artifact.exists():
        with urlopen(str(source["url"])) as response:  # noqa: S310 - fixed public release URL
            artifact.write_bytes(response.read())
    actual_hash = hashlib.sha256(artifact.read_bytes()).hexdigest()
    expected_hash = str(source["artifact_sha256"])
    if actual_hash != expected_hash:
        raise ValueError(
            f"EvalPlus artifact checksum mismatch for {artifact.name}: "
            f"expected {expected_hash}, received {actual_hash}"
        )
    with gzip.open(artifact, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError(f"EvalPlus artifact has no records: {artifact}")
    return rows


def _math_task(
    *,
    task_id: str,
    problem: str,
    answer: str,
    subject: str,
    level: Any,
    split: str,
    source_task_id: str,
    source_name: str,
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
            "source_dataset": source_name,
            "source_task_id": source_task_id,
            "subject": subject,
            "level": level_text,
        },
    }


def _build_math_tasks(math500: Sequence[Mapping[str, Any]], full_math: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Make disjoint MATH development tasks and a fixed MATH-100 held-out test."""

    math500_by_problem = {_normalise_problem(str(row["problem"])) for row in math500}
    if len(math500_by_problem) != 500:
        raise ValueError("MATH-500 source must contain 500 unique problems")

    test_candidates = [
        _math_task(
            task_id=f"math100::{row['unique_id']}",
            problem=str(row["problem"]),
            answer=str(row["answer"]),
            subject=str(row["subject"]),
            level=row["level"],
            split="test",
            source_task_id=str(row["unique_id"]),
            source_name="MATH-500",
        )
        for row in math500
    ]
    selected_test = _stratified_select(test_candidates, count=100, namespace="test")

    development_candidates: list[dict[str, Any]] = []
    for row in full_math:
        problem = str(row["problem"])
        if _normalise_problem(problem) in math500_by_problem:
            continue
        answer = _balanced_boxed_value(str(row["solution"]))
        if answer is None:
            continue
        source_id = _sha256_text(_normalise_problem(problem))[:24]
        development_candidates.append(
            _math_task(
                task_id=f"mathdev::{source_id}",
                problem=problem,
                answer=answer,
                subject=str(row["type"]),
                level=row["level"],
                split="unassigned",
                source_task_id=source_id,
                source_name="MATH (excluding all MATH-500 problems)",
            )
        )
    if len(development_candidates) < 200:
        raise ValueError("insufficient non-MATH-500 development tasks")
    train = _stratified_select(development_candidates, count=140, namespace="development-train")
    train_ids = {task["task_id"] for task in train}
    validation = _stratified_select(
        (task for task in development_candidates if task["task_id"] not in train_ids),
        count=60,
        namespace="development-validation",
    )
    for task in train:
        task["split"] = "train"
    for task in validation:
        task["split"] = "validation"
    tasks = [*train, *validation, *selected_test]
    if len({task["task_id"] for task in tasks}) != len(tasks):
        raise ValueError("duplicate MATH task identifiers")
    return tasks


def _evalplus_inputs(value: Any, *, task_id: str, split: str) -> list[Any]:
    """Normalize official EvalPlus input collections to JSON task fields.

    Mbpp/793 in the pinned v0.2.0 release represents an empty plus split as
    ``{}``.  The upstream runner iterates it as an empty collection, so retain
    that intended no-plus-test meaning as an empty JSON array here.
    """

    if isinstance(value, Mapping) and not value:
        return []
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise ValueError(f"{task_id} has invalid EvalPlus {split} inputs")
    return list(value)


def _code_task(
    *,
    task_id: str,
    prompt: str,
    entry_point: str,
    evaluator_dataset: str,
    reference_code: str,
    base_inputs: Sequence[Any],
    plus_inputs: Sequence[Any],
    atol: Any,
    dataset_id: str,
    source_task_id: str,
    version: str,
) -> dict[str, Any]:
    """Create a private-oracle task from an official EvalPlus release record."""

    if evaluator_dataset not in {"humaneval", "mbpp"}:
        raise ValueError(f"unsupported EvalPlus dataset: {evaluator_dataset}")
    if not reference_code.strip():
        raise ValueError(f"{task_id} has no EvalPlus canonical oracle code")
    normalized_base_inputs = _evalplus_inputs(base_inputs, task_id=task_id, split="base")
    normalized_plus_inputs = _evalplus_inputs(plus_inputs, task_id=task_id, split="plus")
    try:
        tolerance = float(atol)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{task_id} has invalid EvalPlus tolerance") from error
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError(f"{task_id} has non-finite or negative EvalPlus tolerance")
    return {
        "task_id": task_id,
        "domain": "code",
        "split": "test",
        "prompt": prompt,
        "entry_point": entry_point,
        "code_evaluator": "evalplus_differential_v1",
        "evalplus_dataset": evaluator_dataset,
        "evalplus_reference_code": reference_code,
        "evalplus_base_inputs": normalized_base_inputs,
        "evalplus_plus_inputs": normalized_plus_inputs,
        "evalplus_atol": tolerance,
        "scoring": {"timeout_seconds": 10.0},
        "public_metadata": {
            "source_dataset": dataset_id,
            "source_task_id": source_task_id,
            "dataset_version": version,
            "base_input_count": len(normalized_base_inputs),
            "plus_input_count": len(normalized_plus_inputs),
            "evaluation": "EvalPlus base+plus differential inputs",
            "comparison_reference": EVALPLUS_REFERENCE_IMPLEMENTATION,
        },
    }


def _build_humaneval_tasks(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for row in rows:
        source_task_id = str(row["task_id"])
        entry_point = str(row["entry_point"])
        prompt = (
            f"Implement `{entry_point}` as complete Python code. Return only code that defines "
            f"this function, including any standard-library imports it needs.\n\n{row['prompt']}"
        )
        tasks.append(
            _code_task(
                task_id=f"humanevalplus::{source_task_id.replace('/', '_')}",
                prompt=prompt,
                entry_point=entry_point,
                evaluator_dataset="humaneval",
                reference_code=str(row["prompt"]) + str(row["canonical_solution"]),
                base_inputs=row["base_input"],
                plus_inputs=row["plus_input"],
                atol=row["atol"],
                dataset_id="EvalPlus HumanEval+",
                source_task_id=source_task_id,
                version=str(SOURCES["humanevalplus"]["version"]),
            )
        )
    if len(tasks) != 164:
        raise ValueError(f"expected 164 HumanEval+ tasks, received {len(tasks)}")
    return tasks


def _build_mbpp_tasks(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for row in rows:
        source_task_id = str(row["task_id"])
        entry_point = str(row["entry_point"])
        prompt = (
            f"Implement `{entry_point}` as complete Python code. Return only code that defines "
            f"this function, including any standard-library imports it needs.\n\nTask: {row['prompt']}"
        )
        tasks.append(
            _code_task(
                task_id=f"mbppplus::{source_task_id}",
                prompt=prompt,
                entry_point=entry_point,
                evaluator_dataset="mbpp",
                reference_code=str(row["canonical_solution"]),
                base_inputs=row["base_input"],
                plus_inputs=row["plus_input"],
                atol=row["atol"],
                dataset_id="EvalPlus MBPP+",
                source_task_id=source_task_id,
                version=str(SOURCES["mbppplus"]["version"]),
            )
        )
    if len(tasks) != 378:
        raise ValueError(f"expected 378 MBPP+ tasks, received {len(tasks)}")
    return tasks


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _task_document(*, benchmark_id: str, domain: str, tasks: Sequence[Mapping[str, Any]], purpose: str) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "benchmark_id": benchmark_id,
        "domain": domain,
        "purpose": purpose,
        "generator": GENERATOR_VERSION,
        "tasks": list(tasks),
    }


def _formal_manifest(documents: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    source_paths = {
        "math": "benchmark/math/MATH100_tasks.json",
        "humaneval": "benchmark/code/HumanEvalPlus_tasks.json",
        "mbpp": "benchmark/code/MBPPPlus_tasks.json",
    }
    entries: list[dict[str, str]] = []
    for name, tasks in documents.items():
        for task in tasks:
            entries.append(
                {
                    "task_id": str(task["task_id"]),
                    "domain": str(task["domain"]),
                    "source_file": source_paths[name],
                }
            )
    return {
        "schema_version": "1.0",
        "benchmark_id": "roundvalue_formal_realdata_v1",
        "purpose": (
            "Frozen real-data RoundValue experiment: MATH development tasks fit/select the "
            "cross-domain policy; MATH-100, HumanEval+, and MBPP+ are held-out test tasks."
        ),
        "split_protocol": {
            "train": "140 MATH tasks excluding every MATH-500 problem",
            "validation": "60 disjoint MATH tasks excluding every MATH-500 problem",
            "test": "100 fixed MATH-500 tasks plus all HumanEval+ and MBPP+ tasks",
            "note": (
                "Code results are held-out cross-domain policy evaluations. Do not describe this "
                "manifest's train/validation partition as an EvalPlus training split."
            ),
        },
        "tasks": entries,
    }


def _split_counts(tasks: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts = {"train": 0, "validation": 0, "test": 0}
    for task in tasks:
        split = str(task["split"])
        counts[split] = counts.get(split, 0) + 1
    return counts


def build(
    *, output_root: Path, cache_dir: Path | None, evalplus_cache_dir: Path
) -> dict[str, Any]:
    raw_math500 = _load_dataset(SOURCES["math500"], cache_dir=cache_dir)
    raw_math_development = _load_dataset(SOURCES["math_development"], cache_dir=cache_dir)
    raw_humaneval = _load_evalplus_release(
        SOURCES["humanevalplus"], cache_dir=evalplus_cache_dir
    )
    raw_mbpp = _load_evalplus_release(SOURCES["mbppplus"], cache_dir=evalplus_cache_dir)

    math_tasks = _build_math_tasks(raw_math500, raw_math_development)
    humaneval_tasks = _build_humaneval_tasks(raw_humaneval)
    mbpp_tasks = _build_mbpp_tasks(raw_mbpp)
    if _split_counts(math_tasks) != {"train": 140, "validation": 60, "test": 100}:
        raise ValueError("unexpected MATH split sizes")

    math_document = _task_document(
        benchmark_id="MATH-100-RoundValue-v1",
        domain="math",
        tasks=math_tasks,
        purpose=(
            "Real MATH development tasks plus a fixed, stratified 100-question subset of MATH-500. "
            "Only the MATH-100 subset is a held-out test set."
        ),
    )
    human_document = _task_document(
        benchmark_id="EvalPlus-HumanEvalPlus-v0.1.10",
        domain="code",
        tasks=humaneval_tasks,
        purpose=(
            "All 164 official HumanEval+ tasks with private canonical-oracle source and "
            "base+plus inputs retained for RoundValue differential evaluation."
        ),
    )
    mbpp_document = _task_document(
        benchmark_id="EvalPlus-MBPPPlus-v0.2.0",
        domain="code",
        tasks=mbpp_tasks,
        purpose=(
            "All 378 official MBPP+ tasks with private canonical-oracle source and base+plus "
            "inputs retained for RoundValue differential evaluation."
        ),
    )
    manifest = _formal_manifest(
        {"math": math_tasks, "humaneval": humaneval_tasks, "mbpp": mbpp_tasks}
    )
    provenance = {
        "schema_version": "1.0",
        "generator": GENERATOR_VERSION,
        "math100_selection_seed": MATH100_SELECTION_SEED,
        "evalplus_adapter": {
            "evaluator": "evalplus_differential_v1",
            "comparison_reference": EVALPLUS_REFERENCE_IMPLEMENTATION,
            "scope": "RoundValue local adapter; not the upstream container or leaderboard runner",
        },
        "sources": SOURCES,
        "raw_source_record_sha256": {
            "math500": _records_hash(raw_math500),
            "math_development": _records_hash(raw_math_development),
            "humanevalplus": _records_hash(raw_humaneval),
            "mbppplus": _records_hash(raw_mbpp),
        },
        "output_task_counts": {
            "math": _split_counts(math_tasks),
            "humanevalplus": len(humaneval_tasks),
            "mbppplus": len(mbpp_tasks),
            "formal_total": len(manifest["tasks"]),
        },
        "math100_test_task_ids": [
            task["task_id"] for task in math_tasks if task["split"] == "test"
        ],
        "known_canonical_self_check_exceptions": KNOWN_CANONICAL_SELF_CHECK_EXCEPTIONS,
    }
    _write_json(output_root / "math" / "MATH100_tasks.json", math_document)
    _write_json(output_root / "code" / "HumanEvalPlus_tasks.json", human_document)
    _write_json(output_root / "code" / "MBPPPlus_tasks.json", mbpp_document)
    _write_json(output_root / "formal_experiment_v1.json", manifest)
    _write_json(output_root / "formal_experiment_v1.provenance.json", provenance)
    return provenance


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
    parser.add_argument(
        "--evalplus-cache-dir",
        type=Path,
        default=PROJECT_ROOT / ".cache" / "evalplus-official",
        help="Local cache for checksum-pinned official EvalPlus JSONL releases.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    provenance = build(
        output_root=args.output_root.resolve(),
        cache_dir=args.cache_dir,
        evalplus_cache_dir=args.evalplus_cache_dir.resolve(),
    )
    print(
        json.dumps(
            {
                "status": "built",
                "formal_total": provenance["output_task_counts"]["formal_total"],
                "math_splits": provenance["output_task_counts"]["math"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
