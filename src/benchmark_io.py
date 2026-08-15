"""Load JSON benchmark manifests while separating public task inputs from labels."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from config_loader import load_json
from contracts import ConfigurationError, file_hash, require_list, require_object, require_string


PRIVATE_TASK_FIELDS = {
    "reference_answer",
    "expected",
    "gold",
    "label",
    "solution",
}

# These fields are explicitly permitted in runtime prompts/features.  New
# benchmark formats should expose only summary metadata here; raw answer keys,
# test cases, and verifier outcomes remain offline-only.
PUBLIC_OPTIONAL_TASK_FIELDS = {
    "difficulty",
    "public_metadata",
    "public_verifier",
}

# Metadata keys that identify one benchmark task or reveal the size of the
# hidden test suite.  They are available to offline scoring but must never
# reach an Agent, otherwise a model can recall the pinned benchmark entry or
# tune itself against the number of hidden tests.
PUBLIC_METADATA_HIDDEN_KEYS = frozenset(
    {
        "source_task_id",
    }
)


def _safe_project_path(root: Path, supplied: str | Path) -> Path:
    root = root.resolve()
    candidate = Path(supplied)
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ConfigurationError(f"benchmark path must remain inside the project: {supplied}") from error
    if resolved.suffix.casefold() != ".json":
        raise ConfigurationError("benchmark files must use JSON, not another configuration format")
    return resolved


def _validate_task(task: dict[str, Any], *, source: Path) -> dict[str, Any]:
    task_id = require_string(task.get("task_id"), f"{source} task_id")
    domain = require_string(task.get("domain"), f"{source} task {task_id}.domain")
    if domain != "math":
        raise ConfigurationError(
            f"{source} task {task_id} domain must be math; the project is math-only"
        )
    require_string(task.get("prompt"), f"{source} task {task_id}.prompt")
    require_string(task.get("reference_answer"), f"{source} math task {task_id}.reference_answer")
    return task


def _source_tasks(root: Path, source_file: str) -> dict[str, dict[str, Any]]:
    path = _safe_project_path(root, source_file)
    source = load_json(path)
    tasks = require_list(source.get("tasks"), f"{path} tasks")
    default_domain = source.get("domain")
    result: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(tasks):
        candidate = dict(require_object(raw, f"{path} tasks[{index}]"))
        if default_domain is not None:
            candidate.setdefault("domain", default_domain)
        task = _validate_task(candidate, source=path)
        task_id = task["task_id"]
        if task_id in result:
            raise ConfigurationError(f"duplicate task_id {task_id} in {path}")
        result[task_id] = dict(task)
    return result


def load_benchmark(root: Path, supplied_path: str | Path) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    """Resolve a manifest or a self-contained task document into validated task records."""

    path = _safe_project_path(root, supplied_path)
    document = load_json(path)
    if document.get("kind") == "paper_dataset_registry":
        raise ConfigurationError(
            f"{path} is a dataset registry, not runnable tasks; use one of its task_document paths"
        )
    raw_tasks = require_list(document.get("tasks"), f"{path} tasks")
    default_domain = document.get("domain")
    source_cache: dict[str, dict[str, dict[str, Any]]] = {}
    tasks: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_tasks):
        entry = require_object(raw, f"{path} tasks[{index}]")
        if "source_file" in entry:
            task_id = require_string(entry.get("task_id"), f"{path} manifest task_id")
            source_file = require_string(entry.get("source_file"), f"{path} source_file")
            # Do not use ``dict.setdefault(key, _source_tasks(...))`` here:
            # Python evaluates the expensive source-file parse even when the
            # cache already holds it.  Formal manifests intentionally contain
            # one lightweight reference per task, so loading each source once
            # keeps startup proportional to the number of source documents.
            if source_file not in source_cache:
                source_cache[source_file] = _source_tasks(root, source_file)
            source_tasks = source_cache[source_file]
            if task_id not in source_tasks:
                raise ConfigurationError(f"{path} references missing task {task_id} from {source_file}")
            task = dict(source_tasks[task_id])
            manifest_domain = entry.get("domain")
            if manifest_domain is not None and manifest_domain != task["domain"]:
                raise ConfigurationError(f"{path} gives a mismatched domain for {task_id}")
        else:
            candidate = dict(entry)
            if default_domain is not None:
                candidate.setdefault("domain", default_domain)
            task = _validate_task(candidate, source=path)
        task_id = task["task_id"]
        if task_id in seen:
            raise ConfigurationError(f"duplicate task_id {task_id} in benchmark {path}")
        seen.add(task_id)
        tasks.append(task)
    if not tasks:
        raise ConfigurationError(f"benchmark has no tasks: {path}")
    return path, document, tasks


def _public_task_id(task_id: str) -> str:
    """Derive a stable opaque identifier for the Agent-facing task view.

    Benchmark task IDs such as ``math500::test/algebra/2584.json`` name the exact
    upstream entry and can trigger memorized solutions.  The public view
    therefore uses a deterministic hash while all on-disk records keep the
    original ID.
    """

    digest = hashlib.sha256(
        f"roundvalue-public-task-id-v1:{task_id}".encode("utf-8")
    ).hexdigest()
    return f"task_{digest[:16]}"


def public_task(task: dict[str, Any]) -> dict[str, Any]:
    """Return the exact information an Agent may see; offline labels stay on disk only."""

    visible: dict[str, Any] = {
        "task_id": _public_task_id(task["task_id"]),
        "domain": task["domain"],
        "prompt": task["prompt"],
    }
    for key in PUBLIC_OPTIONAL_TASK_FIELDS:
        if key in task and key not in PRIVATE_TASK_FIELDS:
            value = task[key]
            if key == "public_metadata" and isinstance(value, dict):
                value = {
                    metadata_key: metadata_value
                    for metadata_key, metadata_value in value.items()
                    if metadata_key not in PUBLIC_METADATA_HIDDEN_KEYS
                    and metadata_key not in PRIVATE_TASK_FIELDS
                }
            visible[key] = value
    return visible


def freeze_splits(tasks: list[dict[str, Any]], seed: int) -> dict[str, str]:
    """Assign every original task atomically to a split before any trajectory is collected.

    Existing train/validation/test fields are honoured.  All other tasks use a
    stable task-id hash with 60/20/20 buckets, so a task and every future retry
    or continuation remain in the same partition.
    """

    split_by_task: dict[str, str] = {}
    for task in tasks:
        task_id = task["task_id"]
        declared = task.get("split")
        if declared in {"train", "validation", "test"}:
            split_by_task[task_id] = declared
            continue
        digest = hashlib.sha256(f"{seed}:{task_id}".encode("utf-8")).digest()[0]
        split_by_task[task_id] = "train" if digest < 153 else "validation" if digest < 204 else "test"
    return split_by_task


def benchmark_provenance(root: Path, manifest_path: Path, document: dict[str, Any]) -> dict[str, Any]:
    """Record hashes for the manifest and every source JSON it names."""

    root = root.resolve()
    sources: dict[str, str] = {str(manifest_path.relative_to(root)): file_hash(manifest_path)}
    for entry in document.get("tasks", []):
        if isinstance(entry, dict) and isinstance(entry.get("source_file"), str):
            source = _safe_project_path(root, entry["source_file"])
            sources[str(source.relative_to(root))] = file_hash(source)
    return {"files": sources}
