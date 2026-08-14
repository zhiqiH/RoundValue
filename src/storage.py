"""Atomic JSON persistence and immutable run manifests for RoundValue."""

from __future__ import annotations

from datetime import UTC, datetime
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import uuid
from typing import Any

from contracts import canonical_json, file_hash, json_hash, utc_now


_RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,119}\Z")


def write_json(path: Path, value: Any) -> None:
    """Atomically write a UTF-8 JSON document, so interrupted runs remain inspectable."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"JSON file does not exist: {path}") from None
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {path}:{error.lineno}:{error.colno}: {error.msg}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _git_commit(root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            check=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def _git_status(root: Path) -> list[str] | None:
    """Return the exact dirty-worktree status without invoking a shell."""

    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=root,
            capture_output=True,
            check=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return [line for line in completed.stdout.splitlines() if line]


def _source_snapshot(root: Path) -> dict[str, Any]:
    """Freeze readable source text when a run starts, including dirty code.

    A Git commit alone is insufficient during active experiment development.
    This deliberately excludes secrets and generated output, and stores only the
    small set of code and project-contract files that can affect a run.
    """

    candidates: list[Path] = []
    for name in ("pyproject.toml", "README.md", "EXPERIMENT_ARCHITECTURE.md"):
        path = root / name
        if path.is_file():
            candidates.append(path)
    for directory in (root / "src", root / "scripts"):
        if directory.is_dir():
            candidates.extend(sorted(directory.glob("*.py")))
    files: dict[str, dict[str, str]] = {}
    for path in sorted(set(candidates)):
        relative = path.relative_to(root).as_posix()
        content = path.read_text(encoding="utf-8")
        files[relative] = {"sha256": file_hash(path), "content": content}
    return {"schema_version": "1.0", "files": files, "hash": json_hash(files)}


def _package_versions() -> dict[str, str | None]:
    names = ("httpx",)
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def make_run_id() -> str:
    """Compact timestamp + entropy; each side of the output split has a matching ID."""

    return f"{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


def _validated_run_id(run_id: str) -> str:
    """Permit a compact identifier, never a path supplied by a CLI user."""

    if not isinstance(run_id, str) or not _RUN_ID_RE.fullmatch(run_id):
        raise ValueError(
            "run_id must contain only letters, digits, underscores, and hyphens (max 120 chars)"
        )
    return run_id


def result_directory_name(run_id: str) -> str:
    """Keep trajectories compact and results in the requested YYYY-MM-DD time style."""

    try:
        date_part, time_part, suffix = run_id.split("_", 2)
        return f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:]}_{time_part}_{suffix}"
    except ValueError:
        return run_id


def create_run(
    root: Path,
    *,
    command: list[str],
    config_snapshot: dict[str, Any],
    run_id: str | None = None,
) -> dict[str, Any]:
    """Create matching raw-trajectory and aggregate-result directories."""

    root = root.resolve()
    chosen_id = _validated_run_id(run_id or make_run_id())
    trajectory_dir = root / "trajectories" / chosen_id
    result_dir = root / "results" / result_directory_name(chosen_id)
    if trajectory_dir.exists() or result_dir.exists():
        raise FileExistsError(f"run_id already exists: {chosen_id}")
    trajectory_dir.mkdir(parents=True)
    result_dir.mkdir(parents=True)
    source_snapshot = _source_snapshot(root)
    git_status = _git_status(root)
    manifest = {
        "schema_version": "1.0",
        "run_id": chosen_id,
        "status": "created",
        "created_at": utc_now(),
        "command": command,
        "project_root": str(root),
        "trajectory_dir": str(trajectory_dir),
        "result_dir": str(result_dir),
        "git": {
            "commit": _git_commit(root),
            "dirty": bool(git_status),
            "status_porcelain": git_status,
        },
        "python": sys.version,
        "platform": platform.platform(),
        "package_versions": _package_versions(),
        "configs": config_snapshot,
        "config_hash": json_hash(config_snapshot),
        "source_snapshot_hash": source_snapshot["hash"],
        "source_file_hashes": {
            relative: metadata["sha256"]
            for relative, metadata in source_snapshot["files"].items()
        },
    }
    write_json(trajectory_dir / "source_snapshot.json", source_snapshot)
    write_json(trajectory_dir / "run.json", manifest)
    write_json(trajectory_dir / "resolved_config.json", config_snapshot)
    write_json(result_dir / "manifest.json", manifest)
    return manifest


def open_run(root: Path, run_id: str) -> dict[str, Any]:
    """Find a prior run through its trajectory-side manifest, never by fuzzy matching."""

    safe_run_id = _validated_run_id(run_id)
    manifest_path = root.resolve() / "trajectories" / safe_run_id / "run.json"
    manifest = read_json(manifest_path)
    if manifest.get("run_id") != safe_run_id:
        raise ValueError(f"run manifest id mismatch: {manifest_path}")
    return manifest


def update_run_status(manifest: dict[str, Any], status: str, **extra: Any) -> dict[str, Any]:
    """Write the same status transition to both durable output sides."""

    updated = {**manifest, "status": status, "updated_at": utc_now(), **extra}
    trajectory_dir = Path(updated["trajectory_dir"])
    result_dir = Path(updated["result_dir"])
    write_json(trajectory_dir / "run.json", updated)
    write_json(result_dir / "manifest.json", updated)
    return updated


def snapshot_benchmark(manifest: dict[str, Any], benchmark_path: Path) -> dict[str, Any]:
    """Copy the exact benchmark manifest used by this run and retain its source hash."""

    trajectory_dir = Path(manifest["trajectory_dir"])
    content = read_json(benchmark_path)
    snapshot = {
        "source_path": str(benchmark_path.resolve()),
        "source_sha256": file_hash(benchmark_path),
        "content": content,
    }
    write_json(trajectory_dir / "benchmark_snapshot.json", snapshot)
    return snapshot


def task_file_name(task_id: str) -> str:
    """Use a stable hash so arbitrary benchmark identifiers cannot escape the run directory."""

    return f"task_{json_hash(task_id)[:16]}.json"


def write_task_record(manifest: dict[str, Any], task_record: dict[str, Any]) -> Path:
    task = task_record.get("task")
    if not isinstance(task, dict) or not isinstance(task.get("task_id"), str):
        raise ValueError("task record needs task.task_id")
    path = Path(manifest["trajectory_dir"]) / task_file_name(task["task_id"])
    write_json(path, task_record)
    return path


def read_task_records(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    directory = Path(manifest["trajectory_dir"])
    records = [read_json(path) for path in sorted(directory.glob("task_*.json"))]
    return records


def write_result(manifest: dict[str, Any], name: str, value: Any) -> Path:
    if not name.endswith(".json"):
        name = f"{name}.json"
    path = Path(manifest["result_dir"]) / name
    write_json(path, value)
    return path


def reproducibility_index(manifest: dict[str, Any]) -> dict[str, Any]:
    """Return hashes only, suitable for sharing without raw prompts or responses."""

    trajectory_dir = Path(manifest["trajectory_dir"])
    hashes = {
        path.name: file_hash(path)
        for path in sorted(trajectory_dir.glob("*.json"))
        if path.name != "run.json"
    }
    return {
        "run_id": manifest["run_id"],
        "config_hash": manifest.get("config_hash"),
        "source_snapshot_hash": manifest.get("source_snapshot_hash"),
        "trajectory_file_hashes": hashes,
        "generated_at": utc_now(),
        "canonical_manifest_hash": json_hash({key: value for key, value in manifest.items() if key != "updated_at"}),
        "canonical_format": canonical_json({"sort_keys": True, "ensure_ascii": False}),
    }
