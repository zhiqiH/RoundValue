"""Atomic JSON persistence and immutable run manifests for RoundValue."""

from __future__ import annotations

from datetime import datetime
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import uuid
from typing import Any
from zoneinfo import ZoneInfo

from contracts import canonical_json, file_hash, json_hash, utc_now


_RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,119}\Z")
_WINDOWS_ABSOLUTE_PATH_RE = re.compile(r"^[A-Za-z]:[\\/].*")
_DATASET_TOKEN_RE = re.compile(r"^[A-Za-z0-9]+\Z")
_MODEL_TOKEN_RE = re.compile(r"^[a-z0-9][a-z0-9-]*\Z")
_SNAPSHOT_SUFFIX_RE = re.compile(r"-20\d{2}-\d{2}-\d{2}\Z")
RUN_ID_TIMEZONE = ZoneInfo("America/Chicago")


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
    names = ("httpx", "numpy")
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def make_run_id() -> str:
    """Local minute timestamp for Dallas; dataset tag and entropy come from create_run."""

    return datetime.now(RUN_ID_TIMEZONE).strftime("%Y%m%d%H%M")


def canonical_dataset_token(dataset_name: str) -> str:
    """Return the one concise filesystem-safe dataset label.

    The canonical labels are ``MMLUPro50`` and ``MMLUPro500``; the rule removes
    every punctuation character so aliases such as ``MMLU-Pro-50`` can never
    diverge across runs.  New benchmark manifests may instead carry their own
    concise ``dataset_id``, which is used verbatim when already alphanumeric.
    """

    if not isinstance(dataset_name, str) or not dataset_name.strip():
        raise ValueError("dataset_name must be a non-empty string")
    token = re.sub(r"[^A-Za-z0-9]+", "", dataset_name.strip())
    if not _DATASET_TOKEN_RE.fullmatch(token):
        raise ValueError(f"dataset name produces an invalid directory token: {dataset_name!r}")
    return token


def canonical_model_token(requested_model: str) -> str:
    """Return a concise model label from the actual requested model/snapshot.

    A trailing dated API snapshot such as ``-2024-07-18`` is omitted for the
    human-readable directory label; the exact requested model stays in the run
    manifest.  The label is derived only from the configured request model,
    never from a provider response alias.
    """

    if not isinstance(requested_model, str) or not requested_model.strip():
        raise ValueError("requested_model must be a non-empty string")
    token = requested_model.strip().casefold()
    token = _SNAPSHOT_SUFFIX_RE.sub("", token)
    if not _MODEL_TOKEN_RE.fullmatch(token):
        raise ValueError(f"requested model produces an invalid directory token: {requested_model!r}")
    return token


def compose_run_name(
    *,
    dataset_name: str,
    requested_model: str,
    timestamp: str | None = None,
    hex_suffix: str | None = None,
) -> str:
    """The single source of canonical run identity.

    Format is ``YYYYMMDDHHMM_<model>_<dataset>_<hex>``.  The debate topology is
    frozen and the Single-Agent baseline is automatically included in every
    run, so topology never appears in a directory name.  The timestamp and
    suffix are generated exactly once at run creation and are never recomputed
    by later offline analysis.
    """

    run_timestamp = make_run_id() if timestamp is None else timestamp
    if not isinstance(run_timestamp, str) or not re.fullmatch(
        r"\d{12}", run_timestamp
    ):
        raise ValueError("run timestamp must be a 12-digit YYYYMMDDHHMM string")
    suffix = uuid.uuid4().hex[:8] if hex_suffix is None else hex_suffix
    if not isinstance(suffix, str) or not re.fullmatch(r"[0-9a-f]{8}", suffix):
        raise ValueError("run hex suffix must be 8 lowercase hexadecimal characters")
    return (
        f"{run_timestamp}_{canonical_model_token(requested_model)}_"
        f"{canonical_dataset_token(dataset_name)}_{suffix}"
    )


def _validated_run_id(run_id: str) -> str:
    """Permit a compact identifier, never a path supplied by a CLI user."""

    if not isinstance(run_id, str) or not _RUN_ID_RE.fullmatch(run_id):
        raise ValueError(
            "run_id must contain only letters, digits, underscores, and hyphens (max 120 chars)"
        )
    return run_id


def result_directory_name(run_id: str) -> str:
    """Trajectory and result directories share one validated run ID.

    Keeping the directory names identical makes the two sides pair without
    fuzzy matching; ``open_run`` always resolves through the trajectory side.
    """

    return run_id


def create_run(
    root: Path,
    *,
    command: list[str],
    config_snapshot: dict[str, Any],
    run_id: str | None = None,
    dataset_name: str,
    domain: str,
    requested_model: str | None = None,
) -> dict[str, Any]:
    """Create matching raw-trajectory and aggregate-result directories."""

    root = root.resolve()
    # The canonical name is YYYYMMDDHHMM_<model>_<dataset>_<hex> and is
    # generated once here; analysis and visualization reuse the stored id.
    name_components: dict[str, Any] | None = None
    if run_id is None:
        if requested_model is None:
            raise ValueError("auto-named runs require requested_model")
        composed = compose_run_name(
            dataset_name=dataset_name,
            requested_model=requested_model,
        )
        timestamp, model_token, dataset_token, suffix = composed.split("_")
        name_components = {
            "timestamp": timestamp,
            "model": model_token,
            "dataset": dataset_token,
            "hex": suffix,
        }
        chosen_id = _validated_run_id(composed)
    else:
        chosen_id = _validated_run_id(run_id)
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
        "run_name": chosen_id,
        "run_name_components": name_components,
        "dataset": dataset_name,
        "dataset_label": canonical_dataset_token(dataset_name),
        "domain": domain,
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
    root = root.resolve()
    manifest_path = root / "trajectories" / safe_run_id / "run.json"
    manifest = read_json(manifest_path)
    if manifest.get("run_id") != safe_run_id:
        raise ValueError(f"run manifest id mismatch: {manifest_path}")
    # Historical artifacts may record absolute paths from another platform.
    # Never rewrite those artifacts; only use the local sibling directories
    # when the recorded paths are foreign-absolute or do not exist here.
    recorded_trajectory = manifest.get("trajectory_dir")
    if (
        not recorded_trajectory
        or _WINDOWS_ABSOLUTE_PATH_RE.fullmatch(str(recorded_trajectory))
        or not Path(str(recorded_trajectory)).exists()
    ):
        manifest["trajectory_dir"] = str(root / "trajectories" / safe_run_id)
    recorded_result = manifest.get("result_dir")
    if (
        not recorded_result
        or _WINDOWS_ABSOLUTE_PATH_RE.fullmatch(str(recorded_result))
        or not Path(str(recorded_result)).exists()
    ):
        manifest["result_dir"] = str(root / "results" / safe_run_id)
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
