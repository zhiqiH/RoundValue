"""Offline self-check for canonical NEW run directory naming.

Verifies the single-source formatter
``YYYYMMDDHHMM_<model>_<dataset>_<hex>``, matching trajectory and result
directory names, stable manifest components, and continued loading of
historical runs under the previous naming conventions.  The Debate topology is
frozen and the Single-Agent baseline is collected inside every run, so
topology never appears in a directory name.  No files are renamed and no model
API is called.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from storage import (  # noqa: E402
    canonical_dataset_token,
    canonical_model_token,
    compose_run_name,
    create_run,
    open_run,
    update_run_status,
    write_json,
)


def check(condition: bool, label: str) -> None:
    if not condition:
        raise AssertionError(label)
    print(f"PASS {label}")


def _check_tokens() -> None:
    check(canonical_dataset_token("MMLU-Pro-50") == "MMLUPro50", "MMLU-Pro-50 canonicalizes to MMLUPro50")
    check(canonical_dataset_token("MMLU-Pro-500") == "MMLUPro500", "MMLU-Pro-500 canonicalizes to MMLUPro500")
    check(canonical_dataset_token("SmokeTasks") == "SmokeTasks", "smoke benchmark label stays verbatim")
    check(
        canonical_model_token("gpt-4o-mini-2024-07-18") == "gpt-4o-mini",
        "dated snapshot is omitted from the model label",
    )
    check(canonical_model_token("deepseek-v4-flash") == "deepseek-v4-flash", "DeepSeek model label")
    check(canonical_model_token("gpt-5-nano") == "gpt-5-nano", "GPT-5-nano model label")


def _check_compose() -> None:
    cases = (
        (
            "MMLU-Pro-50",
            "deepseek-v4-flash",
            "202608161530",
            "a3f91c2e",
            "202608161530_deepseek-v4-flash_MMLUPro50_a3f91c2e",
        ),
        (
            "MMLU-Pro-50",
            "gpt-4o-mini-2024-07-18",
            "202608161630",
            "03ef916d",
            "202608161630_gpt-4o-mini_MMLUPro50_03ef916d",
        ),
        (
            "MMLU-Pro-500",
            "gpt-5-nano",
            "202608161615",
            "c241bb80",
            "202608161615_gpt-5-nano_MMLUPro500_c241bb80",
        ),
    )
    for dataset, model, timestamp, suffix, expected in cases:
        composed = compose_run_name(
            dataset_name=dataset,
            requested_model=model,
            timestamp=timestamp,
            hex_suffix=suffix,
        )
        check(composed == expected, f"canonical run name for {model}/{dataset}")


def _check_created_run() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        manifest = create_run(
            root,
            command=["roundvalue", "run"],
            config_snapshot={},
            dataset_name="MMLU-Pro-50",
            domain="mmlu_pro",
            requested_model="gpt-4o-mini-2024-07-18",
        )
        run_id = manifest["run_id"]
        parts = run_id.split("_")
        check(
            len(parts) == 4
            and len(parts[0]) == 12
            and parts[0].isdigit()
            and parts[1] == "gpt-4o-mini"
            and parts[2] == "MMLUPro50"
            and len(parts[3]) == 8
            and all(character in "0123456789abcdef" for character in parts[3]),
            "auto-generated name has timestamp/model/dataset/hex components",
        )
        check(
            manifest["run_name"] == run_id
            and manifest["run_name_components"]
            == {
                "timestamp": parts[0],
                "model": parts[1],
                "dataset": parts[2],
                "hex": parts[3],
            },
            "manifest records run name and its components without topology",
        )
        check(
            (root / "trajectories" / run_id).is_dir()
            and (root / "results" / run_id).is_dir(),
            "trajectory and result directories use the exact same run name",
        )
        updated = update_run_status(manifest, "analyze_complete")
        reopened = open_run(root, run_id)
        check(
            reopened["run_id"] == run_id
            and updated["run_id"] == run_id
            and updated["run_name"] == run_id,
            "offline analysis does not regenerate the run identity",
        )


def _check_historical_loadable() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        historical_ids = (
            "202608161016_MMLUPro50_79a9e682",
            "202608161542_MMLUPro50_debate_deepseek-v4-flash_78bd119a",
        )
        for historical_id in historical_ids:
            trajectory_dir = root / "trajectories" / historical_id
            result_dir = root / "results" / historical_id
            trajectory_dir.mkdir(parents=True)
            result_dir.mkdir(parents=True)
            manifest: dict[str, Any] = {
                "run_id": historical_id,
                "dataset": "MMLU-Pro-50",
                "domain": "mmlu_pro",
                "status": "analyze_complete",
                "trajectory_dir": str(trajectory_dir),
                "result_dir": str(result_dir),
            }
            write_json(trajectory_dir / "run.json", manifest)
            write_json(result_dir / "manifest.json", manifest)
            reopened = open_run(root, historical_id)
            check(
                reopened["run_id"] == historical_id
                and reopened["dataset"] == "MMLU-Pro-50",
                f"historical run name {historical_id} remains loadable",
            )

        explicit = create_run(
            root,
            command=["roundvalue", "smoke", "--run-id", "legacy_style_id"],
            config_snapshot={},
            run_id="legacy_style_id",
            dataset_name="SmokeTasks",
            domain="mmlu_pro",
        )
        check(
            explicit["run_id"] == "legacy_style_id"
            and explicit["run_name_components"] is None,
            "an explicit user run id is preserved without regeneration",
        )


def main() -> int:
    _check_tokens()
    _check_compose()
    _check_created_run()
    _check_historical_loadable()
    print("PASS all run-naming self-checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
