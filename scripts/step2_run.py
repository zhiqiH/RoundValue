"""Step 2: run the main experiment (benchmark collection + offline analysis).

This entry merges the former ``step2_collect`` and ``step3_analyze`` stages
into one command.  It first collects the raw trajectories (one-to-five-round
Debate, or one-call single) into ``trajectories/<run_id>/`` under the step1
smoke gate, and then, only if every task completed, runs the deterministic
offline analysis into ``results/<run_id>/``: Debate runs get scoring, label
building, policy fitting, threshold selection, and Test evaluation; single
runs get scoring plus the single-topology aggregate metrics only.

The two stages are never allowed to mix their outputs: trajectories stay raw
and untouched, and every result in ``results/<run_id>/`` can be regenerated
from the saved trajectories.  If collection is incomplete (some trajectories
failed), analysis is skipped and the command exits nonzero with the failed
task IDs, so the same command can be rerun with ``--run-id`` to resume only
the failed or missing tasks before analysis.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from config_loader import load_experiment_config  # noqa: E402
from contracts import ConfigurationError  # noqa: E402
import pipeline as tb  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        required=True,
        help=(
            "Project-relative self-contained dataset document, e.g. "
            "benchmark/mmlu_pro/MMLU-Pro-500.json or "
            "benchmark/mmlu_pro/MMLU-Pro-50.json."
        ),
    )
    parser.add_argument(
        "--run-id",
        help=(
            "ID of an existing collect run to resume before analysis. Only its "
            "failed or missing tasks are recollected. New runs are named "
            "automatically; do not pass this for a fresh run."
        ),
    )
    parser.add_argument(
        "--smoke-run-id",
        required=True,
        help="Passing step1_smoke run ID; collection refuses to start without it.",
    )
    parser.add_argument(
        "--model-id",
        help=(
            "Run-level model profile from configs/model_config.json. Defaults "
            "to deepseek_flash; pass gpt5_nano or gpt4o_mini to run every node "
            "with that profile."
        ),
    )
    parser.add_argument(
        "--topology",
        default=None,
        help=(
            "Run-level topology from configs/topology.json. Defaults to debate; "
            "pass single for the one-call independent solver."
        ),
    )
    args = parser.parse_args(argv)
    args.mode = "collect"
    try:
        experiment = load_experiment_config(
            tb.PROJECT_ROOT,
            model_id=args.model_id,
            topology_id=args.topology,
        )
    except ConfigurationError as error:
        parser.error(str(error))
    state: dict[str, Any] = {"manifest": None}

    def run() -> int:
        collect_code = tb._run_collect(args, experiment, state, score=False)
        if collect_code != 0:
            manifest = state.get("manifest")
            run_id = (
                manifest.get("run_id")
                if isinstance(manifest, dict)
                else None
            )
            tb._emit(
                {
                    "status": "collect_incomplete",
                    "mode": "run",
                    "run_id": run_id,
                    "message": (
                        "analysis skipped because collection is incomplete; "
                        "rerun with --run-id to resume the failed tasks"
                    ),
                }
            )
            return collect_code

        manifest = state.get("manifest")
        if not isinstance(manifest, dict):
            raise RuntimeError("collect completed without a run manifest")
        run_id = manifest["run_id"]
        analyze_args = argparse.Namespace(
            run_id=run_id,
            mode="analyze",
        )
        analyze_code = tb._run_analyze(analyze_args, None, state)
        if analyze_code != 0:
            return analyze_code

        updated = state.get("manifest") or manifest
        tb._emit(
            {
                "status": "run_complete",
                "mode": "run",
                "run_id": run_id,
                "trajectory_dir": updated["trajectory_dir"],
                "result_dir": updated["result_dir"],
            }
        )
        return 0

    return tb.entrypoint("run", run, state)


if __name__ == "__main__":
    raise SystemExit(main())
