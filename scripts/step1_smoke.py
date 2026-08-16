"""Step 1: small-scale real-API smoke acceptance run.

Runs the selected model profile against the independent repository acceptance
tasks, collecting one Debate round plus the automatic Single-Agent baseline for
every task.  Both components must complete and score exactly 1, otherwise this
script exits nonzero and step2 must not start.

Smoke trajectories are stored under their own ``smoke`` split and never enter
the formal paper results.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
        default=tb.DEFAULT_BENCHMARK,
        help="Project-relative smoke benchmark (default: benchmark/test/smoke_tasks.json).",
    )
    parser.add_argument(
        "--model-id",
        help=(
            "Run-level model profile from configs/model_config.json. Defaults "
            "to deepseek_flash; pass gpt5_nano or gpt4o_mini to smoke-test that "
            "OpenAI profile."
        ),
    )
    parser.add_argument("--run-id", help="Optional explicit new smoke run ID.")
    args = parser.parse_args(argv)
    args.mode = "smoke"
    try:
        experiment = load_experiment_config(
            tb.PROJECT_ROOT,
            model_id=args.model_id,
        )
    except ConfigurationError as error:
        parser.error(str(error))
    state: dict = {"manifest": None}
    return tb.entrypoint("smoke", lambda: tb._run_smoke(args, experiment, state), state)


if __name__ == "__main__":
    raise SystemExit(main())
