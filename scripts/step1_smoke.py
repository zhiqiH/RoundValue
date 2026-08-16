"""Step 1: small-scale real-API smoke acceptance run.

Runs one complete fixed Debate round with the real model provider against the
independent repository acceptance tasks.  Every task must complete and
score exactly 1, otherwise this script exits nonzero and step2 must not start.

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
import pipeline as tb  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        default=tb.DEFAULT_BENCHMARK,
        help="Project-relative smoke benchmark (default: benchmark/test/smoke_tasks.json).",
    )
    parser.add_argument("--run-id", help="Optional explicit new smoke run ID.")
    args = parser.parse_args(argv)
    args.mode = "smoke"
    experiment = load_experiment_config(tb.PROJECT_ROOT)
    state: dict = {"manifest": None}
    return tb.entrypoint("smoke", lambda: tb._run_smoke(args, experiment, state), state)


if __name__ == "__main__":
    raise SystemExit(main())
