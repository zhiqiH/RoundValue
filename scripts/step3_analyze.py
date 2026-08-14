"""Step 3: fully offline analysis of collected trajectories.

Reads ``trajectories/<run_id>/`` only: checks completeness, scores every round
deterministically, builds the ΔQ/V/G labels, fits the policy on Train, selects
thresholds on Validation, evaluates Test, and writes scores, labels, policy,
the strategy comparison, and summaries to ``results/<run_id>/``.

This step never calls a model provider and never writes back to trajectories;
every file it produces can be regenerated from the saved trajectories.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

import pipeline as tb  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="Run collected by step2_collect.")
    parser.add_argument(
        "--allow-local-code-evaluation",
        action="store_true",
        help=(
            "Required when the run contains code tasks: the offline scorer executes "
            "candidate code locally. This is not an OS security sandbox."
        ),
    )
    args = parser.parse_args(argv)
    args.mode = "analyze"
    state: dict = {"manifest": None}
    return tb.entrypoint("analyze", lambda: tb._run_analyze(args, None, state), state)


if __name__ == "__main__":
    raise SystemExit(main())
