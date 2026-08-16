"""Step 3: render saved analysis results into readable artifacts.

Reads only ``results/<run_id>/`` (analysis.json written by step2_collect_analyze) and
writes ``task_level_results.csv``, a self-contained ``report.html`` with the
per-round accuracy, token, wall-clock latency, Repair/Neutral/Harm/Recovery,
stop-round, and policy-comparison tables plus SVG charts, and a short
``summary_conclusion.txt``.  It also writes five standalone policy-level PNG
charts (policy quality-vs-tokens, policy quality-vs-latency,
RoundValue-vs-baselines, adaptive stop-round distribution, and oracle
quality regret) into ``charts/``.

Visualization never reads trajectories and cannot affect scoring or policy.
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
    parser.add_argument("--run-id", required=True, help="Run analyzed by step2_collect_analyze.")
    args = parser.parse_args(argv)
    args.mode = "visualize"
    state: dict = {"manifest": None}
    return tb.entrypoint(
        "visualize", lambda: tb._run_visualize(args, None, state), state
    )


if __name__ == "__main__":
    raise SystemExit(main())
