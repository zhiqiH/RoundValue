"""Step 2: collect the formal three-round debate trajectories.

This entry only makes model calls and saves raw trajectories to
``trajectories/<run_id>/``: node outputs, Writer checkpoints, token counts,
latency, retries, and errors.  It does not train a policy, derive labels, or
produce conclusions.  Scoring and every derived result happen offline in
step3_analyze.  ``--benchmark`` selects one self-contained per-dataset JSON
document; its ``dataset_id`` and ``domain`` drive run naming and the smoke
gate, and datasets are never mixed inside a run.

Collection is gated by a passing step1_smoke run: pass its run ID with
``--smoke-run-id``.  If the smoke run failed, or the configs changed since it
ran, this script refuses to start.
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
        required=True,
        help=(
            "Project-relative self-contained dataset document, e.g. "
            "benchmark/math/MATH-500.json or benchmark/code/HumanEvalPlus.json."
        ),
    )
    parser.add_argument(
        "--run-id",
        help=(
            "ID of an existing collect run to resume only its failed or missing "
            "tasks. New runs are named automatically; do not pass this for a "
            "fresh collection."
        ),
    )
    parser.add_argument(
        "--smoke-run-id",
        required=True,
        help="Passing step1_smoke run ID; collection refuses to start without it.",
    )
    parser.add_argument("--model-id", help="Model profile ID from configs/model_config.json.")
    parser.add_argument("--topology-id", help="Topology ID from configs/topology.json.")
    parser.add_argument(
        "--allow-local-code-evaluation",
        action="store_true",
        help=(
            "Required when the benchmark contains code tasks. Local execution is "
            "defense-in-depth, not an OS security sandbox."
        ),
    )
    args = parser.parse_args(argv)
    args.mode = "collect"
    experiment = load_experiment_config(
        tb.PROJECT_ROOT, model_id=args.model_id, topology_id=args.topology_id
    )
    state: dict = {"manifest": None}

    def run() -> int:
        return tb._run_collect(args, experiment, state, score=False)

    return tb.entrypoint("collect", run, state)


if __name__ == "__main__":
    raise SystemExit(main())
