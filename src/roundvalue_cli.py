"""Single ``roundvalue`` console command that aliases the three step scripts.

This file does not add a fourth pipeline step and does not change any step's
flags, exit codes, or gating.  Each subcommand forwards the remaining argv to
the exact same ``main`` used by ``python scripts/step*_*.py``, so the three
documented scripts remain the canonical user-facing entries.

The dispatcher is kept next to the flat ``src/`` modules so it is included in
every run's frozen source snapshot like the rest of the pipeline.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"
SCRIPTS_DIRECTORY = PROJECT_ROOT / "scripts"
for directory in (SRC_DIRECTORY, SCRIPTS_DIRECTORY):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from step1_smoke import main as smoke_main  # noqa: E402
from step2_collect_analyze import main as collect_analyze_main  # noqa: E402
from step3_visualize import main as visualize_main  # noqa: E402

STEP_ENTRIES: dict[str, tuple[str, Callable[[list[str]], int]]] = {
    "smoke": (
        "Run the small real-API acceptance gate (scripts/step1_smoke.py).",
        smoke_main,
    ),
    "collect-analyze": (
        "Collect trajectories, then run offline analysis in one step "
        "(scripts/step2_collect_analyze.py).",
        collect_analyze_main,
    ),
    "visualize": (
        "Render CSV, HTML/SVG and PNG charts, and a conclusion "
        "(scripts/step3_visualize.py).",
        visualize_main,
    ),
}


def main(argv: list[str] | None = None) -> int:
    """Dispatch one subcommand to the matching step entry point."""

    raw = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="roundvalue",
        allow_abbrev=False,
        add_help=False,
        description=(
            "RoundValue experiment runner. Each subcommand is an exact alias "
            "of one scripts/step*.py entry point; run 'roundvalue <step> --help' "
            "for that step's flags."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, (help_text, _) in STEP_ENTRIES.items():
        subparsers.add_parser(name, help=help_text, add_help=False)
    if not raw or raw[0] in ("-h", "--help", "help"):
        parser.print_help()
        return 0
    if not (PROJECT_ROOT / "configs" / "agents.json").is_file():
        parser.error(
            "roundvalue must be installed editable from the repository root "
            "(python -m pip install -e .) so it can locate configs/, scripts/, and src/."
        )
    args, remaining = parser.parse_known_args(raw)
    return STEP_ENTRIES[args.command][1](remaining)


if __name__ == "__main__":
    raise SystemExit(main())
