"""Offline self-check for the single topology, GPT-4o-mini, and comparison.

This script never contacts a model provider.  A scripted fake provider
verifies the one-call independent solver contract, GPT-4o-mini wire
parameters, the offline single analysis, and the fully offline
single-vs-debate comparison (including its compatibility refusals).
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from comparison import (  # noqa: E402
    ComparisonError,
    build_comparison,
    run_topology_id,
    write_comparison,
)
from config_loader import load_experiment_config, select_topology  # noqa: E402
from contracts import ConfigurationError, ModelRequest, ModelResponse  # noqa: E402
from provider import build_provider  # noqa: E402
from scorer import score_single_record  # noqa: E402
from single_analysis import (  # noqa: E402
    build_single_analysis,
    summarize_single_collection,
)
from single_runner import SingleAgentRunner  # noqa: E402
from storage import create_run, update_run_status, write_json  # noqa: E402


def check(condition: bool, label: str) -> None:
    if not condition:
        raise AssertionError(label)
    print(f"PASS {label}")


class FakeProvider:
    def __init__(
        self,
        script: list[tuple[str | None, str]] | None = None,
        *,
        by_node: dict[str, tuple[str | None, str]] | None = None,
    ) -> None:
        self.script = list(script or [])
        self.by_node = dict(by_node or {})
        self.requests: list[ModelRequest] = []
        self.closed = False

    def generate(
        self, request: ModelRequest
    ) -> tuple[ModelResponse, list[dict[str, Any]]]:
        self.requests.append(request)
        node_id = request.metadata.get("node_id")
        if node_id in self.by_node:
            finish_reason, text = self.by_node[node_id]
        elif self.script:
            finish_reason, text = self.script.pop(0)
        else:
            raise AssertionError("fake provider ran out of scripted responses")
        attempts = [
            {
                "status": "succeeded",
                "attempt_index": 1,
                "http_status": 200,
                "latency_ms": 10,
                "input_tokens": 100,
                "output_tokens": max(1, len(text) // 4),
                "input_cache_hit_tokens": 50,
                "input_cache_miss_tokens": 50,
                "request_payload": {
                    "messages": request.messages,
                    "max_output_tokens": request.max_output_tokens,
                },
            }
        ]
        return (
            ModelResponse(
                text=text,
                response_model="fake",
                request_id="fake-1",
                finish_reason=finish_reason,
                input_tokens=100,
                output_tokens=attempts[0]["output_tokens"],
                input_cache_hit_tokens=50,
                input_cache_miss_tokens=50,
                latency_ms=10,
                raw_response={
                    "choices": [
                        {
                            "finish_reason": finish_reason,
                            "message": {"content": text},
                        }
                    ]
                },
            ),
            attempts,
        )

    def close(self) -> None:
        self.closed = True


def _task() -> dict[str, Any]:
    return {
        "task_id": "dev::single::selfcheck",
        "domain": "mmlu_pro",
        "prompt": (
            "What is 2 + 2?\n\nChoices:\n"
            "(A) 2\n(B) 3\n(C) 4\n(D) 5\n(E) 6\n(F) 7\n(G) 8\n(H) 9\n(I) 10\n(J) 11\n\n"
            "Return only the letter of the correct choice (A-J)."
        ),
        "options": ["2", "3", "4", "5", "6", "7", "8", "9", "10", "11"],
        "answer_index": 2,
        "reference_answer": "C",
    }


def _valid_output() -> str:
    return json.dumps(
        {
            "answer": "C",
            "reasoning_summary": (
                "2 + 2 = 4, and 4 is option C; the other options are different numbers."
            ),
        },
        separators=(",", ":"),
    )


def _check_config() -> None:
    document = json.loads((PROJECT_ROOT / "configs" / "topology.json").read_text())
    check(
        set(document["topologies"]) == {"debate", "single"},
        "topology.json contains exactly debate and single",
    )
    check(
        document["default_topology_id"] == "debate",
        "the default topology remains debate",
    )
    default_id, default_topology = select_topology(document)
    single_id, single_topology = select_topology(document, "single")
    check(default_id == "debate" and single_id == "single", "both topologies resolve by name")
    check(
        default_topology == document["topologies"]["debate"],
        "the debate definition is returned unchanged",
    )
    check(
        single_topology["runner"] == "single_solver"
        and len(single_topology["nodes"]) == 1
        and single_topology["packets"] == []
        and single_topology["edges"] == [],
        "single topology is one solver node with no packets or edges",
    )
    experiment = load_experiment_config(PROJECT_ROOT, topology_id="single")
    check(
        experiment["topology_id"] == "single"
        and experiment["model_id"] == "deepseek_flash",
        "single selection keeps the DeepSeek default model",
    )
    default_experiment = load_experiment_config(PROJECT_ROOT)
    check(
        default_experiment["topology_id"] == "debate"
        and default_experiment["topology"] == default_topology,
        "omitting --topology still selects the approved debate topology",
    )
    try:
        select_topology(document, "not_a_topology")
    except ConfigurationError as error:
        check("single" in str(error), "unknown topology fails with configured choices")
    else:
        raise AssertionError("unknown topology was silently accepted")


def _check_runner() -> None:
    experiment = load_experiment_config(PROJECT_ROOT, topology_id="single")
    provider = FakeProvider(by_node={"single_solver": ("stop", _valid_output())})
    runner = SingleAgentRunner(dict(experiment), provider)
    trajectory = runner.run_task(task=_task(), run_id="dev-single-run")
    check(trajectory["status"] == "complete", "single task trajectory completes")
    check(trajectory["topology"] == "single", "trajectory records the single topology")
    check("rounds" not in trajectory, "single trajectory fabricates no rounds")
    check(
        trajectory["prediction"]["answer"] == "C"
        and trajectory["prediction"]["reasoning_summary"],
        "single prediction stores answer and reasoning_summary separately",
    )
    check(
        trajectory["prediction"]["cumulative"]["logical_calls"] == 1,
        "one task accounts for exactly one logical solver call",
    )
    check(
        len(provider.requests) == 1
        and provider.requests[0].model == "deepseek-v4-flash"
        and provider.requests[0].reasoning_enabled
        and provider.requests[0].reasoning_effort == "high",
        "single solver uses the selected DeepSeek profile once",
    )
    node_input = json.loads(provider.requests[0].messages[1]["content"])
    serialized = json.dumps(node_input, ensure_ascii=False, sort_keys=True)
    for forbidden in (
        "planner",
        "analyst",
        "critic",
        "previous_writer_checkpoint",
        "visible_messages",
        "transcript",
        "reference_answer",
        "answer_index",
    ):
        check(
            forbidden not in serialized,
            f"single solver input never contains {forbidden}",
        )
    check("options" not in node_input["task"], "public task hides raw answer options")
    check("debate" not in provider.requests[0].messages[0]["content"].casefold()
          and "baseline" not in provider.requests[0].messages[0]["content"].casefold(),
          "single solver prompt is neutral and never mentions debate or baseline")

    gpt_experiment = load_experiment_config(
        PROJECT_ROOT, model_id="gpt4o_mini", topology_id="single"
    )
    gpt_provider = FakeProvider(by_node={"single_solver": ("stop", _valid_output())})
    gpt_trajectory = SingleAgentRunner(dict(gpt_experiment), gpt_provider).run_task(
        task=_task(), run_id="dev-gpt4o-mini"
    )
    request = gpt_provider.requests[0]
    check(
        request.model == "gpt-4o-mini-2024-07-18"
        and request.temperature == 0
        and request.max_output_tokens == 16384
        and not request.reasoning_enabled
        and request.reasoning_effort is None,
        "GPT-4o-mini request uses the fixed snapshot, temperature 0, wide ceiling, no reasoning",
    )
    check(
        gpt_trajectory["configured_model"]["reasoning_enabled"] is False,
        "GPT-4o-mini trajectory records reasoning.enabled=false",
    )
    previous_env = dict(os.environ)
    os.environ["OPENAI_API_KEY"] = "test-openai-key"
    try:
        gpt_openai_provider = build_provider(dict(gpt_experiment))
    finally:
        if "OPENAI_API_KEY" in previous_env:
            os.environ["OPENAI_API_KEY"] = previous_env["OPENAI_API_KEY"]
        else:
            os.environ.pop("OPENAI_API_KEY", None)
    payload = gpt_openai_provider._payload(request)
    check(
        payload.get("max_completion_tokens") == 16384
        and "max_tokens" not in payload
        and "thinking" not in payload
        and "reasoning_effort" not in payload
        and payload.get("response_format") == {"type": "json_object"},
        "GPT-4o-mini wire payload has max_completion_tokens and no reasoning flags",
    )
    gpt_openai_provider.close()


def _check_repair_and_truncation() -> None:
    experiment = load_experiment_config(PROJECT_ROOT, topology_id="single")

    provider = FakeProvider([("stop", "not json"), ("stop", _valid_output())])
    trajectory = SingleAgentRunner(dict(experiment), provider).run_task(
        task=_task(), run_id="dev-repair"
    )
    check(trajectory["status"] == "complete", "invalid JSON repaired to completion")
    check(trajectory["solver"]["format_repairs"] == 1, "one format repair recorded")
    check(
        trajectory["prediction"]["cumulative"]["logical_calls"] == 1
        and trajectory["prediction"]["cumulative"]["api_attempts"] == 2,
        "format repair is an extra recorded attempt, not another logical call",
    )

    provider = FakeProvider([("length", '{"answer": "C"'), ("stop", _valid_output())])
    trajectory = SingleAgentRunner(dict(experiment), provider).run_task(
        task=_task(), run_id="dev-truncation"
    )
    check(trajectory["status"] == "complete", "truncated JSON repaired to completion")
    check(
        trajectory["solver"]["truncation_encountered"]
        and trajectory["solver"]["truncated_attempts"] == 1
        and trajectory["prediction"]["truncated"],
        "truncation is surfaced on the solver record and the prediction",
    )
    check(
        "TruncatedOutput" in provider.requests[1].messages[2]["content"],
        "truncation repair feedback names the violation",
    )

    provider = FakeProvider([("stop", "bad"), ("stop", "still bad"), ("stop", '"C"')])
    trajectory = SingleAgentRunner(dict(experiment), provider).run_task(
        task=_task(), run_id="dev-fallback"
    )
    check(trajectory["status"] == "complete", "answer-only fallback completes the record")
    check(
        trajectory["solver"]["fallback"]["type"] == "answer_only"
        and trajectory["prediction"]["answer"] == "C"
        and "[answer-only fallback"
        in trajectory["prediction"]["reasoning_summary"],
        "fallback keeps the real answer and marks the omitted reasoning_summary",
    )
    check(trajectory["solver"]["format_repairs"] == 2, "bounded repair budget respected")

    provider = FakeProvider([("stop", "bad"), ("length", '{"answer": "C"'), ("stop", "")])
    trajectory = SingleAgentRunner(dict(experiment), provider).run_task(
        task=_task(), run_id="dev-empty-fallback"
    )
    check(
        trajectory["status"] == "failed"
        and trajectory["solver"]["status"] == "format_error",
        "empty fallback fails honestly instead of fabricating an answer",
    )
    check(
        trajectory["solver"]["truncation_encountered"],
        "truncation remains visible even on a failed trajectory",
    )


def _check_scoring() -> None:
    record = {
        "task": _task(),
        "split": "test",
        "topology": "single",
        "trajectory": {
            "status": "complete",
            "trajectory_id": "dev-score",
            "prediction": {
                "answer": "A",
                "reasoning_summary": "The canonical answer is C, option C is correct.",
                "checkpoint_hash": "hash-1",
                "cumulative": {"logical_calls": 1},
            },
        },
    }
    scores = score_single_record(record)
    check(
        len(scores) == 1
        and scores[0]["quality"] == 0.0
        and "round_index" not in scores[0],
        "single scoring uses only the answer and fabricates no round",
    )
    record["trajectory"]["prediction"]["answer"] = "C"
    scores = score_single_record(record)
    check(scores[0]["quality"] == 1.0, "canonical single answer scores 1")


def _check_analysis() -> None:
    records: list[dict[str, Any]] = []
    for index, (task_id, split, quality, missing_tokens) in enumerate(
        (
            ("dev::analysis::1", "train", 1.0, False),
            ("dev::analysis::2", "train", 0.0, False),
            ("dev::analysis::3", "test", 1.0, True),
            ("dev::analysis::4", "test", 0.0, False),
        )
    ):
        task = _task()
        task["task_id"] = task_id
        cumulative: dict[str, Any] = {
            "output_tokens": 20,
            "wall_clock_ms": 100,
            "api_latency_ms": 90,
            "cost_usd": 0.001,
            "logical_calls": 1,
            "api_attempts": 1,
        }
        if not missing_tokens:
            cumulative["input_tokens"] = 80
        records.append(
            {
                "task": task,
                "split": split,
                "topology": "single",
                "trajectory": {
                    "status": "complete",
                    "trajectory_id": f"dev-analysis-{index}",
                    "prediction": {
                        "answer": "C" if quality else "A",
                        "reasoning_summary": "reasoning",
                        "checkpoint_hash": f"hash-{index}",
                        "finish_reason": "stop",
                        "truncated": False,
                        "cumulative": cumulative,
                    },
                    "solver": {
                        "status": "completed",
                        "format_repairs": 0,
                        "finish_reason": "stop",
                    },
                },
                "scores": [
                    {
                        "task_id": task_id,
                        "quality": quality,
                        "predicted_answer": "C" if quality else "A",
                    }
                ],
            }
        )
    analysis = build_single_analysis({"run_id": "dev-analysis", "model_selection": {}}, records)
    summary = summarize_single_collection(records)
    check(analysis["topology"] == "single", "analysis declares the single topology")
    check(analysis["tasks_total"] == 4 and analysis["tasks_complete"] == 4, "analysis counts tasks")
    check(analysis["accuracy"]["accuracy"] == 0.5, "overall accuracy aggregates correctness")
    check(
        analysis["accuracy_by_split"]["train"]["accuracy"] == 0.5
        and analysis["accuracy_by_split"]["test"]["accuracy"] == 0.5,
        "accuracy is reported by frozen split",
    )
    check(
        analysis["calls"]["logical_calls"] == 4
        and analysis["calls"]["format_repairs"] == 0,
        "logical calls equal the number of tasks",
    )
    check(
        analysis["resources"]["input_tokens"]["n_observed"] == 3
        and analysis["resources"]["input_tokens"]["total"] is None,
        "unknown token counters stay unknown instead of becoming zero",
    )
    check(
        analysis["finish_reason_distribution"] == {"stop": 4},
        "finish-reason distribution is recorded",
    )
    for forbidden in ("labels", "policy", "delta_q", "V", "G", "transitions", "oracle"):
        check(forbidden not in analysis, f"single analysis never builds {forbidden}")
    check("accuracy" in summary and "resources" in summary, "single collection summary is complete")


def _check_manifest() -> None:
    experiment = load_experiment_config(PROJECT_ROOT, topology_id="single")
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        manifest = create_run(
            root,
            command=["roundvalue", "smoke", "--topology", "single"],
            config_snapshot={},
            dataset_name="smoke",
            domain="mmlu_pro",
            topology_id="single",
            requested_model="deepseek-v4-flash",
        )
        updated = update_run_status(
            manifest,
            "running",
            mode="smoke",
            selected_model_id=experiment["model_id"],
            selected_topology_id="single",
            topology_runner="single_solver",
            topology_definition=experiment["topology"],
            topology_hash=json.dumps({"hash": "x"}),
        )
        reloaded = json.loads(
            (root / "trajectories" / updated["run_id"] / "run.json").read_text()
        )
        check(
            reloaded["selected_topology_id"] == "single"
            and reloaded["topology_runner"] == "single_solver"
            and reloaded["topology_definition"] == experiment["topology"],
            "run manifest records topology selector, definition, and runner",
        )
        check(
            re.fullmatch(
                r"\d{12}_smoke_single_deepseek-v4-flash_[0-9a-f]{8}",
                updated["run_id"],
            )
            and (root / "trajectories" / updated["run_id"]).is_dir()
            and (root / "results" / updated["run_id"]).is_dir()
            and updated["run_name_components"]
            == {
                "timestamp": updated["run_id"].split("_")[0],
                "dataset": "smoke",
                "topology": "single",
                "model": "deepseek-v4-flash",
                "hex": updated["run_id"].split("_")[-1],
            },
            "trajectory and result directories share one canonical run name",
        )


def _write_synthetic_run(
    root: Path,
    run_id: str,
    topology: str,
    *,
    benchmark_sha: str,
    model_selection: dict[str, Any],
    task_ids: list[str],
) -> str:
    trajectory_dir = root / "trajectories" / run_id
    result_dir = root / "results" / run_id
    trajectory_dir.mkdir(parents=True)
    result_dir.mkdir(parents=True)
    manifest = {
        "run_id": run_id,
        "dataset": "comparison-benchmark",
        "selected_model_id": "model",
        "model_selection": model_selection,
        "selected_topology_id": topology,
        "trajectory_dir": str(trajectory_dir),
        "result_dir": str(result_dir),
    }
    write_json(trajectory_dir / "run.json", manifest)
    write_json(result_dir / "manifest.json", manifest)
    write_json(
        trajectory_dir / "benchmark_snapshot.json",
        {"source_sha256": benchmark_sha, "content": {"dataset_id": "comparison-benchmark"}},
    )
    write_json(
        trajectory_dir / "frozen_splits.json",
        {"schema_version": "1.0", "split_seed": 1, "splits": {task_id: "test" for task_id in task_ids}},
    )
    task_document = {
        "task_id": task_ids[0],
        "domain": "mmlu_pro",
        "prompt": "prompt",
        "options": ["2", "4"],
        "answer_index": 1,
        "reference_answer": "B",
    }
    if topology == "single":
        write_json(
            trajectory_dir / "task_aaaaaaaaaaaaaaaa.json",
            {
                "task": task_document,
                "split": "test",
                "topology": "single",
                "trajectory": {
                    "status": "complete",
                    "trajectory_id": "single-trajectory",
                    "prediction": {
                        "answer": "B",
                        "reasoning_summary": "reasoning",
                        "checkpoint_hash": "single-hash",
                        "finish_reason": "stop",
                        "truncated": False,
                        "cumulative": {
                            "input_tokens": 80,
                            "output_tokens": 20,
                            "wall_clock_ms": 100,
                            "api_latency_ms": 90,
                            "cost_usd": 0.001,
                            "logical_calls": 1,
                        },
                    },
                },
                "scores": [
                    {"task_id": task_ids[0], "quality": 0.0, "predicted_answer": "A"}
                ],
            },
        )
        write_json(
            result_dir / "scores.json",
            {
                "scores_by_task": {
                    task_ids[0]: [
                        {"task_id": task_ids[0], "quality": 0.0, "predicted_answer": "A"}
                    ]
                }
            },
        )
    else:
        checkpoints = []
        scores = []
        for round_index in range(1, 6):
            quality = 0.0 if round_index == 1 else 1.0
            checkpoints.append(
                {
                    "round_index": round_index,
                    "answer": "A" if quality == 0 else "B",
                    "reasoning_summary": "reasoning",
                    "checkpoint_hash": f"debate-hash-{round_index}",
                    "cumulative": {
                        "input_tokens": 80 * round_index,
                        "output_tokens": 20 * round_index,
                        "wall_clock_ms": 100 * round_index,
                        "api_latency_ms": 90 * round_index,
                        "cost_usd": 0.001 * round_index,
                        "logical_calls": 7 * round_index,
                    },
                }
            )
            scores.append(
                {"task_id": task_ids[0], "round_index": round_index, "quality": quality}
            )
        write_json(
            trajectory_dir / "task_aaaaaaaaaaaaaaaa.json",
            {
                "task": task_document,
                "split": "test",
                "topology": "debate",
                "trajectory": {
                    "status": "complete",
                    "trajectory_id": "debate-trajectory",
                    "checkpoints": checkpoints,
                },
                "scores": scores,
            },
        )
        write_json(
            result_dir / "scores.json",
            {"scores_by_task": {task_ids[0]: scores}},
        )
        write_json(
            result_dir / "test_policy_replay.json",
            {
                "policy_metrics": {
                    "roundvalue": {
                        "accuracy": 0.75,
                        "mean_total_tokens": 1200.0,
                        "mean_wall_clock_ms": 300.0,
                        "mean_api_latency_ms": 280.0,
                        "mean_cost_usd": 0.003,
                        "mean_logical_calls": 14.0,
                        "n_records": 1,
                    },
                    "oracle": {
                        "accuracy": 1.0,
                        "mean_total_tokens": 2500.0,
                        "mean_wall_clock_ms": 500.0,
                        "mean_api_latency_ms": 450.0,
                        "mean_cost_usd": 0.006,
                        "mean_logical_calls": 35.0,
                        "n_records": 1,
                    },
                }
            },
        )
    return str(root)


def _check_comparison() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        benchmark_sha = "a" * 64
        model_selection = {
            "provider": "openai",
            "requested_model": "gpt-4o-mini-2024-07-18",
            "temperature": 0,
            "max_output_tokens": 16384,
            "reasoning": {"enabled": False},
        }
        _write_synthetic_run(
            root, "single-1", "single", benchmark_sha=benchmark_sha,
            model_selection=model_selection, task_ids=["t1"],
        )
        _write_synthetic_run(
            root, "debate-1", "debate", benchmark_sha=benchmark_sha,
            model_selection=model_selection, task_ids=["t1"],
        )
        comparison = build_comparison(root, "single-1", "debate-1")
        check(
            comparison["comparison_type"] == "same_model_topology_comparison",
            "same-model comparison is labeled as a topology comparison",
        )
        check(comparison["single"]["accuracy"] == 0.0, "single accuracy read correctly")
        check(comparison["debate"]["round_1"]["accuracy"] == 0.0, "debate round-1 accuracy read correctly")
        check(comparison["debate"]["round_5"]["accuracy"] == 1.0, "debate round-5 accuracy read correctly")
        check(comparison["debate"]["roundvalue"]["accuracy"] == 0.75, "RoundValue accuracy read when defined")
        check(comparison["debate"]["oracle"]["accuracy"] == 1.0, "Oracle accuracy read when defined")
        check(
            comparison["accuracy_difference_single_minus_debate"]["round_5"] == -1.0,
            "absolute accuracy difference is computed",
        )
        paired_5 = comparison["paired"]["debate_round_5"]
        check(
            paired_5["single_wrong_debate_correct"] == 1
            and paired_5["n_paired"] == 1,
            "paired single-vs-debate outcome counts are computed",
        )
        check(
            comparison["compatibility"]["task_set_match"]
            and comparison["compatibility"]["split_assignment_match"],
            "compatibility checks pass for a matching pair",
        )
        manifest = {
            "run_id": "single-1",
            "result_dir": str(root / "results" / "single-1"),
        }
        path = write_comparison(manifest, comparison, "debate-1")
        check(path.is_file(), "comparison artifact is saved to the visualized run")

        _write_synthetic_run(
            root, "single-mismatch", "single", benchmark_sha="b" * 64,
            model_selection=model_selection, task_ids=["t1"],
        )
        try:
            build_comparison(root, "single-mismatch", "debate-1")
        except ComparisonError as error:
            check("benchmark file hash differs" in str(error), "mismatched benchmark hash is refused")
        else:
            raise AssertionError("mismatched benchmark hash was compared")

        _write_synthetic_run(
            root, "single-tasks", "single", benchmark_sha=benchmark_sha,
            model_selection=model_selection, task_ids=["t1"],
        )
        tasks_dir = root / "trajectories" / "single-tasks"
        frozen = json.loads((tasks_dir / "frozen_splits.json").read_text())
        frozen["splits"] = {"other-task": "test"}
        write_json(tasks_dir / "frozen_splits.json", frozen)
        try:
            build_comparison(root, "single-tasks", "debate-1")
        except ComparisonError as error:
            check("frozen split assignments differ" in str(error), "mismatched split assignment is refused")
        else:
            raise AssertionError("mismatched split assignment was compared")

        cross_model = {
            "provider": "deepseek",
            "requested_model": "deepseek-v4-flash",
            "temperature": 0.2,
            "max_output_tokens": 32768,
            "reasoning": {"enabled": True, "effort": "high"},
        }
        _write_synthetic_run(
            root, "single-cross", "single", benchmark_sha=benchmark_sha,
            model_selection=cross_model, task_ids=["t1"],
        )
        comparison = build_comparison(root, "single-cross", "debate-1")
        check(
            comparison["comparison_type"] == "cross_model_topology_comparison",
            "cross-model comparison is labeled as cross-model + cross-topology",
        )

    check(run_topology_id({}) == "debate", "historical manifests default to debate")


def main() -> int:
    _check_config()
    _check_runner()
    _check_repair_and_truncation()
    _check_scoring()
    _check_analysis()
    _check_manifest()
    _check_comparison()
    print("PASS all single-topology self-checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
