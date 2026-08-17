"""Offline self-check for the automatic Single-Agent baseline.

This script never contacts a model provider.  A scripted fake provider
verifies that the baseline is not a topology, that one independent solver
observation is collected per task alongside the Debate trajectory, that the
two conditions are causally isolated, that scoring uses only ``answer``, and
that the offline Single-Agent aggregates and paired Single-vs-Debate counts
are computed over the same frozen tasks.
"""

from __future__ import annotations

import copy
import inspect
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

from config_loader import load_experiment_config, select_topology  # noqa: E402
from contracts import ModelRequest, ModelResponse  # noqa: E402
from debate_runner import FixedDebateRunner  # noqa: E402
from provider import build_provider  # noqa: E402
from scorer import score_single_observation  # noqa: E402
from single_analysis import (  # noqa: E402
    baseline_table,
    build_single_baseline,
    paired_single_vs_debate,
    single_quality_by_task,
    summarize_single_baseline,
)
from single_runner import SingleAgentRunner  # noqa: E402
from storage import create_run  # noqa: E402

import pipeline as tb  # noqa: E402


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


def _valid_output(answer: str = "C") -> str:
    return json.dumps(
        {
            "answer": answer,
            "reasoning_summary": (
                "2 + 2 = 4, and 4 is option C; the other options are different numbers."
            ),
        },
        separators=(",", ":"),
    )


def _check_config() -> None:
    document = json.loads((PROJECT_ROOT / "configs" / "topology.json").read_text())
    check(
        set(document["topologies"]) == {"debate"},
        "topology.json contains only the frozen debate topology",
    )
    check(
        document["default_topology_id"] == "debate",
        "the default topology remains debate",
    )
    default_id, default_topology = select_topology(document)
    check(
        default_id == "debate" and default_topology == document["topologies"]["debate"],
        "the frozen debate definition resolves unchanged",
    )
    experiment = load_experiment_config(PROJECT_ROOT)
    check(
        experiment["topology_id"] == "debate"
        and experiment["model_id"] == "deepseek_flash",
        "the one experiment selects the frozen debate and the DeepSeek default",
    )
    check(
        "topology_id" not in inspect.signature(load_experiment_config).parameters,
        "load_experiment_config exposes no topology selector",
    )
    role_ids = {role["id"] for role in experiment["agents"]["roles"]}
    check(
        role_ids == {"planner", "analyst", "critic", "writer", "single_solver"},
        "single_solver exists only as a role, not as a topology",
    )


def _check_runner() -> None:
    experiment = load_experiment_config(PROJECT_ROOT)
    # Pin the thinking-mode expectations explicitly so a user can flip the
    # configured reasoning mode without breaking this baseline self-check.
    thinking_experiment = copy.deepcopy(experiment)
    thinking_experiment["model"]["reasoning"] = {"enabled": True, "effort": "high"}
    provider = FakeProvider(by_node={"single_solver": ("stop", _valid_output())})
    runner = SingleAgentRunner(dict(thinking_experiment), provider)
    observation = runner.run_observation(task=_task(), run_id="dev-single-run")
    check(observation["status"] == "complete", "single observation completes")
    check(
        observation["kind"] == "single_agent_baseline"
        and "observation_id" in observation,
        "observation records the single_agent_baseline kind",
    )
    check(
        "rounds" not in observation
        and "checkpoints" not in observation
        and "topology" not in observation,
        "the baseline fabricates no rounds, checkpoints, or topology",
    )
    check(
        observation["prediction"]["answer"] == "C"
        and observation["prediction"]["reasoning_summary"],
        "prediction stores answer and reasoning_summary separately",
    )
    check(
        observation["prediction"]["cumulative"]["logical_calls"] == 1,
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
        "round_index",
    ):
        check(
            forbidden not in serialized,
            f"single solver input never contains {forbidden}",
        )
    check("options" not in node_input["task"], "public task hides raw answer options")
    check(
        "debate" not in provider.requests[0].messages[0]["content"].casefold()
        and "baseline" not in provider.requests[0].messages[0]["content"].casefold(),
        "single solver prompt is neutral and never mentions debate or baseline",
    )

    gpt_experiment = load_experiment_config(PROJECT_ROOT, model_id="gpt4o_mini")
    gpt_provider = FakeProvider(by_node={"single_solver": ("stop", _valid_output())})
    gpt_observation = SingleAgentRunner(dict(gpt_experiment), gpt_provider).run_observation(
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
        gpt_observation["configured_model"]["reasoning_enabled"] is False,
        "GPT-4o-mini observation records reasoning.enabled=false",
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
    experiment = load_experiment_config(PROJECT_ROOT)

    provider = FakeProvider([("stop", "not json"), ("stop", _valid_output())])
    observation = SingleAgentRunner(dict(experiment), provider).run_observation(
        task=_task(), run_id="dev-repair"
    )
    check(observation["status"] == "complete", "invalid JSON repaired to completion")
    check(observation["solver"]["format_repairs"] == 1, "one format repair recorded")
    check(
        observation["prediction"]["cumulative"]["logical_calls"] == 1
        and observation["prediction"]["cumulative"]["api_attempts"] == 2,
        "format repair is an extra recorded attempt, not another logical call",
    )

    provider = FakeProvider([("length", '{"answer": "C"'), ("stop", _valid_output())])
    observation = SingleAgentRunner(dict(experiment), provider).run_observation(
        task=_task(), run_id="dev-truncation"
    )
    check(observation["status"] == "complete", "truncated JSON repaired to completion")
    check(
        observation["solver"]["truncation_encountered"]
        and observation["solver"]["truncated_attempts"] == 1
        and observation["prediction"]["truncated"],
        "truncation is surfaced on the solver record and the prediction",
    )
    check(
        "TruncatedOutput" in provider.requests[1].messages[2]["content"],
        "truncation repair feedback names the violation",
    )

    provider = FakeProvider([("stop", "bad"), ("stop", "still bad"), ("stop", '"C"')])
    observation = SingleAgentRunner(dict(experiment), provider).run_observation(
        task=_task(), run_id="dev-fallback"
    )
    check(observation["status"] == "complete", "answer-only fallback completes the record")
    check(
        observation["solver"]["fallback"]["type"] == "answer_only"
        and observation["prediction"]["answer"] == "C"
        and "[answer-only fallback"
        in observation["prediction"]["reasoning_summary"],
        "fallback keeps the real answer and marks the omitted reasoning_summary",
    )
    check(observation["solver"]["format_repairs"] == 2, "bounded repair budget respected")

    provider = FakeProvider([("stop", "bad"), ("length", '{"answer": "C"'), ("stop", "")])
    observation = SingleAgentRunner(dict(experiment), provider).run_observation(
        task=_task(), run_id="dev-empty-fallback"
    )
    check(
        observation["status"] == "failed"
        and observation["solver"]["status"] == "format_error",
        "empty fallback fails honestly instead of fabricating an answer",
    )
    check(
        observation["solver"]["truncation_encountered"],
        "truncation remains visible even on a failed observation",
    )


def _check_scoring() -> None:
    record = {
        "task": _task(),
        "split": "test",
        "single_agent": {
            "status": "complete",
            "observation_id": "dev-observation",
            "prediction": {
                "answer": "A",
                "reasoning_summary": "The canonical answer is C, option C is correct.",
                "checkpoint_hash": "hash-1",
                "cumulative": {"logical_calls": 1},
            },
        },
    }
    scores = score_single_observation(record)
    check(
        len(scores) == 1
        and scores[0]["quality"] == 0.0
        and "round_index" not in scores[0]
        and scores[0]["observation_id"] == "dev-observation",
        "single scoring uses only the answer and fabricates no round",
    )
    record["single_agent"]["prediction"]["answer"] = "C"
    scores = score_single_observation(record)
    check(scores[0]["quality"] == 1.0, "canonical single answer scores 1")


def _make_record(
    task_id: str,
    split: str,
    single_quality: float,
    round_qualities: list[float],
) -> dict[str, Any]:
    task = _task()
    task["task_id"] = task_id
    checkpoints = []
    scores = []
    for round_index, quality in enumerate(round_qualities, start=1):
        checkpoints.append(
            {
                "round_index": round_index,
                "answer": "C" if quality else "A",
                "reasoning_summary": "reasoning",
                "checkpoint_hash": f"hash-{round_index}",
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
            {"task_id": task_id, "round_index": round_index, "quality": float(quality)}
        )
    return {
        "task": task,
        "split": split,
        "trajectory": {
            "status": "complete",
            "trajectory_id": f"trajectory-{task_id}",
            "checkpoints": checkpoints,
        },
        "scores": scores,
        "single_agent": {
            "status": "complete",
            "observation_id": f"observation-{task_id}",
            "prediction": {
                "answer": "C" if single_quality else "A",
                "reasoning_summary": "reasoning",
                "checkpoint_hash": f"single-hash-{task_id}",
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
            "solver": {
                "status": "completed",
                "format_repairs": 0,
                "finish_reason": "stop",
            },
        },
        "single_agent_scores": [
            {
                "task_id": task_id,
                "quality": float(single_quality),
                "predicted_answer": "C" if single_quality else "A",
            }
        ],
    }


def _check_analysis() -> None:
    records = [
        _make_record("dev::analysis::1", "train", 0.0, [0.0, 1.0, 1.0, 1.0, 1.0]),
        _make_record("dev::analysis::2", "train", 1.0, [1.0, 0.0, 0.0, 1.0, 1.0]),
        _make_record("dev::analysis::3", "test", 0.0, [1.0, 1.0, 1.0, 1.0, 1.0]),
        _make_record("dev::analysis::4", "test", 0.0, [0.0, 0.0, 0.0, 0.0, 0.0]),
    ]
    replay = {
        "policies": {
            "fixed_1": {
                "task_results": [
                    {"task_id": "dev::analysis::1", "quality": 0.0, "available": True},
                    {"task_id": "dev::analysis::2", "quality": 1.0, "available": True},
                    {"task_id": "dev::analysis::3", "quality": 1.0, "available": True},
                    {"task_id": "dev::analysis::4", "quality": 0.0, "available": True},
                ]
            },
            "fixed_5": {
                "task_results": [
                    {"task_id": "dev::analysis::1", "quality": 1.0, "available": True},
                    {"task_id": "dev::analysis::2", "quality": 1.0, "available": True},
                    {"task_id": "dev::analysis::3", "quality": 1.0, "available": True},
                    {"task_id": "dev::analysis::4", "quality": 0.0, "available": True},
                ]
            },
            "roundvalue": {
                "task_results": [
                    {"task_id": "dev::analysis::1", "quality": 1.0, "available": True},
                    {"task_id": "dev::analysis::2", "quality": 0.0, "available": True},
                    {"task_id": "dev::analysis::3", "quality": 1.0, "available": True},
                    {"task_id": "dev::analysis::4", "quality": 0.0, "available": True},
                ]
            },
            "oracle": {
                "task_results": [
                    {"task_id": "dev::analysis::1", "quality": 1.0, "available": True},
                    {"task_id": "dev::analysis::2", "quality": 1.0, "available": True},
                    {"task_id": "dev::analysis::3", "quality": 1.0, "available": True},
                    {"task_id": "dev::analysis::4", "quality": 0.0, "available": True},
                ]
            },
        },
        "policy_metrics": {
            "fixed_1": {"accuracy": 0.5, "mean_total_tokens": 100.0, "n_records": 4},
            "fixed_2": {"accuracy": 0.5, "mean_total_tokens": 200.0, "n_records": 4},
            "fixed_3": {"accuracy": 0.5, "mean_total_tokens": 300.0, "n_records": 4},
            "fixed_4": {"accuracy": 0.5, "mean_total_tokens": 400.0, "n_records": 4},
            "fixed_5": {"accuracy": 0.75, "mean_total_tokens": 500.0, "n_records": 4},
            "roundvalue": {"accuracy": 0.5, "mean_total_tokens": 250.0, "n_records": 4},
            "oracle": {"accuracy": 0.75, "mean_total_tokens": 500.0, "n_records": 4},
        },
    }
    summary = summarize_single_baseline(records)
    analysis = build_single_baseline(
        {
            "run_id": "dev-analysis",
            "selected_model_id": "deepseek_flash",
            "model_selection": {},
        },
        records,
    )
    check(summary["defined"] is True, "the baseline is defined when observations exist")
    check(summary["tasks_total"] == 4 and summary["tasks_complete"] == 4, "baseline counts tasks")
    check(summary["accuracy"]["accuracy"] == 0.25, "overall single accuracy aggregates correctness")
    check(
        summary["accuracy_by_split"]["train"]["accuracy"] == 0.5
        and summary["accuracy_by_split"]["test"]["accuracy"] == 0.0,
        "single accuracy is reported by frozen split",
    )
    check(
        summary["resources"]["logical_calls"]["total"] == 4,
        "logical calls equal the number of tasks",
    )
    check(
        summary["finish_reason_distribution"] == {"stop": 4},
        "finish-reason distribution is recorded",
    )
    for forbidden in ("labels", "policy", "delta_q", "V", "G", "transitions", "oracle"):
        check(forbidden not in analysis, f"single baseline never builds {forbidden}")

    paired = paired_single_vs_debate(records, replay)
    fixed_1 = paired["fixed_1"]
    check(
        fixed_1["defined"]
        and fixed_1["n_paired"] == 4
        and fixed_1["both_correct"] == 1
        and fixed_1["single_correct_debate_wrong"] == 0
        and fixed_1["single_wrong_debate_correct"] == 1
        and fixed_1["both_wrong"] == 2,
        "Single-Agent vs Fixed-1 paired counts are exact",
    )
    fixed_5 = paired["fixed_5"]
    check(
        fixed_5["defined"]
        and fixed_5["single_wrong_debate_correct"] == 2
        and fixed_5["both_correct"] == 1
        and fixed_5["both_wrong"] == 1,
        "Single-Agent vs Fixed-5 paired counts are exact",
    )
    roundvalue = paired["roundvalue"]
    check(
        roundvalue["defined"]
        and roundvalue["single_correct_debate_wrong"] == 1
        and roundvalue["single_wrong_debate_correct"] == 2,
        "Single-Agent vs RoundValue paired counts are exact",
    )
    check(
        paired["oracle"]["defined"]
        and paired["oracle"]["single_wrong_debate_correct"] == 2,
        "Single-Agent vs Oracle paired counts are exact",
    )
    for name, counts in paired.items():
        check(
            set(counts)
            >= {
                "both_correct",
                "single_correct_debate_wrong",
                "single_wrong_debate_correct",
                "both_wrong",
            },
            f"paired {name} exposes the four required outcome counts",
        )
        check(
            "repair" not in counts and "harm" not in counts,
            f"paired {name} does not reuse Repair/Harm terminology",
        )

    single_by_task = single_quality_by_task(records)
    table = baseline_table(summary, replay, single_by_task)
    check(
        [row["display_name"] for row in table]
        == [
            "Single-Agent",
            "Fixed-1",
            "Fixed-2",
            "Fixed-3",
            "Fixed-4",
            "Fixed-5",
            "RoundValue",
            "Oracle",
        ],
        "baseline table orders Single-Agent with the Debate baselines",
    )
    check(
        table[0]["defined"] and table[0]["accuracy"] == 0.25,
        "baseline table carries the Single-Agent accuracy",
    )
    # Same-split regression: when the Debate conditions only cover the test
    # split, the Single-Agent row must report the Single-Agent accuracy over
    # exactly that task-ID set, never the all-task accuracy.
    test_only_replay = {
        "policies": {
            "roundvalue": {
                "task_results": [
                    {"task_id": "dev::analysis::3", "quality": 1.0, "available": True},
                    {"task_id": "dev::analysis::4", "quality": 0.0, "available": True},
                ]
            }
        },
        "policy_metrics": {
            "fixed_1": {"accuracy": 0.5, "n_records": 2},
            "roundvalue": {"accuracy": 0.5, "n_records": 2},
        },
    }
    test_only_table = baseline_table(summary, test_only_replay, single_by_task)
    check(
        test_only_table[0]["accuracy"] == 0.0
        and test_only_table[0]["n_tasks"] == 2,
        "baseline table Single-Agent accuracy uses the same task set as Debate",
    )
    roundvalue_row = next(
        row for row in test_only_table if row["condition"] == "roundvalue"
    )
    check(
        roundvalue_row["paired_single_accuracy"] == 0.0
        and roundvalue_row["n_paired"] == 2,
        "each Debate row carries Single-Agent accuracy over its own task IDs",
    )
    test_only_paired = paired_single_vs_debate(records, test_only_replay)
    check(
        test_only_paired["roundvalue"]["single_accuracy"] == 0.0
        and test_only_paired["roundvalue"]["debate_accuracy"] == 0.5
        and test_only_paired["roundvalue"]["n_paired"] == 2,
        "paired Single-vs-Debate accuracies share one task-ID intersection",
    )


def _check_task_record_and_isolation() -> None:
    experiment = load_experiment_config(PROJECT_ROOT)
    by_node: dict[str, tuple[str | None, str]] = {}
    for node_id in (
        "planner_stage_1",
        "analyst_stage_1",
        "critic_stage_1",
        "planner_stage_2",
        "analyst_stage_2",
        "critic_stage_2",
        "writer",
    ):
        role = node_id.split("_")[0]
        fields = {
            "planner": {
                "plan": "p",
                "assumptions": "a",
                "verification_steps": "v",
                "candidate_answer": "A",
            },
            "analyst": {
                "analysis": "a",
                "candidate_answer": "A",
                "evidence": "e",
            },
            "critic": {
                "issues": "i",
                "evidence": "e",
                "revision_advice": "r",
                "candidate_answer": "A",
            },
            "writer": {
                "answer": "A",
                "reasoning_summary": "unique-debate-writer-text",
            },
        }[role]
        by_node[node_id] = ("stop", json.dumps(fields, separators=(",", ":")))
    by_node["single_solver"] = ("stop", _valid_output("C"))
    provider = FakeProvider(by_node=by_node)
    runner = FixedDebateRunner(dict(experiment), provider)
    single_runner = SingleAgentRunner(dict(experiment), provider)
    record = tb._collect_task(
        runner,
        single_runner,
        task=_task(),
        split="test",
        run_id="dev-collect",
        max_rounds=1,
        score=True,
    )
    check(
        "trajectory" in record
        and "single_agent" in record
        and record["trajectory"]["status"] == "complete"
        and record["single_agent"]["status"] == "complete",
        "one task record holds Debate and Single-Agent siblings",
    )
    check(
        len(record["scores"]) == 1
        and len(record["single_agent_scores"]) == 1
        and record["scores"][0]["quality"] == 0.0
        and record["single_agent_scores"][0]["quality"] == 1.0,
        "Debate and Single-Agent are scored independently",
    )
    single_input = record["single_agent"]["solver"]["input"]
    single_prediction = record["single_agent"]["prediction"]
    check(
        "unique-debate-writer-text" not in json.dumps(single_input)
        and "reference_answer" not in json.dumps(single_input)
        and "previous_writer_checkpoint" not in json.dumps(single_input),
        "Single-Agent input is the sanitized public task only",
    )
    debate_nodes = [
        request
        for request in provider.requests
        if request.metadata.get("node_id") != "single_solver"
    ]
    check(
        all(
            single_prediction["reasoning_summary"] not in json.dumps(request.messages)
            for request in debate_nodes
        )
        and len(debate_nodes) == 7,
        "Debate nodes never see the Single-Agent output or the solver prompt",
    )
    check(
        tb._scored_component_complete(record, component="trajectory")
        and tb._scored_component_complete(record, component="single_agent"),
        "both components satisfy the resume completeness gate",
    )


def _check_manifest() -> None:
    experiment = load_experiment_config(PROJECT_ROOT)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        manifest = create_run(
            root,
            command=["roundvalue", "smoke"],
            config_snapshot={},
            dataset_name="SmokeTasks",
            domain="mmlu_pro",
            requested_model="deepseek-v4-flash",
        )
        check(
            re.fullmatch(
                r"\d{12}_deepseek-v4-flash_SmokeTasks_[0-9a-f]{8}",
                manifest["run_id"],
            )
            and (root / "trajectories" / manifest["run_id"]).is_dir()
            and (root / "results" / manifest["run_id"]).is_dir()
            and manifest["run_name_components"]
            == {
                "timestamp": manifest["run_id"].split("_")[0],
                "model": "deepseek-v4-flash",
                "dataset": "SmokeTasks",
                "hex": manifest["run_id"].split("_")[-1],
            },
            "trajectory and result directories share one topology-free run name",
        )
        check(
            "topology" not in manifest["run_name_components"],
            "topology never appears in run directory components",
        )


def main() -> int:
    _check_config()
    _check_runner()
    _check_repair_and_truncation()
    _check_scoring()
    _check_analysis()
    _check_task_record_and_isolation()
    _check_manifest()
    print("PASS all single-agent-baseline self-checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
