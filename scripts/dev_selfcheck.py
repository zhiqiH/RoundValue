"""Offline self-check for the verify-repair loop in ``FixedDebateRunner``.

This script never contacts a model provider.  A scripted fake provider emits
the exact failure signatures observed in the field (token-length truncation,
early-stop invalid JSON, and missing required fields) and asserts that the
runner repairs them within its bounded retry budget, records every attempt,
and derives each node's output budget from its declared schema.  Per-field
``max_length`` is advisory, so a valid overshoot is accepted without repair
while the hard token budget still bounds the reply.  The terminal repair is a
deterministic answer-only fallback: it asks for nothing but the answer and
then completes the frozen schema around that answer with recorded markers, so
a node cannot fail merely because the model refuses to write a short JSON.
"""

from __future__ import annotations

import copy
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from config_loader import load_experiment_config  # noqa: E402
from contracts import ModelRequest, ModelResponse  # noqa: E402
from debate_runner import FixedDebateRunner  # noqa: E402
from labels import build_labels  # noqa: E402
from policy import replay_policies  # noqa: E402
from scorer import score_trajectory  # noqa: E402


ROLE_OUTPUTS = {
    "planner": {
        "plan": "solve directly",
        "assumptions": "standard",
        "verification_steps": "check once",
        "candidate_answer": "5",
    },
    "analyst": {
        "analysis": "identity holds",
        "candidate_answer": "5",
        "evidence": "substitution",
    },
    "critic": {
        "issues": "none",
        "evidence": "checked",
        "revision_advice": "none",
        "candidate_answer": "5",
    },
    "writer": {
        "answer": "C",
        "reasoning_summary": "2 + 2 = 4, and 4 is option C; no other choice is plausible.",
    },
}


class FakeProvider:
    """Return scripted chat completions and retain every request for checks."""

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

    def generate(self, request: ModelRequest) -> tuple[ModelResponse, list[dict[str, Any]]]:
        self.requests.append(request)
        node_id = request.metadata.get("node_id")
        if node_id in self.by_node:
            finish_reason, text = self.by_node[node_id]
            scripted = True
        else:
            scripted = False
        if not self.script:
            if not scripted:
                raise AssertionError("fake provider ran out of scripted responses")
        elif not scripted:
            finish_reason, text = self.script.pop(0)
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
                "request_payload": {"messages": request.messages, "max_tokens": request.max_output_tokens},
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
                raw_response={"choices": [{"finish_reason": finish_reason, "message": {"content": text}}]},
            ),
            attempts,
        )

    def close(self) -> None:
        self.closed = True


def _task() -> dict[str, Any]:
    return {
        "task_id": "dev::selfcheck",
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


def _node_result(experiment: dict[str, Any], script: list[tuple[str | None, str]]) -> tuple[dict[str, Any], FakeProvider]:
    provider = FakeProvider(script)
    runner = FixedDebateRunner(dict(experiment), provider)
    record = runner._run_node(
        task=_task(),
        round_index=1,
        node_id="writer",
        role_id="writer",
        previous_checkpoint=None,
        visible_messages=[],
    )
    return record, provider


def check(condition: bool, label: str) -> None:
    if not condition:
        raise AssertionError(label)
    print(f"PASS {label}")


def main() -> int:
    experiment = load_experiment_config(PROJECT_ROOT)
    # Pin the thinking-mode expectations explicitly so a user can flip the
    # configured reasoning mode without breaking these runner self-checks.
    thinking_experiment = copy.deepcopy(experiment)
    thinking_experiment["model"]["reasoning"] = {"enabled": True, "effort": "high"}

    valid = json.dumps(ROLE_OUTPUTS["writer"], separators=(",", ":"))
    truncated = '{"answer": "C"'
    overlong = json.dumps(
        {"answer": "x" * 600, "reasoning_summary": "ok"},
        separators=(",", ":"),
    )
    missing_field = "{}"

    record, provider = _node_result(
        experiment, [("stop", "this is not json"), ("stop", valid)]
    )
    check(record["status"] == "completed", "invalid JSON repaired to completed")
    check(record["format_repairs"] == 1, "one format repair recorded")
    check(len(record["repair_records"]) == 1, "failed attempt retained in repair_records")
    check(len(record["attempts"]) == 2, "both attempts retained for accounting")
    check(len(provider.requests[1].messages) == 3, "repair feedback appended to request")
    check("InvalidJSON" in provider.requests[1].messages[2]["content"], "repair message names the violation")

    record, provider = _node_result(
        experiment, [("length", truncated), ("stop", valid)]
    )
    check(record["status"] == "completed", "length truncation repaired to completed")
    check("TruncatedOutput" in provider.requests[1].messages[2]["content"], "length repair message is specific")

    record, provider = _node_result(
        thinking_experiment,
        [("length", truncated), ("length", truncated), ("stop", valid)],
    )
    check(record["status"] == "completed", "repeated truncation recovered by shrinking budget")
    check(record["format_repairs"] == 2, "two shrinking repairs recorded")
    budget_runner = FixedDebateRunner(dict(thinking_experiment), None)
    model_max = int(thinking_experiment["model"]["max_output_tokens"])
    check(
        [request.max_output_tokens for request in provider.requests]
        == [model_max, model_max, model_max],
        "thinking mode keeps the model cap as the wire budget on every attempt",
    )
    check(
        all(
            request.reasoning_enabled and request.reasoning_effort == "high"
            for request in provider.requests
        ),
        "thinking-mode requests carry reasoning plus the configured high effort",
    )
    check(
        "Return ONLY the final answer" in provider.requests[2].messages[1]["content"],
        "terminal repair asks only for the answer",
    )
    check(
        record["fallback"]["type"] == "answer_only"
        and record["output"]["answer"] == "C",
        "answer-only fallback completes the writer schema deterministically",
    )

    no_thinking_experiment = copy.deepcopy(experiment)
    no_thinking_experiment["model"]["reasoning"] = {"enabled": False}
    record, provider = _node_result(
        no_thinking_experiment,
        [("length", truncated), ("length", truncated), ("stop", valid)],
    )
    check(
        record["status"] == "completed",
        "non-thinking truncation repair still completes",
    )
    no_thinking_runner = FixedDebateRunner(dict(no_thinking_experiment), None)
    check(
        all(
            request.max_output_tokens
            == int(no_thinking_experiment["model"]["max_output_tokens"])
            for request in provider.requests
        ),
        "non-thinking requests use the configured wide model ceiling as the wire cap",
    )
    prompt_budget = no_thinking_runner._schema_token_budget("writer")
    check(
        no_thinking_runner._attempt_budgets("writer")
        == [prompt_budget, max(64, prompt_budget // 2)],
        "non-thinking prompt-level visible budgets still shrink across repairs",
    )
    check(
        no_thinking_runner._answer_only_budget() == 128,
        "answer-only fallback prompt target stays at 128 tokens",
    )
    check(
        all(not request.reasoning_enabled for request in provider.requests)
        and all(request.reasoning_effort is None for request in provider.requests),
        "non-thinking requests carry no reasoning flags",
    )

    record, provider = _node_result(
        experiment,
        [("stop", "not json"), ("stop", "still bad"), ("stop", '"63"')],
    )
    check(record["status"] == "completed", "fallback accepts a bare JSON-string answer")
    check(record["output"]["answer"] == "63", "fallback answer is preserved")
    check(
        "[answer-only fallback" in record["output"]["reasoning_summary"],
        "fallback reasoning summary is an explicit recorded placeholder",
    )
    check(record["fallback"]["answer"] == "63", "fallback provenance records the answer")

    record, provider = _node_result(
        experiment, [("stop", overlong)]
    )
    check(record["status"] == "completed", "advisory max_length overshoot accepted")
    check(record["format_repairs"] == 0, "advisory max_length triggers no repair")
    check(len(provider.requests) == 1, "advisory max_length costs exactly one call")

    record, provider = _node_result(
        experiment, [("stop", missing_field), ("stop", valid)]
    )
    check(record["status"] == "completed", "missing required field repaired to completed")
    check("missing required JSON field" in provider.requests[1].messages[2]["content"], "missing-field repair message is specific")

    record, provider = _node_result(
        experiment, [("stop", "bad"), ("length", truncated), ("stop", "")]
    )
    check(record["status"] == "format_error", "empty fallback still fails honestly")
    check(record["format_repairs"] == 2, "retry budget respected")
    check(len(record["repair_records"]) == 2, "both failed repairs recorded")
    check(len(record["attempts"]) == 3, "all three attempts recorded")
    check(record["error"]["type"] == "EmptyFallback", "empty fallback error is specific")
    check("fallback" not in record, "failed fallback is not marked as completed")

    for role_id in ("planner", "analyst", "critic", "writer"):
        runner = FixedDebateRunner(dict(experiment), None)
        budget = runner._schema_token_budget(role_id)
        sequence = runner._attempt_budgets(role_id)
        model_max = int(experiment["model"]["max_output_tokens"])
        check(
            16 <= budget < model_max,
            f"{role_id} budget {budget} derived from schema and below model max {model_max}",
        )
        check(
            sequence[0] == budget
            and sequence == [budget, max(64, budget // 2)]
            and all(
                later <= earlier for earlier, later in zip(sequence, sequence[1:])
            ),
            f"{role_id} attempt budgets escalate downward: {sequence}",
        )
    runner = FixedDebateRunner(dict(experiment), None)
    check(runner._answer_only_budget() == 128, "answer-only fallback cap is 128 tokens")
    analyst_fallback = runner._fallback_output("analyst", "63")
    check(
        analyst_fallback["candidate_answer"] == "63"
        and "[answer-only fallback" in analyst_fallback["analysis"],
        "non-writer fallback keeps the answer and marks omitted fields",
    )

    full_by_node: dict[str, tuple[str | None, str]] = {}
    for node_id in ("planner_stage_1", "analyst_stage_1", "critic_stage_1", "planner_stage_2", "analyst_stage_2", "critic_stage_2", "writer"):
        role = node_id.split("_")[0]
        full_by_node[node_id] = (
            "stop",
            json.dumps(ROLE_OUTPUTS[role], separators=(",", ":")),
        )
    provider = FakeProvider([], by_node=full_by_node)
    runner = FixedDebateRunner(dict(thinking_experiment), provider)
    round_record = runner.run_round(
        task=_task(),
        round_index=1,
        previous_checkpoint=None,
    )
    check(round_record["status"] == "completed", "full seven-node round completes")
    checkpoint = round_record["checkpoint"]
    check(
        checkpoint["answer"] == "C"
        and "reasoning_summary" in checkpoint
        and checkpoint["writer_output"]["answer"] == "C"
        and checkpoint["writer_output"]["reasoning_summary"],
        "writer checkpoint stores answer separately from reasoning_summary",
    )
    model_max = int(thinking_experiment["model"]["max_output_tokens"])
    check(
        all(request.max_output_tokens == model_max for request in provider.requests),
        "thinking-mode node requests reserve the model cap for hidden reasoning",
    )

    # Collect a complete five-round trajectory with the same deterministic
    # fake provider and verify the cross-round checkpoint contract end to end.
    trajectory_provider = FakeProvider([], by_node=full_by_node)
    trajectory_runner = FixedDebateRunner(dict(experiment), trajectory_provider)
    trajectory = trajectory_runner.run_trajectory(
        task=_task(),
        run_id="dev-selfcheck-five-rounds",
        max_rounds=5,
    )
    check(trajectory["status"] == "complete", "five-round trajectory completes")
    check(len(trajectory["checkpoints"]) == 5, "five Writer checkpoints collected")
    check(
        [item["round_index"] for item in trajectory["checkpoints"]] == [1, 2, 3, 4, 5],
        "checkpoint rounds are exactly 1 through 5",
    )
    check(
        len({item["checkpoint_hash"] for item in trajectory["checkpoints"]}) == 5,
        "every round produces a distinct checkpoint hash",
    )
    check(
        trajectory["checkpoints"][-1]["cumulative"]["logical_calls"] == 35,
        "five rounds account for 35 logical model calls",
    )

    # Persist and reload the real trajectory document so the JSON boundary is
    # exercised, not only the in-memory dictionary path.
    with tempfile.TemporaryDirectory() as temporary:
        trajectory_path = Path(temporary) / "trajectory.json"
        trajectory_path.write_text(
            json.dumps(trajectory, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        reloaded = json.loads(trajectory_path.read_text(encoding="utf-8"))
    check(
        reloaded["status"] == "complete"
        and len(reloaded["checkpoints"]) == 5
        and reloaded["checkpoints"][1]["answer"] == "C"
        and reloaded["checkpoints"][1]["reasoning_summary"],
        "serialized five-round trajectory round-trips answer and reasoning_summary",
    )
    replay_record: dict[str, Any] = {
        "schema_version": "1.0",
        "task": _task(),
        "split": "test",
        "trajectory": reloaded,
    }
    replay_scores = score_trajectory(replay_record)
    check(
        [score["quality"] for score in replay_scores] == [1.0] * 5,
        "offline scoring reads only the Writer answer across all five rounds",
    )
    replay_record["scores"] = replay_scores
    replay_record["labels"] = build_labels(
        replay_record, lambda_cost=0.0, mu_latency=0.0
    )
    replay = replay_policies([replay_record], None)
    check(
        replay.get("max_rounds") == 5
        and set(replay["policy_metrics"])
        == {
            "fixed_1",
            "fixed_2",
            "fixed_3",
            "fixed_4",
            "fixed_5",
            "task_only",
            "roundvalue",
            "oracle",
        },
        "five-round serialized trajectory still replays Fixed-1..5 and learned policies",
    )

    inputs_by_round: dict[tuple[int, str], dict[str, Any]] = {}
    for request in trajectory_provider.requests:
        round_index = int(request.metadata.get("round_index"))
        node_id = request.metadata.get("node_id")
        inputs_by_round[(round_index, node_id)] = json.loads(
            request.messages[1]["content"]
        )

    first_input = inputs_by_round[(1, "planner_stage_1")]
    check(
        first_input.get("previous_writer_checkpoint") is None,
        "round 1 receives no previous checkpoint",
    )
    check(
        "previous_writer_checkpoint" not in first_input,
        "round 1 stage-1 input omits the checkpoint field entirely",
    )
    for round_index in range(2, 6):
        previous = trajectory["checkpoints"][round_index - 2]
        for stage_1_node in (
            "planner_stage_1",
            "analyst_stage_1",
            "critic_stage_1",
        ):
            node_input = inputs_by_round[(round_index, stage_1_node)]
            serialized = json.dumps(node_input, ensure_ascii=False, sort_keys=True)
            check(
                "previous_writer_checkpoint" not in node_input,
                f"round {round_index} {stage_1_node} is blind to the checkpoint field",
            )
            check(
                previous["reasoning_summary"] not in serialized
                and previous["checkpoint_hash"] not in serialized,
                f"round {round_index} {stage_1_node} does not leak the previous checkpoint",
            )
            public = node_input.get("task", {})
            check(
                "reference_answer" not in public
                and "answer_index" not in public
                and "options" not in public,
                f"round {round_index} public task hides gold answer fields",
            )
            check(
                node_input.get("visible_messages") == [],
                f"round {round_index} stage-1 node receives no transcript history",
            )
        for stage_2_node in (
            "planner_stage_2",
            "analyst_stage_2",
            "critic_stage_2",
        ):
            node_input = inputs_by_round[(round_index, stage_2_node)]
            visible = node_input.get("previous_writer_checkpoint")
            check(
                isinstance(visible, dict)
                and visible.get("round_index") == round_index - 1
                and visible.get("answer") == previous["answer"]
                and visible.get("reasoning_summary") == previous["reasoning_summary"],
                f"round {round_index} {stage_2_node} sees the round {round_index - 1} checkpoint",
            )
            packets = node_input.get("visible_messages", [])
            check(
                len(packets) == 1
                and packets[0].get("packet_id") == "stage_1_packet"
                and len(packets[0].get("messages", [])) == 3,
                f"round {round_index} {stage_2_node} sees the full stage-1 packet",
            )
        writer_input = inputs_by_round[(round_index, "writer")]
        writer_previous = writer_input.get("previous_writer_checkpoint")
        check(
            isinstance(writer_previous, dict)
            and writer_previous.get("answer") == previous["answer"]
            and writer_previous.get("reasoning_summary") == previous["reasoning_summary"],
            f"round {round_index} Writer still sees the previous checkpoint",
        )

    print("PASS all self-checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
