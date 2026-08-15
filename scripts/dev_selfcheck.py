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

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from config_loader import load_experiment_config  # noqa: E402
from contracts import ModelRequest, ModelResponse  # noqa: E402
from debate_runner import FixedDebateRunner  # noqa: E402


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
    "writer": {"final_answer": "5"},
}


class FakeProvider:
    """Return scripted chat completions and retain every request for checks."""

    def __init__(self, script: list[tuple[str | None, str]]) -> None:
        self.script = list(script)
        self.requests: list[ModelRequest] = []
        self.closed = False

    def generate(self, request: ModelRequest) -> tuple[ModelResponse, list[dict[str, Any]]]:
        self.requests.append(request)
        if not self.script:
            raise AssertionError("fake provider ran out of scripted responses")
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
        "domain": "math",
        "prompt": "Solve for x: 3x + 5 = 20.",
        "reference_answer": "5",
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

    valid = json.dumps(ROLE_OUTPUTS["writer"], separators=(",", ":"))
    truncated = '{"final_answer": "5"'
    overlong = json.dumps({"final_answer": "x" * 600}, separators=(",", ":"))
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
        experiment, [("length", truncated), ("length", truncated), ("stop", valid)]
    )
    check(record["status"] == "completed", "repeated truncation recovered by shrinking budget")
    check(record["format_repairs"] == 2, "two shrinking repairs recorded")
    check(
        [request.max_output_tokens for request in provider.requests] == [348, 174, 128],
        "full-schema budgets shrink, then the answer-only fallback takes over",
    )
    check(
        "Return ONLY the final answer" in provider.requests[2].messages[1]["content"],
        "terminal repair asks only for the answer",
    )
    check(
        record["fallback"]["type"] == "answer_only"
        and record["output"]["final_answer"] == "5",
        "answer-only fallback completes the writer schema deterministically",
    )

    record, provider = _node_result(
        experiment,
        [("stop", "not json"), ("stop", "still bad"), ("stop", '"63"')],
    )
    check(record["status"] == "completed", "fallback accepts a bare JSON-string answer")
    check(record["output"]["final_answer"] == "63", "fallback answer is preserved")
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

    full_script: list[tuple[str | None, str]] = []
    for node_id in ("planner_stage_1", "analyst_stage_1", "critic_stage_1", "planner_stage_2", "analyst_stage_2", "critic_stage_2", "writer"):
        role = node_id.split("_")[0]
        full_script.append(("stop", json.dumps(ROLE_OUTPUTS[role], separators=(",", ":"))))
    provider = FakeProvider(full_script)
    runner = FixedDebateRunner(dict(experiment), provider)
    round_record = runner.run_round(
        task=_task(),
        round_index=1,
        previous_checkpoint=None,
    )
    check(round_record["status"] == "completed", "full seven-node round completes")
    check(round_record["checkpoint"]["final_answer"] == "5", "writer checkpoint preserved")
    check(all(request.max_output_tokens < 4096 for request in provider.requests), "every node request uses a schema-derived budget")

    print("PASS all self-checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
