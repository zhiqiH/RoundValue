"""Independent Single Agent baseline for RoundValue experiments.

The Single Agent is a real one-call baseline: one model request per task
producing ``final_answer`` directly.  It is deliberately not disguised as
round zero of the Debate DAG, never sees other role outputs, and records its
own ``protocol: single_agent`` trajectory so downstream analysis cannot merge
it with debate checkpoints by accident.
"""

from __future__ import annotations

import json
import traceback
from typing import Any

from accounting import node_cumulative
from benchmark_io import public_task
from contracts import (
    ConfigurationError,
    ModelRequest,
    ProviderError,
    json_hash,
    parse_json_object,
    utc_now,
    validate_output_contract,
)
from provider import ProviderAdapter


class SingleAgentRunner:
    """Run one strict JSON model call per task for the Single Agent baseline."""

    def __init__(self, experiment: dict[str, Any], provider: ProviderAdapter):
        self.experiment = experiment
        self.provider = provider
        self.model = experiment["model"]
        self.provider_name = experiment["provider_name"]
        self.role = experiment.get("single_agent")
        if not isinstance(self.role, dict) or not self.role.get("system_prompt"):
            raise ConfigurationError(
                "agents.json must define a validated single_agent baseline block"
            )

    def _node_input(self, task: dict[str, Any]) -> dict[str, Any]:
        return {
            "protocol": "RoundValue single-agent baseline",
            "node_id": "single_agent",
            "role": "single_agent",
            "task": public_task(task),
            "required_output_schema": self.role["output_schema"],
            "instruction": (
                "Return exactly one JSON object. Do not use Markdown fences or "
                "add text outside the object."
            ),
        }

    def _request_for_node(self, node_input: dict[str, Any]) -> ModelRequest:
        return ModelRequest(
            messages=[
                {"role": "system", "content": self.role["system_prompt"]},
                {"role": "user", "content": json.dumps(node_input, ensure_ascii=False, sort_keys=True)},
            ],
            model=self.model["model_name"],
            temperature=float(self.model["temperature"]),
            max_output_tokens=int(self.model["max_output_tokens"]),
            reasoning_enabled=bool(self.model["reasoning"]["enabled"]),
            metadata={"node_id": "single_agent", "role": "single_agent"},
        )

    def _run_node(self, task: dict[str, Any]) -> dict[str, Any]:
        node_input = self._node_input(task)
        request = self._request_for_node(node_input)
        record: dict[str, Any] = {
            "node_id": "single_agent",
            "role": "single_agent",
            "round_index": 1,
            "started_at": utc_now(),
            "input": node_input,
            "request": request.log_view(),
            "status": "started",
            "attempts": [],
        }
        try:
            response, attempts = self.provider.generate(request)
        except ProviderError as error:
            record.update(
                {
                    "ended_at": utc_now(),
                    "status": "provider_error",
                    "attempts": error.attempts,
                    "error": {"type": type(error).__name__, "message": str(error)},
                }
            )
            return record
        except Exception as error:  # Keep unexpected local defects visible in a partial trajectory.
            record.update(
                {
                    "ended_at": utc_now(),
                    "status": "internal_error",
                    "error": {
                        "type": type(error).__name__,
                        "message": str(error),
                        "traceback": traceback.format_exc(limit=5),
                    },
                }
            )
            return record
        parsed = parse_json_object(response.text)
        record.update(
            {
                "ended_at": utc_now(),
                "attempts": attempts,
                "response": response.log_view(),
                "raw_text": response.text,
            }
        )
        if parsed is None:
            record.update(
                {
                    "status": "format_error",
                    "error": {
                        "type": "InvalidJSON",
                        "message": "model output was not a strict JSON object",
                    },
                }
            )
            return record
        output_error = validate_output_contract(
            parsed, self.role["output_schema"], "single_agent"
        )
        if output_error:
            record.update(
                {
                    "status": "format_error",
                    "output": parsed,
                    "error": {"type": "OutputSchemaError", "message": output_error},
                }
            )
            return record
        record.update({"status": "completed", "output": parsed})
        return record

    def run_trajectory(self, *, task: dict[str, Any], run_id: str) -> dict[str, Any]:
        """Collect one real single-call trajectory with a single checkpoint."""

        trajectory_id = json_hash({"run_id": run_id, "task_id": task["task_id"]})[:24]
        trajectory: dict[str, Any] = {
            "schema_version": "1.0",
            "trajectory_id": trajectory_id,
            "task_id": task["task_id"],
            "domain": task["domain"],
            "protocol": "single_agent",
            "status": "running",
            "started_at": utc_now(),
            "max_rounds": 1,
            "configured_model": {
                "model_id": self.experiment["model_id"],
                "provider": self.provider_name,
                "requested_model": self.model["model_name"],
                "temperature": self.model["temperature"],
                "reasoning_enabled": self.model["reasoning"]["enabled"],
            },
            "rounds": [],
            "checkpoints": [],
        }
        node = self._run_node(task)
        round_record: dict[str, Any] = {
            "round_index": 1,
            "started_at": utc_now(),
            "status": "started",
            "nodes": [node],
        }
        if node["status"] != "completed":
            round_record.update(
                {
                    "ended_at": utc_now(),
                    "status": "failed",
                    "cumulative": node_cumulative([node], self.model),
                }
            )
            trajectory["rounds"].append(round_record)
            trajectory.update(
                {
                    "status": "failed",
                    "ended_at": utc_now(),
                    "failure_round": 1,
                    "failure_reason": (
                        "the single-agent model call failed or violated the JSON output contract"
                    ),
                }
            )
            return trajectory
        cumulative = node_cumulative([node], self.model)
        checkpoint = {
            "round_index": 1,
            "final_answer": node["output"]["final_answer"],
            "answer_node_id": "single_agent",
            "single_agent_output": node["output"],
            "nodes": [node],
            "cumulative": cumulative,
        }
        checkpoint["checkpoint_hash"] = json_hash(
            {
                "task_id": task["task_id"],
                "round_index": 1,
                "final_answer": checkpoint["final_answer"],
                "single_agent_output": checkpoint["single_agent_output"],
            }
        )
        round_record.update(
            {
                "ended_at": utc_now(),
                "status": "completed",
                "cumulative": cumulative,
                "checkpoint": checkpoint,
            }
        )
        trajectory["rounds"].append(round_record)
        trajectory["checkpoints"].append(checkpoint)
        trajectory.update({"status": "complete", "ended_at": utc_now()})
        return trajectory
