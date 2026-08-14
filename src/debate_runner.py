"""Execute the immutable two-stage P/A/C-to-Writer Debate graph.

This module never sees benchmark labels.  It receives only a public task view,
which is the boundary preventing gold answers and hidden code tests from leaking
into online stopping features or Agent prompts.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import time
import traceback
from typing import Any

from benchmark_io import public_task
from accounting import node_cumulative
from contracts import (
    ModelRequest,
    ProtocolError,
    ProviderError,
    json_hash,
    parse_json_object,
    utc_now,
    validate_output_contract,
)
from provider import ProviderAdapter


class FixedDebateRunner:
    """Run a fixed 7-node DAG once per communication round."""

    def __init__(self, experiment: dict[str, Any], provider: ProviderAdapter):
        self.experiment = experiment
        self.provider = provider
        self.topology = experiment["topology"]
        self.model = experiment["model"]
        self.provider_name = experiment["provider_name"]
        self.roles = {role["id"]: role for role in experiment["agents"]["roles"]}
        self.packet_nodes = {packet["id"]: packet for packet in self.topology["packets"]}
        self.format_retries = experiment["agents"].get("format_retries", 0)
        if self.format_retries != 0:
            raise ProtocolError(
                "format_retries must be 0 in the fixed seven-call topology; change the topology before enabling it"
            )

    def _node_input(
        self,
        *,
        task: dict[str, Any],
        round_index: int,
        node_id: str,
        role_id: str,
        previous_checkpoint: dict[str, Any] | None,
        visible_messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        role = self.roles[role_id]
        return {
            "protocol": "RoundValue fixed two-stage P/A/C-to-Writer debate",
            "round_index": round_index,
            "node_id": node_id,
            "role": role_id,
            "task": public_task(task),
            "previous_writer_checkpoint": previous_checkpoint,
            "visible_messages": visible_messages,
            "required_output_schema": role["output_schema"],
            "instruction": "Return exactly one JSON object. Do not use Markdown fences or add text outside the object.",
        }

    def _request_for_node(self, node_input: dict[str, Any], role_id: str, round_index: int, node_id: str) -> ModelRequest:
        role = self.roles[role_id]
        return ModelRequest(
            messages=[
                {"role": "system", "content": role["system_prompt"]},
                {"role": "user", "content": json.dumps(node_input, ensure_ascii=False, sort_keys=True)},
            ],
            model=self.model["model_name"],
            temperature=float(self.model["temperature"]),
            max_output_tokens=int(self.model["max_output_tokens"]),
            reasoning_enabled=bool(self.model["reasoning"]["enabled"]),
            metadata={"round_index": str(round_index), "node_id": node_id, "role": role_id},
        )

    def _validate_output(self, output: dict[str, Any], role_id: str) -> str | None:
        return validate_output_contract(output, self.roles[role_id]["output_schema"], role_id)

    def _run_node(
        self,
        *,
        task: dict[str, Any],
        round_index: int,
        node_id: str,
        role_id: str,
        previous_checkpoint: dict[str, Any] | None,
        visible_messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        node_input = self._node_input(
            task=task,
            round_index=round_index,
            node_id=node_id,
            role_id=role_id,
            previous_checkpoint=previous_checkpoint,
            visible_messages=visible_messages,
        )
        request = self._request_for_node(node_input, role_id, round_index, node_id)
        record: dict[str, Any] = {
            "node_id": node_id,
            "role": role_id,
            "round_index": round_index,
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
                    "error": {"type": "InvalidJSON", "message": "model output was not a strict JSON object"},
                }
            )
            return record
        output_error = self._validate_output(parsed, role_id)
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

    def _run_parallel(
        self,
        node_specs: list[dict[str, Any]],
        *,
        task: dict[str, Any],
        round_index: int,
        previous_checkpoint: dict[str, Any] | None,
        visible_messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Parallelize only independent stage peers; store results in topology order."""

        def invoke(spec: dict[str, Any]) -> dict[str, Any]:
            return self._run_node(
                task=task,
                round_index=round_index,
                node_id=spec["id"],
                role_id=spec["role"],
                previous_checkpoint=previous_checkpoint,
                visible_messages=visible_messages,
            )

        with ThreadPoolExecutor(max_workers=len(node_specs), thread_name_prefix="roundvalue") as executor:
            futures = [executor.submit(invoke, spec) for spec in node_specs]
            return [future.result() for future in futures]

    @staticmethod
    def _messages_from_nodes(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "node_id": node["node_id"],
                "role": node["role"],
                "round_index": node["round_index"],
                "output": node.get("output"),
            }
            for node in nodes
        ]

    def _packet(
        self, packet_id: str, completed_nodes: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        """Construct a deterministic JSON packet, never a hidden model invocation."""

        specification = self.packet_nodes[packet_id]
        messages: list[dict[str, Any]] = []
        for source_id in specification["sources"]:
            if source_id in self.packet_nodes:
                source = self._packet(source_id, completed_nodes)
                messages.extend(source["messages"])
                continue
            node = completed_nodes.get(source_id)
            if node is None or node.get("status") != "completed":
                raise ProtocolError(f"packet {packet_id} source {source_id} is unavailable")
            messages.append(
                {
                    "node_id": source_id,
                    "role": node["role"],
                    "round_index": node["round_index"],
                    "output": node["output"],
                }
            )
        return {
            "packet_id": packet_id,
            "kind": "deterministic_json_packet",
            "sources": specification["sources"],
            "messages": messages,
        }

    def _cumulative(
        self, nodes: list[dict[str, Any]], wall_clock_ms: int | float | None = None
    ) -> dict[str, Any]:
        return node_cumulative(nodes, self.model, wall_clock_ms=wall_clock_ms)

    def run_round(
        self,
        *,
        task: dict[str, Any],
        round_index: int,
        previous_checkpoint: dict[str, Any] | None,
    ) -> dict[str, Any]:
        round_started_monotonic = time.monotonic()
        nodes = self.topology["nodes"]
        stage_1_specs = [node for node in nodes if node["stage"] == 1]
        stage_2_specs = [node for node in nodes if node["stage"] == 2]
        writer_spec = next(node for node in nodes if node["id"] == "writer")
        round_record: dict[str, Any] = {
            "round_index": round_index,
            "started_at": utc_now(),
            "status": "started",
            "nodes": [],
        }
        stage_1 = self._run_parallel(
            stage_1_specs,
            task=task,
            round_index=round_index,
            previous_checkpoint=previous_checkpoint,
            visible_messages=[],
        )
        round_record["nodes"].extend(stage_1)
        if any(node["status"] != "completed" for node in stage_1):
            wall_clock_ms = max(0, round((time.monotonic() - round_started_monotonic) * 1000))
            round_record.update(
                {
                    "ended_at": utc_now(),
                    "status": "failed",
                    "wall_clock_ms": wall_clock_ms,
                    "cumulative": self._cumulative(
                        round_record["nodes"], wall_clock_ms=wall_clock_ms
                    ),
                }
            )
            return round_record
        completed_by_id = {node["node_id"]: node for node in stage_1}
        stage_1_packet = self._packet("stage_1_packet", completed_by_id)

        def run_stage_2(spec: dict[str, Any]) -> dict[str, Any]:
            return self._run_node(
                task=task,
                round_index=round_index,
                node_id=spec["id"],
                role_id=spec["role"],
                previous_checkpoint=previous_checkpoint,
                visible_messages=[stage_1_packet],
            )

        with ThreadPoolExecutor(max_workers=len(stage_2_specs), thread_name_prefix="roundvalue") as executor:
            futures = [executor.submit(run_stage_2, spec) for spec in stage_2_specs]
            stage_2 = [future.result() for future in futures]
        round_record["nodes"].extend(stage_2)
        if any(node["status"] != "completed" for node in stage_2):
            wall_clock_ms = max(0, round((time.monotonic() - round_started_monotonic) * 1000))
            round_record.update(
                {
                    "ended_at": utc_now(),
                    "status": "failed",
                    "wall_clock_ms": wall_clock_ms,
                    "cumulative": self._cumulative(
                        round_record["nodes"], wall_clock_ms=wall_clock_ms
                    ),
                }
            )
            return round_record
        node_by_id = {node["node_id"]: node for node in [*stage_1, *stage_2]}
        writer_packet = self._packet("writer_packet", node_by_id)
        writer = self._run_node(
            task=task,
            round_index=round_index,
            node_id=writer_spec["id"],
            role_id=writer_spec["role"],
            previous_checkpoint=previous_checkpoint,
            visible_messages=[writer_packet],
        )
        round_record["nodes"].append(writer)
        wall_clock_ms = max(0, round((time.monotonic() - round_started_monotonic) * 1000))
        cumulative = self._cumulative(round_record["nodes"], wall_clock_ms=wall_clock_ms)
        if writer["status"] != "completed":
            round_record.update(
                {
                    "ended_at": utc_now(),
                    "status": "failed",
                    "wall_clock_ms": wall_clock_ms,
                    "cumulative": cumulative,
                }
            )
            return round_record
        checkpoint = {
            "round_index": round_index,
            "final_answer": writer["output"]["final_answer"],
            "writer_node_id": "writer",
            "writer_output": writer["output"],
            "nodes": round_record["nodes"],
            "cumulative": cumulative,
        }
        checkpoint["checkpoint_hash"] = json_hash(
            {
                "task_id": task["task_id"],
                "round_index": round_index,
                "final_answer": checkpoint["final_answer"],
                "writer_output": checkpoint["writer_output"],
            }
        )
        round_record.update(
            {
                "ended_at": utc_now(),
                "status": "completed",
                "wall_clock_ms": wall_clock_ms,
                "cumulative": cumulative,
                "checkpoint": checkpoint,
            }
        )
        return round_record

    def run_trajectory(
        self,
        *,
        task: dict[str, Any],
        run_id: str,
        max_rounds: int | None = None,
    ) -> dict[str, Any]:
        """Collect a real full trajectory; stopping policies are never used during collection."""

        configured_max = self.topology["max_rounds"]
        rounds_to_run = configured_max if max_rounds is None else max_rounds
        if not isinstance(rounds_to_run, int) or not 1 <= rounds_to_run <= configured_max:
            raise ProtocolError(f"max_rounds must be an integer from 1 to {configured_max}")
        trajectory_id = json_hash({"run_id": run_id, "task_id": task["task_id"]})[:24]
        trajectory_started_monotonic = time.monotonic()
        trajectory: dict[str, Any] = {
            "schema_version": "1.0",
            "trajectory_id": trajectory_id,
            "task_id": task["task_id"],
            "domain": task["domain"],
            "status": "running",
            "started_at": utc_now(),
            "max_rounds": rounds_to_run,
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
        previous_checkpoint: dict[str, Any] | None = None
        cumulative_wall_clock_ms = 0
        for round_index in range(1, rounds_to_run + 1):
            round_record = self.run_round(
                task=task,
                round_index=round_index,
                previous_checkpoint=previous_checkpoint,
            )
            trajectory["rounds"].append(round_record)
            if round_record["status"] != "completed":
                trajectory.update(
                    {
                        "status": "failed",
                        "ended_at": utc_now(),
                        "wall_clock_ms": max(
                            0, round((time.monotonic() - trajectory_started_monotonic) * 1000)
                        ),
                        "failure_round": round_index,
                        "failure_reason": "one or more fixed-DAG node calls failed or violated JSON output contract",
                    }
                )
                return trajectory
            round_wall_clock_ms = int(round_record.get("wall_clock_ms", 0) or 0)
            cumulative_wall_clock_ms += max(0, round_wall_clock_ms)
            checkpoint = round_record["checkpoint"]
            checkpoint["round_cost"] = dict(checkpoint["cumulative"])
            all_nodes = [node for prior_round in trajectory["rounds"] for node in prior_round["nodes"]]
            checkpoint["cumulative"] = self._cumulative(
                all_nodes, wall_clock_ms=cumulative_wall_clock_ms
            )
            trajectory["checkpoints"].append(checkpoint)
            previous_checkpoint = {
                "round_index": checkpoint["round_index"],
                "final_answer": checkpoint["final_answer"],
                "checkpoint_hash": checkpoint["checkpoint_hash"],
            }
        trajectory.update(
            {
                "status": "complete",
                "ended_at": utc_now(),
                "wall_clock_ms": max(
                    0, round((time.monotonic() - trajectory_started_monotonic) * 1000)
                ),
            }
        )
        return trajectory
