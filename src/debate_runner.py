"""Execute the immutable two-stage P/A/C-to-Writer Debate graph.

This module never sees benchmark labels.  It receives only a public task view,
which is the boundary preventing gold answers and hidden code tests from leaking
into online stopping features or Agent prompts.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import traceback
from typing import Any

from benchmark_io import public_task
from contracts import ModelRequest, ProtocolError, ProviderError, json_hash, parse_json_object, utc_now
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
        schema = self.roles[role_id]["output_schema"]
        required = schema["required_fields"]
        missing = [field for field in required if field not in output]
        if missing:
            return f"missing required JSON field(s): {', '.join(missing)}"
        for field in required:
            specification = schema["properties"][field]
            value = output.get(field)
            if specification["type"] == "string":
                if not isinstance(value, str) or len(value.strip()) < specification["min_length"]:
                    return f"{role_id} {field} must be a non-empty string"
            else:
                return f"unsupported configured output type for {role_id}.{field}"
        return None

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

    def _cumulative(self, nodes: list[dict[str, Any]]) -> dict[str, Any]:
        attempts = [attempt for node in nodes for attempt in node.get("attempts", [])]
        successful = [attempt for attempt in attempts if attempt.get("status") == "succeeded"]
        input_values = [attempt.get("input_tokens") for attempt in successful]
        output_values = [attempt.get("output_tokens") for attempt in successful]
        cache_hit_values = [attempt.get("input_cache_hit_tokens") for attempt in successful]
        cache_miss_values = [attempt.get("input_cache_miss_tokens") for attempt in successful]
        latency_values = [attempt.get("latency_ms") for attempt in attempts]
        input_known = bool(successful) and all(isinstance(value, int) for value in input_values)
        output_known = bool(successful) and all(isinstance(value, int) for value in output_values)
        cache_hit_known = bool(successful) and all(isinstance(value, int) for value in cache_hit_values)
        cache_miss_known = bool(successful) and all(isinstance(value, int) for value in cache_miss_values)
        input_tokens = sum(input_values) if input_known else None
        output_tokens = sum(output_values) if output_known else None
        latency_ms = sum(value for value in latency_values if isinstance(value, int))
        pricing = self.model.get("pricing", {})
        input_cache_hit_price = pricing.get("input_cache_hit_per_million") if isinstance(pricing, dict) else None
        input_cache_miss_price = pricing.get("input_cache_miss_per_million") if isinstance(pricing, dict) else None
        output_price = pricing.get("output_per_million") if isinstance(pricing, dict) else None
        cache_hit_tokens = sum(cache_hit_values) if cache_hit_known else None
        cache_miss_tokens = sum(cache_miss_values) if cache_miss_known else None
        cost_usd: float | None = None
        if (
            input_known
            and output_known
            and cache_hit_tokens is not None
            and cache_miss_tokens is not None
            and isinstance(input_cache_hit_price, (int, float))
            and isinstance(input_cache_miss_price, (int, float))
            and isinstance(output_price, (int, float))
        ):
            cost_usd = (
                cache_hit_tokens * float(input_cache_hit_price)
                + cache_miss_tokens * float(input_cache_miss_price)
                + output_tokens * float(output_price)
            ) / 1_000_000
        return {
            "logical_calls": len(nodes),
            "api_attempts": len(attempts),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": latency_ms,
            "cost_usd": cost_usd,
            "input_cache_hit_tokens": cache_hit_tokens,
            "input_cache_miss_tokens": cache_miss_tokens,
            "cost_currency": pricing.get("currency", "USD") if isinstance(pricing, dict) else "USD",
            "usage_complete": input_known and output_known,
            "pricing_complete": cache_hit_known and cache_miss_known,
        }

    def run_round(
        self,
        *,
        task: dict[str, Any],
        round_index: int,
        previous_checkpoint: dict[str, Any] | None,
    ) -> dict[str, Any]:
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
            round_record.update({"ended_at": utc_now(), "status": "failed", "cumulative": self._cumulative(round_record["nodes"])})
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
            round_record.update({"ended_at": utc_now(), "status": "failed", "cumulative": self._cumulative(round_record["nodes"])})
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
        cumulative = self._cumulative(round_record["nodes"])
        if writer["status"] != "completed":
            round_record.update({"ended_at": utc_now(), "status": "failed", "cumulative": cumulative})
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
            {"ended_at": utc_now(), "status": "completed", "cumulative": cumulative, "checkpoint": checkpoint}
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
                        "failure_round": round_index,
                        "failure_reason": "one or more fixed-DAG node calls failed or violated JSON output contract",
                    }
                )
                return trajectory
            checkpoint = round_record["checkpoint"]
            checkpoint["round_cost"] = dict(checkpoint["cumulative"])
            all_nodes = [node for prior_round in trajectory["rounds"] for node in prior_round["nodes"]]
            checkpoint["cumulative"] = self._cumulative(all_nodes)
            trajectory["checkpoints"].append(checkpoint)
            previous_checkpoint = {
                "round_index": checkpoint["round_index"],
                "final_answer": checkpoint["final_answer"],
                "checkpoint_hash": checkpoint["checkpoint_hash"],
            }
        trajectory.update({"status": "complete", "ended_at": utc_now()})
        return trajectory
