"""Execute the immutable two-stage P/A/C-to-Writer Debate graph.

This module never sees benchmark labels.  It receives only a public task view,
which is the boundary preventing gold answers and hidden code tests from leaking
into online stopping features or Agent prompts.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import math
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
        if (
            isinstance(self.format_retries, bool)
            or not isinstance(self.format_retries, int)
            or self.format_retries < 0
        ):
            raise ProtocolError("format_retries must be a non-negative integer")
        try:
            self.format_budget_margin = float(
                experiment["agents"].get("format_budget_margin", 2.0)
            )
        except (TypeError, ValueError) as error:
            raise ProtocolError(
                "format_budget_margin must be a finite number"
            ) from error
        if (
            not math.isfinite(self.format_budget_margin)
            or self.format_budget_margin < 1.0
        ):
            raise ProtocolError("format_budget_margin must be a finite number >= 1.0")

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

    def _request_for_node(
        self,
        node_input: dict[str, Any],
        role_id: str,
        round_index: int,
        node_id: str,
        *,
        token_budget: int | None,
    ) -> ModelRequest:
        role = self.roles[role_id]
        model_max = int(self.model["max_output_tokens"])
        if token_budget is None:
            max_output_tokens = model_max
        else:
            max_output_tokens = min(model_max, max(16, int(token_budget)))
        return ModelRequest(
            messages=[
                {"role": "system", "content": role["system_prompt"]},
                {"role": "user", "content": json.dumps(node_input, ensure_ascii=False, sort_keys=True)},
            ],
            model=self.model["model_name"],
            temperature=float(self.model["temperature"]),
            max_output_tokens=max_output_tokens,
            reasoning_enabled=bool(self.model["reasoning"]["enabled"]),
            metadata={"round_index": str(round_index), "node_id": node_id, "role": role_id},
        )

    def _schema_token_budget(self, role_id: str) -> int:
        """Derive the node output budget from its declared output schema.

        The accepted output is already bounded by the per-field ``max_length``
        values, so a global 4096-token allowance merely invites unbounded
        rambling and the truncation failures that follow.  Convert the schema's
        character budget plus JSON key overhead into tokens conservatively
        (three characters per token covers dense LaTeX) and keep the configured
        margin so a conforming reply is never cut off.
        """

        schema = self.roles[role_id]["output_schema"]
        characters = 2  # opening/closing braces plus padding
        for name in schema["required_fields"]:
            specification = schema["properties"].get(name) or {}
            max_length = specification.get("max_length")
            if (
                isinstance(max_length, int)
                and not isinstance(max_length, bool)
                and max_length > 0
            ):
                characters += max_length
            characters += len(str(name)) + 6  # `"name": ` plus a comma
        token_estimate = max(1, math.ceil(max(1, characters) / 3.0))
        return max(16, math.ceil(token_estimate * self.format_budget_margin))

    def _answer_field(self, role_id: str) -> str:
        """Return the one field whose content is the node's candidate answer."""

        required = self.roles[role_id]["output_schema"]["required_fields"]
        for field in ("answer", "final_answer", "candidate_answer"):
            if field in required:
                return field
        return "candidate_answer"

    def _attempt_budgets(self, role_id: str) -> list[int]:
        """Escalate the full-schema attempts downward before the fallback.

        Repairing at the same budget reproduces the same overlong output at
        temperature zero, as observed in the field.  Non-final repair attempts
        halve the cap; the final repair is a separate answer-only fallback and
        therefore does not appear in this list.
        """

        budgets = [self._schema_token_budget(role_id)]
        for _ in range(max(0, self.format_retries - 1)):
            budgets.append(max(64, budgets[-1] // 2))
        return budgets

    def _answer_only_budget(self) -> int:
        """Cap for the answer-only fallback; roomy for a LaTeX or option answer."""

        return min(int(self.model["max_output_tokens"]), 128)

    def _answer_only_request(
        self,
        base_request: ModelRequest,
        task: dict[str, Any],
        error: dict[str, Any],
    ) -> ModelRequest:
        """Ask for nothing but the final answer on the terminal repair."""

        instruction = {
            "instruction": (
                "Your earlier replies were rejected because they did not form "
                "a complete JSON object. Return ONLY the final answer to the "
                'task as one short JSON string, for example "B" or "42". Do not '
                "derive, restate, explain, or add anything else."
            ),
            "task": public_task(task),
            "rejection_reason": error,
        }
        return ModelRequest(
            messages=[
                base_request.messages[0],
                {
                    "role": "user",
                    "content": json.dumps(instruction, ensure_ascii=False, sort_keys=True),
                },
            ],
            model=base_request.model,
            temperature=base_request.temperature,
            max_output_tokens=self._answer_only_budget(),
            reasoning_enabled=base_request.reasoning_enabled,
            metadata=base_request.metadata,
        )

    @staticmethod
    def _extract_answer(text: str) -> str | None:
        """Read a short answer from a JSON string, object, or bare value."""

        value = parse_json_object(text)
        if value is not None:
            for key in ("final_answer", "candidate_answer", "answer"):
                candidate = value.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    return candidate.strip()
            for candidate in value.values():
                if isinstance(candidate, str) and candidate.strip():
                    return candidate.strip()
            return None
        try:
            decoded = json.loads(text)
            if isinstance(decoded, str) and decoded.strip():
                return decoded.strip()
        except json.JSONDecodeError:
            pass
        stripped = text.strip().strip('"').strip()
        return stripped or None

    def _fallback_output(self, role_id: str, answer: str) -> dict[str, Any]:
        """Deterministically complete the schema around a model-supplied answer.

        This is a recorded degradation, never a silent repair: the trajectory
        stores a ``fallback`` marker, the answer field keeps the model's real
        answer, and every auxiliary field is a self-describing placeholder.
        """

        schema = self.roles[role_id]["output_schema"]
        answer_field = self._answer_field(role_id)
        output: dict[str, Any] = {}
        for name in schema["required_fields"]:
            if name == answer_field:
                output[name] = answer
            else:
                output[name] = "[answer-only fallback: field omitted]"
        return output

    def _repair_message(
        self,
        role_id: str,
        error_type: str,
        message: str,
        token_budget: int,
        offending_text: str | None = None,
    ) -> str:
        """Give the model the concrete contract violation instead of a bare retry."""

        schema = self.roles[role_id]["output_schema"]
        if error_type == "TruncatedOutput":
            lines = [
                "Your previous response was cut off by the token limit before it formed a complete JSON object.",
                "Do not derive, restate, or explain anything. Emit only the required JSON object with one short phrase per field, and put the final answer in its answer field.",
            ]
        elif error_type == "InvalidJSON":
            lines = [
                "Your previous response was not a valid JSON object.",
                "Do not write prose, Markdown, or commentary. Emit only the required JSON object with one short phrase per field.",
            ]
        else:
            lines = [
                "Your previous response was rejected by the protocol.",
                "Fix the reported schema problem and emit only the required JSON object with one short phrase per field.",
            ]
        lines.append(f"Reason: {error_type} - {message}")
        if offending_text:
            sample = offending_text.strip()
            if len(sample) > 600:
                sample = sample[:300] + " ... [omitted] ... " + sample[-300:]
            lines.append(
                f"The rejected output, which may be incomplete, was: {sample}"
            )
        lines.extend(
            [
            "Return exactly one JSON object and nothing else: no Markdown fences, no prose, no commentary outside the object, and do not echo the rejected text back.",
            "Keep fields near these target lengths (short is better than exact):",
            ]
        )
        for name in schema["required_fields"]:
            specification = schema["properties"][name]
            lines.append(
                f'- "{name}": a non-empty string, target at most '
                f'{specification["max_length"]} characters'
            )
        lines.append(
            f"Keep the entire JSON response within {token_budget} tokens; "
            "do not restate the task and do not repeat earlier text."
        )
        return "\n".join(lines)

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
        attempt_budgets = self._attempt_budgets(role_id)
        base_request = self._request_for_node(
            node_input,
            role_id,
            round_index,
            node_id,
            token_budget=attempt_budgets[0],
        )
        record: dict[str, Any] = {
            "node_id": node_id,
            "role": role_id,
            "round_index": round_index,
            "started_at": utc_now(),
            "input": node_input,
            "request": base_request.log_view(),
            "status": "started",
            "attempts": [],
        }
        repair_count = 0
        repair_records: list[dict[str, Any]] = []
        last_error: dict[str, Any] | None = None
        last_response = None
        parsed: dict[str, Any] | None = None
        fallback: dict[str, Any] | None = None
        while True:
            is_fallback = self.format_retries > 0 and repair_count == self.format_retries
            if repair_count == 0:
                request = base_request
            elif is_fallback:
                request = self._answer_only_request(base_request, task, last_error)
            else:
                budget = attempt_budgets[min(repair_count, len(attempt_budgets) - 1)]
                request = ModelRequest(
                    messages=[
                        *base_request.messages,
                        {
                            "role": "user",
                            "content": self._repair_message(
                                role_id,
                                str(last_error["type"]),
                                str(last_error["message"]),
                                token_budget=budget,
                                offending_text=response.text if repair_count else None,
                            ),
                        },
                    ],
                    model=base_request.model,
                    temperature=base_request.temperature,
                    max_output_tokens=budget,
                    reasoning_enabled=base_request.reasoning_enabled,
                    metadata=base_request.metadata,
                )
            try:
                response, attempts = self.provider.generate(request)
            except ProviderError as error:
                record.update(
                    {
                        "ended_at": utc_now(),
                        "status": "provider_error",
                        "attempts": [*record["attempts"], *error.attempts],
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
            record["attempts"].extend(attempts)
            last_response = response
            if is_fallback:
                answer = self._extract_answer(response.text)
                if answer is None:
                    last_error = {
                        "type": "EmptyFallback",
                        "message": "answer-only fallback returned no usable answer",
                    }
                    parsed = None
                else:
                    parsed = self._fallback_output(role_id, answer)
                    fallback = {
                        "type": "answer_only",
                        "reason": last_error,
                        "answer": answer,
                    }
                    last_error = None
            else:
                parsed = parse_json_object(response.text)
                if response.finish_reason == "length":
                    # A provider-side token cutoff can still yield a parseable
                    # but truncated object; it must be repaired instead of scored.
                    parsed = None
                    last_error = {
                        "type": "TruncatedOutput",
                        "message": (
                            "provider stopped with finish_reason=length before "
                            "completing the JSON object"
                        ),
                    }
                elif parsed is None:
                    last_error = {
                        "type": "InvalidJSON",
                        "message": "model output was not a strict JSON object",
                    }
                else:
                    output_error = self._validate_output(parsed, role_id)
                    if output_error:
                        parsed = None
                        last_error = {
                            "type": "OutputSchemaError",
                            "message": output_error,
                        }
                    else:
                        last_error = None
            if last_error is None or repair_count >= self.format_retries:
                break
            repair_records.append(
                {
                    "repair_index": repair_count,
                    "error": last_error,
                    "max_output_tokens": request.max_output_tokens,
                    "response": response.log_view(),
                    "raw_text": response.text,
                }
            )
            repair_count += 1
        record.update(
            {
                "ended_at": utc_now(),
                "attempts": record["attempts"],
                "format_repairs": repair_count,
                "response": last_response.log_view() if last_response is not None else None,
                "raw_text": last_response.text if last_response is not None else None,
            }
        )
        if repair_records:
            record["repair_records"] = repair_records
        if fallback is not None:
            record["fallback"] = fallback
        if last_error is not None:
            record.update({"status": "format_error", "error": last_error})
            if parsed is not None:
                record["output"] = parsed
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
            "answer": writer["output"]["answer"],
            "reasoning_summary": writer["output"]["reasoning_summary"],
            "writer_node_id": "writer",
            "writer_output": writer["output"],
            "nodes": round_record["nodes"],
            "cumulative": cumulative,
        }
        checkpoint["checkpoint_hash"] = json_hash(
            {
                "task_id": task["task_id"],
                "round_index": round_index,
                "answer": checkpoint["answer"],
                "reasoning_summary": checkpoint["reasoning_summary"],
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
                "answer": checkpoint["answer"],
                "reasoning_summary": checkpoint["reasoning_summary"],
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
