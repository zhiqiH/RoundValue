"""One-call independent Single-Agent baseline observation.

This module is deliberately not a topology.  It implements one dedicated
``single_solver`` node that makes exactly one intended logical model call per
benchmark task (bounded format-repair/network attempts are recorded
separately), receives only the sanitized public task, and returns ``answer``
+ ``reasoning_summary`` with the same scoring and repair principles as the
Debate Writer checkpoint.

The observation is a sibling of the Debate trajectory inside one task record:
it never sees a previous Writer checkpoint, a Debate packet, role outputs, a
transcript, or any gold/reference information, and the Debate trajectory never
sees its output.  The two conditions share the same selected run-level model
and the same original task but remain causally independent.
"""

from __future__ import annotations

import json
import math
import time
import traceback
from typing import Any

from accounting import node_cumulative
from benchmark_io import public_task
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


SOLVER_ROLE_ID = "single_solver"
SOLVER_NODE_ID = "single_solver"


class SingleAgentRunner:
    """Run one independent solver call per benchmark task as a baseline."""

    def __init__(self, experiment: dict[str, Any], provider: ProviderAdapter):
        self.experiment = experiment
        self.provider = provider
        self.model = experiment["model"]
        self.provider_name = experiment["provider_name"]
        self.reasoning_enabled = bool(self.model["reasoning"]["enabled"])
        self.reasoning_effort = self.model["reasoning"].get("effort")
        roles = {
            role["id"]: role for role in experiment["agents"]["roles"]
        }
        if SOLVER_ROLE_ID not in roles:
            raise ProtocolError("agents.json must define the single_solver role")
        self.role = roles[SOLVER_ROLE_ID]
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
            raise ProtocolError("format_budget_margin must be a finite number") from error
        if (
            not math.isfinite(self.format_budget_margin)
            or self.format_budget_margin < 1.0
        ):
            raise ProtocolError("format_budget_margin must be a finite number >= 1.0")

    def _node_input(self, task: dict[str, Any]) -> dict[str, Any]:
        """Neutral public input: the task and nothing else."""

        return {
            "protocol": "RoundValue single-agent direct solver",
            "node_id": SOLVER_NODE_ID,
            "role": SOLVER_ROLE_ID,
            "task": public_task(task),
            "required_output_schema": self.role["output_schema"],
            "instruction": (
                "Return exactly one JSON object. Do not use Markdown fences "
                "or add text outside the object."
            ),
        }

    def _request_for_node(self, node_input: dict[str, Any]) -> ModelRequest:
        """Use the configured model-wide ceiling as the wire cap, never a narrow
        schema-derived cap.  Concise output is requested by the role prompt and
        the per-field schema, while ``max_output_tokens`` stays a safety ceiling.
        """

        return ModelRequest(
            messages=[
                {"role": "system", "content": self.role["system_prompt"]},
                {
                    "role": "user",
                    "content": json.dumps(node_input, ensure_ascii=False, sort_keys=True),
                },
            ],
            model=self.model["model_name"],
            temperature=float(self.model["temperature"]),
            max_output_tokens=int(self.model["max_output_tokens"]),
            reasoning_enabled=self.reasoning_enabled,
            reasoning_effort=self.reasoning_effort,
            metadata={"node_id": SOLVER_NODE_ID, "role": SOLVER_ROLE_ID},
        )

    def _schema_token_budget(self) -> int:
        """Prompt-level visible output target derived from the declared schema."""

        schema = self.role["output_schema"]
        characters = 2
        for name in schema["required_fields"]:
            specification = schema["properties"].get(name) or {}
            max_length = specification.get("max_length")
            if (
                isinstance(max_length, int)
                and not isinstance(max_length, bool)
                and max_length > 0
            ):
                characters += max_length
            characters += len(str(name)) + 6
        token_estimate = max(1, math.ceil(max(1, characters) / 3.0))
        return max(16, math.ceil(token_estimate * self.format_budget_margin))

    def _attempt_budgets(self) -> list[int]:
        """Shrink the prompt-level target for non-final repair attempts."""

        budgets = [self._schema_token_budget()]
        for _ in range(max(0, self.format_retries - 1)):
            budgets.append(max(64, budgets[-1] // 2))
        return budgets

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

    def _fallback_output(self, answer: str) -> dict[str, Any]:
        """Deterministically complete the schema around a model-supplied answer."""

        schema = self.role["output_schema"]
        output: dict[str, Any] = {}
        for name in schema["required_fields"]:
            if name == "answer":
                output[name] = answer
            else:
                output[name] = "[answer-only fallback: field omitted]"
        return output

    def _repair_message(
        self,
        error_type: str,
        message: str,
        token_budget: int,
        offending_text: str | None = None,
    ) -> str:
        """Give the model the concrete contract violation instead of a bare retry."""

        schema = self.role["output_schema"]
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

    def _answer_only_request(
        self,
        base_request: ModelRequest,
        task: dict[str, Any],
        error: dict[str, Any],
    ) -> ModelRequest:
        instruction = {
            "instruction": (
                "Your earlier replies were rejected because they did not form "
                "a complete JSON object. Return ONLY the final answer to the "
                'task as one short JSON string, for example "B". Do not derive, '
                "restate, explain, or add anything else."
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
            max_output_tokens=base_request.max_output_tokens,
            reasoning_enabled=base_request.reasoning_enabled,
            reasoning_effort=base_request.reasoning_effort,
            metadata=base_request.metadata,
        )

    def _run_solver(self, task: dict[str, Any]) -> dict[str, Any]:
        node_input = self._node_input(task)
        base_request = self._request_for_node(node_input)
        attempt_budgets = self._attempt_budgets()
        record: dict[str, Any] = {
            "node_id": SOLVER_NODE_ID,
            "role": SOLVER_ROLE_ID,
            "started_at": utc_now(),
            "input": node_input,
            "request": base_request.log_view(),
            "status": "started",
            "attempts": [],
            "format_repairs": 0,
            "truncated_attempts": 0,
            "truncation_encountered": False,
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
                                str(last_error["type"]),
                                str(last_error["message"]),
                                token_budget=budget,
                                offending_text=last_response.text if repair_count else None,
                            ),
                        },
                    ],
                    model=base_request.model,
                    temperature=base_request.temperature,
                    max_output_tokens=base_request.max_output_tokens,
                    reasoning_enabled=base_request.reasoning_enabled,
                    reasoning_effort=base_request.reasoning_effort,
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
            except Exception as error:
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
            if response.finish_reason == "length":
                record["truncated_attempts"] += 1
                record["truncation_encountered"] = True
            if is_fallback:
                answer = self._extract_answer(response.text)
                if answer is None:
                    last_error = {
                        "type": "EmptyFallback",
                        "message": "answer-only fallback returned no usable answer",
                    }
                    parsed = None
                else:
                    parsed = self._fallback_output(answer)
                    fallback = {
                        "type": "answer_only",
                        "reason": last_error,
                        "answer": answer,
                    }
                    last_error = None
            else:
                parsed = parse_json_object(response.text)
                if response.finish_reason == "length":
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
                    output_error = validate_output_contract(
                        parsed, self.role["output_schema"], SOLVER_ROLE_ID
                    )
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
                    "response": response.log_view(),
                    "raw_text": response.text,
                }
            )
            repair_count += 1
        record.update(
            {
                "ended_at": utc_now(),
                "format_repairs": repair_count,
                "response": last_response.log_view() if last_response is not None else None,
                "raw_text": last_response.text if last_response is not None else None,
                "finish_reason": last_response.finish_reason
                if last_response is not None
                else None,
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

    def run_observation(
        self, *, task: dict[str, Any], run_id: str
    ) -> dict[str, Any]:
        """Collect one raw independent Single-Agent observation for one task."""

        observation_id = json_hash(
            {
                "kind": "single_agent_baseline",
                "run_id": run_id,
                "task_id": task["task_id"],
            }
        )[:24]
        started_monotonic = time.monotonic()
        observation: dict[str, Any] = {
            "schema_version": "1.0",
            "kind": "single_agent_baseline",
            "observation_id": observation_id,
            "task_id": task["task_id"],
            "domain": task["domain"],
            "status": "running",
            "started_at": utc_now(),
            "configured_model": {
                "model_id": self.experiment["model_id"],
                "provider": self.provider_name,
                "requested_model": self.model["model_name"],
                "temperature": self.model["temperature"],
                "max_output_tokens": self.model["max_output_tokens"],
                "reasoning_enabled": self.model["reasoning"]["enabled"],
                "reasoning_effort": self.model["reasoning"].get("effort"),
            },
        }
        solver = self._run_solver(task)
        wall_clock_ms = max(0, round((time.monotonic() - started_monotonic) * 1000))
        observation["solver"] = solver
        observation["wall_clock_ms"] = wall_clock_ms
        if solver["status"] != "completed":
            observation.update(
                {
                    "status": "failed",
                    "ended_at": utc_now(),
                    "failure_reason": (
                        "the single solver call failed, violated the JSON output "
                        "contract, or was truncated"
                    ),
                }
            )
            return observation
        cumulative = node_cumulative([solver], self.model, wall_clock_ms=wall_clock_ms)
        prediction = {
            "answer": solver["output"]["answer"],
            "reasoning_summary": solver["output"]["reasoning_summary"],
            "solver_output": solver["output"],
            "cumulative": cumulative,
            "finish_reason": solver.get("finish_reason"),
            "truncated": bool(solver.get("truncation_encountered")),
            "truncated_attempts": int(solver.get("truncated_attempts", 0)),
            "format_repairs": int(solver.get("format_repairs", 0)),
            "fallback": solver.get("fallback"),
        }
        prediction["checkpoint_hash"] = json_hash(
            {
                "task_id": task["task_id"],
                "answer": prediction["answer"],
                "reasoning_summary": prediction["reasoning_summary"],
                "solver_output": prediction["solver_output"],
            }
        )
        observation["prediction"] = prediction
        observation["status"] = "complete"
        observation["ended_at"] = utc_now()
        return observation
