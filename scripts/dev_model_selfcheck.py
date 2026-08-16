"""Offline self-check for run-level model selection and provider isolation.

This script never contacts a model provider.  It verifies that the default
model remains DeepSeek, that an explicit ``gpt5_nano`` selection swaps every
Debate node to GPT-5-nano, that run manifests record the real provider/model,
and that DeepSeek-specific wire parameters never reach the OpenAI adapter (or
vice versa).
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

from config_loader import load_experiment_config  # noqa: E402
from contracts import ConfigurationError, ModelRequest, ModelResponse  # noqa: E402
from debate_runner import FixedDebateRunner  # noqa: E402
from provider import build_provider  # noqa: E402
from scorer import score_task  # noqa: E402
from storage import create_run, read_json, update_run_status  # noqa: E402


ROLE_OUTPUTS = {
    "planner": {
        "plan": "solve directly",
        "assumptions": "standard",
        "verification_steps": "check once",
        "candidate_answer": "B",
    },
    "analyst": {
        "analysis": "two plus two is four",
        "candidate_answer": "B",
        "evidence": "arithmetic",
    },
    "critic": {
        "issues": "none",
        "evidence": "checked",
        "revision_advice": "none",
        "candidate_answer": "B",
    },
    "writer": {
        "answer": "B",
        "reasoning_summary": "Two plus two is four, which is option B.",
    },
}


class FakeProvider:
    """Return scripted chat completions and retain every request for checks."""

    def __init__(self, by_node: dict[str, tuple[str | None, str]] | None = None) -> None:
        self.by_node = dict(by_node or {})
        self.requests: list[ModelRequest] = []
        self.closed = False

    def generate(self, request: ModelRequest) -> tuple[ModelResponse, list[dict[str, Any]]]:
        self.requests.append(request)
        node_id = request.metadata.get("node_id")
        if node_id not in self.by_node:
            raise AssertionError(f"fake provider has no scripted response for {node_id}")
        finish_reason, text = self.by_node[node_id]
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
                "request_payload": {"messages": request.messages},
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


def check(condition: bool, label: str) -> None:
    if not condition:
        raise AssertionError(label)
    print(f"PASS {label}")


def _task() -> dict[str, Any]:
    return {
        "task_id": "dev::model::selfcheck",
        "domain": "mmlu_pro",
        "prompt": (
            "What is 2 + 2?\n\nChoices:\n"
            "(A) 2\n(B) 4\n(C) 6\n(D) 8\n\n"
            "Return only the letter of the correct choice (A-D)."
        ),
        "options": ["2", "4", "6", "8"],
        "answer_index": 1,
        "reference_answer": "B",
    }


def _node_scripts() -> dict[str, tuple[str | None, str]]:
    scripts: dict[str, tuple[str | None, str]] = {}
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
        scripts[node_id] = (
            "stop",
            json.dumps(ROLE_OUTPUTS[role], separators=(",", ":")),
        )
    return scripts


def _check_selection() -> None:
    default_experiment = load_experiment_config(PROJECT_ROOT)
    check(
        default_experiment["model_id"] == "deepseek_flash"
        and default_experiment["provider_name"] == "deepseek"
        and default_experiment["model"]["model_name"] == "deepseek-v4-flash",
        "default model selection remains deepseek_flash",
    )
    gpt_experiment = load_experiment_config(PROJECT_ROOT, model_id="gpt5_nano")
    check(
        gpt_experiment["model_id"] == "gpt5_nano"
        and gpt_experiment["provider_name"] == "openai"
        and gpt_experiment["model"]["model_name"] == "gpt-5-nano",
        "explicit gpt5_nano selection resolves the OpenAI provider and model",
    )
    gpt4o_experiment = load_experiment_config(PROJECT_ROOT, model_id="gpt4o_mini")
    check(
        gpt4o_experiment["model_id"] == "gpt4o_mini"
        and gpt4o_experiment["provider_name"] == "openai"
        and gpt4o_experiment["model"]["model_name"] == "gpt-4o-mini-2024-07-18",
        "explicit gpt4o_mini selection resolves the fixed OpenAI snapshot",
    )
    check(
        gpt4o_experiment["model"]["reasoning"] == {"enabled": False}
        and gpt4o_experiment["model"]["temperature"] == 0
        and gpt4o_experiment["model"]["max_output_tokens"] == 16384,
        "gpt4o_mini is an explicit non-reasoning profile with temperature 0 and a 16384 ceiling",
    )
    check(
        default_experiment["agents"] == gpt_experiment["agents"]
        and default_experiment["topology"] == gpt_experiment["topology"],
        "model selection changes neither agents nor debate topology",
    )
    try:
        load_experiment_config(PROJECT_ROOT, model_id="not_a_model")
    except ConfigurationError as error:
        check("gpt5_nano" in str(error), "unknown model id fails with the configured choices")
    else:
        raise AssertionError("unknown model id was silently accepted")


def _check_wire_payloads() -> None:
    deepseek_experiment = load_experiment_config(PROJECT_ROOT)
    gpt_experiment = load_experiment_config(PROJECT_ROOT, model_id="gpt5_nano")
    gpt4o_experiment = load_experiment_config(PROJECT_ROOT, model_id="gpt4o_mini")
    previous_env = dict(os.environ)
    os.environ["DEEPSEEK_API_KEY"] = "test-deepseek-key"
    os.environ["OPENAI_API_KEY"] = "test-openai-key"
    try:
        deepseek_provider = build_provider(dict(deepseek_experiment))
        openai_provider = build_provider(dict(gpt_experiment))
        gpt4o_provider = build_provider(dict(gpt4o_experiment))
    finally:
        for name in ("DEEPSEEK_API_KEY", "OPENAI_API_KEY"):
            if name in previous_env:
                os.environ[name] = previous_env[name]
            else:
                os.environ.pop(name, None)
    request = ModelRequest(
        messages=[{"role": "system", "content": "system"}, {"role": "user", "content": "{}"}],
        model="request-model",
        temperature=0.2,
        max_output_tokens=32768,
        reasoning_enabled=True,
        reasoning_effort="medium",
    )
    deepseek_payload = deepseek_provider._payload(request)
    openai_payload = openai_provider._payload(request)
    gpt4o_payload = gpt4o_provider._payload(
        ModelRequest(
            messages=[{"role": "system", "content": "system"}, {"role": "user", "content": "{}"}],
            model="gpt-4o-mini-2024-07-18",
            temperature=0,
            max_output_tokens=16384,
            reasoning_enabled=False,
        )
    )
    check(
        deepseek_payload["thinking"] == {"type": "enabled"}
        and deepseek_payload.get("max_tokens") == 32768
        and "max_completion_tokens" not in deepseek_payload,
        "DeepSeek wire payload uses the thinking toggle and max_tokens",
    )
    check(
        "thinking" not in openai_payload
        and openai_payload.get("max_completion_tokens") == 32768
        and "max_tokens" not in openai_payload
        and openai_payload.get("reasoning_effort") == "medium",
        "OpenAI wire payload uses max_completion_tokens and reasoning_effort, never thinking",
    )
    check(
        gpt4o_payload.get("max_completion_tokens") == 16384
        and "max_tokens" not in gpt4o_payload
        and "thinking" not in gpt4o_payload
        and "reasoning_effort" not in gpt4o_payload
        and gpt4o_payload.get("temperature") == 0
        and gpt4o_payload.get("response_format") == {"type": "json_object"},
        "GPT-4o-mini payload uses max_completion_tokens, temperature 0, and no reasoning flags",
    )
    check(
        deepseek_provider.endpoint == "https://api.deepseek.com/chat/completions"
        and openai_provider.endpoint == "https://api.openai.com/v1/chat/completions",
        "each provider targets its own chat-completions endpoint",
    )
    deepseek_provider.close()
    openai_provider.close()
    gpt4o_provider.close()


def _check_gpt4o_mini_runner() -> None:
    experiment = load_experiment_config(PROJECT_ROOT, model_id="gpt4o_mini")
    provider = FakeProvider(_node_scripts())
    runner = FixedDebateRunner(dict(experiment), provider)
    round_record = runner.run_round(
        task=_task(),
        round_index=1,
        previous_checkpoint=None,
    )
    check(round_record["status"] == "completed", "GPT-4o-mini debate round 1 completes")
    check(
        all(
            request.model == "gpt-4o-mini-2024-07-18"
            and request.temperature == 0
            and request.max_output_tokens == 16384
            and not request.reasoning_enabled
            and request.reasoning_effort is None
            for request in provider.requests
        ),
        "every Debate node requests GPT-4o-mini with the wide non-reasoning ceiling",
    )


def _check_usage_parsing() -> None:
    from provider import OpenAICompatibleProvider

    deepseek_hit, deepseek_miss = OpenAICompatibleProvider._cache_usage(
        {
            "usage": {
                "prompt_tokens": 120,
                "completion_tokens": 30,
                "prompt_cache_hit_tokens": 80,
                "prompt_cache_miss_tokens": 40,
            }
        }
    )
    check(
        (deepseek_hit, deepseek_miss) == (80, 40),
        "DeepSeek cache counters are read directly",
    )
    openai_hit, openai_miss = OpenAICompatibleProvider._cache_usage(
        {
            "usage": {
                "prompt_tokens": 120,
                "completion_tokens": 30,
                "prompt_tokens_details": {"cached_tokens": 80},
                "completion_tokens_details": {"reasoning_tokens": 12},
            }
        }
    )
    check(
        (openai_hit, openai_miss) == (80, 40),
        "OpenAI cached prompt tokens are derived from prompt_tokens_details",
    )
    reasoning = OpenAICompatibleProvider._reasoning_tokens(
        {
            "usage": {
                "completion_tokens_details": {"reasoning_tokens": 12},
            }
        }
    )
    check(reasoning == 12, "OpenAI reasoning tokens are read from completion_tokens_details")


def _check_manifest() -> None:
    experiment = load_experiment_config(PROJECT_ROOT, model_id="gpt5_nano")
    model_selection = {
        "model_id": experiment["model_id"],
        "provider": experiment["provider_name"],
        "requested_model": experiment["model"]["model_name"],
        "temperature": experiment["model"]["temperature"],
        "max_output_tokens": experiment["model"]["max_output_tokens"],
        "reasoning": experiment["model"]["reasoning"],
    }
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        manifest = create_run(
            root,
            command=["roundvalue", "run", "--model-id", "gpt5_nano"],
            config_snapshot={},
            dataset_name="smoke",
            domain="mmlu_pro",
            topology_id="debate",
            requested_model="gpt-5-nano",
        )
        updated = update_run_status(
            manifest,
            "running",
            mode="smoke",
            selected_model_id=experiment["model_id"],
            model_selection=model_selection,
        )
        reloaded = read_json(root / "trajectories" / updated["run_id"] / "run.json")
        check(
            reloaded["selected_model_id"] == "gpt5_nano"
            and reloaded["model_selection"] == model_selection,
            "run manifest records the selected provider and model configuration",
        )
        check(
            reloaded["model_selection"]["provider"] == "openai"
            and reloaded["model_selection"]["requested_model"] == "gpt-5-nano",
            "run manifest distinguishes the GPT-5-nano run from a DeepSeek run",
        )
        check(
            re.fullmatch(
                r"\d{12}_smoke_debate_gpt-5-nano_[0-9a-f]{8}",
                updated["run_id"],
            )
            and updated["run_name"] == updated["run_id"]
            and updated["run_name_components"]
            == {
                "timestamp": updated["run_id"].split("_")[0],
                "dataset": "smoke",
                "topology": "debate",
                "model": "gpt-5-nano",
                "hex": updated["run_id"].split("_")[-1],
            },
            "auto-named runs expose timestamp/dataset/topology/model/hex components",
        )


def _check_runner_model() -> None:
    experiment = load_experiment_config(PROJECT_ROOT, model_id="gpt5_nano")
    provider = FakeProvider(_node_scripts())
    runner = FixedDebateRunner(dict(experiment), provider)
    round_record = runner.run_round(
        task=_task(),
        round_index=1,
        previous_checkpoint=None,
    )
    check(round_record["status"] == "completed", "GPT-5-nano round 1 completes")
    check(
        len(provider.requests) == 7
        and all(request.model == "gpt-5-nano" for request in provider.requests),
        "all seven Debate nodes request gpt-5-nano",
    )
    check(
        {request.metadata.get("role") for request in provider.requests}
        == {"planner", "analyst", "critic", "writer"},
        "Planner, Analyst, Critic, and Writer all run through the selected model",
    )
    check(
        all(
            request.reasoning_enabled
            and request.reasoning_effort == "medium"
            for request in provider.requests
        ),
        "GPT-5-nano node requests carry the configured medium reasoning effort",
    )
    trajectory = runner.run_trajectory(
        task=_task(),
        run_id="dev-model-five-rounds",
        max_rounds=5,
    )
    check(
        trajectory["status"] == "complete"
        and len(trajectory["checkpoints"]) == 5
        and all(
            request.model == "gpt-5-nano"
            for request in provider.requests
        ),
        "five-round GPT-5-nano trajectory stays on the selected model",
    )
    check(
        trajectory["configured_model"]
        == {
            "model_id": "gpt5_nano",
            "provider": "openai",
            "requested_model": "gpt-5-nano",
            "temperature": 1.0,
            "max_output_tokens": 32768,
            "reasoning_enabled": True,
            "reasoning_effort": "medium",
        },
        "trajectory metadata records the actual provider and inference settings",
    )
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "trajectory.json"
        path.write_text(json.dumps(trajectory, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        reloaded = json.loads(path.read_text(encoding="utf-8"))
    check(
        reloaded["status"] == "complete"
        and reloaded["configured_model"]["requested_model"] == "gpt-5-nano",
        "GPT-5-nano trajectory serializes and reloads across the JSON boundary",
    )


def _check_scoring() -> None:
    task = _task()
    score = score_task(
        task,
        {"answer": "B", "reasoning_summary": "Two plus two is four, which is option B."},
    )
    check(score["quality"] == 1.0, "MMLU-Pro scoring uses the canonical answer")
    score = score_task(
        task,
        {
            "answer": "A",
            "reasoning_summary": "The correct canonical option is B.",
        },
    )
    check(
        score["quality"] == 0.0,
        "reasoning_summary content never rescues an incorrect answer",
    )


def main() -> int:
    _check_selection()
    _check_wire_payloads()
    _check_usage_parsing()
    _check_manifest()
    _check_runner_model()
    _check_gpt4o_mini_runner()
    _check_scoring()
    print("PASS all model-selection self-checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
