"""Load and validate the three JSON contracts that define an experiment."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from contracts import (
    ConfigurationError,
    file_hash,
    require_list,
    require_object,
    require_string,
)


CONFIG_FILES = ("agents.json", "model_config.json", "topology.json")
DEBATE_ROLE_IDS = {"planner", "analyst", "critic", "writer"}
SINGLE_ROLE_ID = "single_solver"
ALL_ROLE_IDS = DEBATE_ROLE_IDS | {SINGLE_ROLE_ID}
CHECKPOINT_ANSWER_ROLE_IDS = {"writer", SINGLE_ROLE_ID}
FROZEN_TOPOLOGY_ID = "debate"
EXPECTED_NODES = {
    "planner_stage_1": ("planner", 1, "stage_1"),
    "analyst_stage_1": ("analyst", 1, "stage_1"),
    "critic_stage_1": ("critic", 1, "stage_1"),
    "planner_stage_2": ("planner", 2, "stage_2"),
    "analyst_stage_2": ("analyst", 2, "stage_2"),
    "critic_stage_2": ("critic", 2, "stage_2"),
    "writer": ("writer", 3, None),
}
EXPECTED_EDGES = [
    ("planner_stage_1", "stage_1_packet"),
    ("analyst_stage_1", "stage_1_packet"),
    ("critic_stage_1", "stage_1_packet"),
    ("stage_1_packet", "planner_stage_2"),
    ("stage_1_packet", "analyst_stage_2"),
    ("stage_1_packet", "critic_stage_2"),
    ("stage_1_packet", "writer_packet"),
    ("planner_stage_2", "writer_packet"),
    ("analyst_stage_2", "writer_packet"),
    ("critic_stage_2", "writer_packet"),
    ("writer_packet", "writer"),
]
EXPECTED_PACKET_NODES = {
    "stage_1_packet": ["planner_stage_1", "analyst_stage_1", "critic_stage_1"],
    "writer_packet": ["stage_1_packet", "planner_stage_2", "analyst_stage_2", "critic_stage_2"],
}


def load_json(path: Path) -> dict[str, Any]:
    """Read one UTF-8 JSON object and report the precise broken file."""

    if not path.is_file():
        raise ConfigurationError(f"required JSON file does not exist: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ConfigurationError(
            f"invalid JSON in {path}:{error.lineno}:{error.colno}: {error.msg}"
        ) from error
    return require_object(raw, str(path))


def _validate_agents(config: dict[str, Any]) -> None:
    if config.get("schema_version") != "1.0":
        raise ConfigurationError("agents.json schema_version must be '1.0'")
    retries = config.get("format_retries", 0)
    if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
        raise ConfigurationError("agents.json format_retries must be a non-negative integer")
    margin = config.get("format_budget_margin", 2.0)
    if (
        isinstance(margin, bool)
        or not isinstance(margin, (int, float))
        or not math.isfinite(float(margin))
        or float(margin) < 1.0
    ):
        raise ConfigurationError(
            "agents.json format_budget_margin must be a finite number >= 1.0"
        )
    roles = require_list(config.get("roles"), "agents.json roles")
    if len(roles) != len(ALL_ROLE_IDS):
        raise ConfigurationError(
            "agents.json must define exactly the four Debate roles plus "
            "the single_solver role"
        )
    found: set[str] = set()
    for index, value in enumerate(roles):
        role = require_object(value, f"agents.json roles[{index}]")
        role_id = require_string(role.get("id"), f"agents.json roles[{index}].id")
        if role_id not in ALL_ROLE_IDS or role_id in found:
            raise ConfigurationError(
                "agents.json roles must be unique "
                "planner/analyst/critic/writer/single_solver"
            )
        found.add(role_id)
        require_string(role.get("system_prompt"), f"agents.json role {role_id}.system_prompt")
        schema = require_object(role.get("output_schema"), f"agents.json role {role_id}.output_schema")
        fields = require_list(schema.get("required_fields"), f"agents.json role {role_id}.required_fields")
        if not fields or not all(isinstance(field, str) and field for field in fields):
            raise ConfigurationError(f"agents.json role {role_id} has invalid required_fields")
        properties = require_object(schema.get("properties"), f"agents.json role {role_id}.output_schema.properties")
        if set(properties) != set(fields):
            raise ConfigurationError(
                f"agents.json role {role_id}.output_schema.properties must exactly describe required_fields"
            )
        for field in fields:
            field_spec = require_object(properties[field], f"agents.json role {role_id}.{field}")
            if (
                field_spec.get("type") != "string"
                or isinstance(field_spec.get("min_length"), bool)
                or not isinstance(field_spec.get("min_length"), int)
                or field_spec["min_length"] < 1
            ):
                raise ConfigurationError(
                    f"agents.json role {role_id}.{field} must be a non-empty string contract"
                )
        if "prompt_file" in role:
            raise ConfigurationError("prompt files are not allowed; prompts must be stored in agents.json")
    if found != ALL_ROLE_IDS:
        raise ConfigurationError(
            "agents.json must include planner, analyst, critic, writer, and single_solver"
        )
    writer = next(role for role in roles if role["id"] == "writer")
    if "answer" not in writer["output_schema"]["required_fields"]:
        raise ConfigurationError("Writer output_schema must require answer")
    if "reasoning_summary" not in writer["output_schema"]["required_fields"]:
        raise ConfigurationError(
            "Writer output_schema must require reasoning_summary"
        )
    if writer["output_schema"].get("is_checkpoint_answer") is not True:
        raise ConfigurationError("Writer must be explicitly marked as a checkpoint answer")
    single_solver = next(role for role in roles if role["id"] == SINGLE_ROLE_ID)
    if "answer" not in single_solver["output_schema"]["required_fields"]:
        raise ConfigurationError("single_solver output_schema must require answer")
    if "reasoning_summary" not in single_solver["output_schema"]["required_fields"]:
        raise ConfigurationError(
            "single_solver output_schema must require reasoning_summary"
        )
    if single_solver["output_schema"].get("is_checkpoint_answer") is not True:
        raise ConfigurationError(
            "single_solver must be explicitly marked as a checkpoint answer"
        )
    for role in roles:
        if (
            role["id"] not in CHECKPOINT_ANSWER_ROLE_IDS
            and role["output_schema"].get("is_checkpoint_answer") is True
        ):
            raise ConfigurationError(
                "only Writer and single_solver may be marked as checkpoint answers"
            )


def _validate_model_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != "1.0":
        raise ConfigurationError("model_config.json schema_version must be '1.0'")
    providers = require_object(config.get("providers"), "model_config.json providers")
    models = require_object(config.get("models"), "model_config.json models")
    model_id = require_string(config.get("default_model_id"), "model_config.json default_model_id")
    if model_id not in models:
        raise ConfigurationError("default_model_id must exist in models")
    for provider_name, value in providers.items():
        provider = require_object(value, f"provider {provider_name}")
        if provider.get("adapter") != "openai_compatible":
            raise ConfigurationError(
                f"provider {provider_name} has unsupported adapter; add a Python adapter before using it"
            )
        require_string(provider.get("base_url"), f"provider {provider_name}.base_url")
        output_tokens_field = provider.get("max_output_tokens_field", "max_tokens")
        if output_tokens_field not in ("max_tokens", "max_completion_tokens"):
            raise ConfigurationError(
                f"provider {provider_name}.max_output_tokens_field must be "
                "max_tokens or max_completion_tokens"
            )
        thinking_toggle = provider.get("supports_thinking_toggle", False)
        if not isinstance(thinking_toggle, bool):
            raise ConfigurationError(
                f"provider {provider_name}.supports_thinking_toggle must be a boolean"
            )
        key_spec = require_object(provider.get("api_key"), f"provider {provider_name}.api_key")
        require_string(key_spec.get("environment_variable"), f"provider {provider_name}.api_key.environment_variable")
        key_file = require_string(key_spec.get("file"), f"provider {provider_name}.api_key.file")
        if not key_file.replace("\\", "/").startswith(".secret/"):
            raise ConfigurationError(f"provider {provider_name} credential file must be in .secret/")
        require_string(key_spec.get("field"), f"provider {provider_name}.api_key.field")
        attempts = provider.get("max_attempts", 1)
        if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1:
            raise ConfigurationError(f"provider {provider_name}.max_attempts must be >= 1")
    for configured_id, value in models.items():
        model = require_object(value, f"model {configured_id}")
        provider_name = require_string(model.get("provider"), f"model {configured_id}.provider")
        if provider_name not in providers:
            raise ConfigurationError(f"model {configured_id} references unknown provider {provider_name}")
        require_string(model.get("model_name"), f"model {configured_id}.model_name")
        temperature = model.get("temperature")
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(float(temperature))
            or not 0.0 <= float(temperature) <= 2.0
        ):
            raise ConfigurationError(
                f"model {configured_id} temperature must be a finite number between 0 and 2"
            )
        reasoning = require_object(model.get("reasoning"), f"model {configured_id}.reasoning")
        reasoning_enabled = reasoning.get("enabled")
        if not isinstance(reasoning_enabled, bool):
            raise ConfigurationError(f"model {configured_id}.reasoning.enabled must be a boolean")
        reasoning_effort = reasoning.get("effort")
        if reasoning_enabled:
            if reasoning_effort not in (
                "none",
                "minimal",
                "low",
                "medium",
                "high",
                "xhigh",
                "max",
            ):
                raise ConfigurationError(
                    f"model {configured_id}.reasoning.effort must be "
                    "none/minimal/low/medium/high/xhigh/max while reasoning is enabled"
                )
        elif reasoning_effort is not None:
            raise ConfigurationError(
                f"model {configured_id} must not set reasoning.effort while reasoning is disabled"
            )
        max_tokens = model.get("max_output_tokens")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ConfigurationError(f"model {configured_id}.max_output_tokens must be positive")
        pricing = model.get("pricing", {})
        if pricing:
            pricing = require_object(pricing, f"model {configured_id}.pricing")
            for key in ("input_cache_hit_per_million", "input_cache_miss_per_million", "output_per_million"):
                price = pricing.get(key)
                if price is not None and (
                    isinstance(price, bool) or not isinstance(price, (int, float)) or price < 0
                ):
                    raise ConfigurationError(f"model {configured_id}.pricing.{key} must be null or non-negative")
        defaults = model.get("request_defaults", {})
        if not isinstance(defaults, dict):
            raise ConfigurationError(f"model {configured_id}.request_defaults must be a JSON object")
        if any(
            key in defaults
            for key in (
                "model",
                "messages",
                "temperature",
                "max_tokens",
                "max_completion_tokens",
                "thinking",
                "reasoning_effort",
                "stream",
            )
        ):
            raise ConfigurationError(
                f"model {configured_id}.request_defaults may not override the "
                "model, messages, sampling, output cap, thinking, reasoning effort, or streaming"
            )
        response_format = defaults.get("response_format")
        if response_format is not None:
            response_format = require_object(response_format, f"model {configured_id}.request_defaults.response_format")
            if response_format.get("type") != "json_object":
                raise ConfigurationError(f"model {configured_id} must request JSON-object output")


def _validate_debate_topology(topology: dict[str, Any], topology_id: str) -> None:
    """Validate the frozen Debate protocol behind its named runner."""

    if topology.get("runner") != "two_stage_pac_writer":
        raise ConfigurationError(f"topology {topology_id} must use runner two_stage_pac_writer")
    if topology.get("max_rounds") != 5:
        raise ConfigurationError(f"topology {topology_id}.max_rounds must be exactly 5")
    nodes = require_list(topology.get("nodes"), f"topology {topology_id}.nodes")
    node_ids = {
        require_string(require_object(node, "topology node").get("id"), "topology node.id")
        for node in nodes
    }
    if node_ids != set(EXPECTED_NODES) or len(nodes) != 7:
        raise ConfigurationError("Debate must contain exactly P1/A1/C1/P2/A2/C2/Writer")
    for node in nodes:
        node_obj = require_object(node, "topology node")
        expected_role, expected_stage, expected_group = EXPECTED_NODES[node_obj["id"]]
        if (node_obj.get("role"), node_obj.get("stage"), node_obj.get("parallel_group")) != (
            expected_role,
            expected_stage,
            expected_group,
        ):
            raise ConfigurationError(f"Debate node {node_obj['id']} does not match its frozen role/stage")
    packet_ids = set(EXPECTED_PACKET_NODES)
    all_ids = node_ids | packet_ids
    actual_edges: list[tuple[str, str]] = []
    for edge in require_list(topology.get("edges"), f"topology {topology_id}.edges"):
        pair = require_list(edge, f"topology {topology_id} edge")
        if len(pair) != 2 or not all(isinstance(value, str) and value for value in pair):
            raise ConfigurationError("each topology edge must be a [source, destination] pair")
        if pair[0] not in all_ids or pair[1] not in all_ids:
            raise ConfigurationError("topology edge references an unknown node or packet")
        actual_edges.append((pair[0], pair[1]))
    if actual_edges != EXPECTED_EDGES:
        raise ConfigurationError("Debate edge set/order must remain fixed")
    packets = require_list(topology.get("packets"), f"topology {topology_id}.packets")
    if len(packets) != len(EXPECTED_PACKET_NODES):
        raise ConfigurationError("Debate must contain exactly its two deterministic packets")
    actual_packets: dict[str, list[Any]] = {}
    for packet in packets:
        packet_obj = require_object(packet, f"topology {topology_id} packet")
        packet_id = require_string(packet_obj.get("id"), f"topology {topology_id} packet.id")
        actual_packets[packet_id] = require_list(
            packet_obj.get("sources"), f"topology {topology_id} {packet_id}.sources"
        )
    if actual_packets != EXPECTED_PACKET_NODES:
        raise ConfigurationError("Debate deterministic packet source order must remain fixed")


def select_topology(document: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Resolve the one frozen Debate topology; there is no other topology."""

    if document.get("schema_version") != "1.0":
        raise ConfigurationError("topology.json schema_version must be '1.0'")
    topologies = require_object(document.get("topologies"), "topology.json topologies")
    default_id = require_string(
        document.get("default_topology_id"), "topology.json default_topology_id"
    )
    if set(topologies) != {FROZEN_TOPOLOGY_ID}:
        raise ConfigurationError(
            "topology.json must describe only the frozen debate topology"
        )
    if default_id != FROZEN_TOPOLOGY_ID:
        raise ConfigurationError("topology.json default_topology_id must be 'debate'")
    selected = require_object(
        topologies[FROZEN_TOPOLOGY_ID], f"topology {FROZEN_TOPOLOGY_ID}"
    )
    _validate_debate_topology(selected, FROZEN_TOPOLOGY_ID)
    return FROZEN_TOPOLOGY_ID, selected


def load_experiment_config(
    project_root: Path,
    model_id: str | None = None,
) -> dict[str, Any]:
    """Return the validated configuration resolved for one run-level model.

    ``model_id=None`` keeps the frozen ``default_model_id`` so existing runs
    stay DeepSeek-backed.  The approved delayed-checkpoint Debate topology is
    the only topology in the project and is never selectable from the CLI.
    """

    root = project_root.resolve()
    config_dir = root / "configs"
    agents = load_json(config_dir / "agents.json")
    model_config = load_json(config_dir / "model_config.json")
    topology_document = load_json(config_dir / "topology.json")
    _validate_agents(agents)
    _validate_model_config(model_config)
    selected_topology_id, topology = select_topology(topology_document)
    selected_id = model_config["default_model_id"] if model_id is None else model_id
    models = model_config["models"]
    if selected_id not in models:
        available = ", ".join(sorted(models))
        raise ConfigurationError(
            f"unknown model id {selected_id!r}; configured models are: {available}"
        )
    model = models[selected_id]
    provider = model_config["providers"][model["provider"]]
    return {
        "root": str(root),
        "agents": agents,
        "model_config": model_config,
        "topology_document": topology_document,
        "topology": topology,
        "topology_id": selected_topology_id,
        "model_id": selected_id,
        "model": model,
        "provider_name": model["provider"],
        "provider": provider,
    }


def config_snapshot(project_root: Path) -> dict[str, Any]:
    """Capture exact JSON values and hashes for an immutable experiment manifest."""

    root = project_root.resolve()
    files: dict[str, Any] = {}
    for name in CONFIG_FILES:
        path = root / "configs" / name
        files[name] = {"sha256": file_hash(path), "content": load_json(path)}
    return files
