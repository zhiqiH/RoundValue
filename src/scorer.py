"""Deterministic, dependency-free offline scorer for RoundValue math checkpoints.

The public functions in this module operate on plain JSON-shaped dictionaries.
They deliberately never contact a model provider and never mutate a saved
trajectory.  A scorer may therefore be used during ``fit``, ``evaluate``, or
``reproduce`` without creating a new experiment run.

Every benchmark task carries ``reference_answer`` (or ``accepted_answers``)
and is scored by conservative normalized / numeric equivalence.  There is no
code execution path: the project is a math-only benchmark.

The canonical trajectory shape is documented in :func:`score_trajectory`,
but compatibility readers also accept common older flat checkpoint fields.
"""

from __future__ import annotations

import ast
import json
import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation, localcontext
from typing import Any

SCORER_VERSION = "roundvalue-offline-scorer-v3"
_FINAL_ANSWER_RE = re.compile(r"(?:final[_ ]?answer|answer)\s*:\s*([^\r\n]+)", re.I)
# A comma is treated as a thousands separator only inside a bare digit run
# such as ``1,234`` or ``11,111,100``.  It is never removed inside a tuple,
# interval, or set, so ``(1,2)`` stays distinct from ``(12)`` and ``12``.
_THOUSANDS_SEPARATOR_RE = re.compile(
    r"(?<![0-9(\[{])([0-9]{1,3})(?:,([0-9]{3}))+(?![0-9)\]}\{\[])"
)
# The pinned canonical oracles are trusted upstream code and may legitimately
# be much slower than a model candidate (for example Mbpp/599's official
# reference sums ``range`` up to 1e8).  The reference gets its own generous
# budget while untrusted candidate code keeps the short configured timeout.
def _mapping(value: Any) -> Mapping[str, Any] | None:
    """Return *value* as a mapping, or ``None`` without coercing it."""

    return value if isinstance(value, Mapping) else None


def _string(value: Any) -> str | None:
    """Return a nonempty textual value while preserving numeric answers."""

    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, int | float | Decimal) and not isinstance(value, bool):
        return str(value)
    return None


def _nested_value(mapping: Mapping[str, Any], paths: Sequence[Sequence[str]]) -> Any:
    """Read the first present dotted path from an untrusted JSON object."""

    for path in paths:
        current: Any = mapping
        found = True
        for name in path:
            if not isinstance(current, Mapping) or name not in current:
                found = False
                break
            current = current[name]
        if found and current is not None:
            return current
    return None


def _last_boxed_value(text: str) -> str | None:
    """Extract the final balanced ``\\boxed{...}`` expression if present."""

    marker = "\\boxed{"
    values: list[str] = []
    start = 0
    while True:
        marker_at = text.find(marker, start)
        if marker_at < 0:
            break
        position = marker_at + len(marker)
        depth = 1
        cursor = position
        while cursor < len(text) and depth:
            character = text[cursor]
            if character == "{" and (cursor == 0 or text[cursor - 1] != "\\"):
                depth += 1
            elif character == "}" and (cursor == 0 or text[cursor - 1] != "\\"):
                depth -= 1
            cursor += 1
        if depth == 0:
            candidate = text[position : cursor - 1].strip()
            if candidate:
                values.append(candidate)
        start = marker_at + len(marker)
    return values[-1] if values else None


def extract_final_answer(value: Any) -> str | None:
    """Extract a Writer answer from a JSON object, scalar, or text response.

    Writer output is expected to be ``{"final_answer": ...}``, but old saved
    records sometimes put the answer under ``answer``, ``content``, or a nested
    ``writer`` object.  The function is intentionally read-only and does not
    attempt to repair malformed JSON.
    """

    value_map = _mapping(value)
    if value_map is not None:
        nested = _nested_value(
            value_map,
            (
                ("final_answer",),
                ("answer",),
                ("writer", "final_answer"),
                ("writer", "answer"),
                ("writer_output", "final_answer"),
                ("output", "final_answer"),
                ("response", "final_answer"),
                ("content",),
                ("text",),
                ("output",),
            ),
        )
        if nested is value:
            return None
        return extract_final_answer(nested)

    text = _string(value)
    if text is None:
        return None
    # A provider may have returned an otherwise valid Writer JSON object as
    # text.  Parsing it avoids treating its braces as part of the answer.
    if text.startswith("{") and text.endswith("}"):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, Mapping):
            nested = extract_final_answer(parsed)
            if nested is not None:
                return nested
    matches = list(_FINAL_ANSWER_RE.finditer(text))
    if matches:
        answer = matches[-1].group(1).strip()
        if answer:
            return answer
    boxed = _last_boxed_value(text)
    return boxed if boxed is not None else text


def _strip_math_delimiters(value: str) -> str:
    value = value.strip()
    pairs = (("$", "$"), ("\\(", "\\)"), ("\\[", "\\]"))
    changed = True
    while changed:
        changed = False
        for left, right in pairs:
            if (
                value.startswith(left)
                and value.endswith(right)
                and len(value) >= len(left) + len(right)
            ):
                value = value[len(left) : len(value) - len(right)].strip()
                changed = True
    return value


def _fix_fracs(value: str) -> str:
    """Expand shorthand LaTeX fractions like ``\\frac12`` to ``\\frac{1}{2}``."""

    substrs = value.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if not substr:
                return value
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return value
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    return new_str


def _fix_a_slash_b(value: str) -> str:
    """Convert a pure ``a/b`` integer form to ``\\frac{a}{b}``."""

    if len(value.split("/")) != 2:
        return value
    a, b = value.split("/")
    try:
        a_int, b_int = int(a), int(b)
        assert value == f"{a_int}/{b_int}"
        return f"\\frac{{{a_int}}}{{{b_int}}}"
    except (ValueError, AssertionError):
        return value


def _remove_right_units(value: str) -> str:
    if "\\text{ " in value:
        splits = value.split("\\text{ ")
        if len(splits) == 2:
            return splits[0]
    return value


def _fix_sqrt(value: str) -> str:
    """Expand ``\\sqrt3`` to ``\\sqrt{3}`` while leaving braced forms intact."""

    if "\\sqrt" not in value:
        return value
    splits = value.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if not split:
            return value
        if split[0] != "{":
            new_substr = "\\sqrt{" + split[0] + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def _strip_string(value: str) -> str:
    """Canonical Hendrycks MATH / PRM800K answer normalization.

    This is the grading convention used by the MATH-500 evaluation harness, so
    aligning with it keeps RoundValue's math accuracy comparable to published
    numbers.  Fractions, square roots, degrees, units and percentages are
    canonicalized before the two answers are compared exactly.
    """

    value = value.replace("\n", "")
    value = value.replace("\\!", "")
    value = value.replace("\\\\", "\\")
    value = value.replace("tfrac", "frac")
    value = value.replace("dfrac", "frac")
    value = value.replace("\\left", "")
    value = value.replace("\\right", "")
    value = value.replace("^{\\circ}", "")
    value = value.replace("^\\circ", "")
    value = value.replace("\\$", "")
    value = _remove_right_units(value)
    value = value.replace("\\%", "")
    # A bare ``%`` is deliberately preserved: stripping it would turn a wrong
    # ``5%`` into ``5`` and therefore into a false correct answer.  The
    # numeric fallback still interprets a trailing percent as x/100, so
    # ``50%`` and ``0.5`` remain equivalent while ``5%`` and ``5`` do not.
    value = value.replace(" .", " 0.")
    value = value.replace("{.", "{0.")
    if not value:
        return value
    if value[0] == ".":
        value = "0" + value
    if len(value.split("=")) == 2 and len(value.split("=")[0]) <= 2:
        value = value.split("=")[1]
    value = _fix_sqrt(value)
    value = value.replace(" ", "")
    value = _fix_fracs(value)
    if value == "0.5":
        value = "\\frac{1}{2}"
    value = _fix_a_slash_b(value)
    return value


def normalize_math_answer(value: str) -> str:
    """Normalize a math answer with the canonical MATH-500 convention.

    The core transformation is the Hendrycks MATH / PRM800K normalization used
    by the MATH-500 harness.  On top of it we keep only conservative, safe
    extensions: NFKC unicode normalization, math-delimiter stripping, dash
    unification, thousands-separator removal, and a trailing-period cleanup.
    Case is preserved because it is meaningful for symbolic answers.
    """

    text = unicodedata.normalize("NFKC", value)
    text = _strip_math_delimiters(text)
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    match = re.search(r"^\\text\{(?P<text>.+?)\}$", text, re.S)
    if match:
        text = match.group("text").strip()
    text = _strip_string(text)
    if text.endswith(".") and not re.fullmatch(r"[+-]?\d+\.", text):
        text = text[:-1]
    text = _THOUSANDS_SEPARATOR_RE.sub(
        lambda match: match.group(0).replace(",", ""), text
    )
    return text


def _balanced_group(value: str, start: int) -> tuple[str, int] | None:
    if start >= len(value) or value[start] != "{":
        return None
    depth = 1
    cursor = start + 1
    while cursor < len(value) and depth:
        if value[cursor] == "{":
            depth += 1
        elif value[cursor] == "}":
            depth -= 1
        cursor += 1
    if depth:
        return None
    return value[start + 1 : cursor - 1], cursor


def _replace_latex_fractions(value: str) -> str | None:
    """Translate bounded ``\\frac{a}{b}`` fragments to parenthesized division."""

    token = "\\frac"
    while token in value:
        token_at = value.rfind(token)
        first = _balanced_group(value, token_at + len(token))
        if first is None:
            return None
        numerator, cursor = first
        second = _balanced_group(value, cursor)
        if second is None:
            return None
        denominator, cursor = second
        value = value[:token_at] + f"(({numerator})/({denominator}))" + value[cursor:]
    return value


def _numeric_expression(value: str) -> Decimal | None:
    """Evaluate a small numeric expression without ``eval`` or third parties."""

    candidate = _replace_latex_fractions(value)
    if candidate is None:
        return None
    candidate = candidate.replace("^", "**")
    if candidate.endswith("%"):
        base = _numeric_expression(candidate[:-1])
        return None if base is None else base / Decimal(100)
    # ``x=5`` is accepted only for the simple answer-format convention.
    equation = re.fullmatch(r"[a-zA-Z][a-zA-Z_0-9]*=(.+)", candidate)
    if equation:
        candidate = equation.group(1)
    if len(candidate) > 300:
        return None
    try:
        tree = ast.parse(candidate, mode="eval")
    except SyntaxError:
        return None

    def calculate(node: ast.AST) -> Decimal:
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, int | float)
            and not isinstance(node.value, bool)
        ):
            result = Decimal(str(node.value))
            if not result.is_finite():
                raise ValueError("nonfinite numeric constant")
            return result
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd | ast.USub):
            result = calculate(node.operand)
            return result if isinstance(node.op, ast.UAdd) else -result
        if isinstance(node, ast.BinOp):
            left = calculate(node.left)
            right = calculate(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                if right != right.to_integral_value() or abs(right) > 100:
                    raise ValueError("unsupported exponent")
                return left ** int(right)
        raise ValueError("unsupported numeric expression")

    try:
        with localcontext() as context:
            context.prec = 80
            result = calculate(tree.body)
            return result if result.is_finite() else None
    except (ArithmeticError, InvalidOperation, ValueError, OverflowError):
        return None


def _as_nonnegative_decimal(value: Any) -> Decimal | None:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return result if result.is_finite() and result >= 0 else None


def _math_equivalent(prediction: str, reference: str, task: Mapping[str, Any]) -> tuple[bool, str]:
    predicted = normalize_math_answer(prediction)
    expected = normalize_math_answer(reference)
    if not predicted or not expected:
        return False, "empty_answer"
    if predicted == expected:
        return True, "exact_normalized_match"
    left = _numeric_expression(predicted)
    right = _numeric_expression(expected)
    if left is None or right is None:
        return False, "not_equivalent"
    tolerance_config = _mapping(task.get("scoring")) or {}
    absolute = _as_nonnegative_decimal(tolerance_config.get("abs_tol", task.get("abs_tol", 0)))
    relative = _as_nonnegative_decimal(tolerance_config.get("rel_tol", task.get("rel_tol", 0)))
    if absolute is None or relative is None:
        return False, "invalid_tolerance"
    difference = abs(left - right)
    allowed = max(absolute, relative * max(abs(left), abs(right)))
    return (difference <= allowed, "numeric_match" if difference <= allowed else "not_equivalent")


def _reference_answers(task: Mapping[str, Any]) -> list[str]:
    values: list[Any] = []
    for key in ("accepted_answers", "reference_answers"):
        candidate = task.get(key)
        if isinstance(candidate, Sequence) and not isinstance(candidate, str | bytes | bytearray):
            values.extend(candidate)
    for key in ("reference_answer", "gold_answer", "expected_answer", "answer"):
        if key in task:
            values.append(task[key])
    scoring = _mapping(task.get("scoring"))
    if scoring is not None:
        for key in ("accepted_answers", "reference_answers", "reference_answer"):
            candidate = scoring.get(key)
            if isinstance(candidate, Sequence) and not isinstance(
                candidate, str | bytes | bytearray
            ):
                values.extend(candidate)
            elif candidate is not None:
                values.append(candidate)
    answers: list[str] = []
    for value in values:
        answer = extract_final_answer(value)
        if answer is not None and answer not in answers:
            answers.append(answer)
    return answers


def _task_id(task: Mapping[str, Any]) -> str | None:
    value = task.get("task_id", task.get("id"))
    return str(value) if value is not None else None


def _result(
    task: Mapping[str, Any],
    *,
    domain: str,
    answer: str | None,
    quality: float | None,
    reason: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "task_id": _task_id(task),
        "domain": domain,
        "predicted_answer": answer,
        "quality": float(quality) if quality is not None else None,
        "is_correct": bool(quality == 1.0) if quality is not None else None,
        "reason": reason,
        "scorer_version": SCORER_VERSION,
    }
    if extra:
        result.update(dict(extra))
    return result


def score_math(task: Mapping[str, Any], final_answer: Any) -> dict[str, Any]:
    """Score one mathematical Writer answer against public offline labels."""

    answer = extract_final_answer(final_answer)
    if answer is None:
        return _result(task, domain="math", answer=None, quality=0.0, reason="missing_final_answer")
    references = _reference_answers(task)
    if not references:
        return _result(
            task, domain="math", answer=answer, quality=0.0, reason="missing_reference_answer"
        )
    for reference in references:
        matched, reason = _math_equivalent(answer, reference, task)
        if matched:
            return _result(task, domain="math", answer=answer, quality=1.0, reason=reason)
    return _result(task, domain="math", answer=answer, quality=0.0, reason="not_equivalent")


def score_task(
    task: Mapping[str, Any],
    final_answer: Any,
) -> dict[str, Any]:
    """Score one Writer answer with the math-only offline scorer."""

    if not isinstance(task, Mapping):
        raise TypeError("task must be a JSON object")
    return score_math(task, final_answer)


def _checkpoint_answer(checkpoint: Mapping[str, Any]) -> Any:
    return _nested_value(
        checkpoint,
        (
            ("final_answer",),
            ("writer", "final_answer"),
            ("writer_output", "final_answer"),
            ("output", "final_answer"),
            ("answer",),
            ("answer_text",),
        ),
    )


def score_checkpoint(
    task: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    *,
    trajectory_id: str | None = None,
) -> dict[str, Any]:
    """Score a single saved Writer checkpoint and retain its stable identity."""

    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint must be a JSON object")
    score = score_task(
        task,
        _checkpoint_answer(checkpoint),
    )
    round_index = checkpoint.get("round_index", checkpoint.get("round"))
    if isinstance(round_index, bool) or not isinstance(round_index, int):
        raise ValueError("checkpoint requires integer round_index")
    score["round_index"] = round_index
    record_trajectory_id = trajectory_id or checkpoint.get("trajectory_id")
    if record_trajectory_id is not None:
        score["trajectory_id"] = str(record_trajectory_id)
    for key in ("checkpoint_hash", "checkpoint_id"):
        if checkpoint.get(key) is not None:
            score[key] = str(checkpoint[key])
    return score


def score_trajectory(
    task_record: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Generate deterministic score records for every saved checkpoint.

    Canonical input::

        {
          "task": {...}, "split": "test",
          "trajectory": {
            "status": "complete", "trajectory_id": "...",
            "checkpoints": [{"round_index": 1, "final_answer": "..."}, ...]
          }
        }

    For migration convenience, ``checkpoints`` and ``task`` may also be at the
    top level.  The returned list is sorted by round and is deliberately not
    written back into the input record.
    """

    if not isinstance(task_record, Mapping):
        raise TypeError("task_record must be a JSON object")
    task_candidate = task_record.get("task", task_record.get("task_spec"))
    task = _mapping(task_candidate)
    if task is None:
        # A direct task-plus-trajectory object remains unambiguous when it has
        # public scoring fields; avoid accepting an arbitrary trajectory alone.
        if "reference_answer" in task_record:
            task = task_record
        else:
            raise ValueError("task_record requires a task object")
    trajectory = _mapping(task_record.get("trajectory")) or task_record
    checkpoints = trajectory.get("checkpoints", task_record.get("checkpoints"))
    if not isinstance(checkpoints, Sequence) or isinstance(checkpoints, str | bytes | bytearray):
        raise ValueError("trajectory requires a checkpoints array")
    trajectory_id_value = trajectory.get("trajectory_id", task_record.get("trajectory_id"))
    trajectory_id = str(trajectory_id_value) if trajectory_id_value is not None else None
    ordered: list[Mapping[str, Any]] = []
    for checkpoint in checkpoints:
        if not isinstance(checkpoint, Mapping):
            raise ValueError("every checkpoint must be a JSON object")
        ordered.append(checkpoint)
    ordered.sort(key=lambda item: item.get("round_index", item.get("round", -1)))
    seen_rounds: set[int] = set()
    scores: list[dict[str, Any]] = []
    for checkpoint in ordered:
        round_index = checkpoint.get("round_index", checkpoint.get("round"))
        if isinstance(round_index, bool) or not isinstance(round_index, int) or round_index < 1:
            raise ValueError("checkpoint round_index must be a positive integer")
        if round_index in seen_rounds:
            raise ValueError(f"duplicate checkpoint round_index: {round_index}")
        seen_rounds.add(round_index)
        score = score_checkpoint(
            task,
            checkpoint,
            trajectory_id=trajectory_id,
        )
        if task_record.get("split") is not None:
            score["split"] = str(task_record["split"])
        scores.append(score)
    return scores


# Names kept intentionally explicit for scripts and external analysis notebooks.
score_task_answer = score_task
score_task_record = score_trajectory


__all__ = [
    "SCORER_VERSION",
    "extract_final_answer",
    "normalize_math_answer",
    "score_checkpoint",
    "score_math",
    "score_task",
    "score_task_answer",
    "score_task_record",
    "score_trajectory",
]
