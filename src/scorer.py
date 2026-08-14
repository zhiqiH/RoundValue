"""Deterministic, dependency-free offline scorers for RoundValue checkpoints.

The public functions in this module operate on plain JSON-shaped dictionaries.
They deliberately never contact a model provider and never mutate a saved
trajectory.  A scorer may therefore be used during ``fit``, ``evaluate``, or
``reproduce`` without creating a new experiment run.

Supported task shapes are intentionally small:

* Math tasks carry ``reference_answer`` (or ``accepted_answers``) and are
  scored by conservative normalized / numeric equivalence.
* Code tasks either carry an ``entry_point`` and public ``test_cases``, or use
  the private ``evalplus_differential_v1`` fields for pinned EvalPlus
  base/plus-oracle comparison.  Local execution is opt-in; candidates run in
  a temporary-cwd child process with a secret-stripped environment and a
  wall-clock timeout.  This is not a security boundary for hostile code.

The canonical trajectory shape is documented in :func:`score_trajectory`,
but compatibility readers also accept common older flat checkpoint fields.
"""

from __future__ import annotations

import ast
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import unicodedata
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation, localcontext
from typing import Any

SCORER_VERSION = "roundvalue-offline-scorer-v3"
_FINAL_ANSWER_RE = re.compile(r"(?:final[_ ]?answer|answer)\s*:\s*([^\r\n]+)", re.I)
_FENCE_RE = re.compile(r"```(?:python|py)?\s*\r?\n(.*?)```", re.I | re.S)
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
_EVALPLUS_REFERENCE_TIMEOUT_SECONDS = 180.0


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


def extract_code(value: Any) -> str | None:
    """Extract a code answer, preferring a fenced Python block when supplied."""

    if isinstance(value, Mapping):
        direct = _nested_value(
            value,
            (
                ("code",),
                ("final_answer",),
                ("answer",),
                ("writer", "final_answer"),
                ("output", "code"),
                ("output", "final_answer"),
            ),
        )
        if direct is not None:
            return extract_code(direct)
    text = _string(value)
    if text is None:
        return None
    fences = _FENCE_RE.findall(text)
    return fences[-1].strip() if fences else text.strip()


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


def normalize_math_answer(value: str) -> str:
    """Conservatively normalize a math answer before an equivalence check."""

    text = unicodedata.normalize("NFKC", value)
    text = _strip_math_delimiters(text)
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    text = text.replace("\\,", "").replace("\\!", "")
    text = _THOUSANDS_SEPARATOR_RE.sub(
        lambda match: match.group(0).replace(",", ""), text
    )
    text = "".join(text.split())
    if text.endswith(".") and not re.fullmatch(r"[+-]?\d+\.", text):
        text = text[:-1]
    return text.casefold()


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


_SIMPLE_FIXTURE_ALLOWED_MODULES = (
    "collections",
    "functools",
    "itertools",
    "math",
    "re",
    "string",
)

# Candidate code for the formal EvalPlus runs may import only these modules.
# The set covers every module used by the pinned canonical solutions and a
# few benign standard-library additions.  ``sys`` is exposed through a
# read-only facade that blocks its module/loader/tracing attributes.
# Deliberately excluded: ``os``, ``io``, ``subprocess``, ``socket``,
# ``pathlib``, ``importlib``, ``ctypes``, ``pickle``, ``numpy``, and similar
# process/file/loader surfaces.  Reference oracles and trusted test programs
# do not use this allowlist.
_CANDIDATE_ALLOWED_MODULES = (
    "bisect",
    "cmath",
    "collections",
    "copy",
    "dataclasses",
    "datetime",
    "decimal",
    "enum",
    "fractions",
    "functools",
    "hashlib",
    "heapq",
    "itertools",
    "math",
    "operator",
    "random",
    "re",
    "statistics",
    "string",
    "sys",
    "typing",
)

# Shared restricted-execution prologue embedded in every code-evaluator
# child process.  Candidate source is parsed, rejected on forbidden syntax or
# unapproved imports, and executed with a reduced builtins namespace plus a
# dunder-attribute ban.  ``eval`` is replaced by an arithmetic-only evaluator;
# ``exec``/``compile``/``open``/``__import__`` are unreachable.  This is
# defense-in-depth for untrusted model output, not an OS security boundary;
# callers must keep the explicit local-execution opt-in and run untrusted code
# in a real sandbox when that is required.
_CODE_GUARD_TEMPLATE = r"""
import ast


FORBIDDEN_NODES = (
    ast.AsyncFunctionDef,
    ast.AsyncWith,
    ast.Await,
    ast.ClassDef,
    ast.Delete,
    ast.Global,
    ast.Nonlocal,
    ast.With,
)
FORBIDDEN_ATTRIBUTES = {
    "modules", "settrace", "setprofile", "addaudithook", "monitoring",
}


def load_candidate(source, entry_point, allowed_modules):
    import importlib as _importlib
    import sys as _worker_sys

    safe_modules = {}
    for name in allowed_modules:
        safe_modules[name] = _importlib.import_module(name)

    class _SafeSys:
        __slots__ = ()

        def __getattr__(self, name):
            if name in FORBIDDEN_ATTRIBUTES:
                raise AttributeError("forbidden attribute: sys." + name)
            return getattr(_worker_sys, name)

        def __setattr__(self, name, value):
            raise AttributeError("sys attributes are read-only in candidate code")

    _safe_sys = _SafeSys()

    def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        root = name.split(".", 1)[0]
        if level != 0 or root not in safe_modules:
            raise ImportError("only approved modules may be imported")
        if root == "sys":
            return _safe_sys
        return __import__(name, globals, locals, fromlist, level)

    def safe_eval(source):
        # Evaluate only numeric literals and arithmetic operators.
        tree = ast.parse(source, mode="eval")
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant):
                if isinstance(node.value, bool) or not isinstance(
                    node.value, (int, float, complex)
                ):
                    raise ValueError("unsupported numeric literal")
            elif not isinstance(
                node,
                (
                    ast.Expression,
                    ast.BinOp,
                    ast.UnaryOp,
                    ast.operator,
                    ast.unaryop,
                    ast.Load,
                ),
            ):
                raise ValueError("unsupported eval expression")
        return eval(compile(tree, "<safe-eval>", "eval"), {"__builtins__": {}})

    safe_builtins = {
        "abs": abs, "all": all, "any": any, "AssertionError": AssertionError,
        "ascii": ascii, "bin": bin, "bool": bool, "bytes": bytes, "bytearray": bytearray,
        "callable": callable, "chr": chr, "complex": complex, "dict": dict,
        "dir": dir, "divmod": divmod, "enumerate": enumerate, "Exception": Exception,
        "eval": safe_eval, "filter": filter, "float": float, "format": format,
        "frozenset": frozenset, "hash": hash, "hex": hex, "int": int,
        "hasattr": hasattr, "id": id, "isinstance": isinstance, "iter": iter,
        "len": len, "list": list,
        "map": map, "max": max, "min": min, "next": next, "object": object,
        "oct": oct, "ord": ord, "pow": pow, "print": print, "range": range, "repr": repr,
        "reversed": reversed, "round": round, "set": set, "slice": slice,
        "sorted": sorted, "str": str, "sum": sum, "tuple": tuple,
        "type": type, "TypeError": TypeError, "ValueError": ValueError,
        "zip": zip, "__import__": safe_import,
    }

    class Guard(ast.NodeVisitor):
        def visit(self, node):
            if isinstance(node, FORBIDDEN_NODES):
                raise ValueError("forbidden syntax: " + type(node).__name__)
            return super().visit(node)

        def visit_Name(self, node):
            if node.id.startswith("__"):
                raise ValueError("forbidden name")
            self.generic_visit(node)

        def visit_Attribute(self, node):
            if node.attr.startswith("_") or node.attr in FORBIDDEN_ATTRIBUTES:
                raise ValueError("forbidden attribute")
            self.generic_visit(node)

        def visit_Import(self, node):
            if any(
                alias.name.split(".", 1)[0] not in safe_modules
                or any(segment in FORBIDDEN_ATTRIBUTES for segment in alias.name.split("."))
                for alias in node.names
            ):
                raise ValueError("unapproved import")
            self.generic_visit(node)

        def visit_ImportFrom(self, node):
            root = (node.module or "").split(".", 1)[0]
            if (
                node.level != 0
                or root not in safe_modules
                or any(
                    segment in FORBIDDEN_ATTRIBUTES
                    for segment in (node.module or "").split(".")
                )
                or any(alias.name in FORBIDDEN_ATTRIBUTES for alias in node.names)
            ):
                raise ValueError("unapproved import")
            self.generic_visit(node)

    tree = ast.parse(source, mode="exec")
    Guard().visit(tree)
    namespace = {"__builtins__": safe_builtins, "__name__": "candidate"}
    exec(compile(tree, "<candidate>", "exec"), namespace, namespace)
    function = namespace.get(entry_point)
    if not callable(function):
        raise ValueError("entry_point_not_callable")
    return function
"""


_CODE_WORKER = _CODE_GUARD_TEMPLATE + r"""
import ast
import contextlib
import io
import json
import math
import sys


def equivalent(actual, expected, abs_tol, rel_tol):
    if isinstance(actual, bool) or isinstance(expected, bool):
        return type(actual) is type(expected) and actual == expected
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return math.isclose(float(actual), float(expected), abs_tol=abs_tol, rel_tol=rel_tol)
    if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        return len(actual) == len(expected) and all(
            equivalent(left, right, abs_tol, rel_tol) for left, right in zip(actual, expected)
        )
    if isinstance(actual, dict) and isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            equivalent(actual[key], expected[key], abs_tol, rel_tol) for key in actual
        )
    return actual == expected


def main():
    payload = json.loads(sys.stdin.read())
    sink = io.StringIO()
    try:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            function = load_candidate(
                payload["source"], payload["entry_point"], payload.get("allowed_modules", [])
            )
    except ValueError as error:
        print(json.dumps({"status": "rejected", "detail": str(error)}))
        return
    except BaseException as error:
        print(json.dumps({"status": "error", "detail": type(error).__name__}))
        return
    passed = 0
    tests = payload["tests"]
    for index, test in enumerate(tests):
        args = test.get("args", [])
        kwargs = test.get("kwargs", {})
        try:
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                actual = function(*args, **kwargs)
        except BaseException as error:
            print(json.dumps({
                "status": "failed", "passed": passed, "total": len(tests),
                "failed_test": index, "detail": "raised_" + type(error).__name__,
            }))
            return
        if not equivalent(actual, test["expected"], payload["abs_tol"], payload["rel_tol"]):
            print(json.dumps({
                "status": "failed", "passed": passed, "total": len(tests),
                "failed_test": index, "detail": "wrong_output",
            }))
            return
        passed += 1
    print(json.dumps({"status": "passed", "passed": passed, "total": len(tests)}))


if __name__ == "__main__":
    main()
"""


# EvalPlus ships each task's complete, trusted test program instead of a small
# JSON args/expected fixture.  Keep the test program intact: flattening it
# would lose its comparison, mutation, and special-type semantics.  Candidate
# code runs through the shared ``load_candidate`` guard; the trusted test
# program executes in its own namespace with normal builtins.  This is
# deliberately *not* a security sandbox; callers must retain the explicit
# local-execution opt-in.
_EVALPLUS_WORKER = _CODE_GUARD_TEMPLATE + r"""
import ast
import contextlib
import io
import json
import sys

try:
    import numpy
except BaseException:
    numpy = None


def emit(value):
    sys.stdout.write(json.dumps(value, ensure_ascii=False) + "\n")


def main():
    payload = json.loads(sys.stdin.read())
    if numpy is None:
        emit({"status": "error", "detail": "numpy_not_installed"})
        return
    captured = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            candidate = load_candidate(
                payload["source"], payload["entry_point"], payload.get("allowed_modules", [])
            )
    except ValueError as error:
        emit({"status": "rejected", "detail": str(error)})
        return
    except BaseException as error:
        emit({"status": "error", "detail": "candidate_" + type(error).__name__})
        return
    try:
        setup = "\n".join(payload.get("test_imports", []))
        test = payload["test"]
        test_namespace = {
            "__builtins__": __builtins__,
            "__name__": "evalplus_test",
            "candidate": candidate,
        }
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            if setup:
                exec(compile(setup, "<evalplus-test-imports>", "exec"), test_namespace, test_namespace)
            exec(compile(test, "<evalplus-test>", "exec"), test_namespace, test_namespace)
            check = test_namespace.get("check")
            if callable(check):
                check(candidate)
    except AssertionError:
        emit({"status": "failed", "detail": "assertion_failed"})
        return
    except BaseException as error:
        emit({"status": "failed", "detail": "test_" + type(error).__name__})
        return
    emit({"status": "passed"})


if __name__ == "__main__":
    main()
"""


# The official EvalPlus releases contain differential-test inputs and a
# canonical oracle rather than only a fixture with expected JSON values.  This
# worker mirrors the public v0.3.1 comparison semantics closely enough for the
# RoundValue protocol while keeping candidate code and the private oracle in
# distinct namespaces.  The trusted oracle executes with normal builtins;
# model-generated candidate code executes through the shared restricted
# ``load_candidate`` guard.  It is intentionally a local-execution adapter,
# not a replacement for EvalPlus's container/leaderboard runner.
_EVALPLUS_DIFFERENTIAL_WORKER = _CODE_GUARD_TEMPLATE + r"""
import ast
import contextlib
import copy
import io
import json
import math
import os
import sys
import threading

try:
    import numpy as np
except BaseException:
    np = None

CAPTURED = io.StringIO()
EXIT_REFERENCE_TIMEOUT = 42
EXIT_CANDIDATE_TIMEOUT = 43


class PhaseWatchdog:
    # Abruptly exit this throwaway worker when a phase exceeds its budget.
    # ``os._exit`` interrupts even a C-level loop such as ``sum(range(...))``,
    # which a Python exception raised from a signal handler cannot reliably
    # do.  The parent maps the exit code back to the timed-out phase.

    def __init__(self, seconds, exit_code):
        self.seconds = seconds
        self.exit_code = exit_code
        self.ready = threading.Event()
        self.thread = None

    def __enter__(self):
        def run():
            if not self.ready.wait(self.seconds):
                os._exit(self.exit_code)

        self.thread = threading.Thread(
            target=run, name="roundvalue-phase-watchdog", daemon=True
        )
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.ready.set()
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        return False


MBPP_OUTPUT_NOT_NONE_TASKS = {"check_str", "text_match_three", "text_starta_endb"}
MBPP_OUTPUT_SET_EQ_TASKS = {
    "similar_elements", "find_char_long", "common_in_nested_lists", "extract_singly",
    "larg_nnum", "intersection_array", "find_dissimilar", "Diff",
}

def emit(value):
    sys.stdout.write(json.dumps(value, ensure_ascii=False) + "\n")

def is_floats(value):
    if isinstance(value, float):
        return True
    if isinstance(value, (list, tuple)) and value:
        return all(isinstance(item, float) for item in value)
    if isinstance(value, np.ndarray):
        return value.dtype == np.float64 or value.dtype == np.float32
    return False

def surface_area(base_edge, height):
    slant_height = math.sqrt((base_edge / 2) ** 2 + height**2)
    return round(base_edge**2 + 4 * (base_edge * slant_height) / 2)

def digit_distance_nums(num1, num2):
    left, right = str(num1), str(num2)
    length = max(len(left), len(right))
    return sum(abs(int(a) - int(b)) for a, b in zip(left.zfill(length), right.zfill(length)))

def poly(coefficients, value):
    return sum(coefficient * math.pow(value, power) for power, coefficient in enumerate(coefficients))

def mbpp_deserialize_inputs(task_id, inputs):
    task_number = int(task_id.split("/")[-1])
    if task_number in {
        2, 116, 132, 143, 222, 261, 273, 394, 399, 421, 424, 429,
        470, 560, 579, 596, 616, 630, 726, 740, 744, 809,
    }:
        return [[tuple(item) for item in item_input] for item_input in inputs]
    if task_number in {63, 64, 70, 94, 120, 237, 272, 299, 400, 409, 417, 438, 473, 614, 780}:
        return [[[tuple(item) for item in item_list] for item_list in item_input] for item_input in inputs]
    if task_number in {75, 413, 444, 753}:
        return [[[tuple(item) for item in item_input[0]]] + [item_input[1]] for item_input in inputs]
    if task_number in {106, 750}:
        return [[item_input[0]] + [tuple(item_input[1])] for item_input in inputs]
    if task_number == 115:
        return [
            [[set(item) if isinstance(item, list) and len(item) else {} for item in item_input[0]]]
            for item_input in inputs
        ]
    if task_number == 124:
        return [(float(item_input[0]), complex(item_input[1])) for item_input in inputs]
    if task_number in {250, 405, 446, 617, 720, 763, 808}:
        return [[tuple(item_input[0])] + [item_input[1]] for item_input in inputs]
    if task_number in {259, 401, 445}:
        converted = [[[tuple(item) for item in item_list] for item_list in item_input] for item_input in inputs]
        return [[tuple(item) for item in item_input] for item_input in converted]
    if task_number == 278:
        converted = [
            [[tuple(item) if isinstance(item, list) else item for item in item_input[0]]]
            for item_input in inputs
        ]
        return [[tuple(item) for item in item_input] for item_input in converted]
    if task_number == 307:
        return [[tuple(item_input[0])] + [item_input[1], item_input[2]] for item_input in inputs]
    if task_number == 722:
        return [[{key: tuple(value) for key, value in item_input[0].items()}] + item_input[1:] for item_input in inputs]
    if task_number == 252:
        return [[complex(item_input[0])] for item_input in inputs]
    if task_number in {580, 615, 791}:
        def lists_to_tuples(value):
            if isinstance(value, list):
                return tuple(lists_to_tuples(item) for item in value)
            return value
        return [lists_to_tuples(item_input) for item_input in inputs]
    return inputs

def reference_outputs(function, inputs, output_not_none):
    outputs = []
    for item_input in inputs:
        output = function(*copy.deepcopy(item_input))
        outputs.append(output is not None if output_not_none else output)
    return outputs

def check_outputs(dataset, entry_point, function, inputs, expected, atol):
    effective_atol = atol
    for index, (item_input, expected_output) in enumerate(zip(inputs, expected)):
        try:
            with contextlib.redirect_stdout(CAPTURED), contextlib.redirect_stderr(CAPTURED):
                output = function(*copy.deepcopy(item_input))
            exact_match = output == expected_output
            if dataset == "mbpp":
                if entry_point == "are_equivalent":
                    exact_match = exact_match or True
                elif entry_point == "sum_div":
                    exact_match = exact_match or output == 0
                elif entry_point == "surface_Area":
                    exact_match = exact_match or abs(output - surface_area(*item_input)) <= effective_atol
                elif entry_point == "digit_distance_nums":
                    exact_match = exact_match or output == digit_distance_nums(*item_input)
                elif entry_point in MBPP_OUTPUT_SET_EQ_TASKS:
                    exact_match = set(output) == set(expected_output)
                elif entry_point in MBPP_OUTPUT_NOT_NONE_TASKS:
                    if isinstance(output, bool):
                        exact_match = output == expected_output
                    else:
                        exact_match = expected_output == (output is not None)
            if dataset == "humaneval" and entry_point == "find_zero":
                assert abs(poly(*item_input, output)) <= effective_atol
                continue
            if effective_atol == 0 and is_floats(expected_output):
                effective_atol = 1e-6
            if not exact_match and effective_atol != 0:
                assert type(output) == type(expected_output)
                if isinstance(expected_output, (list, tuple)):
                    assert len(output) == len(expected_output)
                assert np.allclose(output, expected_output, rtol=1e-07, atol=effective_atol)
            else:
                assert exact_match
        except BaseException as error:
            return False, index, type(error).__name__
    return True, len(inputs), None

def main():
    if np is None:
        emit({"status": "error", "detail": "numpy_not_installed"})
        return
    payload = json.loads(sys.stdin.read())
    dataset = payload["dataset"]
    entry_point = payload["entry_point"]
    reference_seconds = float(payload.get("reference_timeout_seconds", 180.0))
    candidate_seconds = float(payload.get("candidate_timeout_seconds", 10.0))
    try:
        with PhaseWatchdog(reference_seconds, EXIT_REFERENCE_TIMEOUT):
            base_inputs = payload["base_inputs"]
            plus_inputs = payload["plus_inputs"]
            if dataset == "mbpp":
                base_inputs = mbpp_deserialize_inputs(payload["source_task_id"], base_inputs)
                plus_inputs = mbpp_deserialize_inputs(payload["source_task_id"], plus_inputs)
            reference_namespace = {"__name__": "reference"}
            with contextlib.redirect_stdout(CAPTURED), contextlib.redirect_stderr(CAPTURED):
                exec(compile(payload["reference_code"], "<evalplus-reference>", "exec"), reference_namespace, reference_namespace)
            reference_function = reference_namespace.get(entry_point)
            if not callable(reference_function):
                raise ValueError("reference_entry_point_not_callable")
            output_not_none = dataset == "mbpp" and entry_point in MBPP_OUTPUT_NOT_NONE_TASKS
            expected_base = reference_outputs(reference_function, base_inputs, output_not_none)
            expected_plus = reference_outputs(reference_function, plus_inputs, output_not_none)
    except BaseException as error:
        emit({"status": "error", "detail": "reference_" + type(error).__name__})
        return
    try:
        with PhaseWatchdog(candidate_seconds, EXIT_CANDIDATE_TIMEOUT):
            with contextlib.redirect_stdout(CAPTURED), contextlib.redirect_stderr(CAPTURED):
                candidate_function = load_candidate(
                    payload["source"], entry_point, payload.get("allowed_modules", [])
                )
            tolerance = float(payload["atol"])
            base_ok, base_count, base_detail = check_outputs(
                dataset, entry_point, candidate_function, base_inputs, expected_base, tolerance
            )
            if not base_ok:
                emit({"status": "failed", "split": "base", "failed_test": base_count, "detail": base_detail})
                return
            plus_ok, plus_count, plus_detail = check_outputs(
                dataset, entry_point, candidate_function, plus_inputs, expected_plus, tolerance
            )
            if not plus_ok:
                emit({"status": "failed", "split": "plus", "failed_test": plus_count, "detail": plus_detail})
                return
    except ValueError as error:
        emit({"status": "rejected", "detail": str(error)})
        return
    except BaseException as error:
        emit({"status": "rejected", "detail": "candidate_" + type(error).__name__})
        return
    emit({"status": "passed", "base_count": len(base_inputs), "plus_count": len(plus_inputs)})

if __name__ == "__main__":
    main()
"""


def _code_tests(task: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Normalize simple public function-test fixtures to a child-safe shape."""

    raw: Any = task.get("test_cases", task.get("tests"))
    scoring = _mapping(task.get("scoring"))
    if raw is None and scoring is not None:
        raw = scoring.get("test_cases", scoring.get("tests"))
    if not isinstance(raw, Sequence) or isinstance(raw, str | bytes | bytearray):
        return []
    normalized: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            return []
        if "expected" in item:
            expected = item["expected"]
        elif "expected_output" in item:
            expected = item["expected_output"]
        elif "output" in item:
            expected = item["output"]
        elif "returns" in item:
            expected = item["returns"]
        else:
            return []
        args = item.get("args")
        if args is None:
            args = [item["input"]] if "input" in item else []
        kwargs = item.get("kwargs", {})
        if not isinstance(args, list) or not isinstance(kwargs, Mapping):
            return []
        # JSON round-tripping both validates fixture serializability and detaches
        # it from the caller's mutable record.
        try:
            safe = json.loads(
                json.dumps({"args": args, "kwargs": dict(kwargs), "expected": expected})
            )
        except (TypeError, ValueError):
            return []
        normalized.append(safe)
    return normalized


def _entry_point(task: Mapping[str, Any]) -> str | None:
    scoring = _mapping(task.get("scoring")) or {}
    value = task.get("entry_point", task.get("function_name", scoring.get("entry_point")))
    text = _string(value)
    return text if text and text.isidentifier() else None


def _evalplus_test_program(task: Mapping[str, Any]) -> tuple[str, list[str]] | None:
    """Return a validated embedded EvalPlus test program, if this task uses one."""

    if task.get("code_evaluator") != "evalplus_embedded_v1":
        return None
    test = _string(task.get("evalplus_test"))
    raw_imports = task.get("evalplus_test_imports", [])
    if test is None or not isinstance(raw_imports, Sequence) or isinstance(
        raw_imports, str | bytes | bytearray
    ):
        return None
    imports: list[str] = []
    for value in raw_imports:
        text = _string(value)
        if text is None:
            return None
        imports.append(text)
    return test, imports


def _evalplus_differential_data(task: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return validated private fields for an official EvalPlus differential task."""

    if task.get("code_evaluator") != "evalplus_differential_v1":
        return None
    dataset = _string(task.get("evalplus_dataset"))
    reference_code = _string(task.get("evalplus_reference_code"))
    base_inputs = task.get("evalplus_base_inputs")
    plus_inputs = task.get("evalplus_plus_inputs")
    try:
        atol = float(task.get("evalplus_atol"))
    except (TypeError, ValueError):
        return None
    metadata = _mapping(task.get("public_metadata")) or {}
    source_task_id = _string(metadata.get("source_task_id")) or _string(task.get("task_id"))
    if (
        dataset not in {"humaneval", "mbpp"}
        or reference_code is None
        or source_task_id is None
        or not isinstance(base_inputs, list)
        or not isinstance(plus_inputs, list)
        or not math.isfinite(atol)
        or atol < 0
    ):
        return None
    return {
        "dataset": dataset,
        "reference_code": reference_code,
        "base_inputs": base_inputs,
        "plus_inputs": plus_inputs,
        "atol": atol,
        "source_task_id": source_task_id,
    }


def _positive_float(value: Any, default: float, maximum: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return min(maximum, result) if math.isfinite(result) and result > 0 else default


def _nonnegative_float(value: Any, default: float, maximum: float) -> float:
    """Parse a bounded tolerance while preserving an explicit exact-zero value."""

    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return min(maximum, result) if math.isfinite(result) and result >= 0 else default


def _code_runner_environment(temp_directory: str) -> dict[str, str]:
    """Build the child environment without inheriting user credentials.

    The candidate process receives only a small set of operating-system values
    required to start Python.  In particular, it does not inherit the parent
    environment, ``PYTHONPATH``, or any provider/API key.  This deliberately
    reduces accidental exposure; it is not an OS-level security sandbox.
    """

    environment: dict[str, str] = {
        "PYTHONIOENCODING": "utf-8",
        "TMP": temp_directory,
        "TEMP": temp_directory,
        "TMPDIR": temp_directory,
        "HOME": temp_directory,
        "USERPROFILE": temp_directory,
        "APPDATA": temp_directory,
        "LOCALAPPDATA": temp_directory,
    }
    # These are operating-system loader/shell settings, not user secrets.  Do
    # not copy PATH: the executable is invoked by absolute path and a reduced
    # child environment is safer than an inherited one.
    for name in ("SystemRoot", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "LANG", "LC_ALL", "TZ"):
        value = os.environ.get(name)
        if value:
            environment[name] = value
    return environment


def _score_evalplus_embedded(
    task: Mapping[str, Any],
    code: str,
    *,
    timeout_seconds: float,
    allow_local_code_execution: bool,
) -> dict[str, Any]:
    """Evaluate one checkpoint with a bundled, pinned EvalPlus test program.

    The published HumanEval+ and MBPP+ assets store their complete test program
    as Python source.  We execute that trusted program against untrusted model
    code in the same secret-stripped child-process model as ordinary fixtures.
    The implementation intentionally reports its evaluator identity so results
    cannot be mistaken for a different test suite or a security sandbox.
    """

    entry_point = _entry_point(task)
    program = _evalplus_test_program(task)
    if entry_point is None:
        return _result(task, domain="code", answer=code, quality=0.0, reason="missing_entry_point")
    if program is None:
        return _result(
            task,
            domain="code",
            answer=code,
            quality=0.0,
            reason="missing_evalplus_test_program",
        )
    if not allow_local_code_execution:
        return _result(
            task,
            domain="code",
            answer=code,
            quality=None,
            reason="local_code_execution_disabled",
            extra={
                "scorer_status": "not_evaluated",
                "evaluator": "evalplus_embedded_v1",
                "execution_isolation": "disabled_by_default",
            },
        )
    test, test_imports = program
    scoring = _mapping(task.get("scoring")) or {}
    timeout = _positive_float(scoring.get("timeout_seconds", timeout_seconds), timeout_seconds, 30.0)
    payload = {
        "source": code,
        "entry_point": entry_point,
        "test": test,
        "test_imports": test_imports,
        "allowed_modules": _CANDIDATE_ALLOWED_MODULES,
    }
    try:
        with tempfile.TemporaryDirectory(prefix="roundvalue-evalplus-") as temporary_cwd:
            completed = subprocess.run(
                [sys.executable, "-I", "-c", _EVALPLUS_WORKER],
                input=json.dumps(payload),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                check=False,
                cwd=temporary_cwd,
                env=_code_runner_environment(temporary_cwd),
            )
    except subprocess.TimeoutExpired:
        return _result(task, domain="code", answer=code, quality=0.0, reason="evalplus_timeout")
    except OSError as error:
        return _result(
            task,
            domain="code",
            answer=code,
            quality=0.0,
            reason="evalplus_runner_error",
            extra={"runner_error": type(error).__name__},
        )
    try:
        response = json.loads(completed.stdout)
    except json.JSONDecodeError:
        response = {"status": "error", "detail": "invalid_runner_response"}
    if not isinstance(response, Mapping):
        response = {"status": "error", "detail": "invalid_runner_response"}
    status = str(response.get("status", "error"))
    extra = {
        "evaluator": "evalplus_embedded_v1",
        "execution_isolation": "temporary_cwd_and_secret_stripped_environment_not_security_sandbox",
    }
    metadata = _mapping(task.get("public_metadata")) or {}
    for key in ("source_dataset", "dataset_version"):
        if metadata.get(key) is not None:
            extra[key] = str(metadata[key])
    if status == "passed":
        return _result(
            task,
            domain="code",
            answer=code,
            quality=1.0,
            reason="evalplus_tests_passed",
            extra=extra,
        )
    detail = str(response.get("detail", "evalplus_test_failure"))
    reason = {
        "rejected": "evalplus_candidate_rejected",
        "failed": "evalplus_test_failure",
    }.get(status, "evalplus_evaluator_error")
    extra["runner_detail"] = detail
    return _result(task, domain="code", answer=code, quality=0.0, reason=reason, extra=extra)


def _score_evalplus_differential(
    task: Mapping[str, Any],
    code: str,
    *,
    timeout_seconds: float,
    allow_local_code_execution: bool,
) -> dict[str, Any]:
    """Score a candidate with pinned official EvalPlus base/plus inputs.

    Expected outputs are recomputed from the private canonical oracle inside a
    fresh child process.  The oracle and the candidate run in two separately
    time-boxed phases of the same worker: a slow official reference (for
    example Mbpp/599) gets a generous budget, while untrusted candidate code
    keeps the short configured timeout.  The candidate receives neither the
    oracle source nor the hidden inputs through its agent-facing task view.
    This deliberately has a distinct result identity from the upstream
    EvalPlus runner: its process isolation and timing policy are RoundValue's
    local adapter policy.
    """

    entry_point = _entry_point(task)
    data = _evalplus_differential_data(task)
    if entry_point is None:
        return _result(task, domain="code", answer=code, quality=0.0, reason="missing_entry_point")
    if data is None:
        return _result(
            task,
            domain="code",
            answer=code,
            quality=0.0,
            reason="missing_evalplus_differential_data",
        )
    if not allow_local_code_execution:
        return _result(
            task,
            domain="code",
            answer=code,
            quality=None,
            reason="local_code_execution_disabled",
            extra={
                "scorer_status": "not_evaluated",
                "evaluator": "evalplus_differential_v1",
                "execution_isolation": "disabled_by_default",
            },
        )
    scoring = _mapping(task.get("scoring")) or {}
    timeout = _positive_float(scoring.get("timeout_seconds", timeout_seconds), timeout_seconds, 60.0)
    reference_timeout = _positive_float(
        scoring.get("reference_timeout_seconds", _EVALPLUS_REFERENCE_TIMEOUT_SECONDS),
        _EVALPLUS_REFERENCE_TIMEOUT_SECONDS,
        3600.0,
    )
    payload = {
        "source": code,
        "entry_point": entry_point,
        "allowed_modules": _CANDIDATE_ALLOWED_MODULES,
        "candidate_timeout_seconds": timeout,
        "reference_timeout_seconds": reference_timeout,
        **data,
    }
    try:
        with tempfile.TemporaryDirectory(prefix="roundvalue-evalplus-differential-") as temporary_cwd:
            completed = subprocess.run(
                [sys.executable, "-I", "-c", _EVALPLUS_DIFFERENTIAL_WORKER],
                input=json.dumps(payload),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout + reference_timeout + 15,
                check=False,
                cwd=temporary_cwd,
                env=_code_runner_environment(temporary_cwd),
            )
    except subprocess.TimeoutExpired:
        return _result(
            task,
            domain="code",
            answer=code,
            quality=0.0,
            reason="evalplus_differential_timeout",
            extra={"evaluator": "evalplus_differential_v1"},
        )
    except OSError as error:
        return _result(
            task,
            domain="code",
            answer=code,
            quality=0.0,
            reason="evalplus_differential_runner_error",
            extra={"runner_error": type(error).__name__, "evaluator": "evalplus_differential_v1"},
        )
    if completed.returncode == 42:
        return _result(
            task,
            domain="code",
            answer=code,
            quality=0.0,
            reason="evalplus_reference_timeout",
            extra={
                "evaluator": "evalplus_differential_v1",
                "reference_timeout_seconds": reference_timeout,
            },
        )
    if completed.returncode == 43:
        return _result(
            task,
            domain="code",
            answer=code,
            quality=0.0,
            reason="evalplus_differential_timeout",
            extra={
                "evaluator": "evalplus_differential_v1",
                "candidate_timeout_seconds": timeout,
            },
        )
    try:
        response = json.loads(completed.stdout)
    except json.JSONDecodeError:
        response = {"status": "error", "detail": "invalid_runner_response"}
    if not isinstance(response, Mapping):
        response = {"status": "error", "detail": "invalid_runner_response"}
    status = str(response.get("status", "error"))
    extra: dict[str, Any] = {
        "evaluator": "evalplus_differential_v1",
        "execution_isolation": "temporary_cwd_and_secret_stripped_environment_not_security_sandbox",
        "evaluation_protocol": "official_evalplus_release_base_plus_oracle_roundvalue_adapter",
    }
    metadata = _mapping(task.get("public_metadata")) or {}
    for key in ("source_dataset", "dataset_version", "source_task_id"):
        if metadata.get(key) is not None:
            extra[key] = str(metadata[key])
    for key in ("split", "failed_test", "base_count", "plus_count"):
        if key in response:
            extra[f"evalplus_{key}"] = response[key]
    if status == "passed":
        return _result(
            task,
            domain="code",
            answer=code,
            quality=1.0,
            reason="evalplus_differential_passed",
            extra=extra,
        )
    detail = str(response.get("detail", "evalplus_differential_failure"))
    extra["runner_detail"] = detail
    reason = {
        "rejected": "evalplus_candidate_error",
        "failed": "evalplus_differential_failure",
    }.get(status, "evalplus_differential_error")
    return _result(task, domain="code", answer=code, quality=0.0, reason=reason, extra=extra)


def score_code(
    task: Mapping[str, Any],
    final_answer: Any,
    *,
    timeout_seconds: float = 3.0,
    max_code_chars: int = 30_000,
    allow_local_code_execution: bool = False,
) -> dict[str, Any]:
    """Run a code benchmark's local evaluator against a Writer answer.

    Local execution is opt-in because generated code is untrusted input.  When
    enabled, the candidate runs in a temporary working directory with a
    secret-stripped environment and a wall-clock timeout.  Every evaluator path
    now applies the same guard to model-generated code: a bounded AST policy,
    reduced builtins (with an arithmetic-only ``eval``), a dunder-attribute
    ban, and a curated module allowlist.  Trusted canonical oracles and test
    programs are exempt from that guard.  This is defense-in-depth, not an
    OS/container security sandbox.
    """

    code = extract_code(final_answer)
    if code is None:
        return _result(task, domain="code", answer=None, quality=0.0, reason="missing_final_answer")
    if len(code) > max_code_chars:
        return _result(task, domain="code", answer=code, quality=0.0, reason="code_too_large")
    if task.get("code_evaluator") == "evalplus_differential_v1":
        return _score_evalplus_differential(
            task,
            code,
            timeout_seconds=timeout_seconds,
            allow_local_code_execution=allow_local_code_execution,
        )
    if task.get("code_evaluator") == "evalplus_embedded_v1":
        return _score_evalplus_embedded(
            task,
            code,
            timeout_seconds=timeout_seconds,
            allow_local_code_execution=allow_local_code_execution,
        )
    entry_point = _entry_point(task)
    if entry_point is None:
        return _result(task, domain="code", answer=code, quality=0.0, reason="missing_entry_point")
    tests = _code_tests(task)
    if not tests:
        return _result(task, domain="code", answer=code, quality=0.0, reason="missing_public_tests")
    if not allow_local_code_execution:
        return _result(
            task,
            domain="code",
            answer=code,
            quality=None,
            reason="local_code_execution_disabled",
            extra={
                "scorer_status": "not_evaluated",
                "execution_isolation": "disabled_by_default",
            },
        )
    scoring = _mapping(task.get("scoring")) or {}
    payload = {
        "source": code,
        "entry_point": entry_point,
        "tests": tests,
        "allowed_modules": _SIMPLE_FIXTURE_ALLOWED_MODULES,
        "abs_tol": _nonnegative_float(scoring.get("abs_tol", task.get("abs_tol", 1e-9)), 1e-9, 1.0),
        "rel_tol": _nonnegative_float(scoring.get("rel_tol", task.get("rel_tol", 1e-9)), 1e-9, 1.0),
    }
    timeout = _positive_float(timeout_seconds, 3.0, 30.0)
    try:
        # ``TemporaryDirectory`` gives the evaluator a cwd outside the project
        # tree.  ``env=...`` replaces, rather than augments, the parent
        # environment, so API credentials cannot leak through child variables.
        with tempfile.TemporaryDirectory(prefix="roundvalue-code-") as temporary_cwd:
            completed = subprocess.run(
                [sys.executable, "-I", "-c", _CODE_WORKER],
                input=json.dumps(payload),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                check=False,
                cwd=temporary_cwd,
                env=_code_runner_environment(temporary_cwd),
            )
    except subprocess.TimeoutExpired:
        return _result(task, domain="code", answer=code, quality=0.0, reason="code_timeout")
    except OSError as error:
        return _result(
            task,
            domain="code",
            answer=code,
            quality=0.0,
            reason="code_runner_error",
            extra={"runner_error": type(error).__name__},
        )
    try:
        response = json.loads(completed.stdout)
    except json.JSONDecodeError:
        response = {"status": "error", "detail": "invalid_runner_response"}
    if not isinstance(response, Mapping):
        response = {"status": "error", "detail": "invalid_runner_response"}
    status = str(response.get("status", "error"))
    extra = {
        "tests_passed": int(response.get("passed", 0))
        if isinstance(response.get("passed", 0), int)
        else 0,
        "tests_total": len(tests),
        "execution_isolation": "temporary_cwd_and_secret_stripped_environment_not_security_sandbox",
    }
    if isinstance(response.get("failed_test"), int):
        extra["failed_test"] = int(response["failed_test"])
    if status == "passed":
        return _result(
            task, domain="code", answer=code, quality=1.0, reason="code_tests_passed", extra=extra
        )
    detail = str(response.get("detail", "code_test_failure"))
    reason = {
        "rejected": "code_rejected",
        "failed": "code_test_failure",
        "error": "code_execution_error",
    }.get(status, "code_execution_error")
    extra["runner_detail"] = detail
    return _result(task, domain="code", answer=code, quality=0.0, reason=reason, extra=extra)


def task_domain(task: Mapping[str, Any]) -> str:
    """Return the scorer domain from an explicit task field or task shape."""

    raw = task.get("domain", task.get("task_type", task.get("type", "")))
    domain = str(raw).strip().casefold()
    if domain in {"code", "programming", "python", "coding"}:
        return "code"
    if domain in {"math", "mathematics", "gsm", "algebra"}:
        return "math"
    if (
        _entry_point(task) is not None
        or task.get("test_cases") is not None
        or task.get("tests") is not None
    ):
        return "code"
    return "math"


def score_task(
    task: Mapping[str, Any],
    final_answer: Any,
    *,
    code_timeout_seconds: float = 3.0,
    allow_local_code_execution: bool = False,
) -> dict[str, Any]:
    """Dispatch a Writer answer to the task's deterministic offline scorer."""

    if not isinstance(task, Mapping):
        raise TypeError("task must be a JSON object")
    if task_domain(task) == "code":
        return score_code(
            task,
            final_answer,
            timeout_seconds=code_timeout_seconds,
            allow_local_code_execution=allow_local_code_execution,
        )
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
    code_timeout_seconds: float = 3.0,
    allow_local_code_execution: bool = False,
) -> dict[str, Any]:
    """Score a single saved Writer checkpoint and retain its stable identity."""

    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint must be a JSON object")
    score = score_task(
        task,
        _checkpoint_answer(checkpoint),
        code_timeout_seconds=code_timeout_seconds,
        allow_local_code_execution=allow_local_code_execution,
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
    *,
    code_timeout_seconds: float = 3.0,
    allow_local_code_execution: bool = False,
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
        if any(key in task_record for key in ("reference_answer", "entry_point", "test_cases")):
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
            code_timeout_seconds=code_timeout_seconds,
            allow_local_code_execution=allow_local_code_execution,
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
    "extract_code",
    "extract_final_answer",
    "normalize_math_answer",
    "score_checkpoint",
    "score_code",
    "score_math",
    "score_task",
    "score_task_answer",
    "score_task_record",
    "score_trajectory",
    "task_domain",
]
