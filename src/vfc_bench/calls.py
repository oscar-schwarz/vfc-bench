"""Turning a system's raw output into executed function calls.

The same steps apply to every system, so results stay comparable:

1. strip everything before the first ``[`` and after the last ``]``
2. parse the rest as JSON                  (fails the whole output)
3. check each item is ``{"name": str, "arguments": object}``  (per call)
4. check name and arguments against the schema                (per call)
5. execute the call in the environment                        (per call)

Execution continues after a failing call, like real devices would.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

from vfc_bench.environment import (
    Environment,
    ToolCallError,
    ToolSchemaError,
    ToolSpec,
)

__all__ = [
    "CallResult",
    "ParsedCall",
    "ParsedOutput",
    "execute_calls",
    "format_instructions",
    "parse_calls",
]

ParseStatus = Literal["ok", "no_json", "invalid_json"]
CallStatus = Literal["ok", "malformed_call", "schema_error", "rejected"]


@dataclass(frozen=True)
class ParsedCall:
    """One item of the parsed array; ``error`` is set if it is malformed."""

    raw: Any
    name: str | None = None
    arguments: dict[str, Any] | None = None
    error: str | None = None


@dataclass(frozen=True)
class ParsedOutput:
    """Result of steps 1-3."""

    status: ParseStatus
    stripped: bool = False
    """True if text had to be removed around the array (preamble, code fences)."""
    calls: tuple[ParsedCall, ...] = ()
    error: str | None = None


@dataclass(frozen=True)
class CallResult:
    """Result of steps 3-5 for one call."""

    index: int
    status: CallStatus
    name: str | None
    arguments: Any
    error: str | None = None
    result: Any = None
    """Return value of the function, e.g. a sensor reading."""


def _check_item(item: Any) -> ParsedCall:
    if not isinstance(item, dict):
        return ParsedCall(
            raw=item, error=f"expected an object, got {type(item).__name__}"
        )
    keys = set(item)
    if keys != {"name", "arguments"}:
        return ParsedCall(
            raw=item,
            error=f"expected keys ['arguments', 'name'], got {sorted(keys)}",
        )
    name, arguments = item["name"], item["arguments"]
    if not isinstance(name, str):
        return ParsedCall(raw=item, error=f"'name' must be a string, got {name!r}")
    if not isinstance(arguments, dict):
        return ParsedCall(
            raw=item, error=f"'arguments' must be an object, got {arguments!r}"
        )
    return ParsedCall(raw=item, name=name, arguments=arguments)


def parse_calls(text: str) -> ParsedOutput:
    """Steps 1-3: extract the JSON array of calls from raw output."""
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end < start:
        return ParsedOutput(status="no_json", error="no [...] found in output")
    stripped = bool(text[:start].strip() or text[end + 1 :].strip())
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        return ParsedOutput(status="invalid_json", stripped=stripped, error=str(exc))
    if not isinstance(
        data, list
    ):  # unreachable for valid JSON in [...], kept as a guard
        return ParsedOutput(
            status="invalid_json", stripped=stripped, error="not an array"
        )
    return ParsedOutput(
        status="ok", stripped=stripped, calls=tuple(_check_item(i) for i in data)
    )


def execute_calls(
    parsed: ParsedOutput, env: Environment[Any]
) -> tuple[CallResult, ...]:
    """Steps 4-5: validate each call against the schema and execute it in order.

    Exceptions other than :class:`ToolCallError` are bugs in the environment and
    are not caught.
    """
    results = []
    for index, call in enumerate(parsed.calls):
        if call.error is not None or call.name is None:
            results.append(
                CallResult(index, "malformed_call", None, call.raw, error=call.error)
            )
            continue
        try:
            value = env.call(call.name, call.arguments)
        except ToolSchemaError as exc:
            results.append(
                CallResult(index, "schema_error", call.name, call.arguments, str(exc))
            )
        except ToolCallError as exc:
            results.append(
                CallResult(index, "rejected", call.name, call.arguments, str(exc))
            )
        else:
            results.append(
                CallResult(index, "ok", call.name, call.arguments, result=value)
            )
    return tuple(results)


def format_instructions(tools: Iterable[ToolSpec]) -> str:
    """The shared prompt text describing the output format and the functions.

    Prompt-based systems should include it verbatim so that systems are compared
    on the same instructions.
    """
    schemas = json.dumps([t.json_schema() for t in tools], indent=2)
    return (
        "Respond with a JSON array of the function calls to execute, in order, and "
        'nothing else. Each element is an object with exactly two keys: "name", the '
        'function name, and "arguments", an object mapping argument names to values. '
        "If no function should be called, respond with [].\n\n"
        'Example: [{"name": "some_function", "arguments": {"some_argument": 1}}]\n\n'
        f"Available functions:\n{schemas}"
    )
