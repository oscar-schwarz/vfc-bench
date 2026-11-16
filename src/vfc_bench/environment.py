"""Executable function sets.

A function set is a subclass of :class:`Environment`. Its public methods are the
functions a system under test may call, and its instance holds the state those
calls change::

    @dataclass
    class LampState:
        on: bool = False

    class Lamp(Environment[LampState]):
        def __init__(self) -> None:
            self.state = LampState()

        def lamp_power_state(self, state: bool) -> None:
            \"\"\"Turns the lamp on or off.\"\"\"
            self.state.on = state

The method signature *is* the schema: type hints become JSON Schema types,
``Literal[...]`` becomes an enum, parameters with a default are optional and the
docstring is the description. Supported argument types are ``bool``, ``int``,
``float``, ``str``, ``Literal`` of one of those, and ``X | None``.

Methods starting with ``_`` are helpers, not tools. Methods must validate their
input *before* mutating state, so a rejected call leaves the environment as it was.

All state lives in ``self.state``, ideally a dataclass: test case expectations
receive it, and :meth:`Environment.snapshot` serializes it.
"""

from __future__ import annotations

import dataclasses
import inspect
import types
import typing
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, Generic, Literal, TypeVar

__all__ = ["ABSENT", "Environment", "ToolArg", "ToolCallError", "ToolSpec", "diff"]

S = TypeVar("S")


class ToolCallError(Exception):
    """A call the environment rejected: unknown function, bad arguments or an
    action the current state does not allow (e.g. an unknown room)."""


_JSON_TYPES: dict[type, str] = {
    bool: "boolean",
    int: "integer",
    float: "number",
    str: "string",
}


@dataclass(frozen=True)
class ToolArg:
    """One parameter of a tool, derived from a method signature."""

    name: str
    type: type
    required: bool
    default: Any = None
    options: tuple[Any, ...] | None = None
    nullable: bool = False

    @classmethod
    def _from_parameter(cls, param: inspect.Parameter, annotation: Any) -> ToolArg:
        nullable = False
        if typing.get_origin(annotation) in (typing.Union, types.UnionType):
            members = [a for a in typing.get_args(annotation) if a is not type(None)]
            if len(members) != 1 or len(members) == len(typing.get_args(annotation)):
                raise TypeError(f"{param.name}: only `X | None` unions are supported")
            nullable, annotation = True, members[0]

        options = None
        if typing.get_origin(annotation) is Literal:
            options = typing.get_args(annotation)
            kinds = {type(o) for o in options}
            if len(kinds) != 1:
                raise TypeError(f"{param.name}: Literal values must share one type")
            annotation = kinds.pop()

        if annotation not in _JSON_TYPES:
            raise TypeError(f"{param.name}: unsupported type {annotation!r}")

        has_default = param.default is not inspect.Parameter.empty
        return cls(
            name=param.name,
            type=annotation,
            required=not has_default,
            default=param.default if has_default else None,
            options=options,
            nullable=nullable,
        )

    def json_schema(self) -> dict[str, Any]:
        json_type = _JSON_TYPES[self.type]
        schema: dict[str, Any] = {
            "type": [json_type, "null"] if self.nullable else json_type
        }
        if self.options is not None:
            schema["enum"] = list(self.options)
        if not self.required and self.default is not None:
            schema["default"] = self.default
        return schema

    def validate(self, value: Any) -> Any:
        """Return ``value`` (ints widened to float where needed) or raise."""
        if value is None and self.nullable:
            return None
        # bool is a subclass of int, so it has to be excluded explicitly.
        if self.type is bool:
            ok = isinstance(value, bool)
        elif self.type is float:
            ok = isinstance(value, (int, float)) and not isinstance(value, bool)
            if isinstance(value, int) and ok:
                value = float(value)
        else:
            ok = isinstance(value, self.type) and not isinstance(value, bool)
        if not ok:
            raise ToolCallError(
                f"argument {self.name!r} must be {_JSON_TYPES[self.type]}, got {value!r}"
            )
        if self.options is not None and value not in self.options:
            raise ToolCallError(
                f"argument {self.name!r} must be one of {list(self.options)}, got {value!r}"
            )
        return value


@dataclass(frozen=True)
class ToolSpec:
    """A callable function of an environment, derived from a method."""

    name: str
    description: str
    args: dict[str, ToolArg]

    @classmethod
    def _from_method(cls, method: Callable[..., Any]) -> ToolSpec:
        doc = inspect.getdoc(method)
        if not doc:
            raise TypeError(
                f"tool {method.__name__!r} needs a docstring as its description"
            )
        hints = typing.get_type_hints(method)
        args = {}
        for param in list(inspect.signature(method).parameters.values())[
            1:
        ]:  # skip self
            if param.kind not in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY):
                raise TypeError(
                    f"tool {method.__name__!r}: *args/**kwargs not supported"
                )
            if param.name not in hints:
                raise TypeError(
                    f"tool {method.__name__!r}: {param.name!r} needs a type hint"
                )
            args[param.name] = ToolArg._from_parameter(param, hints[param.name])
        return cls(name=method.__name__, description=doc, args=args)

    def json_schema(self) -> dict[str, Any]:
        """Provider-neutral schema: ``{"name", "description", "parameters"}``."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    name: arg.json_schema() for name, arg in self.args.items()
                },
                "required": [name for name, arg in self.args.items() if arg.required],
                "additionalProperties": False,
            },
        }

    def bind(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        """Validate raw call arguments and return them as keyword arguments."""
        unknown = set(arguments) - set(self.args)
        if unknown:
            raise ToolCallError(f"{self.name}: unknown argument(s) {sorted(unknown)}")
        missing = [n for n, a in self.args.items() if a.required and n not in arguments]
        if missing:
            raise ToolCallError(f"{self.name}: missing argument(s) {missing}")
        return {
            name: self.args[name].validate(value) for name, value in arguments.items()
        }


class Environment(Generic[S]):
    """Base class for function sets. See the module docstring."""

    _tools: ClassVar[dict[str, ToolSpec]] = {}
    state: S

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        tools: dict[str, ToolSpec] = {}
        for klass in reversed(cls.__mro__):
            if not issubclass(klass, Environment) or klass is Environment:
                continue
            for name, attr in vars(klass).items():
                if name.startswith("_") or name in vars(Environment):
                    continue
                if inspect.isfunction(attr):
                    tools[name] = ToolSpec._from_method(attr)
        cls._tools = tools

    @classmethod
    def tools(cls) -> dict[str, ToolSpec]:
        """All callable functions, in definition order."""
        return dict(cls._tools)

    @classmethod
    def tool_schemas(cls) -> list[dict[str, Any]]:
        """Provider-neutral JSON schemas of all tools."""
        return [spec.json_schema() for spec in cls._tools.values()]

    @classmethod
    def openai_tools(cls) -> list[dict[str, Any]]:
        """Tools in the format of the OpenAI chat completions ``tools`` parameter."""
        return [
            {"type": "function", "function": schema} for schema in cls.tool_schemas()
        ]

    def call(self, name: str, arguments: Mapping[str, Any] | None = None) -> Any:
        """Execute one function call. Raises :class:`ToolCallError` if rejected."""
        spec = self._tools.get(name)
        if spec is None:
            raise ToolCallError(f"unknown function {name!r}")
        return getattr(self, name)(**spec.bind(arguments or {}))

    def snapshot(self) -> dict[str, Any]:
        """The current state as plain data, for diffs and reports.

        The default works for a dataclass ``self.state``; override it otherwise.
        """
        if dataclasses.is_dataclass(self.state) and not isinstance(self.state, type):
            return dataclasses.asdict(self.state)
        raise NotImplementedError(f"{type(self).__name__} must implement snapshot()")


class _Absent:
    def __repr__(self) -> str:
        return "<absent>"


ABSENT: Any = _Absent()
"""Marks a path missing on one side of a :func:`diff`, e.g. a newly started timer."""


def _flatten(data: Any, prefix: str, out: dict[str, Any]) -> dict[str, Any]:
    if isinstance(data, dict):
        items = data.items()
    elif isinstance(data, (list, tuple)):
        items = enumerate(data)
    else:
        out[prefix] = data
        return out
    for key, value in items:
        _flatten(value, f"{prefix}.{key}" if prefix else str(key), out)
    return out


def diff(before: dict[str, Any], after: dict[str, Any]) -> dict[str, tuple[Any, Any]]:
    """What changed between two snapshots, as ``{path: (before, after)}``::

    {"rooms.bedroom.heater.on": (False, True), "timers.0.seconds": (ABSENT, 60)}
    """
    old, new = _flatten(before, "", {}), _flatten(after, "", {})
    return {
        path: (old.get(path, ABSENT), new.get(path, ABSENT))
        for path in sorted(old.keys() | new.keys())
        if old.get(path, ABSENT) != new.get(path, ABSENT)
    }
