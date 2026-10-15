"""Access to the benchmark dataset bundled with the library.

The data lives in ``src/vfc_bench/data/`` and is shipped inside the wheel, so it
is readable from an installed package without a checkout:

    >>> from vfc_bench import load_test_case
    >>> case = load_test_case("1-heater")
    >>> case.function_set.functions["heater_power_state"].description
    'Turns the heater on or off.'

On-disk layout::

    data/
      function-sets/<name>.json     one file per function set
      testcases/<id>/<case>.json    one directory per test case
      testcases/<id>/<case>.mp3     audio referenced by the json

Everything is read through ``importlib.resources``, so it also works when the
package is installed as a zip.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from importlib.resources import as_file, files
from importlib.resources.abc import Traversable
from typing import Any

__all__ = [
    "FunctionArg",
    "FunctionSpec",
    "FunctionSet",
    "VoiceCommand",
    "TestCase",
    "list_function_sets",
    "load_function_set",
    "list_test_cases",
    "load_test_case",
    "load_test_cases",
]

_ROOT = "data"
_FUNCTION_SETS = "function-sets"
_TESTCASES = "testcases"


def _dataset_root() -> Traversable:
    return files(__package__) / _ROOT


@dataclass(frozen=True)
class FunctionArg:
    """A single argument of a callable function."""

    name: str
    type: str
    default: Any = None
    options: tuple[str, ...] | None = None

    @classmethod
    def _parse(cls, name: str, raw: dict[str, Any]) -> FunctionArg:
        options = raw.get("options")
        return cls(
            name=name,
            type=raw["type"],
            default=raw.get("default"),
            options=tuple(options) if options is not None else None,
        )


@dataclass(frozen=True)
class FunctionSpec:
    """One function a solution under test is allowed to call."""

    name: str
    description: str
    args: dict[str, FunctionArg] = field(default_factory=dict)

    @classmethod
    def _parse(cls, name: str, raw: dict[str, Any]) -> FunctionSpec:
        return cls(
            name=name,
            description=raw["description"],
            args={
                arg_name: FunctionArg._parse(arg_name, arg)
                for arg_name, arg in raw.get("args", {}).items()
            },
        )


@dataclass(frozen=True)
class FunctionSet:
    """A named collection of functions, e.g. ``smart-home``."""

    name: str
    functions: dict[str, FunctionSpec] = field(default_factory=dict)

    @classmethod
    def _parse(cls, name: str, raw: dict[str, Any]) -> FunctionSet:
        return cls(
            name=name,
            functions={
                fn_name: FunctionSpec._parse(fn_name, fn) for fn_name, fn in raw.items()
            },
        )


@dataclass(frozen=True)
class VoiceCommand:
    """The spoken input of a test case, plus its reference transcription."""

    transcription: str
    file_name: str
    _resource: Traversable

    def read_bytes(self) -> bytes:
        """Return the raw audio bytes."""
        return self._resource.read_bytes()

    def as_file(self):
        """Context manager yielding a real filesystem :class:`~pathlib.Path`.

        Use this for APIs that insist on a path instead of a file object::

            with case.voice_command.as_file() as path:
                client.audio.transcriptions.create(file=path.open("rb"), ...)
        """
        return as_file(self._resource)

    def open(self):
        """Open the audio for binary reading."""
        return self._resource.open("rb")


@dataclass(frozen=True)
class TestCase:
    """One benchmark item: audio in, an expected outcome out."""

    id: str
    voice_command: VoiceCommand
    function_set: FunctionSet
    expected_result: str

    @classmethod
    def _parse(
        cls, case_id: str, directory: Traversable, raw: dict[str, Any]
    ) -> TestCase:
        command = raw["voice-command"]
        file_name = command["file"]
        return cls(
            id=case_id,
            voice_command=VoiceCommand(
                transcription=command["transcription"],
                file_name=file_name,
                _resource=directory / file_name,
            ),
            function_set=load_function_set(raw["function-set"]),
            expected_result=raw["expected-result"],
        )


def _read_json(resource: Traversable) -> Any:
    return json.loads(resource.read_text(encoding="utf-8"))


def list_function_sets() -> list[str]:
    """Names of every bundled function set, sorted."""
    directory = _dataset_root() / _FUNCTION_SETS
    return sorted(
        entry.name.removesuffix(".json")
        for entry in directory.iterdir()
        if entry.name.endswith(".json")
    )


def load_function_set(name: str) -> FunctionSet:
    """Load a function set by name, e.g. ``"smart-home"``."""
    resource = _dataset_root() / _FUNCTION_SETS / f"{name}.json"
    if not resource.is_file():
        raise KeyError(
            f"unknown function set {name!r}; available: {', '.join(list_function_sets())}"
        )
    return FunctionSet._parse(name, _read_json(resource))


def list_test_cases() -> list[str]:
    """Ids of every bundled test case, sorted."""
    directory = _dataset_root() / _TESTCASES
    return sorted(entry.name for entry in directory.iterdir() if entry.is_dir())


def load_test_case(case_id: str) -> TestCase:
    """Load a single test case by its directory name, e.g. ``"1-heater"``."""
    directory = _dataset_root() / _TESTCASES / case_id
    if not directory.is_dir():
        raise KeyError(
            f"unknown test case {case_id!r}; available: {', '.join(list_test_cases())}"
        )
    specs = sorted(
        (entry for entry in directory.iterdir() if entry.name.endswith(".json")),
        key=lambda entry: entry.name,
    )
    if not specs:
        raise ValueError(f"test case {case_id!r} contains no json spec")
    return TestCase._parse(case_id, directory, _read_json(specs[0]))


def load_test_cases() -> Iterator[TestCase]:
    """Load every bundled test case, in id order."""
    for case_id in list_test_cases():
        yield load_test_case(case_id)
