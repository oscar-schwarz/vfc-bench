"""Access to the benchmark dataset bundled with the library.

The data lives in ``src/vfc_bench/data/`` and is shipped inside the wheel::

    >>> from vfc_bench import load_test_case
    >>> case = load_test_case("heater_bedroom")
    >>> env = case.setup()
    >>> env.call("heater_power_state", {"state": True, "room": "bedroom"})
    >>> env.call("heater_set_temperature", {"value": 22, "room": "bedroom"})
    >>> [str(result) for result in case.check(env)]
    ['PASS heater is on', 'PASS bedroom is warm']

On-disk layout::

    data/
      function_sets/<name>.py        an Environment subclass per function set
      testcases/<id>/case.py         defines CASE = TestCase(...)
      testcases/<id>/*.mp3|wav|...   recordings of the command; each is a variant

Test case ids are directory names and must be valid Python identifiers, since
``case.py`` is imported as ``vfc_bench.data.testcases.<id>.case``.
"""

from __future__ import annotations

import copy
import dataclasses
import importlib
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from importlib.resources import as_file, files
from importlib.resources.abc import Traversable
from typing import Any, Generic, TypeVar

from vfc_bench.environment import Environment

__all__ = [
    "CheckResult",
    "Expectation",
    "Recording",
    "TestCase",
    "list_test_cases",
    "load_test_case",
    "load_test_cases",
]

_TESTCASES_PACKAGE = "vfc_bench.data.testcases"
_CASE_FILE = "case.py"
_AUDIO_SUFFIXES = (".mp3", ".wav", ".flac", ".ogg", ".opus", ".m4a", ".webm")

S = TypeVar("S")

Expectation = Callable[[S], bool]
"""A condition on the final state, e.g. ``lambda s: s.rooms["bedroom"].heater.on``."""


@dataclass(frozen=True)
class CheckResult:
    """The outcome of one named expectation against a final state."""

    name: str
    passed: bool
    error: str | None = None

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"{status} {self.name}" + (f" ({self.error})" if self.error else "")


def _run_expectation(
    name: str, expectation: Expectation[Any], state: Any
) -> CheckResult:
    # A crashing expectation is a failed check, not a crashed benchmark run.
    try:
        result = expectation(state)
    except Exception as exc:  # noqa: BLE001
        return CheckResult(name, passed=False, error=f"{type(exc).__name__}: {exc}")
    if not isinstance(result, bool):
        return CheckResult(
            name, passed=False, error=f"returned {result!r}, expected a bool"
        )
    return CheckResult(name, passed=result)


@dataclass(frozen=True)
class Recording:
    """One audio rendition of a test case's voice command."""

    file_name: str
    _resource: Traversable = field(repr=False, compare=False)

    def read_bytes(self) -> bytes:
        """Return the raw audio bytes."""
        return self._resource.read_bytes()

    def open(self):
        """Open the audio for binary reading."""
        return self._resource.open("rb")

    def as_file(self):
        """Context manager yielding a real filesystem :class:`~pathlib.Path`."""
        return as_file(self._resource)


@dataclass(frozen=True)
class TestCase(Generic[S]):
    """One benchmark item: a spoken command, a starting state and named
    expectations on the state after the system's function calls were executed::

        CASE = TestCase(
            transcription="turn on the bedroom heater",
            setup=lambda: SmartHome(rooms=["bedroom"]),
            expect={
                "heater on": lambda s: s.rooms["bedroom"].heater.on,
            },
        )

    ``setup`` must build a *fresh* environment on every call, since each run
    (recording x system x repetition) mutates its own copy. Each expectation gets
    the environment's ``state`` and must return a bool; the names identify it in
    results, so keep them stable once results exist.

    ``id`` and ``recordings`` are filled in by the loader; leave them out in
    ``case.py``.
    """

    __test__ = False  # not a pytest test class

    transcription: str
    setup: Callable[[], Environment[S]]
    expect: Mapping[str, Expectation[S]]
    id: str = ""
    recordings: tuple[Recording, ...] = ()

    def __post_init__(self) -> None:
        if not self.expect:
            raise ValueError("a test case needs at least one expectation")
        for name, expectation in self.expect.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError(
                    f"expectation names must be non-empty strings, got {name!r}"
                )
            if not callable(expectation):
                raise TypeError(f"expectation {name!r} is not callable")

    def check(self, env: Environment[S]) -> list[CheckResult]:
        """Check every expectation against the current state of ``env``.

        Expectations see a copy, so a careless one cannot change the state.
        """
        state = copy.deepcopy(env.state)
        return [_run_expectation(name, fn, state) for name, fn in self.expect.items()]


def _testcases_root() -> Traversable:
    return files(_TESTCASES_PACKAGE)


def list_test_cases() -> list[str]:
    """Ids of every bundled test case, sorted."""
    return sorted(
        entry.name
        for entry in _testcases_root().iterdir()
        if entry.is_dir() and (entry / _CASE_FILE).is_file()
    )


def _recordings(directory: Traversable) -> tuple[Recording, ...]:
    return tuple(
        Recording(file_name=entry.name, _resource=entry)
        for entry in sorted(directory.iterdir(), key=lambda e: e.name)
        if entry.is_file() and entry.name.lower().endswith(_AUDIO_SUFFIXES)
    )


def load_test_case(case_id: str) -> TestCase:
    """Load a single test case by its directory name, e.g. ``"heater_bedroom"``."""
    directory = _testcases_root() / case_id
    if not (directory / _CASE_FILE).is_file():
        raise KeyError(
            f"unknown test case {case_id!r}; available: {', '.join(list_test_cases())}"
        )
    if not case_id.isidentifier():
        raise ValueError(f"test case id {case_id!r} must be a valid Python identifier")

    module = importlib.import_module(f"{_TESTCASES_PACKAGE}.{case_id}.case")
    case = getattr(module, "CASE", None)
    if not isinstance(case, TestCase):
        raise TypeError(f"{case_id}/{_CASE_FILE} must define CASE = TestCase(...)")

    recordings = _recordings(directory)
    if not recordings:
        raise ValueError(f"test case {case_id!r} has no recordings")
    return dataclasses.replace(case, id=case_id, recordings=recordings)


def load_test_cases(ids: Iterable[str] | None = None) -> Iterator[TestCase]:
    """Load the given test cases, or every bundled one, in id order."""
    for case_id in list_test_cases() if ids is None else ids:
        yield load_test_case(case_id)
