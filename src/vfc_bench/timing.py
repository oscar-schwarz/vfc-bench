"""Library-owned timing of system runs.

Systems never measure time themselves. The runner times the whole
:meth:`System.run` call; inside it, a system only *labels* sections::

    with self.stage("llm"):
        text = await self.complete(prompt)

Rules:
- the total is measured by the runner and is authoritative;
- stages are a breakdown of it; time outside any stage shows up as ``other_ns``;
- using a stage name twice adds up the durations and counts the uses;
- stages must not be nested or overlap within one run.

Each run records into its own recorder, held in a :class:`~contextvars.ContextVar`,
so concurrent runs of the same system instance do not mix up their timings.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Self

__all__ = ["StageTiming", "Timing", "stage"]

_RESERVED = {"total", "other"}


@dataclass(frozen=True)
class StageTiming:
    """Accumulated time of one stage name within a run."""

    ns: int
    count: int


@dataclass(frozen=True)
class Timing:
    """Timing of one system run, all in nanoseconds."""

    total_ns: int
    stages: dict[str, StageTiming] = field(default_factory=dict)

    @property
    def other_ns(self) -> int:
        """Time inside :meth:`System.run` not covered by any stage."""
        return self.total_ns - sum(s.ns for s in self.stages.values())


class _Recorder:
    def __init__(self) -> None:
        self.stages: dict[str, StageTiming] = {}
        self.open: str | None = None

    def add(self, name: str, ns: int) -> None:
        prev = self.stages.get(name, StageTiming(0, 0))
        self.stages[name] = StageTiming(prev.ns + ns, prev.count + 1)


_current: ContextVar[_Recorder | None] = ContextVar("vfc_bench_recorder", default=None)


@contextmanager
def stage(name: str) -> Iterator[None]:
    """Time the enclosed block as stage ``name`` of the current run."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"stage name must be a non-empty string, got {name!r}")
    if name in _RESERVED:
        raise ValueError(f"stage name {name!r} is reserved")
    recorder = _current.get()
    if recorder is None:
        raise RuntimeError(
            "stage() can only be used while the runner executes a system"
        )
    if recorder.open is not None:
        raise RuntimeError(
            f"stage {name!r} started while stage {recorder.open!r} is open"
        )
    recorder.open = name
    start = time.perf_counter_ns()
    try:
        yield
    finally:
        recorder.add(name, time.perf_counter_ns() - start)
        recorder.open = None


class _Measurement:
    """Used by the runner: times a block and collects the stages inside it."""

    def __init__(self) -> None:
        self._recorder = _Recorder()
        self._token = None
        self._start = 0
        self.timing: Timing | None = None

    def __enter__(self) -> Self:
        self._token = _current.set(self._recorder)
        self._start = time.perf_counter_ns()
        return self

    def __exit__(self, *exc: object) -> None:
        total = time.perf_counter_ns() - self._start
        assert self._token is not None
        _current.reset(self._token)
        self.timing = Timing(total_ns=total, stages=dict(self._recorder.stages))
