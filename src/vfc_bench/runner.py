"""Running systems against test cases.

The runner is called by the application, which decides what to run and what to
do with the results; nothing here prints or writes files::

    async for result in run(load_test_cases(), [system_a, system_b], repetitions=3):
        ui.show(result)                       # or: file.write(json.dumps(result.to_dict()))

One run is one test case x one recording x one system x one repetition:

1. ``case.setup()`` builds a fresh environment and the state is snapshotted
2. the system runs, timed by the runner (see :mod:`vfc_bench.timing`)
3. its output is parsed and the calls are executed (see :mod:`vfc_bench.calls`)
4. the state change is diffed and the expectations are checked
"""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import AsyncIterator, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from vfc_bench.calls import CallResult, ParsedOutput, execute_calls, parse_calls
from vfc_bench.dataset import CheckResult, Recording, TestCase
from vfc_bench.environment import ABSENT, diff
from vfc_bench.system import System, SystemOutput
from vfc_bench.timing import Timing, _Measurement

__all__ = ["RunResult", "run", "run_case"]


@dataclass(frozen=True)
class RunResult:
    """Everything recorded about one run. Plain data; see :meth:`to_dict`."""

    case_id: str
    recording: str
    system: str
    repetition: int
    timing: Timing
    output: SystemOutput | None = None
    system_error: str | None = None
    parsed: ParsedOutput | None = None
    calls: tuple[CallResult, ...] = ()
    changes: dict[str, tuple[Any, Any]] = dataclasses.field(default_factory=dict)
    checks: tuple[CheckResult, ...] = ()

    @property
    def passed(self) -> bool:
        """The output was parseable and every expectation holds.

        Unparseable output never passes, even if the expectations happen to
        hold for the unchanged state (e.g. a case where nothing should happen).
        """
        return (
            self.system_error is None
            and self.parsed is not None
            and self.parsed.status == "ok"
            and all(check.passed for check in self.checks)
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable form, e.g. for one line of a JSONL results file."""
        changes = []
        for path, (before, after) in self.changes.items():
            change: dict[str, Any] = {"path": path}
            if before is not ABSENT:
                change["before"] = before
            if after is not ABSENT:
                change["after"] = after
            changes.append(change)
        return {
            "case_id": self.case_id,
            "recording": self.recording,
            "system": self.system,
            "repetition": self.repetition,
            "passed": self.passed,
            "system_error": self.system_error,
            "output": dataclasses.asdict(self.output) if self.output else None,
            "parsed": (
                {
                    "status": self.parsed.status,
                    "stripped": self.parsed.stripped,
                    "error": self.parsed.error,
                }
                if self.parsed
                else None
            ),
            "calls": [dataclasses.asdict(call) for call in self.calls],
            "changes": changes,
            "checks": [
                {"name": c.name, "passed": c.passed, "error": c.error}
                for c in self.checks
            ],
            "timing": {
                "total_ns": self.timing.total_ns,
                "other_ns": self.timing.other_ns,
                "stages": {
                    name: {"ns": s.ns, "count": s.count}
                    for name, s in self.timing.stages.items()
                },
            },
        }


async def run_case(
    case: TestCase[Any],
    recording: Recording,
    system: System,
    *,
    repetition: int = 0,
) -> RunResult:
    """Execute one run. A failing system is recorded, not raised."""
    env = case.setup()
    before = env.snapshot()
    tools = list(type(env).tools().values())

    output: SystemOutput | None = None
    error: str | None = None
    with _Measurement() as measurement:
        try:
            output = await system.run(recording, tools)
            if not isinstance(output, SystemOutput):
                raise TypeError(
                    f"run() must return SystemOutput, got {type(output).__name__}"
                )
        except Exception as exc:  # noqa: BLE001 -- any system failure is a result
            output, error = None, f"{type(exc).__name__}: {exc}"
    assert measurement.timing is not None

    common = {
        "case_id": case.id,
        "recording": recording.file_name,
        "system": system.display_name,
        "repetition": repetition,
        "timing": measurement.timing,
    }
    if output is None:
        return RunResult(**common, system_error=error)

    parsed = parse_calls(output.text)
    calls = execute_calls(parsed, env)
    return RunResult(
        **common,
        output=output,
        parsed=parsed,
        calls=calls,
        changes=diff(before, env.snapshot()),
        checks=tuple(case.check(env)),
    )


async def run(
    cases: Iterable[TestCase[Any]],
    systems: Sequence[System],
    *,
    repetitions: int = 1,
    concurrency: int = 1,
) -> AsyncIterator[RunResult]:
    """Run every recording of every case against every system, ``repetitions``
    times each, yielding results as they finish.

    ``concurrency`` limits how many runs are in flight at once. Stopping the
    iteration (``break``) cancels the runs still pending.
    """
    if repetitions < 1 or concurrency < 1:
        raise ValueError("repetitions and concurrency must be at least 1")
    jobs = [
        (case, recording, system, repetition)
        for case in cases
        for recording in case.recordings
        for system in systems
        for repetition in range(repetitions)
    ]
    semaphore = asyncio.Semaphore(concurrency)

    async def job(case, recording, system, repetition) -> RunResult:
        async with semaphore:
            return await run_case(case, recording, system, repetition=repetition)

    tasks = [asyncio.ensure_future(job(*args)) for args in jobs]
    try:
        for next_done in asyncio.as_completed(tasks):
            yield await next_done
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
