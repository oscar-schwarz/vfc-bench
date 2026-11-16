import asyncio
import json

from vfc_bench import System, SystemOutput, load_test_case, run, run_case

GOOD = (
    '[{"name": "heater_power_state", "arguments": {"state": true, "room": "bedroom"}},'
    ' {"name": "heater_set_temperature", "arguments": {"value": 22, "room": "bedroom"}}]'
)


class Fixed(System):
    """Returns a fixed text, pretending to transcribe and think."""

    def __init__(self, text, name=None, delay=0.0):
        self.text, self.name, self.delay = text, name, delay

    async def run(self, recording, tools):
        assert recording.read_bytes()
        assert {t.name for t in tools} >= {"heater_power_state"}
        with self.stage("transcription"):
            await asyncio.sleep(self.delay)
        with self.stage("llm"):
            await asyncio.sleep(self.delay)
        return SystemOutput(text=self.text, transcript="heat my bedroom")


class Broken(System):
    async def run(self, recording, tools):
        raise ConnectionError("api down")


def _run_one(system):
    case = load_test_case("heater_bedroom")
    return asyncio.run(run_case(case, case.recordings[0], system))


def test_passing_run():
    result = _run_one(Fixed(GOOD, name="good"))
    assert result.passed
    assert result.system == "good"
    assert [c.status for c in result.calls] == ["ok", "ok"]
    assert result.changes["rooms.bedroom.heater.on"] == (False, True)
    assert set(result.timing.stages) == {"transcription", "llm"}
    json.dumps(result.to_dict())  # serializable


def test_failing_checks():
    result = _run_one(Fixed("[]"))
    assert not result.passed
    assert result.system == "Fixed"
    assert result.parsed.status == "ok"
    assert result.changes == {}
    assert [c.passed for c in result.checks] == [False, False]


def test_unparseable_output_never_passes():
    result = _run_one(Fixed("I have turned on the heater."))
    assert result.parsed.status == "no_json"
    assert not result.passed


def test_system_error_is_recorded_not_raised():
    result = _run_one(Broken())
    assert not result.passed
    assert result.system_error == "ConnectionError: api down"
    assert result.timing.total_ns > 0
    assert result.to_dict()["output"] is None


def test_stage_misuse_is_a_system_error():
    class Nested(System):
        async def run(self, recording, tools):
            with self.stage("a"), self.stage("b"):
                return SystemOutput("[]")

    assert "is open" in _run_one(Nested()).system_error


def test_run_covers_all_combinations_concurrently():
    case = load_test_case("heater_bedroom")
    systems = [Fixed(GOOD, name="a", delay=0.05), Fixed("[]", name="b", delay=0.05)]

    async def main():
        return [r async for r in run([case], systems, repetitions=3, concurrency=6)]

    results = asyncio.run(main())
    assert len(results) == len(case.recordings) * 2 * 3
    assert sorted((r.system, r.repetition) for r in results) == sorted(
        (s, k) for s in "ab" for k in range(3)
    )
    assert all(r.passed == (r.system == "a") for r in results)
    # each run sees only its own stages despite running in parallel
    assert all(r.timing.stages["llm"].count == 1 for r in results)


def test_breaking_out_cancels_pending_runs():
    case = load_test_case("heater_bedroom")
    started = []

    class Slow(System):
        async def run(self, recording, tools):
            started.append(1)
            await asyncio.sleep(0 if len(started) == 1 else 10)
            return SystemOutput("[]")

    async def main():
        async for _ in run([case], [Slow()], repetitions=5, concurrency=5):
            break

    asyncio.run(asyncio.wait_for(main(), timeout=2))
