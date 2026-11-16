import asyncio
import time

import pytest

from vfc_bench.timing import _Measurement, stage


def test_stages_are_a_breakdown_of_the_total():
    with _Measurement() as m:
        with stage("a"):
            time.sleep(0.02)
        time.sleep(0.01)
        with stage("b"):
            pass
        with stage("a"):
            time.sleep(0.01)
    t = m.timing
    assert set(t.stages) == {"a", "b"}
    assert t.stages["a"].count == 2 and t.stages["b"].count == 1
    assert t.stages["a"].ns >= 30_000_000
    assert t.other_ns >= 10_000_000
    assert t.total_ns == sum(s.ns for s in t.stages.values()) + t.other_ns


def test_stage_outside_a_run_is_an_error():
    with pytest.raises(RuntimeError, match="only be used while"), stage("a"):
        pass


def test_nested_stages_are_an_error():
    with (
        _Measurement(),
        stage("outer"),
        pytest.raises(RuntimeError, match="is open"),
        stage("inner"),
    ):
        pass


@pytest.mark.parametrize("name", ["", "total", "other"])
def test_invalid_stage_names(name):
    with _Measurement(), pytest.raises(ValueError), stage(name):
        pass


def test_stage_is_recorded_when_the_block_raises():
    with _Measurement() as m, pytest.raises(KeyError), stage("a"):
        raise KeyError
    assert m.timing.stages["a"].count == 1


def test_concurrent_runs_do_not_mix_timings():
    async def one(name, delay):
        with _Measurement() as m, stage(name):
            await asyncio.sleep(delay)
        return m.timing

    async def main():
        return await asyncio.gather(one("x", 0.03), one("y", 0.01))

    x, y = asyncio.run(main())
    assert set(x.stages) == {"x"} and set(y.stages) == {"y"}
    assert x.stages["x"].ns > y.stages["y"].ns
