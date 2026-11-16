import pytest

from vfc_bench import (
    ABSENT,
    TestCase,
    diff,
    list_test_cases,
    load_test_case,
    load_test_cases,
)
from vfc_bench.data.function_sets.smart_home import SmartHome


def test_bundled_cases_load():
    cases = list(load_test_cases())
    assert [c.id for c in cases] == list_test_cases()
    for case in cases:
        assert case.transcription
        assert case.recordings
        assert case.recordings[0].read_bytes()


def test_setup_builds_a_fresh_environment_each_time():
    for case in load_test_cases():
        assert case.setup() is not case.setup()


def test_heater_case_passes_with_correct_calls():
    case = load_test_case("heater_bedroom")
    env = case.setup()
    env.call("heater_power_state", {"state": True, "room": "bedroom"})
    env.call("heater_set_temperature", {"value": 22, "room": "bedroom"})
    assert all(r.passed for r in case.check(env))


def test_heater_case_fails_with_wrong_room():
    case = load_test_case("heater_bedroom")
    env = case.setup()
    env.call("heater_power_state", {"state": True, "room": "kitchen"})
    env.call("heater_set_temperature", {"value": 22, "room": "kitchen"})
    results = case.check(env)
    assert not any(r.passed for r in results)
    assert [str(r) for r in results] == ["FAIL heater is on", "FAIL bedroom is warm"]


def test_heater_case_fails_without_any_calls():
    case = load_test_case("heater_bedroom")
    assert not all(r.passed for r in case.check(case.setup()))


def test_unknown_case():
    with pytest.raises(KeyError, match="unknown test case"):
        load_test_case("nope")


def _case(**expect):
    return TestCase(transcription="x", setup=SmartHome, expect=expect)


def test_crashing_or_non_bool_expectation_fails_with_reason():
    case = _case(
        typo=lambda s: s.rooms["attic"].heater.on,
        no_bool=lambda s: s.rooms["bedroom"].heater,
        fine=lambda s: not s.rooms["bedroom"].heater.on,
    )
    assert [str(r) for r in case.check(SmartHome())] == [
        "FAIL typo (KeyError: 'attic')",
        "FAIL no_bool (returned Heater(on=False, target_c=20.0, mode='comfort'), expected a bool)",
        "PASS fine",
    ]


def test_expectations_cannot_mutate_state():
    def sneaky(s):
        s.rooms["bedroom"].heater.on = True
        return True

    env = SmartHome()
    _case(sneaky=sneaky).check(env)
    assert env.state.rooms["bedroom"].heater.on is False


def test_case_needs_named_expectations():
    with pytest.raises(ValueError, match="at least one"):
        _case()
    with pytest.raises(ValueError, match="non-empty"):
        TestCase(transcription="x", setup=SmartHome, expect={"": lambda s: True})


def test_diff_shows_what_calls_changed():
    env = SmartHome()
    before = env.snapshot()
    env.call("heater_power_state", {"state": True, "room": "kitchen"})
    env.call("timer_start", {"seconds": 60})
    assert diff(before, env.snapshot()) == {
        "rooms.kitchen.heater.on": (False, True),
        "timers.0.label": (ABSENT, None),
        "timers.0.seconds": (ABSENT, 60),
    }
