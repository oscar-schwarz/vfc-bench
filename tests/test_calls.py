import pytest

from vfc_bench import execute_calls, format_instructions, parse_calls
from vfc_bench.data.function_sets.smart_home import SmartHome

ON = '{"name": "heater_power_state", "arguments": {"state": true, "room": "bedroom"}}'


@pytest.mark.parametrize(
    ("text", "status", "stripped", "n_calls"),
    [
        (f"[{ON}]", "ok", False, 1),
        (f"  [{ON}, {ON}]\n", "ok", False, 2),
        ("[]", "ok", False, 0),
        (f"```json\n[{ON}]\n```", "ok", True, 1),
        (f"Sure! Here you go: [{ON}] Hope that helps.", "ok", True, 1),
        ("I turned on the heater.", "no_json", False, 0),
        ("] backwards [", "no_json", False, 0),
        (f"[{ON},]", "invalid_json", False, 0),
        (f"[{ON}] and also [{ON}]", "invalid_json", False, 0),
    ],
)
def test_parse_calls(text, status, stripped, n_calls):
    parsed = parse_calls(text)
    assert (parsed.status, parsed.stripped, len(parsed.calls)) == (
        status,
        stripped,
        n_calls,
    )
    assert (parsed.error is None) == (status == "ok")


@pytest.mark.parametrize(
    ("item", "error"),
    [
        ("heater_power_state", "expected an object"),
        ({"name": "x"}, "expected keys"),
        ({"name": "x", "arguments": {}, "extra": 1}, "expected keys"),
        ({"name": 1, "arguments": {}}, "'name' must be a string"),
        ({"name": "x", "arguments": "{}"}, "'arguments' must be an object"),
    ],
)
def test_malformed_items(item, error):
    import json

    (call,) = parse_calls(json.dumps([item])).calls
    assert error in call.error


def test_execute_calls_statuses_and_continues_after_errors():
    env = SmartHome()
    parsed = parse_calls(
        "["
        '"garbage",'
        '{"name": "teleport", "arguments": {}},'
        '{"name": "heater_power_state", "arguments": {"state": "yes", "room": "bedroom"}},'
        '{"name": "heater_power_state", "arguments": {"state": true, "room": "attic"}},'
        f"{ON},"
        '{"name": "sensor_read", "arguments": {"sensor": "humidity", "room": "kitchen"}}'
        "]"
    )
    results = execute_calls(parsed, env)
    assert [r.status for r in results] == [
        "malformed_call",
        "schema_error",
        "schema_error",
        "rejected",
        "ok",
        "ok",
    ]
    assert "unknown function" in results[1].error
    assert "unknown room" in results[3].error
    assert results[5].result == {"value": 45, "unit": "percent"}
    assert env.state.rooms["bedroom"].heater.on is True


def test_execute_nothing_on_parse_failure():
    assert execute_calls(parse_calls("nope"), SmartHome()) == ()


def test_format_instructions_lists_every_function():
    text = format_instructions(SmartHome.tools().values())
    assert "respond with []" in text
    for name in SmartHome.tools():
        assert f'"{name}"' in text
