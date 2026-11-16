from dataclasses import dataclass
from typing import Literal

import pytest

from vfc_bench import Environment, ToolCallError
from vfc_bench.data.function_sets.smart_home import Heater, Room, SmartHome


def test_schema_is_derived_from_signature():
    schema = SmartHome.tools()["heater_set_temperature"].json_schema()
    assert schema == {
        "name": "heater_set_temperature",
        "description": "Sets the target temperature of the heater, in the unit currently configured via heater_set_unit.",
        "parameters": {
            "type": "object",
            "properties": {"value": {"type": "number"}, "room": {"type": "string"}},
            "required": ["value", "room"],
            "additionalProperties": False,
        },
    }


def test_literal_becomes_enum_and_defaults_are_optional():
    tools = SmartHome.tools()
    assert tools["heater_set_unit"].json_schema()["parameters"]["properties"][
        "unit"
    ] == {
        "type": "string",
        "enum": ["celsius", "fahrenheit"],
    }
    door = tools["door_lock_state"].json_schema()["parameters"]
    assert door["required"] == ["locked"]
    assert door["properties"]["door"] == {"type": "string", "default": "front"}
    vacuum_room = tools["vacuum_set_state"].json_schema()["parameters"]["properties"][
        "room"
    ]
    assert vacuum_room == {"type": ["string", "null"]}


def test_helpers_and_base_methods_are_not_tools():
    names = set(SmartHome.tools())
    assert len(names) == 26
    assert not {"call", "snapshot", "tools", "_room", "__init__"} & names


def test_openai_format():
    tool = SmartHome.openai_tools()[0]
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "heater_power_state"


@pytest.mark.parametrize(
    ("name", "args", "message"),
    [
        ("teleport", {}, "unknown function"),
        ("heater_power_state", {"state": True}, "missing argument"),
        (
            "heater_power_state",
            {"state": True, "room": "bedroom", "x": 1},
            "unknown argument",
        ),
        ("heater_power_state", {"state": "on", "room": "bedroom"}, "must be boolean"),
        (
            "light_set_brightness",
            {"brightness": True, "room": "bedroom"},
            "must be integer",
        ),
        (
            "light_set_brightness",
            {"brightness": 50.5, "room": "bedroom"},
            "must be integer",
        ),
        (
            "light_set_brightness",
            {"brightness": 101, "room": "bedroom"},
            "between 0 and 100",
        ),
        ("heater_set_unit", {"unit": "kelvin"}, "must be one of"),
        ("heater_power_state", {"state": True, "room": "attic"}, "unknown room"),
        ("alarm_clock_set", {"time": "7:30"}, "HH:MM"),
        ("scene_activate", {"name": "party"}, "unknown scene"),
    ],
)
def test_invalid_calls_are_rejected(name, args, message):
    home = SmartHome()
    before = home.snapshot()
    with pytest.raises(ToolCallError, match=message):
        home.call(name, args)
    assert home.snapshot() == before


def test_room_names_are_normalized():
    home = SmartHome()
    home.call("light_power_state", {"state": True, "room": "Living Room"})
    assert home.snapshot()["rooms"]["living_room"]["light"]["on"] is True


def test_temperature_is_stored_in_celsius():
    home = SmartHome()
    home.call("heater_set_unit", {"unit": "fahrenheit"})
    home.call("heater_set_temperature", {"value": 70, "room": "bedroom"})
    assert home.snapshot()["rooms"]["bedroom"]["heater"]["target_c"] == 21.1
    # 70 in celsius is out of range, not silently accepted
    with pytest.raises(ToolCallError):
        SmartHome().call("heater_set_temperature", {"value": 70, "room": "bedroom"})


def test_int_is_accepted_for_float():
    home = SmartHome()
    home.call("heater_set_temperature", {"value": 22, "room": "bedroom"})
    assert home.snapshot()["rooms"]["bedroom"]["heater"]["target_c"] == 22.0


def test_sensor_read_returns_value_in_configured_unit():
    home = SmartHome(unit="fahrenheit")
    assert home.call("sensor_read", {"sensor": "temperature", "room": "kitchen"}) == {
        "value": 66.2,
        "unit": "fahrenheit",
    }


def test_initial_state_from_mapping():
    home = SmartHome(rooms={"Bedroom": Room(heater=Heater(on=True))})
    assert home.snapshot()["rooms"]["bedroom"]["heater"]["on"] is True


def test_tool_without_docstring_is_an_error():
    with pytest.raises(TypeError, match="docstring"):

        class Broken(Environment):
            def do(self, x: int) -> None:
                pass


def test_unsupported_type_is_an_error():
    with pytest.raises(TypeError, match="unsupported type"):

        class Broken(Environment):
            def do(self, x: list) -> None:
                """Does it."""


def test_custom_environment():
    @dataclass
    class LampState:
        on: bool = False

    class Lamp(Environment[LampState]):
        def __init__(self) -> None:
            self.state = LampState()

        def lamp(self, state: Literal["on", "off"]) -> None:
            """Switches the lamp."""
            self.state.on = state == "on"

    lamp = Lamp()
    assert list(Lamp.tools()) == ["lamp"]
    lamp.call("lamp", {"state": "on"})
    assert lamp.snapshot() == {"on": True}
