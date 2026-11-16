"""A simulated smart home.

State paths for expectations look like ``rooms.bedroom.heater.target_c`` or
``doors.front.locked``. Room and door names are normalized to lowercase with
underscores, so ``"Living Room"`` is ``rooms.living_room``.

Conventions:
- Temperatures are stored in celsius (``*_c``) no matter which unit is configured;
  ``heater_set_temperature(70)`` with unit fahrenheit stores ``21.1``.
- Setters never change the power state: setting a temperature does not turn the
  heater on.
- Out-of-range values are rejected, not clamped.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from vfc_bench.environment import Environment, ToolRejectedError

__all__ = [
    "AC",
    "AlarmClock",
    "Blinds",
    "Door",
    "Fan",
    "Heater",
    "HomeState",
    "Humidifier",
    "Light",
    "Media",
    "Room",
    "Sensors",
    "SmartHome",
    "Timer",
    "Vacuum",
]

Unit = Literal["celsius", "fahrenheit"]


@dataclass
class Heater:
    on: bool = False
    target_c: float = 20.0
    mode: Literal["eco", "comfort", "boost", "schedule", "frost_protection"] = "comfort"


@dataclass
class AC:
    on: bool = False
    target_c: float = 22.0
    fan_speed: Literal["auto", "low", "medium", "high"] = "auto"


@dataclass
class Light:
    on: bool = False
    brightness: int = 100
    color: str | None = None  # None means white at `kelvin`
    kelvin: int = 2700


@dataclass
class Blinds:
    position: int = 100  # 0 closed, 100 open


@dataclass
class Fan:
    on: bool = False
    speed: Literal["low", "medium", "high"] = "medium"


@dataclass
class Humidifier:
    on: bool = False
    target_humidity: int = 45


@dataclass
class Media:
    playback: Literal["playing", "paused", "stopped"] = "stopped"
    volume: int = 30
    content: str | None = None
    track_offset: int = 0  # net next/previous presses


@dataclass
class Sensors:
    temperature_c: float = 19.0
    humidity: int = 45
    air_quality: int = 20  # AQI
    motion: bool = False
    door_contact: bool = False  # True = open
    power_usage_w: float = 150.0


@dataclass
class Room:
    heater: Heater = field(default_factory=Heater)
    ac: AC = field(default_factory=AC)
    light: Light = field(default_factory=Light)
    blinds: Blinds = field(default_factory=Blinds)
    fan: Fan = field(default_factory=Fan)
    humidifier: Humidifier = field(default_factory=Humidifier)
    media: Media = field(default_factory=Media)
    sensors: Sensors = field(default_factory=Sensors)


@dataclass
class Door:
    locked: bool = True


@dataclass
class Vacuum:
    status: Literal["docked", "cleaning", "paused", "idle"] = "docked"
    room: str | None = None  # None = whole home


@dataclass
class Timer:
    seconds: int
    label: str | None = None


@dataclass
class AlarmClock:
    time: str
    repeat_daily: bool = False


@dataclass
class HomeState:
    rooms: dict[str, Room]
    doors: dict[str, Door]
    scenes: tuple[str, ...]
    unit: Unit = "celsius"
    alarm_system: Literal["disarmed", "armed_home", "armed_away"] = "disarmed"
    vacuum: Vacuum = field(default_factory=Vacuum)
    active_scene: str | None = None
    timers: list[Timer] = field(default_factory=list)
    alarm_clocks: list[AlarmClock] = field(default_factory=list)


def _key(name: str) -> str:
    return re.sub(r"[\s\-]+", "_", name.strip().lower())


def _named(items: Iterable[str] | Mapping[str, Any], factory: type) -> dict[str, Any]:
    if isinstance(items, Mapping):
        return {_key(name): value for name, value in items.items()}
    return {_key(name): factory() for name in items}


def _in_range(name: str, value: float, low: float, high: float) -> None:
    if not low <= value <= high:
        raise ToolRejectedError(f"{name} must be between {low} and {high}, got {value}")


class SmartHome(Environment[HomeState]):
    """A home with rooms full of devices, some doors and home-wide systems.

    ``rooms`` and ``doors`` take names, or a mapping of name to an initial
    :class:`Room` / :class:`Door` to start from a non-default state::

        SmartHome(rooms={"bedroom": Room(heater=Heater(on=True)), "kitchen": Room()})
    """

    def __init__(
        self,
        rooms: Iterable[str] | Mapping[str, Room] = (
            "living room",
            "bedroom",
            "kitchen",
            "bathroom",
        ),
        doors: Iterable[str] | Mapping[str, Door] = ("front", "back", "garage"),
        unit: Unit = "celsius",
        scenes: Iterable[str] = ("movie night", "good morning", "good night", "away"),
    ) -> None:
        self.state = HomeState(
            rooms=_named(rooms, Room),
            doors=_named(doors, Door),
            scenes=tuple(_key(s) for s in scenes),
            unit=unit,
        )

    # -- helpers ---------------------------------------------------------------

    def _room(self, name: str) -> Room:
        room = self.state.rooms.get(_key(name))
        if room is None:
            raise ToolRejectedError(
                f"unknown room {name!r}; known: {sorted(self.state.rooms)}"
            )
        return room

    def _to_celsius(self, value: float) -> float:
        if self.state.unit == "fahrenheit":
            return round((value - 32) * 5 / 9, 1)
        return value

    def _from_celsius(self, value: float) -> float:
        if self.state.unit == "fahrenheit":
            return round(value * 9 / 5 + 32, 1)
        return value

    # -- climate ---------------------------------------------------------------

    def heater_power_state(self, state: bool, room: str) -> None:
        """Turns the heater on or off."""
        self._room(room).heater.on = state

    def heater_set_temperature(self, value: float, room: str) -> None:
        """Sets the target temperature of the heater, in the unit currently configured via heater_set_unit."""
        heater = self._room(room).heater
        celsius = self._to_celsius(value)
        _in_range("heater temperature", celsius, 5, 30)
        heater.target_c = celsius

    def heater_set_unit(self, unit: Literal["celsius", "fahrenheit"]) -> None:
        """Sets the temperature unit used by the heating system."""
        self.state.unit = unit

    def heater_set_mode(
        self,
        mode: Literal["eco", "comfort", "boost", "schedule", "frost_protection"],
        room: str,
    ) -> None:
        """Sets the operating mode of the heater."""
        self._room(room).heater.mode = mode

    def ac_power_state(self, state: bool, room: str) -> None:
        """Turns the AC on or off."""
        self._room(room).ac.on = state

    def ac_set_temperature(self, value: float, room: str) -> None:
        """Sets the target temperature of the AC, in the unit currently configured via heater_set_unit."""
        ac = self._room(room).ac
        celsius = self._to_celsius(value)
        _in_range("AC temperature", celsius, 16, 30)
        ac.target_c = celsius

    def ac_set_fan_speed(
        self, speed: Literal["auto", "low", "medium", "high"], room: str
    ) -> None:
        """Sets the fan speed of the AC."""
        self._room(room).ac.fan_speed = speed

    def fan_power_state(self, state: bool, room: str) -> None:
        """Turns the fan on or off."""
        self._room(room).fan.on = state

    def fan_set_speed(self, speed: Literal["low", "medium", "high"], room: str) -> None:
        """Sets the speed of the fan."""
        self._room(room).fan.speed = speed

    def humidifier_power_state(self, state: bool, room: str) -> None:
        """Turns the humidifier on or off."""
        self._room(room).humidifier.on = state

    def humidifier_set_humidity(self, humidity: int, room: str) -> None:
        """Sets the target relative humidity in percent, from 30 to 70."""
        humidifier = self._room(room).humidifier
        _in_range("humidity", humidity, 30, 70)
        humidifier.target_humidity = humidity

    # -- light & blinds --------------------------------------------------------

    def light_power_state(self, state: bool, room: str) -> None:
        """Turns the lights on or off."""
        self._room(room).light.on = state

    def light_set_brightness(self, brightness: int, room: str) -> None:
        """Sets the brightness of the lights, from 0 to 100 percent."""
        light = self._room(room).light
        _in_range("brightness", brightness, 0, 100)
        light.brightness = brightness

    def light_set_color(self, color: str, room: str) -> None:
        """Sets the color of the lights, as a color name or a hex code like #ff8800."""
        self._room(room).light.color = color.strip().lower()

    def light_set_color_temperature(self, kelvin: int, room: str) -> None:
        """Sets the white color temperature of the lights in kelvin, from 2000 (warm) to 6500 (cold)."""
        light = self._room(room).light
        _in_range("color temperature", kelvin, 2000, 6500)
        light.kelvin = kelvin
        light.color = None

    def blinds_set_position(self, position: int, room: str) -> None:
        """Moves the blinds to a position, from 0 (fully closed) to 100 (fully open)."""
        blinds = self._room(room).blinds
        _in_range("blinds position", position, 0, 100)
        blinds.position = position

    # -- media -----------------------------------------------------------------

    def media_playback_state(
        self, state: Literal["play", "pause", "stop", "next", "previous"], room: str
    ) -> None:
        """Controls playback on the media player."""
        media = self._room(room).media
        if state == "play":
            media.playback = "playing"
        elif state == "pause":
            media.playback = "paused"
        elif state == "stop":
            media.playback = "stopped"
        else:
            media.track_offset += 1 if state == "next" else -1

    def media_set_volume(self, volume: int, room: str) -> None:
        """Sets the volume of the media player, from 0 to 100 percent."""
        media = self._room(room).media
        _in_range("volume", volume, 0, 100)
        media.volume = volume

    def media_play_content(self, query: str, room: str) -> None:
        """Plays the song, album, playlist, radio station or podcast matching the query."""
        media = self._room(room).media
        media.content = query
        media.playback = "playing"

    # -- security --------------------------------------------------------------

    def door_lock_state(self, locked: bool, door: str = "front") -> None:
        """Locks or unlocks a door."""
        target = self.state.doors.get(_key(door))
        if target is None:
            raise ToolRejectedError(
                f"unknown door {door!r}; known: {sorted(self.state.doors)}"
            )
        target.locked = locked

    def alarm_system_set_state(
        self, state: Literal["disarmed", "armed_home", "armed_away"]
    ) -> None:
        """Arms or disarms the security alarm system."""
        self.state.alarm_system = state

    # -- misc ------------------------------------------------------------------

    def vacuum_set_state(
        self,
        state: Literal["start", "pause", "stop", "return_to_dock"],
        room: str | None = None,
    ) -> None:
        """Controls the robot vacuum. If a room is given, it cleans only that room."""
        vacuum = self.state.vacuum
        if state == "start":
            if room is not None:
                self._room(room)
            vacuum.status, vacuum.room = "cleaning", _key(room) if room else None
        elif state == "pause":
            vacuum.status = "paused"
        elif state == "stop":
            vacuum.status = "idle"
        else:
            vacuum.status, vacuum.room = "docked", None

    def scene_activate(self, name: str) -> None:
        """Activates a predefined scene, for example 'movie night', 'good morning' or 'away'."""
        if _key(name) not in self.state.scenes:
            raise ToolRejectedError(
                f"unknown scene {name!r}; known: {list(self.state.scenes)}"
            )
        self.state.active_scene = _key(name)

    def timer_start(self, seconds: int, label: str | None = None) -> None:
        """Starts a countdown timer."""
        _in_range("timer seconds", seconds, 1, 24 * 3600)
        self.state.timers.append(Timer(seconds=seconds, label=label))

    def alarm_clock_set(self, time: str, repeat_daily: bool = False) -> None:
        """Sets an alarm clock for a time of day in 24h HH:MM format."""
        if not re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", time):
            raise ToolRejectedError(f"time must be HH:MM in 24h format, got {time!r}")
        self.state.alarm_clocks.append(AlarmClock(time=time, repeat_daily=repeat_daily))

    def sensor_read(
        self,
        sensor: Literal[
            "temperature",
            "humidity",
            "air_quality",
            "motion",
            "door_contact",
            "power_usage",
        ],
        room: str,
    ) -> dict[str, Any]:
        """Reads the current value of a sensor."""
        sensors = self._room(room).sensors
        if sensor == "temperature":
            return {
                "value": self._from_celsius(sensors.temperature_c),
                "unit": self.state.unit,
            }
        if sensor == "humidity":
            return {"value": sensors.humidity, "unit": "percent"}
        if sensor == "air_quality":
            return {"value": sensors.air_quality, "unit": "aqi"}
        if sensor == "motion":
            return {"value": sensors.motion}
        if sensor == "door_contact":
            return {"value": "open" if sensors.door_contact else "closed"}
        return {"value": sensors.power_usage_w, "unit": "watt"}
