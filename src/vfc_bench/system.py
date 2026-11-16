"""The interface every system under test implements.

A system turns one recording into raw text describing function calls. It does
*not* parse or validate its output; the library does that the same way for every
system (see :mod:`vfc_bench.calls`). Systems are implemented and configured by
the application using this library, not by the library itself::

    class Cascade(System):
        def __init__(self, stt, llm) -> None:
            self.name = f"{stt.model} + {llm.model}"
            ...

        async def run(self, recording, tools):
            with self.stage("transcription"):
                transcript = await self.stt(recording.read_bytes())
            with self.stage("llm"):
                text = await self.llm(format_instructions(tools), transcript)
            return SystemOutput(text=text, transcript=transcript)

Systems that use native tool calling render the provider's calls as the JSON
array described by :func:`~vfc_bench.calls.format_instructions`, copying the
argument strings verbatim instead of parsing them.

A system only receives the recording and the tools, never the test case, so it
cannot see the reference transcription or the expectations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass

from vfc_bench import timing
from vfc_bench.dataset import Recording
from vfc_bench.environment import ToolSpec

__all__ = ["System", "SystemOutput"]


@dataclass(frozen=True)
class SystemOutput:
    """What a system returns for one recording."""

    text: str
    """Raw output describing the calls, parsed by :func:`~vfc_bench.calls.parse_calls`."""

    transcript: str | None = None
    """The intermediate transcript, if the system has a speech-to-text step."""


class System(ABC):
    """Base class for systems under test. See the module docstring."""

    name: str | None = None
    """Shown in results; defaults to the class name. Set it per configuration."""

    @property
    def display_name(self) -> str:
        return self.name or type(self).__name__

    @abstractmethod
    async def run(
        self, recording: Recording, tools: Sequence[ToolSpec]
    ) -> SystemOutput:
        """Turn one recording into raw function-call text."""

    def stage(self, name: str) -> AbstractContextManager[None]:
        """Time the enclosed block as stage ``name``; see :mod:`vfc_bench.timing`."""
        return timing.stage(name)
