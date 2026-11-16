"""VFC benchmarking library."""

__version__ = "0.1.0"

from vfc_bench.calls import (
    CallResult,
    ParsedCall,
    ParsedOutput,
    execute_calls,
    format_instructions,
    parse_calls,
)
from vfc_bench.dataset import (
    CheckResult,
    Expectation,
    Recording,
    TestCase,
    list_test_cases,
    load_test_case,
    load_test_cases,
)
from vfc_bench.environment import (
    ABSENT,
    Environment,
    ToolArg,
    ToolCallError,
    ToolRejectedError,
    ToolSchemaError,
    ToolSpec,
    diff,
)
from vfc_bench.runner import RunResult, run, run_case
from vfc_bench.system import System, SystemOutput
from vfc_bench.timing import StageTiming, Timing

__all__ = [
    "ABSENT",
    "CallResult",
    "CheckResult",
    "Environment",
    "Expectation",
    "ParsedCall",
    "ParsedOutput",
    "Recording",
    "RunResult",
    "StageTiming",
    "System",
    "SystemOutput",
    "TestCase",
    "Timing",
    "ToolArg",
    "ToolCallError",
    "ToolRejectedError",
    "ToolSchemaError",
    "ToolSpec",
    "__version__",
    "diff",
    "execute_calls",
    "format_instructions",
    "list_test_cases",
    "load_test_case",
    "load_test_cases",
    "parse_calls",
    "run",
    "run_case",
]
