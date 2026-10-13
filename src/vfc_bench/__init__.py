"""VFC benchmarking library."""

__version__ = "0.1.0"

__all__ = ["__version__"]


import time


class VfcBench:
    def __init(self, data, functions, expected):
        self.overall_time_ns = 0
        self.transcribe_time_ns = 0
        self.llm_step_time_ns = 0
        self.data = data
        self.functions = functions
        self.expected = expected
        self.scratch = ""

    def run(self):
        self.overall_time_ns = time.perf_counter_ns()
        self.voice_to_function_call()
        self.validate_schema()
        self.overall_time_ns = time.perf_counter_ns() - self.overall_time_ns()

    def voice_to_function_call(self):
        self.transcribe_time_ns = time.perf_counter_ns()
        self.transcribe()
        self.transcribe_time_ns = time.perf_counter_ns() - self.transcribe_time_ns

        self.llm_step_time_ns = time.perf_counter_ns()
        self.llm_step()
        self.llm_step_time_ns = time.perf_counter_ns() - self.llm_step_time_ns

    def transcribe(self):
        pass

    def llm_step(self):
        pass
