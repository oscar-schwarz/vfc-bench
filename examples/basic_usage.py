"""How an application uses the library. Run with: python examples/basic_usage.py

Needs a .env in the repo root -- copy .env.example and fill it in.

The systems are defined here, not in the library: an application (or a UI)
configures them and hands them to the runner.
"""

import asyncio
import json
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

import vfc_bench
from vfc_bench import System, SystemOutput, format_instructions, load_test_cases, run


class Cascade(System):
    """Speech-to-text, then an LLM.

    ``native_tools=False``: the LLM is prompted with the library's shared format
    instructions and answers with a JSON array as text.
    ``native_tools=True``: the provider's tool calling is used and its calls are
    rendered into the same JSON array, with the argument strings copied verbatim.
    """

    def __init__(self, client, stt_model, llm_model, native_tools=False):
        self.client = client
        self.stt_model, self.llm_model, self.native_tools = (
            stt_model,
            llm_model,
            native_tools,
        )
        mode = "native tools" if native_tools else "prompted"
        self.name = f"{stt_model} -> {llm_model} ({mode})"

    async def run(self, recording, tools):
        with self.stage("transcription"):
            transcript = await self.client.audio.transcriptions.create(
                model=self.stt_model, file=recording.read_bytes()
            )
        system_prompt = "You control a smart home. Call the functions needed to fulfil the user's request."

        if self.native_tools:
            with self.stage("llm"):
                res = await self.client.chat.completions.create(
                    model=self.llm_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": transcript.text},
                    ],
                    tools=[t.openai_schema() for t in tools],
                )
            calls = res.choices[0].message.tool_calls or []
            # Copy name and argument string verbatim; the library validates them.
            text = (
                "["
                + ", ".join(
                    '{"name": '
                    + json.dumps(c.function.name)
                    + ', "arguments": '
                    + (c.function.arguments or "{}")
                    + "}"
                    for c in calls
                )
                + "]"
            )
        else:
            with self.stage("llm"):
                res = await self.client.chat.completions.create(
                    model=self.llm_model,
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt
                            + "\n\n"
                            + format_instructions(tools),
                        },
                        {"role": "user", "content": transcript.text},
                    ],
                )
            text = res.choices[0].message.content or ""

        return SystemOutput(text=text, transcript=transcript.text)


async def main() -> None:
    load_dotenv()
    client = AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.getenv("OPENAI_BASE_URL", "").strip()
        or "https://api.openai.com/v1",
    )
    print(f"vfc-bench {vfc_bench.__version__}, talking to {client.base_url}")

    systems = [
        Cascade(client, "alias-stt", "deepseek-ai/DeepSeek-V4.1-Flash"),
        Cascade(
            client, "alias-stt", "deepseek-ai/DeepSeek-V4.1-Flash", native_tools=True
        ),
    ]

    # Blocking writes are fine for a script; a UI app would store results its own way.
    with open("results.jsonl", "w") as results_file:  # noqa: ASYNC230
        async for result in run(
            load_test_cases(), systems, repetitions=1, concurrency=4
        ):
            results_file.write(json.dumps(result.to_dict()) + "\n")

            t = result.timing
            stages = ", ".join(f"{n} {s.ns / 1e9:.2f}s" for n, s in t.stages.items())
            print(
                f"\n{'PASS' if result.passed else 'FAIL'}  {result.case_id} / {result.recording} / {result.system}"
            )
            print(
                f"  time: {t.total_ns / 1e9:.2f}s ({stages}, other {t.other_ns / 1e9:.2f}s)"
            )
            if result.system_error:
                print(f"  system error: {result.system_error}")
                continue
            print(f"  transcript: {result.output.transcript}")
            print(
                f"  parse: {result.parsed.status}{' (stripped)' if result.parsed.stripped else ''}"
            )
            for call in result.calls:
                print(
                    f"  call {call.status:<14} {call.name} {call.arguments} {call.error or ''}"
                )
            for path, (before, after) in result.changes.items():
                print(f"  change {path}: {before!r} -> {after!r}")
            for check in result.checks:
                print(f"  {check}")


if __name__ == "__main__":
    asyncio.run(main())
