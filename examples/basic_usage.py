"""How you would use the library. Run with: python examples/basic_usage.py

Needs a .env in the repo root -- copy .env.example and fill it in.
"""

import json
import os

from dotenv import load_dotenv
from openai import OpenAI

import vfc_bench
from vfc_bench import ToolCallError, diff, load_test_case


def main() -> None:
    load_dotenv()
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.getenv("OPENAI_BASE_URL", "").strip()
        or "https://api.openai.com/v1",
    )

    print(f"vfc-bench {vfc_bench.__version__}")
    print(f"talking to {client.base_url}")

    case = load_test_case("heater_bedroom")
    env = case.setup()
    recording = case.recordings[0]

    print(f"loaded testcase {case.id} ({recording.file_name})")
    print("reference transcription: " + case.transcription)

    print("step 1) transcription")
    restran = client.audio.transcriptions.create(
        model="alias-stt",
        file=recording.read_bytes(),
    )
    print("stt returned: " + restran.text)

    print("step 2) inference with native tool calling")
    res = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V4.1-Flash",
        messages=[
            {
                "role": "system",
                "content": "You control a smart home. Call the functions needed to fulfil the user's request.",
            },
            {"role": "user", "content": restran.text},
        ],
        tools=env.openai_tools(),
    )
    tool_calls = res.choices[0].message.tool_calls or []
    before = env.snapshot()

    print(f"step 3) executing {len(tool_calls)} call(s) against the simulated home")
    for call in tool_calls:
        name = call.function.name
        try:
            result = env.call(name, json.loads(call.function.arguments))
            print(f"  ok       {name}({call.function.arguments}) -> {result}")
        except (ToolCallError, json.JSONDecodeError) as exc:
            print(f"  rejected {name}({call.function.arguments}): {exc}")

    print("step 4) state changes")
    for path, (old, new) in diff(before, env.snapshot()).items():
        print(f"  {path}: {old!r} -> {new!r}")

    print("step 5) checking expectations on the final state")
    results = case.check(env)
    for check in results:
        print("  " + str(check))
    print("PASSED" if all(r.passed for r in results) else "FAILED")


if __name__ == "__main__":
    main()
