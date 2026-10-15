"""How you would use the library. Run with: python examples/basic_usage.py

Needs a .env in the repo root -- copy .env.example and fill it in.
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

import vfc_bench
from vfc_bench.dataset import load_test_case


def main() -> None:
    load_dotenv()
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.getenv("OPENAI_BASE_URL", "").strip() or "https://api.openai.com/v1",
    )
    
    print(f"vfc-bench {vfc_bench.__version__}")
    print(f"talking to {client.base_url}")
    
    testcase = load_test_case("1-heater")

    print("loaded testcase")
    print("transcription: "+testcase.voice_command.transcription)
    print("expected-result: "+testcase.expected_result)

    print("step 1) transcription:")
    restran = client.audio.transcriptions.create(
        model="alias-stt",
        file=testcase.voice_command.read_bytes()
    )

    print("stt returned: "+restran.text)


    print("step 2) prompt assembly")

    system_prompt = "You are a tool caller program. Based on the prompt you will be given you return a json array of functions to be called. Each object must have 'name' of the function an an 'args' object which maps argument name to argument value."

    system_prompt += "\n Here are all available functions: "+str(testcase.function_set)


    print("system-prompt: "+system_prompt)


    print("step 3) inference")
    
    res = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V4.1-Flash",  # the model you want
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": restran.text},
        ],
    )

    print("llm output: "+res.choices[0].message.content)


    print("step 4) schema validation -> skipped, trust me bro")

    print("step 5) llm judge")

    res = client.chat.completions.create(
        model="alias-huge",
        messages=[{"role":"system","content": "You judge YES or NO if the expected result was reached. Add no other text"},
         {"role": "user", "content": "prompt: "+testcase.voice_command.transcription+"; called functions: "+res.choices[0].message.content+"; expected result: "+testcase.expected_result}
        ]
    )

    print("expected result? -> "+res.choices[0].message.content)


if __name__ == "__main__":
    main()
