"""How you would use the library. Run with: python examples/basic_usage.py

Needs a .env in the repo root -- copy .env.example and fill it in.
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

import vfc_bench


def main() -> None:
    load_dotenv()
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.getenv("OPENAI_BASE_URL", "").strip() or "https://api.openai.com/v1",
    )

    print(f"vfc-bench {vfc_bench.__version__}")
    print(f"talking to {client.base_url}")


if __name__ == "__main__":
    main()
