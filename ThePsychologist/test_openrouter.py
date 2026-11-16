"""Quick connection test for OpenRouter with a free reasoning model.

Usage:
    pip install requests
    export OPENROUTER_API_KEY="sk-or-..."     (Windows PowerShell: $env:OPENROUTER_API_KEY="sk-or-...")
    python test_openrouter.py
"""
import json
import os
import sys

import requests

URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"

def load_env_file():
    """Read OPENROUTER_API_KEY from a .env file next to this script or in the current folder."""
    here = os.path.dirname(os.path.abspath(__file__))
    for folder in (here, os.getcwd()):
        path = os.path.join(folder, ".env")
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                name, value = line.split("=", 1)
                name = name.replace("export ", "").strip()
                value = value.strip().strip('"').strip("'")
                if name == "OPENROUTER_API_KEY" and value:
                    return value, path
    return None, None


API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not API_KEY:
    API_KEY, env_path = load_env_file()
    if API_KEY:
        print(f"Loaded key from {env_path}")
if not API_KEY:
    sys.exit(
        "No key found. Put this line in a file named .env next to the script:\n"
        "OPENROUTER_API_KEY=sk-or-..."
    )

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}


def call(messages):
    """POST to OpenRouter and return the assistant message dict, or exit with a clear error."""
    resp = requests.post(
        URL,
        headers=HEADERS,  # the second call in your snippet was missing this
        data=json.dumps({
            "model": MODEL,
            "messages": messages,
            "reasoning": {"enabled": True},
        }),
        timeout=120,
    )

    if resp.status_code != 200:
        hints = {
            401: "Invalid API key.",
            402: "Negative credit balance (even free models return this).",
            404: "Model not found. Free models rotate; check openrouter.ai/models?q=free",
            429: "Rate limited (free: 20 req/min, 50 req/day). Wait and retry.",
        }
        print(f"HTTP {resp.status_code}: {hints.get(resp.status_code, '')}")
        print(resp.text)
        sys.exit(1)

    data = resp.json()
    if "error" in data:
        print("API error:", json.dumps(data["error"], indent=2))
        sys.exit(1)

    return data["choices"][0]["message"]


def main():
    question = "How many r's are in the word 'strawberry'?"

    print("=== Call 1 ===")
    msg1 = call([{"role": "user", "content": question}])
    print("Answer:", msg1.get("content"))
    print("Reasoning returned:", bool(msg1.get("reasoning_details") or msg1.get("reasoning")))

    messages = [
        {"role": "user", "content": question},
        {
            "role": "assistant",
            "content": msg1.get("content"),
            "reasoning_details": msg1.get("reasoning_details"),  # pass back unmodified
        },
        {"role": "user", "content": "Are you sure? Think carefully."},
    ]

    print("\n=== Call 2 (continues reasoning) ===")
    msg2 = call(messages)
    print("Answer:", msg2.get("content"))

    print("\nConnection works.")


if __name__ == "__main__":
    main()