"""Minimal OpenRouter chat client (key from OPENROUTER_API_KEY or .env), used by make_data.py."""
import os
import sys
import time

import requests
from dotenv import load_dotenv

from config import OPENROUTER_MAX_TOKENS, OPENROUTER_REASONING, OPENROUTER_RETRIES, TEMPERATURE, TOP_P

URL = "https://openrouter.ai/api/v1/chat/completions"
load_dotenv()


def chat(model, system, user):
    """One chat completion; returns the reply text. Retries rate limits, empty replies and transient errors."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        sys.exit("OPENROUTER_API_KEY missing: put OPENROUTER_API_KEY=sk-or-... in .env")
    body = {"model": model, "temperature": TEMPERATURE, "top_p": TOP_P, "max_tokens": OPENROUTER_MAX_TOKENS,
            "reasoning": {"enabled": OPENROUTER_REASONING},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    for attempt in range(OPENROUTER_RETRIES):
        wait = 5 * 2 ** attempt
        try:
            resp = requests.post(URL, headers={"Authorization": f"Bearer {key}"}, json=body, timeout=120)
        except requests.RequestException as e:
            print(f"  request failed ({e}); retrying in {wait}s", flush=True)
            time.sleep(wait)
            continue
        if resp.status_code in (401, 402, 404):  # retrying cannot fix these
            sys.exit(f"HTTP {resp.status_code}: {resp.text}")
        data = resp.json() if resp.status_code == 200 else {}
        if resp.status_code == 200 and "error" not in data:
            text = (data["choices"][0]["message"].get("content") or "").strip()
            if text:
                return text
            print("  empty reply (reasoning may have used all tokens); retrying", flush=True)
            continue
        if resp.status_code == 429 and "per-day" in resp.text:  # daily quota: waiting a minute will not help
            sys.exit(f"Daily OpenRouter limit reached: {resp.text[:200]}\nRows so far are saved; rerun tomorrow to resume.")
        wait = 60 if resp.status_code == 429 else wait  # free tier is 20 requests/min
        print(f"  HTTP {resp.status_code} {resp.text[:200]}; retrying in {wait}s", flush=True)
        time.sleep(wait)
    sys.exit(f"OpenRouter failed {OPENROUTER_RETRIES} times in a row; rerun to resume from the saved rows")
