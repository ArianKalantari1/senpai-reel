#!/usr/bin/env python3
"""Test available Cerebras models for JSON extraction."""
import sys
sys.path.insert(0, ".")

import tomllib
with open(".streamlit/secrets.toml", "rb") as f:
    s = tomllib.load(f)

key = s.get("CEREBRAS_API_KEY", "")
from cerebras.cloud.sdk import Cerebras
import json

c = Cerebras(api_key=key)

test_transcript = (
    "Always tailor your resume with keywords from the job description. "
    "Quantify achievements with numbers. Use the STAR method in behavioural interviews."
)
system = (
    'Extract 1-3 tips as JSON with this shape: {"units":[{"text":"...","claim":"...","topic":"Resume","content_type":"tip","confidence":0.9}]}'
)

for model in ["qwen-3-235b-a22b-instruct-2507", "llama3.1-8b"]:
    print(f"Testing {model}...")
    try:
        resp = c.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"Transcript: {test_transcript}"},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=500,
        )
        content = resp.choices[0].message.content
        parsed = json.loads(content)
        units = parsed.get("units", [])
        first = units[0]["text"][:60] if units else "none"
        print(f"  OK: {len(units)} units — {first}")
        print(f"  Tokens: {resp.usage.prompt_tokens} in + {resp.usage.completion_tokens} out")
        print(f"  WINNER: {model}")
        break
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
