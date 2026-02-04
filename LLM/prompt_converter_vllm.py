#!/usr/bin/env python3
# prompt_converter_vllm_http.py — vLLM prompt converter for comm_manager
# API:
#   POST /prompts  {"sentence":"..."}  -> {"prompts":[...]}
#   GET  /health

import os
import json
import requests
from flask import Flask, request, jsonify

# ---------------- config ----------------
VLLM_URL   = os.getenv("VLLM_URL", "http://127.0.0.1:8000").rstrip("/")
VLLM_MODEL = os.getenv("VLLM_MODEL", "espressor/meta-llama.Llama-3.2-3B-Instruct_W4A16")
TIMEOUT_S  = float(os.getenv("VLLM_TIMEOUT", "20"))

MAX_TOKENS   = int(os.getenv("CONVERTER_MAX_TOKENS", "32"))
TEMPERATURE  = float(os.getenv("CONVERTER_TEMPERATURE", "0.2"))

PROMPT_PREFIX = (
    "Extract unique object names from the text."
    "Return only a lowercase JSON array. No extra text. "
    "Remove colors, sizes, materials, and adjectives: "
)

# Reuse connections for speed
SESSION = requests.Session()

app = Flask(__name__)


def extract_prompts(sentence: str) -> list[str]:
    payload = {
        "model": VLLM_MODEL,
        "messages": [
            {"role": "user", "content": PROMPT_PREFIX + sentence}
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
    }

    r = SESSION.post(
        f"{VLLM_URL}/v1/chat/completions",
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=TIMEOUT_S,
    )
    r.raise_for_status()
    j = r.json()

    content = j["choices"][0]["message"]["content"].strip()

    # Trust model: must be JSON array
    arr = json.loads(content)

    if not isinstance(arr, list):
        raise ValueError(f"Expected JSON array, got: {content}")

    # tiny cleanup: lowercase + unique (minimal)
    out, seen = [], set()
    for x in arr:
        if not isinstance(x, str):
            continue
        x = x.strip().lower()
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


@app.post("/prompts")
def prompts():
    data = request.get_json(silent=True) or {}
    sentence = (data.get("sentence") or data.get("caption") or "").strip()
    if not sentence:
        return jsonify({"error": "missing 'sentence'"}), 400

    try:
        prompts_list = extract_prompts(sentence)
        return jsonify({"prompts": prompts_list}), 200
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"upstream vLLM error: {e}"}), 502
    except Exception as e:
        # includes JSON parse errors if model didn't comply
        return jsonify({"error": str(e)}), 500


@app.get("/health")
def health():
    return jsonify({"ok": True}), 200


if __name__ == "__main__":
    # comm_manager expects Jetson2 endpoint like http://<ip>:5050/prompts
    # so we listen on 0.0.0.0:5050 by default
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "5050"))
    print(f"[prompt_converter_vllm_http] listening on {host}:{port}")
    print(f"  vllm_url={VLLM_URL}")
    print(f"  model={VLLM_MODEL}")
    app.run(host=host, port=port, debug=False)
