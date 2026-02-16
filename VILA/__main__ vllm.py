#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VILA server/CLI:
1) CLI mode (interactive prompts as before).
2) HTTP API server mode (--server) to accept image_path (and optional question),
   run VILA, and append to a local JSON-by-image file.

Changes for new architecture:
- Keep per-image JSON *locally* only (no remote forwarding of JSON or images).
- Optional notify hook: --notify-url (text/plain) sends ONLY the VLM textual output
  to a local comm-manager endpoint on every generation.
"""

import os
import sys
import time
import json
import signal
import logging
import threading

import urllib.request, urllib.error
from urllib.parse import urlparse

from termcolor import cprint
import numpy as np
from types import SimpleNamespace

# NanoLLM stack (assumes your existing environment)
try:
    from nano_llm import NanoLLM, ChatHistory, ChatTemplates, BotFunctions
    from nano_llm.utils import ImageExtensions, ArgParser, KeyboardInterrupt, load_prompts, print_table
except Exception:
    # Provide light fallbacks so --api vllm mode can run without nano_llm installed.
    import argparse
    ImageExtensions = ["jpg", "jpeg", "png", "webp", "bmp", "tiff", "tif", "gif"]

    class ArgParser(argparse.ArgumentParser):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)

    class KeyboardInterrupt:
        def __init__(self):
            pass
        def reset(self):
            pass
        def __bool__(self):
            return False

    def load_prompts(x):
        if not x:
            return []
        # If x is a filename, try to read lines; otherwise treat as single prompt
        try:
            if os.path.exists(x):
                with open(x, "r", encoding="utf-8") as f:
                    return [l.strip() for l in f if l.strip()]
        except Exception:
            pass
        return [x]

    def print_table(x):
        return

    class NanoLLM:
        @staticmethod
        def from_pretrained(*a, **kw):
            raise RuntimeError("nano_llm not installed; only --api vllm mode works with fallback")

    class ChatTemplates:
        pass

    class BotFunctions:
        @staticmethod
        def run(*a, **kw):
            return None

    class ChatHistory:
        def __init__(self, model=None, chat_template=None, system_prompt=None):
            self.messages = []
            self.kv_cache = None
            self.template = SimpleNamespace(stop=[])
        def append(self, role, content):
            self.messages.append({"role": role, "content": content})
        def reset(self):
            self.messages = []
        def embed_chat(self, *a, **kw):
            return None, None
        def turn(self, who="user"):
            return True

# ---------------------------
# Lightweight HTTP client (stdlib)
# ---------------------------
import urllib.request
import urllib.error

def _http_post_text(url: str, text: str, timeout: float = 6.0) -> tuple[int, str]:
    """
    POST plain text to 'url' as text/plain; utf-8.
    Returns: (status_code, response_text) or (-1, err) on network errors.
    """
    req = urllib.request.Request(
        url=url,
        data=(text or "").encode("utf-8"),
        headers={"Content-Type": "text/plain; charset=utf-8"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = getattr(resp, "status", 200)
            body = resp.read().decode("utf-8", errors="replace")
            return status, body
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace") if e.fp else str(e)
        return e.code, body
    except Exception as e:
        return -1, str(e)

# 1. Update the helper to handle JSON
def _http_post_json_vila(url: str, payload: dict, timeout: float = 45.0):
    import json
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8")
    except Exception as e:
        return -1, str(e)


# ---------------------------
# Helpers for JSON by image (local only)
# ---------------------------

def _ext_of(path: str) -> str:
    """Return the lowercase extension of a local path or URL."""
    p = path.strip().strip("'").strip('"')
    if p.lower().startswith(("http://", "https://")):
        parsed = urlparse(p)
        _, ext = os.path.splitext(parsed.path)
        return ext.lower()
    _, ext = os.path.splitext(p)
    return ext.lower()

def _is_image_path_or_url(user_text: str) -> bool:
    """Detect if the user_text looks like an image path or image URL."""
    if not user_text:
        return False
    ext = _ext_of(user_text)
    normalized = {e if e.startswith(".") else f".{e}" for e in (ImageExtensions if isinstance(ImageExtensions, (list, set, tuple)) else [])}
    if not normalized:
        # Fallback if ImageExtensions is not provided
        normalized = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif", ".gif"}
    return ext in normalized

def _json_path_for_image(image_path_or_url: str) -> str:
    """Return the JSON filename that corresponds to the image (next to it or derived from URL)."""
    p = image_path_or_url.strip().strip("'").strip('"')
    if p.lower().startswith(("http://", "https://")):
        parsed = urlparse(p)
        base = os.path.basename(parsed.path) or "image"
        name, _ = os.path.splitext(base)
        return f"{name}.json"
    name, _ = os.path.splitext(p)
    return f"{name}.json"


# ---------------------------
# Arguments
# ---------------------------

parser = ArgParser()

# Colors and features
parser.add_argument("--prompt-color", type=str, default="blue", help="termcolor name for user prompts")
parser.add_argument("--reply-color", type=str, default="green", help="termcolor name for model replies")
parser.add_argument("--enable-tools", action="store_true", help="allow tool/function calls")

# Streaming and stats
parser.add_argument("--disable-automatic-generation", action="store_false", dest="automatic_generation", help="wait for 'generate' command")
parser.add_argument("--disable-streaming", action="store_true", help="disable token streaming output")
parser.add_argument("--disable-stats", action="store_true", help="suppress generation performance stats")

# Save JSON by image toggles (local only)
parser.add_argument("--save-json-by-image", action="store_true",
                    help="After each bot reply, append JSON bound to the last image path/URL provided in chat. JSON filename is <image_path>.json")
parser.add_argument("--json-indent", type=int, default=2, help="Indentation for JSON (0 to minify)")

# HTTP server mode
parser.add_argument("--server", action="store_true",
                    help="Run as HTTP server that accepts image_path/question and triggers VILA, saving JSON per image like CLI.")
parser.add_argument("--port", type=int, default=8080, help="Port for --server mode (default: 8080)")

# vLLM HTTP server (optional)
parser.add_argument("--vllm-url", type=str, default="http://127.0.0.1:8080",
                    help="Base URL for a running vLLM server (e.g. http://127.0.0.1:8080)")

# NEW: notify comm-manager (message-only)
parser.add_argument(
    "--notify-url",
    type=str,
    default="",
    help="If set, POST the VLM textual output (text/plain) to this URL after generation (e.g., http://127.0.0.1:5050/from_vila)."
)

args = parser.parse_args()

prompts = load_prompts(args.prompt)
interrupt = KeyboardInterrupt()
tool_response = None

# Track the most recent image the user provided (used for JSON filename)
last_image_path = None

# ---------------------------
# Load Model or configure vLLM HTTP client
# ---------------------------

if getattr(args, "api", "").lower() == "vllm":
    # Thin local model placeholder so other code can read repo_id/name
    model = SimpleNamespace()
    model.repo_id = args.model
    model.name = args.model
    model.api = "vllm"
    class _cfg: pass
    model.config = SimpleNamespace(max_length=getattr(args, "max_context_len", 2048))
    model.stats = {}
else:
    model = NanoLLM.from_pretrained(
        args.model,
        api=args.api,
        quantization=args.quantization,
        max_context_len=args.max_context_len,
        vision_api=args.vision_api,
        vision_model=args.vision_model,
        vision_scaling=args.vision_scaling,
    )


def _call_vllm_chat_completion(messages, *, model_name: str, vllm_url: str, max_tokens: int = 128, temperature: float = 1.0, top_p: float = 1.0):
    """
    messages: list of {role: 'user'|'assistant'|'system', content: [ {type:text|image_url, ...}, ... ] }
    Returns: (success_bool, reply_text or error)
    """
    url = vllm_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "max_num_seqs": 1,
    }
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = resp.read().decode("utf-8")
            obj = json.loads(body)
    except Exception as e:
        return False, str(e)

    # Extract text content from response choices
    try:
        choice = obj.get("choices", [])[0]
        msg = choice.get("message") if isinstance(choice, dict) else None
        content = msg.get("content") if isinstance(msg, dict) else None
        # content may be an array of items or a string
        reply_text = ""
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    reply_text += str(item)
                else:
                    if item.get("type") == "text":
                        reply_text += item.get("text", "")
                    elif item.get("type") == "image_url":
                        # skip images in textual reply
                        continue
                    else:
                        reply_text += item.get("text", "") if "text" in item else str(item)
        elif isinstance(content, str):
            reply_text = content
        else:
            # fallback attempt
            reply_text = json.dumps(content, ensure_ascii=False)
        return True, reply_text
    except Exception as e:
        return False, f"failed to parse vLLM response: {e} -> {obj}"

# ---------------------------
# Chat history
# ---------------------------

chat_history = ChatHistory(model, args.chat_template, args.system_prompt)

# ---------------------------
# Append (local JSON only)
# ---------------------------

def _append_entry_to_json(
    json_path: str,
    image_path_or_url: str,
    model,
    prompt_text: str,
    reply_text: str,
    indent: int = 2
):
    """
    Append a single {timestamp, prompt, response} record into the per-image JSON file (local only).
    NOTE: No remote forwarding from here.
    """
    record = {
        "timestamp": int(time.time()),
        "prompt": prompt_text,
        "response": reply_text,
    }

    doc = None
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                doc = json.load(f)
        except Exception:
            doc = None

    if not isinstance(doc, dict):
        doc = {
            "image_path": image_path_or_url.strip().strip("'").strip('"'),
            "model": getattr(model, "repo_id", None) or getattr(model, "name", None),
            "api": getattr(model, "api", None),
            "entries": []
        }

    doc.setdefault("entries", [])
    doc["entries"].append(record)

    # Persist locally
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=indent)

    cprint(f"[saved] {json_path}", "cyan")


# ---------------------------
# Single "run cycle"
# ---------------------------

_run_lock = threading.Lock()

def process_user_prompt(user_prompt: str, *, generate: bool = True) -> str:
    """
    Execute one cycle:
    - Detect if prompt is an image path/URL and update last_image_path.
    - Append user prompt.
    - Optionally embed_chat + generate with the model.
    - Append bot reply.
    - Optionally save JSON bound to last_image_path (local).
    - Optionally notify comm-manager with message-only text.
    Returns the textual reply produced by the model (or "" if generate=False).
    """
    global last_image_path

    # Detect image path/URL
    if _is_image_path_or_url(user_prompt):
        last_image_path = user_prompt.strip().strip("'").strip('"')

    # Append user message into chat history
    chat_history.append("user", user_prompt)

    # If we only want to append (no generation), exit early
    if not generate:
        return ""

    # Generate step: support native NanoLLM or remote vLLM HTTP API
    gen_start = time.perf_counter()
    # If using vLLM HTTP API (set via --api vllm), build the conversation and POST
    if getattr(args, "api", "").lower() == "vllm":
        # Build messages for vLLM from chat_history.messages
        messages = []
        for m in getattr(chat_history, "messages", []):
            role = m.get("role", "user")
            content_text = m.get("content", "")
            if _is_image_path_or_url(content_text):
                content = [{"type": "image_url", "image_url": {"url": content_text}}]
            else:
                content = [{"type": "text", "text": content_text}]
            vrole = "user" if role == "user" else ("assistant" if role in ("bot", "assistant") else "system")
            messages.append({"role": vrole, "content": content})

        ok, result = _call_vllm_chat_completion(
            messages,
            model_name=args.model,
            vllm_url=args.vllm_url,
            max_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        if not ok:
            cprint(f"[error] vLLM request failed: {result}", "red")
            chat_history.append("bot", f"[error] generation failed: {result}")
            if args.notify_url:
                _http_post_text(args.notify_url.strip(), f"[error] generation failed: {result}", timeout=10.0)
            return f"[error] generation failed: {result}"

        reply_text = result
        # Print reply (no streaming support for HTTP backend)
        cprint(reply_text, args.reply_color)
        gen_end = time.perf_counter()
        print(f"[TICTOK] vllm_generate_total: {(gen_end - gen_start):.3f}s")
    else:
        # Native NanoLLM path (existing behavior)
        try:
            embedding, position = chat_history.embed_chat(
                max_tokens=model.config.max_length - args.max_new_tokens,
                wrap_tokens=args.wrap_tokens,
                use_cache=getattr(model, "has_embed", False) and chat_history.kv_cache,
            )
        except Exception:
            embedding, position = None, None

        try:
            reply = model.generate(
                embedding,
                streaming=not args.disable_streaming,
                kv_cache=chat_history.kv_cache,
                cache_position=position,
                stop_tokens=chat_history.template.stop,
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=args.min_new_tokens,
                do_sample=args.do_sample,
                repetition_penalty=args.repetition_penalty,
                temperature=args.temperature,
                top_p=args.top_p,
            )
        except Exception as e:
            cprint(f"[error] generate() failed: {e}", "red")
            chat_history.append("bot", f"[error] generation failed: {e}")
            # notify error as text if needed
            if args.notify_url:
                _http_post_text(args.notify_url.strip(), f"[error] generation failed: {e}", timeout=10.0)
            return f"[error] generation failed: {e}"

        reply_text = ""
        if args.disable_streaming:
            # Non-streaming mode: reply is a single string
            reply_text = reply
            cprint(reply_text, args.reply_color)
            gen_end = time.perf_counter()
            print(f"[TICTOK] generate_total: {(gen_end - gen_start):.3f}s")
        else:
            # Streaming mode: reply yields tokens
            first_token_time = None
            token_count = 0
            for token in reply:
                now = time.perf_counter()
                if first_token_time is None:
                    first_token_time = now
                    print(f"[TICTOK] TTFT: {(first_token_time - gen_start)*1000:.2f} ms")
                cprint(token, args.reply_color, end="", flush=True)
                reply_text += token
                token_count += 1
                if interrupt:
                    try:
                        reply.stop()
                    except Exception:
                        pass
                    interrupt.reset()
                    break

            gen_end = time.perf_counter()
            total_time = gen_end - gen_start
            if token_count > 0:
                throughput = token_count / (gen_end - (first_token_time or gen_start))
                print(f"\n[TICTOK] generate_total: {total_time:.3f}s | tokens: {token_count} | throughput: {throughput:.2f} tok/s")

    print("")  # newline after generation

    if not args.disable_stats:
        print_table(model.stats)
        print("")

    # Append bot reply to chat history
    chat_history.append("bot", reply_text)

    # ---- Local JSON persistence (optional) ----
    if args.save_json_by_image:
        if last_image_path:
            json_path = _json_path_for_image(last_image_path)
            _append_entry_to_json(
                json_path=json_path,
                image_path_or_url=last_image_path,
                model=model,
                prompt_text=user_prompt,
                reply_text=reply_text,
                indent=(None if args.json_indent == 0 else args.json_indent),
            )
        else:
            cprint("[warn] --save-json-by-image is enabled, but no image path/URL was provided yet.", "red")

    # ---- Notify comm-manager (message-only) ----
    if args.notify_url and reply_text.strip():
        payload = {
            "caption": reply_text.strip(),
            "image_name": os.path.basename(last_image_path) if last_image_path else None
        }
        status, body = _http_post_json_vila(args.notify_url.strip(), payload, timeout=45.0)
        # status, body = _http_post_text(args.notify_url.strip(), reply_text.strip(), timeout=10.0)
        if status in (200, 201):
            cprint(f"[notify] sent caption to {args.notify_url} (status {status})", "cyan")
        else:
            cprint(f"[notify][warn] failed to notify {args.notify_url} (status {status}): {body}", "yellow")

    return reply_text


# ---------------------------
# HTTP Server mode (Flask)
# ---------------------------

if args.server:
    # Lazy import so Flask is only required in --server mode.
    from flask import Flask, request, jsonify

    app = Flask(__name__)

    @app.route("/describe", methods=["POST"])
    def describe():
        """
        JSON in:
          {
            "image_path": "/data/images/01.jpg",   # required
            "question": "optional follow-up"       # optional
          }

        Behavior:
          - Hard reset of chat context for every request (prevents leakage).
          - Append the image (no generation).
          - Auto-inject "Describe the image" and generate.
          - If 'question' provided, ask it as a second turn and generate.
          - Local JSON-by-image saving remains intact via process_user_prompt.
          - If --notify-url is set, each generation is sent as text/plain to it.
        """
        body = request.get_json(force=True, silent=False) or {}
        image_path = (body.get("image_path") or "").strip()
        question   = (body.get("question")   or "").strip()

        if not image_path:
            return jsonify({"error": "image_path is required"}), 400

        with _run_lock:
            # RESET between requests
            chat_history.reset()
            globals()['last_image_path'] = None

            # 1) add the image to history (no generation)
            process_user_prompt(image_path, generate=False)

            # 2) auto prompt
            auto_prompt = "Describe the objects in the image"
            resp_describe = process_user_prompt(auto_prompt, generate=True)

            # 3) optional follow-up
            resp_question = None
            if question:
                resp_question = process_user_prompt(question, generate=True)

        return jsonify({
            "ok": True,
            "image_path": image_path,
            "auto_prompt": auto_prompt,
            "response_describe": resp_describe,
            "response_question": resp_question
        })

    @app.get("/health")
    def health():
        """Simple health endpoint to verify server is up."""
        return jsonify({"ok": True, "time": int(time.time())})

    # Start the server and exit the CLI flow
    app.run(host="0.0.0.0", port=args.port)
    sys.exit(0)


# ---------------------------
# CLI mode (unchanged behavior)
# ---------------------------

while True:
    if chat_history.turn("user"):
        # Fetch next prompt
        if isinstance(prompts, list):
            if len(prompts) > 0:
                user_prompt = prompts.pop(0)
                cprint(f">> PROMPT: {user_prompt}", args.prompt_color)
            else:
                break
        else:
            cprint(">> PROMPT: ", args.prompt_color, end="", flush=True)
            user_prompt = sys.stdin.readline().strip()

        print("")

        # Load from file or reset if needed
        if user_prompt.lower().endswith((".txt", ".json")):
            user_prompt = " ".join(load_prompts(user_prompt))
        elif user_prompt.lower() in ("reset", "clear"):
            logging.info("resetting chat history")
            chat_history.reset()
            last_image_path = None
            continue

        # Process one cycle with the given prompt
        process_user_prompt(user_prompt)

    # Optional tool functions
    if args.enable_tools:
        tool_response = BotFunctions.run(
            chat_history.messages[-1]["content"] if chat_history.messages else "",
            template=chat_history.template
        )
        if tool_response:
            chat_history.append("tool_response", tool_response)
            cprint(tool_response, "yellow")
