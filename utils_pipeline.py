#!/usr/bin/env python3
import argparse
import datetime
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import requests
@dataclass
class VlmResult:
    ok: bool
    caption: Optional[str]
    error: Optional[str]


def log(msg: str):
    print(msg, flush=True)


def pick_latest_subdir(parent: str, method: str, exclude_suffix: str = "_ann") -> str:
    if not os.path.isdir(parent):
        raise FileNotFoundError(f"Parent dir does not exist: {parent}")

    subdirs = []
    for name in os.listdir(parent):
        if exclude_suffix and name.endswith(exclude_suffix):
            continue
        p = os.path.join(parent, name)
        if os.path.isdir(p):
            subdirs.append(p)

    if not subdirs:
        raise FileNotFoundError(f"No subfolders found under: {parent} (after excluding '{exclude_suffix}')")

    if method == "mtime":
        return max(subdirs, key=lambda p: os.path.getmtime(p))

    return max(subdirs, key=lambda p: os.path.basename(p))



def remap_path(path: str, src: Optional[str], dst: Optional[str]) -> str:
    """
    If your VLM server runs in a different container / mount namespace,
    you can remap local path -> server-visible path.

    Example:
      src=/home/user/jetson-containers/data
      dst=/mnt/VLM/jetson-data

    Then:
      /home/user/jetson-containers/data/R2/... -> /mnt/VLM/jetson-data/R2/...
    """
    if not src or not dst:
        return path

    src = os.path.abspath(src)
    path_abs = os.path.abspath(path)

    try:
        rel = os.path.relpath(path_abs, src)
    except Exception:
        return path

    # If rel begins with .., the path is not under src; don't remap
    if rel.startswith(".."):
        return path

    return os.path.normpath(os.path.join(dst, rel))


def load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj: dict):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def _try_parse_json_string(s: str) -> Optional[dict]:
    s = (s or "").strip()
    if not s:
        return None
    if not (s.startswith("{") and s.endswith("}")):
        return None
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def extract_prompt_response(vlm_caption_or_payload: str) -> Tuple[str, str]:
    """
    Accepts either:
    - plain caption string
    - JSON string with fields like auto_prompt/response_describe/...
    Returns: (prompt, response)
    """
    payload = _try_parse_json_string(vm := (vlm_caption_or_payload or ""))

    if payload:
        prompt = payload.get("auto_prompt") or payload.get("prompt") or "Describe the objects in the image"
        # Prefer your server's describe field names
        response = (
            payload.get("response_describe")
            or payload.get("response")
            or payload.get("text")
            or payload.get("caption")
            or ""
        )
        return str(prompt), str(response)

    # Not JSON → treat as raw caption
    return ("Describe the objects in the image", vm)



def already_captioned(js: dict) -> bool:
    # Your sidecar likely uses "vlm_text"; handle a couple of variants safely.
    if js.get("vlm_text"):
        return True
    vlm = js.get("vlm", {})
    if isinstance(vlm, dict) and vlm.get("text"):
        return True
    return False


def parse_caption_from_response(resp: requests.Response) -> str:
    """
    Try common response shapes:
      - {"text": "..."}
      - {"caption": "..."}
      - {"description": "..."}
      - or plain text body
    """
    ctype = (resp.headers.get("Content-Type") or "").lower()
    if "application/json" in ctype:
        data = resp.json()
        if isinstance(data, dict):
            for k in ("text", "caption", "description", "result"):
                v = data.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()
        # fallback: stringify
        return json.dumps(data, ensure_ascii=False)

    # plain text fallback
    return resp.text.strip()
