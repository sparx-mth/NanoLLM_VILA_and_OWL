#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
comm_manager.py

Flow:
  1) Receive caption from VILA via POST /from_vila  (Content-Type: text/plain)
  2) Forward caption to Jetson2 /prompts -> expect {"prompts":[...]}
  3) Only after prompts are received, find the latest captured image under --captures-root
  4) POST to NanoOWL /infer with multipart form:
        -F image=@/path/to/image.jpg
        -F prompts='["one","two","..."]'
        -F annotate=0/1
  5) Write OWL output into the sidecar JSON of that image under key "nanoowl"
  6) **NEW:** Auto-annotate the image with OpenCV (draw BBox + label),
     writing <basename>_ann.jpg next to the original image.

Notes:
- This manager does NOT send images or JSON to remote machines other than:
    * Jetson2 (/prompts) for prompts
    * NanoOWL (/infer) for detections
- It stores OWL results locally in the image's JSON and renders an annotated image.


run command both process folder and comm original
python3 comm_manager_2.py   --host 0.0.0.0    --port 5050     --jetson2-endpoint http://192.168.131.21:5050/prompts       --captures-root /home/user/jetson-containers/data/R2/    --nanoowl-endpoint http://192.168.131.22:5060/infer  --forward-timeout 45   --forward-retries 3     --nanoowl-timeout 70   --nanoowl-annotate 0     --forward-json-url http://192.168.131.23:9090/ingest --endpoint http://192.168.131.22:8080/describe    --watch-interval 5.0 --sleep-between 20 --vlm-timeout 60

"""
import datetime
import queue
import signal
import sys
import threading
from collections.abc import Callable

from typing import Optional, Tuple
from flask import Flask, request, jsonify
from pathlib import Path
import os
import json
import time
import glob
import shutil
import argparse
from collections import deque
import hashlib
import re
import urllib.request, urllib.error  # for Jetson2 JSON POST
import requests                      # for NanoOWL multipart
import cv2                           # for drawing boxes

from utils_pipeline import log, load_json, already_captioned, extract_prompt_response, remap_path, \
    parse_caption_from_response, VlmResult

shutdown_event = threading.Event()
vlm_caption_queue = queue.Queue()
app = Flask(__name__)

# --- Runtime configuration (populated from CLI args) ---
NANOOWL_ENDPOINT = None      # e.g., http://172.16.17.11:5060/infer
CAPTURES_ROOT = None         # e.g., /home/user/jetson-containers/data/images/captures

NANOOWL_TIMEOUT = 45.0       # NanoOWL infer timeout
NANOOWL_ANNOTATE = 0         # annotate flag sent to NanoOWL (0/1)

_ANN_RE = re.compile(r"_ann\.(jpg|jpeg|png)$", re.IGNORECASE)

FORWARD_JSON_URL = None       # e.g., http://172.17.16.9:9090/ingest
FORWARD_JSON_TIMEOUT = 8.0
FORWARD_JSON_RETRIES = 3

# --- Simple in-memory log/state for quick debugging ---
HISTORY = deque(maxlen=200)
LAST = {
    "vila_caption": None,        # {"ts": int, "text": str}
    "last_forward_status": None, # {"status": int, "body": str/dict}
    "last_image_path": None,     # str
    "nanoowl_result": None,      # {"status": int, "body": any}
}


VLLM_URL = None
VLLM_MODEL = None
VLLM_TIMEOUT = 20.0
VLLM_MAX_TOKENS = 32
VLLM_TEMPERATURE = 0.2

VLLM_PROMPT_PREFIX = (
    "Extract unique object names from the text."
    "Return only a lowercase JSON array. No extra text. "
    "Remove colors, sizes, materials, and adjectives: "
)

_VLLM_SESSION = requests.Session()

# -------------------- Helpers --------------------

def _is_in_ann_folder(fp: str) -> bool:
    p = Path(fp)
    return any(str(part).lower().endswith("_ann") for part in p.parents)

def _find_latest_image_and_json(root_dir: str):
    if not root_dir or not os.path.isdir(root_dir):
        print("Failed #1")
        return None, None

    root_path = Path(root_dir)
    assert root_path.exists(), f"{root_dir} does not exist"
    latest_folder_path = root_path / "latest"
    if not latest_folder_path.exists():
        return None, None
    assert latest_folder_path.is_dir(), f"{latest_folder_path} is not a dir"
    latest_folder = str(latest_folder_path)

    latest_img = None
    latest_mtime = -1.0
    print(f"latest folder: {latest_folder}")

    for fp in latest_folder_path.iterdir():
        if not fp.is_file() or fp.suffix not in [".jpg", ".jpeg", ".png"]:
            continue
        print(f"Processing {fp}")

        try:
            ctime = os.path.getctime(fp)
            if ctime > latest_mtime:
                latest_mtime = ctime
                latest_img = fp
        except Exception as exp:
            print(f"Failed #3 {exp}")
            pass
    if not latest_img:
        return None, None

    latest_img = str(latest_img)
    print(f"latest image: {latest_img}")

    base, _ = os.path.splitext(latest_img)
    sidecar_json = base + ".json"
    return latest_img, sidecar_json


def _update_sidecar_json(json_path: str, updater: dict):
    """
    Robustly updates the JSON by merging new data with existing content on disk.
    """
    for attempt in range(5):  # Retry up to 5 times if file is busy
        try:
            obj = {}
            if os.path.isfile(json_path):
                with open(json_path, "r", encoding="utf-8") as f:
                    obj = json.load(f)

            # Deep merge the new data (OWL or VLM results)
            for k, v in updater.items():
                if isinstance(v, dict) and k in obj and isinstance(obj[k], dict):
                    obj[k].update(v)
                else:
                    obj[k] = v

            # Write to a unique temp file first to prevent corruption
            tmp = f"{json_path}.{os.getpid()}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)

            os.replace(tmp, json_path)
            return True
        except (IOError, json.JSONDecodeError) as e:
            time.sleep(0.1)  # Wait for other process to release file
    return False



def _post_nanoowl_multipart(endpoint: str, image_path: str, prompts: list[str],
                            annotate: int, timeout: float):
    """
    Send multipart/form-data to NanoOWL:
      files: image=@<path>
      data:  prompts='["a","b"]', annotate='0'/'1'
    Returns (status_code, response_json_or_text)
    """
    if not endpoint:
        return -1, "nanoowl endpoint not configured"
    if not (image_path and os.path.isfile(image_path)):
        return -1, f"image not found: {image_path}"
    files = {"image": (os.path.basename(image_path), open(image_path, "rb"), "application/octet-stream")}
    data = {"prompts": json.dumps(prompts or []), "annotate": str(int(annotate))}
    try:
        r = requests.post(endpoint, files=files, data=data, timeout=timeout)
        try:
            body = r.json()
        except Exception:
            body = r.text
        return r.status_code, body
    except Exception as e:
        return -1, str(e)


def _ann_outpath_for_image(image_path: str) -> str:
    """
    Return output path for annotated image inside a *run-level* folder named <run_dir>_ann.
    Example:
      image_path = /.../captures/2025_10_19___15_53_28/x-010y017z055yaw0000000___2025_10_19___15_54_07.jpg
      => /.../captures/2025_10_19___15_53_28_ann/x-010y017z055yaw0000000___2025_10_19___15_54_07_ann.jpg
    """
    base_dir = os.path.dirname(image_path)                     # e.g. .../captures/2025_10_19___15_53_28
    parent_dir = os.path.dirname(base_dir)                     # e.g. .../captures

    run_name_path = Path(base_dir)# e.g. 2025_10_19___15_53_28
    assert run_name_path.is_symlink(), f"Expected symlink {run_name_path} to point to a run folder"
    run_name = str(run_name_path.resolve().name)
    ann_dir = os.path.join(parent_dir, f"{run_name}_ann")      # e.g. .../captures/2025_10_19___15_53_28_ann
    os.makedirs(ann_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    base_name = re.sub(r"_ann$", "", base_name, flags=re.IGNORECASE)
    out_name = f"{base_name}_ann.jpg"

    return os.path.join(ann_dir, out_name)

def _vllm_extract_prompts(sentence: str) -> list[str]:
    """
    Extracts a list of prompts from a given sentence using the vLLM API.

    This function communicates with a vLLM service to derive an array of prompts
    based on the input sentence. It sends the sentence along with specific parameters
    to an API endpoint and expects a response containing a JSON array.

    Parameters:
    sentence (str): The input sentence to process and retrieve prompts for.

    Returns:
    list[str]: A list of unique, lowercase prompts extracted from the input sentence.

    Raises:
    ValueError: If the response from the vLLM API is not formatted as a JSON array.
    """
    if not VLLM_URL:
        return []

    payload = {
        "model": VLLM_MODEL,
        "messages": [{"role": "user", "content": VLLM_PROMPT_PREFIX + sentence}],
        "max_tokens": int(VLLM_MAX_TOKENS),
        "temperature": float(VLLM_TEMPERATURE),
    }

    r = _VLLM_SESSION.post(
        f"{VLLM_URL.rstrip('/')}/v1/chat/completions",
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=float(VLLM_TIMEOUT),
    )
    r.raise_for_status()
    j = r.json()
    content = j["choices"][0]["message"]["content"].strip()

    arr = json.loads(content)  # trusting model output is JSON array

    if not isinstance(arr, list):
        raise ValueError(f"Expected JSON array from vLLM, got: {content}")

    # tiny cleanup: lowercase + unique
    out, seen = [], set()
    for x in arr:
        if isinstance(x, str):
            x2 = x.strip().lower()
            if x2 and x2 not in seen:
                seen.add(x2)
                out.append(x2)
    return out



# -------------------- Annotation utilities (OpenCV) --------------------

def _color_for_label(label: str):
    """
    Deterministic BGR color from label string.
    """
    h = hashlib.md5(label.encode("utf-8")).hexdigest()
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    return (b, g, r)  # OpenCV uses BGR

def _extract_detections(nanoowl_result):
    """
    Normalize detections into:
      [{"label": str, "score": float|None, "bbox": [x1,y1,x2,y2]}]
    Accepts either:
      - {"detections": [ ... ]}
      - [ ... ] (plain list)
      - {"items": [ ... ]} (fallback)
    """
    if nanoowl_result is None:
        return []

    if isinstance(nanoowl_result, dict) and "detections" in nanoowl_result:
        dets = nanoowl_result.get("detections") or []
    elif isinstance(nanoowl_result, list):
        dets = nanoowl_result
    elif isinstance(nanoowl_result, dict) and "items" in nanoowl_result:
        dets = nanoowl_result["items"]
    else:
        return []

    norm = []
    for d in dets:
        if not isinstance(d, dict):
            continue
        label = d.get("label") or d.get("name") or d.get("text") or "object"
        score = d.get("score") or d.get("confidence") or None
        bbox  = d.get("bbox") or d.get("box") or d.get("xyxy") or None
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            continue
        try:
            x1, y1, x2, y2 = [float(v) for v in bbox]
        except Exception:
            continue
        norm.append({
            "label": str(label),
            "score": (float(score) if score is not None else None),
            "bbox": [x1, y1, x2, y2]
        })
    return norm

def _scale_if_normalized(bbox, W, H):
    """
    If bbox looks normalized ([0..1]), scale to pixel coordinates.
    """
    x1, y1, x2, y2 = bbox
    if 0.0 <= x1 <= 1.0 and 0.0 <= y1 <= 1.0 and 0.0 <= x2 <= 1.0 and 0.0 <= y2 <= 1.0:
        x1 *= W; x2 *= W
        y1 *= H; y2 *= H
    return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))

def _draw_label_box(img, x1, y1, text, color):
    """
    Draw a filled background for readable label text.
    """
    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.1
    thick = 2
    (tw, th), bl = cv2.getTextSize(text, font, scale, thick)
    cv2.rectangle(img, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), color, thickness=-1)
    cv2.putText(img, text, (x1 + 3, y1 - 4), font, scale, (255, 255, 255), thick, cv2.LINE_AA)

def _annotate_from_json(image_path: str, json_path: str):
    """
    Read sidecar JSON (expects json["nanoowl"]["result"]), draw boxes + labels
    and write <basename>_ann.jpg next to the original image.
    """
    if not (image_path and os.path.isfile(image_path)):
        print(f"[annotate][skip] missing image: {image_path}")
        return False
    if not (json_path and os.path.isfile(json_path)):
        print(f"[annotate][skip] missing json: {json_path}")
        return False

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception as e:
        print(f"[annotate][warn] failed to read json: {e}")
        return False

    nano = meta.get("nanoowl") or {}
    result = nano.get("result")
    dets = _extract_detections(result)
    if not dets:
        print("[annotate] no detections; skipping")
        return False

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        print("[annotate][warn] failed to read image")
        return False

    H, W = img.shape[:2]
    for d in dets:
        x1, y1, x2, y2 = _scale_if_normalized(d["bbox"], W, H)
        x1 = max(0, min(W - 1, x1)); x2 = max(0, min(W - 1, x2))
        y1 = max(0, min(H - 1, y1)); y2 = max(0, min(H - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        label = d["label"]
        score = d["score"]
        text  = f"{label}" + (f" {score:.2f}" if isinstance(score, float) else "")
        color = _color_for_label(label)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=7)
        _draw_label_box(img, x1, y1, text, color)

    out_path = _ann_outpath_for_image(image_path)

    ok = cv2.imwrite(out_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if ok:
        print(f"[annotate] wrote {out_path}")
        return True

    print("[annotate][error] failed to write annotated image")
    return False


def _load_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _has_any_bbox(nanoowl_section: dict) -> bool:
    """
    Returns True iff the nanoowl result contains at least one detection (bbox).
    Uses the same normalization as _extract_detections().
    """
    if not isinstance(nanoowl_section, dict):
        return False
    result = nanoowl_section.get("result")
    dets = _extract_detections(result)
    return len(dets) > 0


def _post_full_json(url: str, obj: dict, timeout: float, retries: int = 3, headers: dict | None = None):
    if not (url and isinstance(obj, dict)):
        return -1, "invalid url or payload"

    last_status, last_body = None, None
    for attempt in range(1, int(retries or 1) + 1):
        try:
            data = {"meta": json.dumps(obj, ensure_ascii=False)}
            r = requests.post(url, data=data, timeout=timeout, headers=headers or {})
            try:
                body = r.json()
            except Exception:
                body = r.text
            last_status, last_body = r.status_code, body
            if 200 <= r.status_code < 300:
                return last_status, last_body
            time.sleep(min(1.5 * attempt, 4.0))
        except Exception as e:
            last_status, last_body = -1, str(e)
            time.sleep(min(1.5 * attempt, 4.0))
    return last_status, last_body


def _update_sidecar_json_robust(json_path: str, new_data: dict):
    """
    Ensures steps 1-4 are merged correctly without deleting existing keys.
    """
    for _ in range(5):  # Retry loop to handle file locking
        try:
            current_data = {}
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    current_data = json.load(f)

            # Deep merge the new results into the existing JSON
            # This preserves 'pose', 'entries', and 'vlm' while adding 'nanoowl'
            for key, value in new_data.items():
                if isinstance(value, dict) and key in current_data:
                    current_data[key].update(value)
                else:
                    current_data[key] = value

            # Atomic write
            tmp_path = json_path + ".tmp"
            with open(tmp_path, 'w') as f:
                json.dump(current_data, f, indent=2)
            os.replace(tmp_path, json_path)
            return current_data  # Return the FULL json for the planner
        except (json.JSONDecodeError, IOError):
            time.sleep(0.05)
    return None

def call_vlm(endpoint: str, image_path: str, timeout_s: float, retries: int, retry_sleep_s: float) -> VlmResult:
    payload = {"image_path": image_path}

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            t0 = time.time()
            resp = requests.post(endpoint, json=payload, timeout=timeout_s)
            dt = time.time() - t0

            if resp.status_code != 200:
                last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
                log(f"[vlm] attempt {attempt}/{retries} failed in {dt:.2f}s: {last_err}")
            else:
                caption = parse_caption_from_response(resp)
                if not caption:
                    last_err = "Empty caption"
                    log(f"[vlm] attempt {attempt}/{retries} got empty caption in {dt:.2f}s")
                else:
                    return VlmResult(ok=True, caption=caption, error=None)

        except Exception as e:
            last_err = str(e)
            log(f"[vlm] attempt {attempt}/{retries} exception: {last_err}")

        if shutdown_event.is_set():
            return VlmResult(ok=False, caption=None, error="Shutdown requested")
        if attempt < retries:
            time.sleep(retry_sleep_s)

    return VlmResult(ok=False, caption=None, error=last_err)

def process_folder(
    folder: str,
    new_folder: str,
    endpoint: str,
    path_src: Optional[str],
    path_dst: Optional[str],
    timeout_s: float,
    retries: int,
    retry_sleep_s: float,
    force: bool,
    sleep_between_s: float,
) -> Tuple[int, int, int]:
    # if not hasattr(process_folder, "villa_response"):
    #    process_folder.villa_response=None
    #    update_cb(villa_response_callback)

    jpgs = sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith(".jpg") and not f.lower().endswith("_ann.jpg")
    ])

    if not jpgs:
        log(f"[worker] No JPGs found in: {folder}")
        return (0, 0, 0)
    print(jpgs)
    done = skipped = failed = 0
    new_folder_path = Path(new_folder)
    new_folder_path.mkdir(parents=True, exist_ok=True)

    folder_path = Path(folder)
    assert folder_path.is_symlink(), f"{folder_path} is not a symlink"
    folder = folder_path.resolve()

    latest_link = new_folder_path.parent / "latest"
    try:
        if os.path.lexists(latest_link):
            os.remove(latest_link)
        os.symlink(new_folder_path, latest_link)
    except Exception as e:
        print(f"Failed to create symlink {latest_link}: {e}")

    for jpg in jpgs:
        jpg_path = Path(folder) / jpg
        shutil.copy(jpg_path, Path(new_folder))
        jpg_path = new_folder_path / jpg
        base = os.path.splitext(jpg)[0]
        json_path = os.path.join(folder, base + ".json")
        js = load_json(json_path)
        json_path = new_folder_path / f"{base}.json"


        new_js = {}
        pose = js.get("pose")
        new_js.setdefault("pose", pose)
        new_js.setdefault("image", os.path.basename(str(jpg_path)))


        # tmp = json_path + ".tmp"
        with open(json_path, "w") as f:
            print(f)
            json.dump(new_js, f, indent=2)
        # os.replace(tmp, new_json_path)
        print("finished dump")
        if (not force) and already_captioned(js):
            skipped += 1
            continue

        img_for_vlm = remap_path(str(jpg_path), path_src, path_dst)
        log(f"[vlm] POST {endpoint}  image_path={img_for_vlm}")

        t0 = time.time()
        res = call_vlm(endpoint, img_for_vlm, timeout_s=timeout_s, retries=retries, retry_sleep_s=retry_sleep_s)
        dt = time.time() - t0
        log(f"[vlm] took {dt:.2f}s  ok={res.ok}")

        vlm_caption = vlm_caption_queue.get()
        print("%%%%%%%%%  vlm_caption ^^^^^^^^^^^", vlm_caption)


        # Update JSON
        js.setdefault("image", os.path.basename(jpg_path))
        # Ensure list exists
        entries = js.get("entries")
        if not isinstance(entries, list):
            entries = []
            js["entries"] = entries

        if res.ok and res.caption:
            prompt, response = extract_prompt_response(res.caption)

            entries.append({
                "timestamp": int(time.time()),
                "prompt": prompt,
                "response": response,
            })
        if res.ok:
            done += 1
        else:
            failed += 1

        if shutdown_event.is_set():
            return (done, skipped, failed)
        if sleep_between_s > 0:
            time.sleep(sleep_between_s)

    return (done, skipped, failed)


def run_process_once(args):
    latest = Path(CAPTURES_ROOT) / "latest"

    parent = latest.parent
    new_folder_name = datetime.datetime.now().strftime("%Y_%m_%d___%H_%M_%S")
    new_folder_path = Path(parent / new_folder_name)

    log(f"[worker] Latest folder: {latest}")
    done, skipped, failed = process_folder(
        folder=latest.as_posix(),
        new_folder=new_folder_path.as_posix(),
        endpoint=args.endpoint,
        path_src=args.path_src,
        path_dst=args.path_dst,
        timeout_s=args.vlm_timeout,
        retries=args.retries,
        retry_sleep_s=args.retry_sleep,
        force=args.force,
        sleep_between_s=args.sleep_between,
    )
    log(f"[worker] summary: done={done} skipped={skipped} failed={failed}")
    return latest

def update_cb(cb):
    update_cb.VILLA_RESULTS_CB = cb
    print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&777 update cb", cb)
# -------------------- HTTP API --------------------
@app.post("/from_vila")
def from_vila():
    """
    Entry point called by VILA (text/plain body = caption).
    We MUST:
      1) Forward caption to Jetson2 /prompts and WAIT for prompts.
      2) Only then find the latest image and call NanoOWL with (image, prompts).
      3) Store NanoOWL output into the sidecar JSON next to that image.
      4) **NEW:** Render annotated image _ann.jpg next to the original.
    """

    if not hasattr(update_cb, "VILLA_RESULTS_CB"):
        print("VILLA_RESULTS_CB is defined as None")
        update_cb.VILLA_RESULTS_CB = None
    print("hello")
    caption = request.get_data(as_text=True, parse_form_data=False).strip()
    if not caption:
        print(f"not captoin")
        return jsonify({"ok": False, "error": "empty caption"}), 400
    vlm_caption_queue.put(caption)

    ts = int(time.time())
    print(f"[from_vila][{ts}] {caption}")
    LAST["vila_caption"] = {"ts": ts, "text": caption}
    HISTORY.appendleft({"src": "vila", "ts": ts, "text": caption})
    if update_cb.VILLA_RESULTS_CB is not None:
        print(f"[villa_cb] _*************************_ vila_cb is NOT None")
        update_cb.VILLA_RESULTS_CB(caption)
    else:
        print(f"[villa_cb] ____________________________ vila_cb is None!!!")

    # ---- 1) Get prompts directly from vLLM (single hop) ----
    try:
        prompts = _vllm_extract_prompts(caption)
        print(f"[vllm][prompts] {prompts}")
    except Exception as e:
        print(f"[vllm][error] {e}")
        prompts = None

    if not prompts:
        return jsonify({
            "ok": True,
            "note": "prompts missing; NanoOWL not called",
            "prompts": None
        })

    # ---- 2) Find latest image + sidecar JSON ----
    img_path, json_path = _find_latest_image_and_json(CAPTURES_ROOT)
    LAST["last_image_path"] = img_path
    if not img_path:
        print(f"[nanoowl][warn] no image found under {CAPTURES_ROOT}")
        return jsonify({
            "ok": False,
            "error": f"no image found under {CAPTURES_ROOT}",
            "prompts": prompts
        }), 500
    if not json_path:
        base, _ = os.path.splitext(img_path)
        json_path = base + ".json"

    # ---- 3) Call NanoOWL ----
    status, body = _post_nanoowl_multipart(
        endpoint=NANOOWL_ENDPOINT,
        image_path=img_path,
        prompts=prompts,
        annotate=NANOOWL_ANNOTATE,
        timeout=NANOOWL_TIMEOUT
    )
    LAST["nanoowl_result"] = {"status": status, "body": body if not isinstance(body, str) else body[:2000]}
    print(f"[nanoowl] status={status} body_type={'json' if isinstance(body, dict) else 'text'}")

    # ---- 4) Write NanoOWL result to sidecar JSON ----
    now = time.time()
    iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))
    nano_payload = {
        "ts": now,
        "iso_time": iso,
        "endpoint": NANOOWL_ENDPOINT,
        "status": status,
        "prompts": prompts,
        "annotate": int(NANOOWL_ANNOTATE),
        "result": body
    }
    try:
        _update_sidecar_json(json_path, {"nanoowl": nano_payload})
        print(f"[nanoowl][json] updated: {json_path}")

        # ---- 4.1) If has BBOX, forward the FULL JSON to remote machine ----
        try:
            meta = _load_json(json_path)
            if meta and _has_any_bbox(meta.get("nanoowl")):
                if FORWARD_JSON_URL:
                    # 1) take the local sidecar basename (no folders)
                    sidecar_basename = os.path.basename(json_path)  # e.g. x0200...__11_31_15.json

                    # 2) embed it in the payload so the receiver can save with the SAME name
                    meta["_sidecar_basename"] = sidecar_basename

                    # 3) (optional) also send as HTTP header for convenience
                    headers = {"X-Sidecar-Basename": sidecar_basename}

                    # 4) post
                    s, b = _post_full_json(
                        url=FORWARD_JSON_URL,
                        obj=meta,
                        timeout=FORWARD_JSON_TIMEOUT,
                        retries=FORWARD_JSON_RETRIES,
                        headers=headers,
                    )
                    print(f"[forward-json] url={FORWARD_JSON_URL} status={s} body={b}")

        except Exception as e:
            print(f"[forward-json][error] {e}")

    except Exception as e:
        print(f"[nanoowl][json][error] failed to update {json_path}: {e}")

    # ---- 5) **Auto-annotate** and write <basename>_ann.jpg ----
    ann_ok = _annotate_from_json(img_path, json_path)

    return jsonify({
        "ok": True,
        "caption": caption,
        "prompts": prompts,
        "image_path": img_path,
        "nanoowl_status": status,
        "nanoowl_body": body,
        "sidecar_json": json_path,
        "annotated": bool(ann_ok)
    })

def _find_specific_image(root_dir, filename):
    """
    Locates a specific file within the recursive captures tree.
    """
    for p in Path(root_dir).rglob(filename):
        img_path = str(p)
        json_path = str(p.with_suffix('.json'))
        return img_path, json_path
    return None, None


@app.get("/latest")
def latest():
    return jsonify({"ok": True, "last": LAST})


@app.get("/health")
def health():
    return jsonify({"ok": True, "time": int(time.time())})


# -------------------- Main --------------------

def main():
    global NANOOWL_ENDPOINT, CAPTURES_ROOT
    global NANOOWL_TIMEOUT, NANOOWL_ANNOTATE

    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=5050)
    # set root dir
    p.add_argument("--captures-root", required=True,
                   help="Root where capture_frames saves images+json (used to find latest image)")
    # process_folder
    p.add_argument("--endpoint", required=True, help="VLM endpoint, e.g. http://192.168.131.22:8080/describe")
    p.add_argument("--latest-by", choices=["name", "mtime"], default="name", help="How to choose the latest folder")
    p.add_argument("--path-src", default=None, help="Local root path to remap from (optional)")
    p.add_argument("--path-dst", default=None, help="Server-visible root path to remap to (optional)")
    p.add_argument("--vlm-timeout", type=float, default=10.0, help="HTTP timeout per attempt")
    p.add_argument("--retries", type=int, default=3, help="Retries per image")
    p.add_argument("--retry-sleep", type=float, default=0.2, help="Sleep between retries")
    p.add_argument("--sleep-between", type=float, default=3.0, help="Sleep between images (throttle)")
    p.add_argument("--force", action="store_true", help="Re-caption even if vlm_text exists")
    p.add_argument("--watch", action="store_true", help="Keep running and process the newest folder repeatedly")
    p.add_argument("--watch-interval", type=float, default=2.0, help="Seconds between scans in --watch mode")
    p.add_argument("--exclude-suffix", default="_ann",
                   help="Ignore folders ending with this suffix (set '' to disable)")

    # llm prompt converter
    p.add_argument("--vllm-url", required=True, help="e.g. http://192.168.131.21:8000")
    p.add_argument("--vllm-model", required=True, help="model name as served by vLLM")
    p.add_argument("--vllm-timeout", type=float, default=20.0)
    p.add_argument("--vllm-max-tokens", type=int, default=32)
    p.add_argument("--vllm-temperature", type=float, default=0.2)

    # nanoowl
    p.add_argument("--nanoowl-endpoint", required=True,
                   help="NanoOWL endpoint, e.g. http://172.16.17.11:5060/infer")

    p.add_argument("--forward-timeout", type=float, default=30.0,
                   help="Timeout (sec) for POST to Jetson-2")
    p.add_argument("--forward-retries", type=int, default=3,
                   help="Retries for POST to Jetson-2 on failure/timeout")

    p.add_argument("--nanoowl-timeout", type=float, default=45.0,
                   help="Timeout (sec) for NanoOWL POST")
    p.add_argument("--nanoowl-annotate", type=int, default=0,
                   help="Pass annotate=0/1 to NanoOWL")

    # send json
    p.add_argument("--forward-json-url", default="http://172.17.16.9:9090/ingest",
                   help="If set, forward the FULL sidecar JSON here, but only when NanoOWL has BBOX detections")
    p.add_argument("--forward-json-timeout", type=float, default=10.0,
                   help="Timeout (sec) for forwarding full JSON")
    p.add_argument("--forward-json-retries", type=int, default=3,
                   help="Retries for forwarding full JSON")



    args = p.parse_args()

    CAPTURES_ROOT = args.captures_root.strip()
    NANOOWL_ENDPOINT = args.nanoowl_endpoint.strip()

    NANOOWL_TIMEOUT = args.nanoowl_timeout
    NANOOWL_ANNOTATE = int(args.nanoowl_annotate)

    global FORWARD_JSON_URL, FORWARD_JSON_TIMEOUT, FORWARD_JSON_RETRIES
    FORWARD_JSON_URL = (args.forward_json_url or "").strip()
    FORWARD_JSON_TIMEOUT = float(args.forward_json_timeout)
    FORWARD_JSON_RETRIES = int(args.forward_json_retries)

    global VLLM_URL, VLLM_MODEL, VLLM_TIMEOUT, VLLM_MAX_TOKENS, VLLM_TEMPERATURE
    VLLM_URL = args.vllm_url.strip()
    VLLM_MODEL = args.vllm_model.strip()
    VLLM_TIMEOUT = args.vllm_timeout
    VLLM_MAX_TOKENS = args.vllm_max_tokens
    VLLM_TEMPERATURE = args.vllm_temperature

    print(f"[comm_manager] listening on {args.host}:{args.port}")
    print(f"  captures_root    = {CAPTURES_ROOT}")
    print(f"  nanoowl_endpoint = {NANOOWL_ENDPOINT} (annotate={NANOOWL_ANNOTATE})")
    # Start the folder processing loop in a background thread
    worker_thread = threading.Thread(target=run_process_once, args=(args,), daemon=True)
    worker_thread.start()

    def handle_shutdown(signum, frame):
        log(f"[Main] Shutdown signal ({signum}) received. Releasing worker...")
        shutdown_event.set()
        sys.exit(0)


    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)


    app.run(host=args.host, port=args.port)
    run_process_once(args)




if __name__ == "__main__":
    main()





