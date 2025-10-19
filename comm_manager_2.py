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
"""

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

app = Flask(__name__)

# --- Runtime configuration (populated from CLI args) ---
JETSON2_ENDPOINT = None      # e.g., http://172.16.17.11:5050/prompts
NANOOWL_ENDPOINT = None      # e.g., http://172.16.17.11:5060/infer
CAPTURES_ROOT = None         # e.g., /home/user/jetson-containers/data/images/captures

FORWARD_TIMEOUT = 20.0       # Jetson2 prompts timeout
FORWARD_RETRIES = 3          # Jetson2 prompts retries

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
    "jetson2_prompts": None,     # {"ts": int, "prompts": [str]}
    "last_forward_status": None, # {"status": int, "body": str/dict}
    "last_image_path": None,     # str
    "nanoowl_result": None,      # {"status": int, "body": any}
}

# -------------------- Helpers --------------------


def _http_post_json(url: str, payload: dict, timeout: float = 6.0):
    """
    POST JSON using stdlib (no requests). Returns (status_code, response_text).
    """
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=data,
        headers={"Content-Type": "application/json"},
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

def _is_in_ann_folder(fp: str) -> bool:
    p = Path(fp)
    return any(str(part).lower().endswith("_ann") for part in p.parents)

def _find_latest_image_and_json(root_dir: str):
    if not root_dir or not os.path.isdir(root_dir):
        return None, None

    patterns = ["**/*.jpg", "**/*.jpeg", "**/*.png"]
    latest_img = None
    latest_mtime = -1.0

    for pat in patterns:
        for fp in glob.glob(os.path.join(root_dir, pat), recursive=True):
            if _ANN_RE.search(fp):
                continue
            if _is_in_ann_folder(fp):
                continue
            try:
                mtime = os.path.getmtime(fp)
                if mtime > latest_mtime:
                    latest_mtime = mtime
                    latest_img = fp
            except Exception:
                pass

    if not latest_img:
        return None, None

    base, _ = os.path.splitext(latest_img)
    sidecar_json = base + ".json"
    return latest_img, sidecar_json


def _update_sidecar_json(json_path: str, updater: dict):
    """
    Safe write/update to sidecar JSON:
      - read existing dict or start a new one
      - merge 'updater' keys
      - write atomically via *.tmp then replace
    """
    obj = {}
    if os.path.isfile(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception:
            obj = {}

    # Merge/update top-level keys in-place
    for k, v in updater.items():
        obj[k] = v

    tmp = json_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, json_path)


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
    run_name = os.path.basename(base_dir)                      # e.g. 2025_10_19___15_53_28

    ann_dir = os.path.join(parent_dir, f"{run_name}_ann")      # e.g. .../captures/2025_10_19___15_53_28_ann
    os.makedirs(ann_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    base_name = re.sub(r"_ann$", "", base_name, flags=re.IGNORECASE)
    out_name = f"{base_name}_ann.jpg"

    return os.path.join(ann_dir, out_name)


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
    print("hello")
    caption = request.get_data(as_text=True, parse_form_data=False).strip()
    if not caption:
        print(f"not captoin")
        return jsonify({"ok": False, "error": "empty caption"}), 400

    ts = int(time.time())
    print(f"[from_vila][{ts}] {caption}")
    LAST["vila_caption"] = {"ts": ts, "text": caption}
    HISTORY.appendleft({"src": "vila", "ts": ts, "text": caption})

    # ---- 1) Send to Jetson2 and wait for prompts ----
    f_status, f_body, prompts = None, None, None
    if JETSON2_ENDPOINT:
        last_err = None
        for attempt in range(1, int(FORWARD_RETRIES or 1) + 1):
            f_status, f_body = _http_post_json(
                JETSON2_ENDPOINT, {"sentence": caption}, timeout=float(FORWARD_TIMEOUT or 10.0)
            )
            if f_status not in (-1, 408, 504) and not (isinstance(f_status, int) and f_status >= 500):
                break
            last_err = f_body
            print(f"[forward->jetson2] attempt {attempt} failed: status={f_status} body={str(f_body)[:180]}")
            time.sleep(min(2.0 * attempt, 6.0))

        print(f"[forward->jetson2] status={f_status} body={f_body[:180] if isinstance(f_body, str) else f_body}")
        try:
            data = json.loads(f_body) if isinstance(f_body, str) else {}
            if isinstance(data, dict) and isinstance(data.get("prompts"), list):
                prompts = [str(x) for x in data["prompts"]]
        except Exception:
            prompts = None

        LAST["last_forward_status"] = {"status": f_status, "body": f_body}
        if prompts:
            LAST["jetson2_prompts"] = {"ts": int(time.time()), "prompts": prompts}
            HISTORY.appendleft({"src": "jetson2", "ts": int(time.time()), "prompts": prompts})
            print(f"[jetson2][prompts] {prompts}")
        else:
            print("[jetson2][warn] no prompts parsed")

    if not prompts:
        return jsonify({
            "ok": True,
            "note": "prompts missing; NanoOWL not called",
            "forward_status": f_status,
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
                    headers=headers,             #  <<<<< add
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


@app.get("/latest")
def latest():
    return jsonify({"ok": True, "last": LAST})


@app.get("/health")
def health():
    return jsonify({"ok": True, "time": int(time.time())})


# -------------------- Main --------------------

def main():
    global JETSON2_ENDPOINT, NANOOWL_ENDPOINT, CAPTURES_ROOT
    global FORWARD_TIMEOUT, FORWARD_RETRIES, NANOOWL_TIMEOUT, NANOOWL_ANNOTATE

    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=5050)

    p.add_argument("--jetson2-endpoint", required=True,
                   help="URL to Jetson-2 prompts endpoint, e.g. http://172.16.17.11:5050/prompts")
    p.add_argument("--captures-root", required=True,
                   help="Root where capture_frames saves images+json (used to find latest image)")
    p.add_argument("--nanoowl-endpoint", required=True,
                   help="NanoOWL endpoint, e.g. http://172.16.17.11:5060/infer")

    p.add_argument("--forward-timeout", type=float, default=20.0,
                   help="Timeout (sec) for POST to Jetson-2")
    p.add_argument("--forward-retries", type=int, default=3,
                   help="Retries for POST to Jetson-2 on failure/timeout")

    p.add_argument("--nanoowl-timeout", type=float, default=45.0,
                   help="Timeout (sec) for NanoOWL POST")
    p.add_argument("--nanoowl-annotate", type=int, default=0,
                   help="Pass annotate=0/1 to NanoOWL")


    p.add_argument("--forward-json-url", default="http://172.17.16.9:9090/ingest",
                   help="If set, forward the FULL sidecar JSON here, but only when NanoOWL has BBOX detections")
    p.add_argument("--forward-json-timeout", type=float, default=8.0,
                   help="Timeout (sec) for forwarding full JSON")
    p.add_argument("--forward-json-retries", type=int, default=3,
                   help="Retries for forwarding full JSON")

    args = p.parse_args()

    JETSON2_ENDPOINT = args.jetson2_endpoint.strip()
    CAPTURES_ROOT = args.captures_root.strip()
    NANOOWL_ENDPOINT = args.nanoowl_endpoint.strip()

    FORWARD_TIMEOUT = args.forward_timeout
    FORWARD_RETRIES = args.forward_retries
    NANOOWL_TIMEOUT = args.nanoowl_timeout
    NANOOWL_ANNOTATE = int(args.nanoowl_annotate)

    global FORWARD_JSON_URL, FORWARD_JSON_TIMEOUT, FORWARD_JSON_RETRIES
    FORWARD_JSON_URL = (args.forward_json_url or "").strip()
    FORWARD_JSON_TIMEOUT = float(args.forward_json_timeout)
    FORWARD_JSON_RETRIES = int(args.forward_json_retries)

    print(f"[comm_manager] listening on {args.host}:{args.port}")
    print(f"  jetson2_endpoint = {JETSON2_ENDPOINT}")
    print(f"  captures_root    = {CAPTURES_ROOT}")
    print(f"  nanoowl_endpoint = {NANOOWL_ENDPOINT} (annotate={NANOOWL_ANNOTATE})")

    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()


