#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
comm_manager.py

Flow:
  1) Receive caption from VLM
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
python3 comm_manager.py   --host 0.0.0.0    --port 5050     --jetson2-endpoint http://192.168.131.21:5050/prompts       --captures-root /home/user/jetson-containers/data/R2/    --nanoowl-endpoint http://192.168.131.22:5060/infer  --forward-timeout 45   --forward-retries 3     --nanoowl-timeout 70   --nanoowl-annotate 0     --forward-json-url http://192.168.131.23:9090/ingest --endpoint http://192.168.131.22:8080/describe    --watch-interval 5.0 --sleep-between 20 --vlm-timeout 60

"""
import datetime
import signal
import threading
import numpy as np

from collections.abc import Callable

from typing import Optional, Tuple
from flask import Flask, request, jsonify
from pathlib import Path
import os
import json
import time
import argparse
from collections import deque
import hashlib
import re
import urllib.request, urllib.error  # for Jetson2 JSON POST
import requests                      # for NanoOWL multipart
import cv2                           # for drawing boxes
import base64

from utils_pipeline import log, load_json, already_captioned, extract_prompt_response, remap_path, \
    parse_caption_from_response, VlmResult

import logging
from logging.handlers import RotatingFileHandler
from concurrent.futures import ThreadPoolExecutor, as_completed

shutdown_event = threading.Event()
app = Flask(__name__)

# --- Runtime configuration (populated from CLI args) ---
NANOOWL_ENDPOINT = None      # e.g., http://172.16.17.11:5060/infer
CAPTURES_ROOT = None         # e.g., /home/user/jetson-containers/data/images/captures

NANOOWL_TIMEOUT = 45.0       # NanoOWL infer timeout
NANOOWL_ANNOTATE = 0         # annotate flag sent to NanoOWL (0/1)

_ANN_RE = re.compile(r"_ann\.(jpg|jpeg|png)$", re.IGNORECASE)

FORWARD_JSON_URL = None       # e.g., http://172.17.16.9:9090/ingest
FORWARD_JSON_TIMEOUT = 8.0
FORWARD_JSON_RETRIES = 1

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
VLLM_MAX_TOKENS = 512
VLLM_TEMPERATURE = 0.1

TILES_ROOT_DEFAULT = "/home/user/jetson-containers/data/tiles"
OUT_ROOT_DEFAULT   = "/home/user/jetson-containers/data/R2"
TILES_ANN_SUBDIR   = "____ANN"



_VLLM_SESSION = requests.Session()

LOG_PATH = os.environ.get("COMM_LOG", "comm_timings.log")
logger = logging.getLogger("comm")
logger.setLevel(logging.INFO)

fmt = logging.Formatter("%(asctime)s | %(message)s", "%Y-%m-%d %H:%M:%S")

fh = RotatingFileHandler(LOG_PATH, maxBytes=10*1024*1024, backupCount=5)
fh.setFormatter(fmt)

ch = logging.StreamHandler()
ch.setFormatter(fmt)

if not logger.handlers:
    logger.addHandler(fh)
    logger.addHandler(ch)

# -------------------- Helpers --------------------

_BULLET_RE = re.compile(r"^\s*[-•*]\s+")
_NUM_RE    = re.compile(r"^\s*\d+\s*[\.\)]\s*")

_DROP_WORDS = {
    "black","white","yellow","red","blue","green","orange","pink","purple","brown","gray","grey",
    "light","dark","wood","metal","plastic","cardboard",
    "small","big","large","tiny",
    "partially","visible","likely"
}


def caption_to_owl_prompts(caption: str) -> list[str]:
    """
    Convert bullet/numbered caption into NanoOWL prompts.
    No LLM. Heuristic cleanup only.
    """
    if not caption:
        return []

    lines = []
    for raw in caption.splitlines():
        s = raw.strip()
        if not s:
            continue
        s = _BULLET_RE.sub("", s)
        s = _NUM_RE.sub("", s)
        if not s:
            continue

        # remove parentheses notes:  "box (with 'Maggi')" -> "box"
        s = re.sub(r"\([^)]*\)", "", s).strip()

        # remove quotes content if it’s just a logo mention
        # (optional) keep simple
        s = s.replace('"', "").replace("'", "").strip()

        # normalize: "floor tiles" -> keep as-is (OWL can detect plural too)
        s = s.lower()

        # light cleanup of adjectives (optional)
        tokens = [t for t in re.split(r"\s+", s) if t]
        tokens = [t for t in tokens if t not in _DROP_WORDS]
        s = " ".join(tokens).strip()

        # remove trailing punctuation
        s = s.strip(" .,:;")

        if s:
            lines.append(s)

    # dedupe preserving order + add a/an prefix
    out, seen = [], set()
    for item in lines:
        if item not in seen:
            seen.add(item)
            out.append(item)

    return out

def _to_image_url(image_ref: str) -> str:
    """
    Accepts either:
      - http(s) URL  -> returned as-is
      - local path   -> converted to data:<mime>;base64,...
    """
    s = (image_ref or "").strip()
    if s.startswith("http://") or s.startswith("https://"):
        return s
    
    # assume local file path
    if not os.path.isfile(s):
        raise FileNotFoundError(f"Image not found: {s}")

    img = cv2.imread(s)
    if img is None:
        raise ValueError(f"Failed to decode image: {s}")

    target_dim = (640, 360) 
    img_small = cv2.resize(img, target_dim, interpolation=cv2.INTER_AREA)

    success, buffer = cv2.imencode('.jpg', img_small, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    
    if not success:
        raise ValueError("Failed to encode resized image")

    b64 = base64.b64encode(buffer).decode("ascii")
    
    return f"data:image/jpeg;base64,{b64}"


def _post_nanoowl_multipart(endpoint: str, image_path: str, prompts: list[str],
                            annotate: int, timeout: float):
    if not endpoint:
        return -1, "nanoowl endpoint not configured", 0.0
    if not (image_path and os.path.isfile(image_path)):
        return -1, f"image not found: {image_path}", 0.0

    t0 = time.perf_counter()
    data = {"prompts": json.dumps(prompts or []), "annotate": str(int(annotate))}

    try:
        with open(image_path, "rb") as f:
            files = {"image": (os.path.basename(image_path), f, "application/octet-stream")}
            r = requests.post(endpoint, files=files, data=data, timeout=timeout)

        dt = time.perf_counter() - t0
        try:
            body = r.json()
        except Exception:
            body = r.text
        return r.status_code, body, dt

    except Exception as e:
        dt = time.perf_counter() - t0
        return -1, str(e), dt


def annotate_mosaic_from_global_dets(
    mosaic_path: str,
    out_path: str,
    dets: list[dict],
    thickness: int = 6,
    font_scale: float = 1.2,
    show_score: bool = True,
) -> bool:
    """
    Draw global detections (already in mosaic pixel coords) on the mosaic image.
    Expects each det:
      { "label": str, "score": float|None, "bbox": [x1,y1,x2,y2] }
    """
    if not os.path.isfile(mosaic_path):
        print(f"[mosaic_ann][skip] missing mosaic: {mosaic_path}")
        return False

    img = cv2.imread(mosaic_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"[mosaic_ann][skip] failed to read: {mosaic_path}")
        return False

    H, W = img.shape[:2]

    for d in dets or []:
        bbox = d.get("bbox")
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            continue
        try:
            x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
        except Exception:
            continue

        # clamp
        x1 = max(0, min(W - 1, x1))
        x2 = max(0, min(W - 1, x2))
        y1 = max(0, min(H - 1, y1))
        y2 = max(0, min(H - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        label = str(d.get("label", "object"))
        score = d.get("score", None)

        if show_score and isinstance(score, (int, float)):
            text = f"{label} {float(score):.2f}"
        else:
            text = label

        color = _color_for_label(label)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=thickness)
        _draw_label_box(img, x1, y1, text, color)

    ok = cv2.imwrite(out_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if ok:
        print(f"[mosaic_ann] wrote {out_path}")
        return True
    print(f"[mosaic_ann][error] failed to write {out_path}")
    return False


_TILE_RE = re.compile(r"^(.*)_tile_(\d+)_(\d+)$", re.IGNORECASE)

def _parse_group_and_rc(stem: str):
    m = _TILE_RE.match(stem)
    if not m:
        return None, None, None
    return m.group(1), int(m.group(2)), int(m.group(3))

def _bbox_shift_xyxy(b, dx, dy):
    x1, y1, x2, y2 = b
    return [x1 + dx, y1 + dy, x2 + dx, y2 + dy]

def _iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0

def _dedupe_by_iou(dets, iou_thr=0.6):
    # keep highest score first
    dets = sorted(dets, key=lambda d: float(d.get("score") or 0.0), reverse=True)
    kept = []
    for d in dets:
        lbl = d.get("label") or "object"
        bb = d.get("bbox")
        if not bb:
            continue
        is_dup = False
        for k in kept:
            if (k.get("label") or "object") != lbl:
                continue
            if _iou_xyxy(bb, k["bbox"]) >= iou_thr:
                is_dup = True
                break
        if not is_dup:
            kept.append(d)
    return kept


def merge_tiles_and_send(
    tiles_folder_path: str,
    out_folder_path: str,
    out_ann_folder_path: str,
    overlap: float,
    depth_endpoint: str,
    forward_json_url: str,
    forward_timeout: float,
    forward_retries: int,
    depth_timeout: float = 30.0,
    iou_thr: float = 0.6,
):
    tiles_folder_path = Path(tiles_folder_path)
    out_folder_path   = Path(out_folder_path)
    ann_dir   = Path(out_ann_folder_path)


    if not tiles_folder_path.exists():
        print("[merge] tiles folder not found:", tiles_folder_path)
        return []

    out_folder_path.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)


    # --- collect tiles from tiles_folder_path ---
    tiles = []
    for jpg_path in tiles_folder_path.glob("*.jpg"):
        if jpg_path.name.lower().endswith("_ann.jpg"):
            continue
        gid, r, c = _parse_group_and_rc(jpg_path.stem)
        if gid is None:
            continue

        json_path = jpg_path.with_suffix(".json")
        meta = _load_json(str(json_path)) or {}
        nano = meta.get("nanoowl") or {}
        dets = _extract_detections(nano.get("result"))

        img = cv2.imread(str(jpg_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        H, W = img.shape[:2]

        tiles.append({
            "gid": gid,
            "r": r,
            "c": c,
            "jpg": jpg_path,
            "json": json_path,
            "img": img,
            "W": W,
            "H": H,
            "dets": dets,
        })

    if not tiles:
        print("[merge] no tiles found in", tiles_folder_path)
        return []

    # --- group by gid ---
    by_gid = {}
    for t in tiles:
        by_gid.setdefault(t["gid"], []).append(t)

    outputs = []

    for gid, gtiles in by_gid.items():
        gtiles.sort(key=lambda x: (x["r"], x["c"]))

        max_r = max(t["r"] for t in gtiles)
        max_c = max(t["c"] for t in gtiles)

        tileW = gtiles[0]["W"]
        tileH = gtiles[0]["H"]

        strideX = int(round(tileW * (1.0 - float(overlap))))
        strideY = int(round(tileH * (1.0 - float(overlap))))

        mosaicW = tileW + max_c * strideX
        mosaicH = tileH + max_r * strideY

        # --- stitch mosaic ---
        mosaic = np.zeros((mosaicH, mosaicW, 3), dtype=np.uint8)
        for t in gtiles:
            x0 = t["c"] * strideX
            y0 = t["r"] * strideY
            h, w = t["H"], t["W"]
            mosaic[y0:y0+h, x0:x0+w] = t["img"]

        # ✅ FINAL output names (display-friendly)
        final_img_name = f"{gid}.jpg"
        final_json_name = f"{gid}.json"
        mosaic_path = out_folder_path / final_img_name
        cv2.imwrite(str(mosaic_path), mosaic, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

        # --- shift detections to global coords ---
        all_global = []
        for t in gtiles:
            dx = t["c"] * strideX
            dy = t["r"] * strideY
            for d in t["dets"]:
                bb = d.get("bbox")
                if not bb or len(bb) != 4:
                    continue

                sx1, sy1, sx2, sy2 = _scale_if_normalized(bb, t["W"], t["H"])
                gb = _bbox_shift_xyxy([float(sx1), float(sy1), float(sx2), float(sy2)], dx, dy)

                all_global.append({
                    "label": d.get("label") or "object",
                    "score": d.get("score"),
                    "bbox": gb,
                    "src": {"tile_r": t["r"], "tile_c": t["c"], "tile": t["jpg"].name},
                })

        merged = _dedupe_by_iou(all_global, iou_thr=iou_thr)

        # --- FINAL ann (written to out_folder_path) ---
        mosaic_ann_path = ann_dir / f"{gid}_ann.jpg"
        annotate_mosaic_from_global_dets(
            mosaic_path=str(mosaic_path),
            out_path=str(mosaic_ann_path),
            dets=merged,
            thickness=7,
            font_scale=1.3,
            show_score=True,
        )

        depth_dets = {
            "detections": [
                {"label": d["label"], "score": d.get("score"), "bbox": d["bbox"]}
                for d in merged
            ]
        }


        # write merged JSON with prompts + detections + metadata
        now = time.time()
        iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))

        mosaic_prompts = []
        seen = set()
        for d in (merged or []):
            lbl = str(d.get("label") or "").strip()
            if not lbl:
                continue
            if lbl not in seen:
                seen.add(lbl)
                mosaic_prompts.append(lbl)

        merged_json = {
            "pose": {},
            "image": final_img_name,
            "entries": [
                {
                    "timestamp": int(now),
                    "prompt": "Describe the objects in the image",
                    "response": "\n".join([f"- {p}" for p in mosaic_prompts]),
                }
            ],
            "nanoowl": {
                "ts": now,
                "iso_time": iso,
                "endpoint": "MERGED_FROM_TILES",
                "status": 200,
                "prompts": mosaic_prompts,
                "annotate": 0,
                "result": {
                    "detections": [
                        {
                            "bbox": [
                                int(round(d["bbox"][0])),
                                int(round(d["bbox"][1])),
                                int(round(d["bbox"][2])),
                                int(round(d["bbox"][3])),
                            ],
                            "label": d.get("label") or "object",
                            "score": (float(d["score"]) if d.get("score") is not None else None),
                        }
                        for d in (merged or [])
                        if isinstance(d.get("bbox"), (list, tuple)) and len(d["bbox"]) == 4
                    ]
                }
            },

            "mosaic": {
                "group_id": gid,
                "rows": max_r + 1,
                "cols": max_c + 1,
                "overlap": float(overlap),
                "tile_size": {"w": tileW, "h": tileH},
            }
        }


        # --- call depth on FINAL mosaic ---
        if depth_endpoint:
            try:
                with open(str(mosaic_path), "rb") as f_img:
                    files = {
                        "image": (final_img_name, f_img, "image/jpeg"),
                        "detections": ("detections.json", json.dumps(depth_dets), "application/json"),
                        "image_dir": (None, str(out_folder_path)),
                    }
                    r_depth = requests.post(depth_endpoint, files=files, timeout=depth_timeout)
                    if r_depth.status_code == 200:
                        merged_json["DEPTH_ANYTHING"] = r_depth.json()
                        print(f"[merge][depth] ok gid={gid}")
                    else:
                        merged_json["DEPTH_ANYTHING_error"] = {
                            "status": r_depth.status_code,
                            "text": r_depth.text[:500],
                        }
                        print(f"[merge][depth] failed gid={gid} status={r_depth.status_code}")
            except Exception as e:
                merged_json["DEPTH_ANYTHING_error"] = {"exception": str(e)}
                print(f"[merge][depth] exception gid={gid}: {e}")

        # --- write FINAL merged json ---
        merged_path = out_folder_path / final_json_name
        with open(merged_path, "w", encoding="utf-8") as f:
            json.dump(merged_json, f, ensure_ascii=False, indent=2)

        # --- forward merged json ---
        if forward_json_url:
            try:
                headers = {"X-Sidecar-Basename": merged_path.name}
                s, b = _post_full_json(
                    forward_json_url,
                    merged_json,
                    timeout=forward_timeout,
                    retries=forward_retries,
                    headers=headers
                )
                print(f"[merge][forward] gid={gid} status={s}")
                merged_json["_forward_status"] = s
                merged_json["_forward_body"] = b if isinstance(b, (dict, list)) else str(b)[:500]

                # update FINAL json with forward result
                with open(merged_path, "w", encoding="utf-8") as f:
                    json.dump(merged_json, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"[merge][forward] exception gid={gid}: {e}")

        outputs.append({
            "gid": gid,
            "final_image": str(mosaic_path),
            "final_json": str(merged_path),
            "final_ann": str(mosaic_ann_path),
        })

    return outputs


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
    scale = 1.5
    thick = 2
    (tw, th), bl = cv2.getTextSize(text, font, scale, thick)
    cv2.rectangle(img, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), color, thickness=-1)
    cv2.putText(img, text, (x1 + 3, y1 - 4), font, scale, (255, 255, 255), thick, cv2.LINE_AA)


def _annotate_tile_to_path(image_path: str, json_path: str, out_path: str) -> bool:
    """
    Like _annotate_from_json, but writes to explicit out_path (no symlink logic).
    """
    if not (image_path and os.path.isfile(image_path)):
        return False
    if not (json_path and os.path.isfile(json_path)):
        return False

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return False

    nano = meta.get("nanoowl") or {}
    result = nano.get("result")
    dets = _extract_detections(result)
    if not dets:
        return False

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
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

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    return bool(cv2.imwrite(out_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 92]))



def _load_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


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



def call_vlm(endpoint: str, image_path: str, timeout_s: float, retries: int, retry_sleep_s: float) -> VlmResult:
    """
    Replaces VILA /describe with vLLM Qwen3-VL OpenAI-compatible endpoint:
      POST {endpoint}/v1/chat/completions

    Keeps minimal pipeline changes by:
      - returning VlmResult(caption=...)
      - ALSO pushing caption into vlm_caption_queue so existing code that does
        vlm_caption_queue.get() continues to work unchanged.
    """
    last_err = None

    # your "question"/instruction for Qwen3-VL
    user_text = (
        "Extract ONLY object names from the image.\n"
        "Output a bullet list.\n"
        "Each bullet must be exactly: 'A <object>' or 'An <object>'.\n"
        "No adjectives. No verbs. No extra text."
    )

    for attempt in range(1, retries + 1):
        try:
            t0 = time.time()

            img_url = _to_image_url(image_path)

            payload = {
                "model": VLLM_MODEL,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {"type": "image_url", "image_url": {"url": img_url}},
                    ]
                }],
                "max_tokens": 64,
                "temperature": 0.1,
            }

            url = f"{endpoint.rstrip('/')}/v1/chat/completions"
            resp = requests.post(url, json=payload, timeout=timeout_s)
            dt = time.time() - t0

            if resp.status_code != 200:
                last_err = f"HTTP {resp.status_code}: {resp.text[:300]}"
                log(f"[vlm-qwen] attempt {attempt}/{retries} failed in {dt:.2f}s: {last_err}")
            else:
                j = resp.json()
                caption = (j.get("choices", [{}])[0]
                             .get("message", {})
                             .get("content", "") or "").strip()

                if not caption:
                    last_err = "Empty caption from vLLM"
                    log(f"[vlm-qwen] attempt {attempt}/{retries} got empty caption in {dt:.2f}s")
                else:
                    return VlmResult(ok=True, caption=caption, error=None)

        except Exception as e:
            last_err = str(e)
            log(f"[vlm-qwen] attempt {attempt}/{retries} exception: {last_err}")

        if shutdown_event.is_set():
            return VlmResult(ok=False, caption=None, error="Shutdown requested")
        if attempt < retries:
            time.sleep(retry_sleep_s)

    return VlmResult(ok=False, caption=None, error=last_err)

def process_folder(
    folder: str,
    tiles_folder: str,
    tiles_ann_folder: str,
    out_folder: str,
    out_ann_folder: str, 
    endpoint: str,
    depth_endpoint: str,
    path_src: Optional[str],
    path_dst: Optional[str],
    timeout_s: float,
    retries: int,
    retry_sleep_s: float,
    force: bool,
    sleep_between_s: float,
    args=None
) -> Tuple[int, int, int]:

    # -------------------------
    # Stage -1: scan source folder
    # -------------------------
    jpgs = sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith(".jpg") and not f.lower().endswith("_ann.jpg")
    ])

    if not jpgs:
        log(f"[worker] No JPGs found in: {folder}")
        return (0, 0, 0)

    folder_path = Path(folder)
    if folder_path.is_symlink():
        folder = str(folder_path.resolve())

    tiles_folder_path = Path(tiles_folder)
    tiles_folder_path.mkdir(parents=True, exist_ok=True)

    tiles_ann_folder_path = Path(tiles_ann_folder)
    tiles_ann_folder_path.mkdir(parents=True, exist_ok=True)

    out_folder_path = Path(out_folder)
    out_folder_path.mkdir(parents=True, exist_ok=True)

    out_ann_folder_path = Path(out_ann_folder)
    out_ann_folder_path.mkdir(parents=True, exist_ok=True)

    log(f"[worker] source folder      = {folder}")
    log(f"[worker] tiles_folder      = {tiles_folder_path}")
    log(f"[worker] tiles_ann_folder  = {tiles_ann_folder_path}")
    log(f"[worker] out_folder        = {out_folder_path}")
    log(f"[worker] out_ann_folder    = {out_ann_folder_path}")

    # -------------------------
    # Stage 0: prepare tasks (copy tile + init tile json)
    # -------------------------
    tasks = []
    for jpg in jpgs:
        dest_jpg_path = Path(folder) / jpg        
        base = os.path.splitext(jpg)[0]
        tile_json_path = Path(folder) / f"{base}.json"

        js0 = load_json(str(tile_json_path)) or {}
        js = {
            "pose": js0.get("pose", {}) if isinstance(js0.get("pose", {}), dict) else {},
            "image": js0.get("image", dest_jpg_path.name),
            "entries": js0.get("entries", []) if isinstance(js0.get("entries", []), list) else [],
        }

        need_vlm = True
        if (not force) and already_captioned(js):
            need_vlm = False

        img_for_vlm = remap_path(str(dest_jpg_path), path_src, path_dst)

        tasks.append({
            "jpg": jpg,
            "base": base,
            "dest_jpg_path": dest_jpg_path,
            "tile_json_path": tile_json_path,
            "new_js": js,              
            "need_vlm": need_vlm,
            "img_for_vlm": img_for_vlm,
        })
    # -------------------------
    # Stage 1: VLM in parallel (BATCH)
    # -------------------------
    vlm_results = {}

    def _vlm_job(task):
        jpg = task["jpg"]
        img_for_vlm = task["img_for_vlm"]
        log(f"[vlm] POST {endpoint} image_path={img_for_vlm}")
        t0 = time.time()
        res = call_vlm(endpoint, img_for_vlm, timeout_s=timeout_s, retries=retries, retry_sleep_s=retry_sleep_s)
        dt = time.time() - t0
        log(f"[vlm] {jpg} took {dt:.2f}s ok={res.ok}")
        return jpg, res

    workers = max(1, int(getattr(args, "vlm_workers", 1)))
    vlm_tasks = [t for t in tasks if t["need_vlm"]]

    if vlm_tasks and workers > 1:
        log(f"[vlm] running batch with workers={workers} (tasks={len(vlm_tasks)})")
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_vlm_job, t) for t in vlm_tasks]
            for fut in as_completed(futs):
                if shutdown_event.is_set():
                    break
                jpg, res = fut.result()
                vlm_results[jpg] = res
    else:
        for t in vlm_tasks:
            if shutdown_event.is_set():
                break
            jpg, res = _vlm_job(t)
            vlm_results[jpg] = res

    # -------------------------
    # Stage 2: Per-tile OWL + write tile JSON + tile ANN
    # -------------------------
    done = skipped = failed = 0

    for t in tasks:
        if shutdown_event.is_set():
            break

        jpg = t["jpg"]
        base = t["base"]
        dest_jpg_path = t["dest_jpg_path"]
        tile_json_path = t["tile_json_path"]
        new_js = t["new_js"]

        if not t["need_vlm"]:
            skipped += 1
            continue

        res = vlm_results.get(jpg)
        if not (res and res.ok and res.caption):
            failed += 1
            continue

        vlm_caption = res.caption

        prompts = caption_to_owl_prompts(vlm_caption)
        log(f"[worker] Prompts: {prompts}")

        status, body, _ = _post_nanoowl_multipart(
            endpoint=NANOOWL_ENDPOINT,
            image_path=str(dest_jpg_path),
            prompts=prompts,
            annotate=NANOOWL_ANNOTATE,
            timeout=NANOOWL_TIMEOUT
        )

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
        new_js["nanoowl"] = nano_payload

        prompt_str = "Describe the objects in the image"
        response_str = vlm_caption

        new_js["entries"].append({
            "timestamp": int(time.time()),
            "prompt": prompt_str,
            "response": response_str,
        })

        # write TILE json
        try:
            with open(tile_json_path, "w", encoding="utf-8") as f:
                json.dump(new_js, f, indent=2)
        except Exception as e:
            log(f"[worker] failed saving tile json: {e}")

        # write TILE ann into tiles_ann_folder (NOT into R2, NOT via symlink)
        try:
            tile_ann_path = tiles_ann_folder_path / f"{base}_ann.jpg"
            ok = _annotate_tile_to_path(
                image_path=str(dest_jpg_path),
                json_path=str(tile_json_path),
                out_path=str(tile_ann_path),
            )
            if ok:
                log(f"[tile_ann] wrote {tile_ann_path}")
        except Exception as e:
            log(f"[tile_ann][error] {e}")

        done += 1

        if sleep_between_s > 0:
            time.sleep(sleep_between_s)

    # -------------------------
    # Stage 3: Merge tiles -> FINAL in out_folder (mosaic + merged json + mosaic ann + depth + forward)
    # -------------------------
    try:
        overlap_val = float(getattr(args, "overlap", 0.0)) if args is not None else 0.0

        out = merge_tiles_and_send(
            tiles_folder_path=tiles_folder_path.as_posix(),
            out_folder_path=out_folder_path.as_posix(),
            out_ann_folder_path=Path(out_ann_folder).as_posix(), 
            overlap=overlap_val,
            depth_endpoint=depth_endpoint,
            forward_json_url=FORWARD_JSON_URL,
            forward_timeout=FORWARD_JSON_TIMEOUT,
            forward_retries=FORWARD_JSON_RETRIES,
            depth_timeout=30.0,
            iou_thr=0.6,
        )
        log(f"[merge] outputs: {out}")
    except Exception as e:
        log(f"[merge][fatal] {e}")

    return (done, skipped, failed)



def run_process_once(args):
    tiles_latest = Path(args.tiles_root) / "latest"
    if not tiles_latest.exists():
        log("[worker] No tiles latest symlink/folder")
        return None

    tiles_run_dir = tiles_latest.resolve()
    if not tiles_run_dir.is_dir():
        log(f"[worker] tiles_latest is not a dir: {tiles_run_dir}")
        return None

    run_id = tiles_run_dir.name

    src_jpgs = sorted([
        f.name for f in tiles_run_dir.glob("*.jpg")
        if f.suffix.lower() == ".jpg" and not f.name.lower().endswith("_ann.jpg")
    ])
    if not src_jpgs:
        log("[worker] No JPGs in tiles/latest")
        return tiles_run_dir

    tiles_ann_dir = Path(args.tiles_root) / TILES_ANN_SUBDIR / run_id
    out_root      = Path(args.out_root)
    out_run_dir   = out_root / run_id
    out_ann_dir   = out_root / f"{run_id}_ANN"

    tiles_ann_dir.mkdir(parents=True, exist_ok=True)
    out_run_dir.mkdir(parents=True, exist_ok=True)
    out_ann_dir.mkdir(parents=True, exist_ok=True)

    log(f"[worker] run_id={run_id}")
    log(f"[worker] tiles_run_dir(in-place)={tiles_run_dir}")
    log(f"[worker] tiles_ann_dir={tiles_ann_dir}")
    log(f"[worker] out_run_dir={out_run_dir}")

    done, skipped, failed = process_folder(
        folder=tiles_run_dir.as_posix(),           
        tiles_folder=tiles_run_dir.as_posix(),      
        tiles_ann_folder=tiles_ann_dir.as_posix(),
        out_folder=out_run_dir.as_posix(),
        out_ann_folder=out_ann_dir.as_posix(),
        endpoint=args.endpoint,
        depth_endpoint=args.depth_endpoint,
        path_src=args.path_src,
        path_dst=args.path_dst,
        timeout_s=args.vlm_timeout,
        retries=args.retries,
        retry_sleep_s=args.retry_sleep,
        force=args.force,
        sleep_between_s=args.sleep_between,
        args=args,
    )

    # update OUT latest symlink
    latest_link = out_root / "latest"
    try:
        if os.path.lexists(latest_link):
            os.remove(latest_link)
        os.symlink(out_run_dir, latest_link)
        log(f"[worker] Updated OUT latest -> {out_run_dir}")
    except Exception as e:
        log(f"[worker] Failed to update OUT latest symlink: {e}")

    log(f"[worker] summary: done={done} skipped={skipped} failed={failed}")
    return tiles_run_dir

def update_cb(cb):
    update_cb.VILLA_RESULTS_CB = cb
    print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&777 update cb", cb)



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
    p.add_argument("--endpoint", required=True, help="VLM endpoint, e.g. http://192.168.131.22:8080")
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
    p.add_argument("--vllm-url", default=None, help="e.g. http://192.168.131.21:8000") 
    p.add_argument("--vllm-model", required=True, help="model name as served by vLLM")
    p.add_argument("--vllm-timeout", type=float, default=20.0)
    p.add_argument("--vllm-max-tokens", type=int, default=32)
    p.add_argument("--vllm-temperature", type=float, default=0.2)

    # nanoowl
    p.add_argument("--nanoowl-endpoint", default=None,
               help="NanoOWL endpoint, e.g. http://172.16.17.11:5060/infer")


    p.add_argument("--forward-timeout", type=float, default=30.0,
                   help="Timeout (sec) for POST to Jetson-2")
    p.add_argument("--forward-retries", type=int, default=3,
                   help="Retries for POST to Jetson-2 on failure/timeout")

    p.add_argument("--nanoowl-timeout", type=float, default=45.0,
                   help="Timeout (sec) for NanoOWL POST")
    p.add_argument("--nanoowl-annotate", type=int, default=0,
                   help="Pass annotate=0/1 to NanoOWL")
    
    #depth

    p.add_argument("--depth-endpoint", default="http://127.0.0.1:5070/bbox_depth", 
               help="Endpoint for Depth Anything service")

    # send json
    p.add_argument("--forward-json-url", default="http://172.17.16.9:9090/ingest",
                   help="If set, forward the FULL sidecar JSON here, but only when NanoOWL has BBOX detections")
    p.add_argument("--forward-json-timeout", type=float, default=10.0,
                   help="Timeout (sec) for forwarding full JSON")
    p.add_argument("--forward-json-retries", type=int, default=3,
                   help="Retries for forwarding full JSON")

    p.add_argument("--config", default="config/networks.yaml",
                        help="Path to networks config YAML")
    p.add_argument("--profile", default=None,
                        help="Profile name (adsl|robotican). Overrides R2_PROFILE and defaults.profile")
    p.add_argument("--no-config", action="store_true",
                        help="Disable config resolution and use CLI flags as-is")


    p.add_argument("--vlm-workers", type=int, default=4,
               help="How many VLM requests to run in parallel (batch concurrency)")

    p.add_argument("--overlap", type=float, default=0.0,
               help="Tile overlap used in capture (e.g. 0.3)")


    p.add_argument("--tiles-root", default=TILES_ROOT_DEFAULT,
               help="Where to store all tiles + tile JSONs")
    p.add_argument("--out-root", default=OUT_ROOT_DEFAULT,
               help="Where to store FINAL mosaic + merged json")

    args = p.parse_args()

    from config.profile_loader import load_profile

    if not args.no_config:
        net = load_profile(args.config, args.profile)

        # Override relevant args automatically
        # (keep CLI override possible only if you want — right now config wins)
        args.vllm_url = net.vllm_url
        args.nanoowl_endpoint = net.nanoowl_infer_url
        args.forward_json_url = net.ingest_json_url
        args.endpoint = net.vila_describe_url

        # Optional: print summary once
        print(
            f"[comm_manager] profile={net.name} | VLLM={args.vllm_url} | VILA={args.endpoint} | OWL={args.nanoowl_endpoint} | INGEST={args.forward_json_url}")
    # ---- validate required settings after config resolution ----
    missing = []
    if not args.captures_root:
        missing.append("--captures-root")
    if not args.endpoint:
        missing.append("--endpoint (VILA /describe)")
    if not args.vllm_url:
        missing.append("--vllm-url (or profile must provide vllm)")
    if not args.nanoowl_endpoint:
        missing.append("--nanoowl-endpoint (or profile must provide nanoowl)")
    if not args.vllm_model:
        missing.append("--vllm-model (or set a default / put in config)")

    if missing:
        p.error("Missing required args after config resolution: " + ", ".join(missing))

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

    def handle_shutdown(signum, frame):
        log(f"[Main] Shutdown signal ({signum}) received. Releasing worker...")
        shutdown_event.set()

    def worker_loop(args):
        while not shutdown_event.is_set():
            try:
                run_process_once(args)
            except Exception as e:
                log(f"[worker][fatal] {e}")
            time.sleep(args.watch_interval)

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)
    worker_thread = threading.Thread(target=worker_loop, args=(args,), daemon=True)
    worker_thread.start()

    app.run(host=args.host, port=args.port)



if __name__ == "__main__":
    main()






