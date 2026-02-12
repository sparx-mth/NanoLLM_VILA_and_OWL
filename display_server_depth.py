#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VLM Ingest DISPLAY Server — Cute Web GUI for viewing ingested images + captions

What it does
------------
- Serves a lightweight web UI that shows every <image + description> pair received
  into an "ingested" directory (from your receiver `/ingest`).
- Auto-refreshes every 2 seconds to pick up new arrivals.
- Click a card to open a modal with a large preview and the raw JSON metadata.

Assumptions about files in ROOT_DIR
-----------------------------------
- For each image there is a matching JSON file with the SAME BASENAME.
  Example:
    ROOT_DIR/
      2025-10-12_13-05-11_image.jpg
      2025-10-12_13-05-11_image.json
- Image extension can vary (.jpg/.png/.jpeg/...).
- JSON schema is flexible. We attempt to extract a human description from common keys.

Usage
-----
python3 display_server.py --root /path/to/ingested --host 0.0.0.0 --port 8090

Dependencies
------------
- Flask only (no sockets or DB needed)
  pip install flask

Notes
-----
- Designed to run on the same machine that stores the ingested folder, 
  but can also run on a different host if it mounts that folder.
- If your receiver saves with different names, you can adapt the MATCH_GLOB below.
"""

import argparse
import json
import mimetypes
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from flask import Flask, jsonify, render_template_string, send_from_directory, request, abort

# -----------------
# Config & parsing
# -----------------

def parse_args():
    p = argparse.ArgumentParser(description="VLM Ingest DISPLAY Server")
    p.add_argument("--root", required=True, help="Directory holding ingested files (images + json)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--scan-interval", type=float, default=2.0, help="Seconds between UI auto-refreshes")
    p.add_argument("--latest-only", action="store_true",help="Show only the most-recent subfolder under --root (auto-updates on refresh)")
    p.add_argument("--static", dest="static_dir", default=None,
                   help="Directory for static assets (logo, etc.). Defaults to './static' next to this script.")
    return p.parse_args()

# -----------------
# Model
# -----------------

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
JSON_EXT = ".json"

_TILE_RE = re.compile(r"^(frame_\d{4}_\d{2}_\d{2}___\d{2}_\d{2}_\d{2})_tile_(\d+)_(\d+)$")


@dataclass
class Item:
    basename: str              # file base w/o extension
    image_rel: str             # relative path from ROOT
    json_rel: Optional[str]    # relative path from ROOT (may be None)
    ctime: float               # latest ctime among image/json
    text: str                  # extracted caption/answer (best-effort)
    vlm_terms: List[str] = None      #  LLM (nanoowl.prompts)
    owl_labels: List[str] = None     # OWL labels
    depth_rel: Optional[str] = None # relative path to depth image (if exists)


# -----------------
# Utilities
# -----------------

def _depth_variant(path: Path) -> Optional[Path]:
    parent = path.parent
    stem = path.stem

    sibling_depth_dir = parent.with_name(parent.name + "_depth")
    if sibling_depth_dir.exists() and sibling_depth_dir.is_dir():
        for cand in [
            sibling_depth_dir / (stem + "_depth.png"),
            sibling_depth_dir / (stem + "_depth.jpg"),
        ]:
            if cand.exists():
                return cand
    return None

def _group_tiles(items: List[Item], root: Path) -> List[dict]:
    """
    Group tiles into puzzles using filename pattern or JSON tile_index.
    Returns list of groups sorted by newest ctime.
    """
    groups: Dict[str, dict] = {}

    for it in items:
        stem = Path(it.basename).stem  # already stem
        m = _TILE_RE.match(stem)
        if not m:
            # Not a tile – you can skip or treat as single-item group
            continue

        group_id, r_s, c_s = m.group(1), m.group(2), m.group(3)
        r, c = int(r_s), int(c_s)

        # Try read JSON to get authoritative tile_index (optional)
        if it.json_rel:
            try:
                with open(root / it.json_rel, "r", encoding="utf-8") as f:
                    doc = json.load(f)
                ti = doc.get("tile_index")
                if isinstance(ti, list) and len(ti) == 2:
                    r, c = int(ti[0]), int(ti[1])
            except Exception:
                pass

        g = groups.get(group_id)
        if g is None:
            g = {
                "group_id": group_id,
                "ctime": it.ctime,
                "tiles": [],
                "max_r": r,
                "max_c": c,
            }
            groups[group_id] = g

        g["ctime"] = max(g["ctime"], it.ctime)
        g["max_r"] = max(g["max_r"], r)
        g["max_c"] = max(g["max_c"], c)

        g["tiles"].append({
            "r": r,
            "c": c,
            "basename": it.basename,
            "image": f"/img/{it.image_rel}",
            "json": (f"/meta/{it.json_rel}" if it.json_rel else None),
            "text": it.text,  # ✅ ADD THIS
            "vlm_terms": it.vlm_terms or [],
            "owl_labels": it.owl_labels or [],
            "ctime": it.ctime,
            "depth": (f"/img/{it.depth_rel}" if getattr(it, "depth_rel", None) else None),

        })
    # finalize rows/cols and sort tiles
    out = []
    for g in groups.values():
        g["rows"] = g["max_r"] + 1
        g["cols"] = g["max_c"] + 1
        g["tiles"].sort(key=lambda t: (t["r"], t["c"]))
        # cleanup
        g.pop("max_r", None)
        g.pop("max_c", None)
        out.append(g)

    out.sort(key=lambda x: x["ctime"], reverse=True)
    return out

def _extract_vlm_terms(doc: Dict) -> List[str]:
    try:
        terms = doc.get("nanoowl", {}).get("prompts", [])
        if isinstance(terms, list):
            seen = set()
            out = []
            for t in terms:
                s = str(t).strip()
                if s and s not in seen:
                    seen.add(s)
                    out.append(s)
            return out
    except Exception:
        pass
    return []


def _extract_owl_labels(doc: Dict) -> List[str]:
    labels = []
    try:
        dets = doc.get("nanoowl", {}).get("result", {}).get("detections", [])
        if isinstance(dets, list):
            for d in dets:
                lab = d.get("label")
                if isinstance(lab, str) and lab.strip():
                    labels.append(lab.strip())
        seen = set()
        uniq = []
        for x in labels:
            if x not in seen:
                seen.add(x)
                uniq.append(x)
        return uniq
    except Exception:
        return []



def _ann_variant(path: Path) -> Path:
    """Prefer annotated image under a sibling '<parent>_ann' directory.
       Fallback to '<basename>_ann.jpg' next to the original."""
    parent = path.parent
    stem = path.stem

    sibling_ann_dir = parent.with_name(parent.name + "_ann")
    if sibling_ann_dir.exists() and sibling_ann_dir.is_dir():
        cand1 = sibling_ann_dir / (stem + "_ann.jpg")
        if cand1.exists():
            return cand1
        cand2 = sibling_ann_dir / path.name
        if cand2.exists():
            return cand2

    return path.with_suffix("").with_name(stem + "_ann").with_suffix(".jpg")


def _latest_run_dir(root: Path) -> Optional[Path]:
    latest = root / "latest"
    return latest if latest.exists() and latest.is_dir() else None
    """Return newest immediate subdirectory under root (by ctime)."""
    try:
        subdirs = [d for d in root.iterdir() if d.is_dir()]
        if not subdirs:
            return None
        subdirs.sort(key=lambda d: d.stat().st_ctime, reverse=True)
        return subdirs[0]
    except Exception:
        return None

def _is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def _best_json_for_image(img_path: Path) -> Optional[Path]:
    """Return JSON that sits next to the image (same basename + .json)."""
    cand = img_path.with_suffix(JSON_EXT)
    return cand if cand.exists() else None

def _extract_text(doc: dict) -> str:
    """
    Extract clean human-readable text for display.
    Priority: entries[0].response → response_describe → fallback keys.
    """
    try:
        entries = doc.get("entries")
        if isinstance(entries, list) and len(entries) > 0:
            v = entries[0].get("response")
            if isinstance(v, str) and v.strip():
                return v.strip().replace("</s>", "").strip()
    except Exception:
        pass

    v = doc.get("response_describe")
    if isinstance(v, str) and v.strip():
        return v.strip().replace("</s>", "").strip()

    for key in ("description", "caption", "summary", "output", "response"):
        v = doc.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip().replace("</s>", "").strip()

    return "(no textual description found in JSON)"



def _collect_items(root: Path, rel_root: Path) -> List[Item]:
    items: List[Item] = []
    seen_keys = set()  
    imgs_list = [p for p in root.glob("**/*")]
    print(f"[_collect_items] images list: {imgs_list}")
    for img_path in root.glob("**/*"):
        if not img_path.is_file():
            continue
        if not _is_image(img_path):
            continue

        # skip depth images (they should be attached to the RGB/ANN tile, not shown as standalone)
        if img_path.parent.name.endswith("_depth") or img_path.stem.endswith("_depth"):
            continue

        if img_path.stem.endswith("_ann"):
            continue

        print(f"[_collect_items] Found image: {img_path}")

        ann_path = _ann_variant(img_path)
        use_path = ann_path if ann_path.exists() else img_path

        key = img_path.stem
        if key in seen_keys:
            continue
        seen_keys.add(key)

        json_path = _best_json_for_image(img_path)

        text = ""
        vlm_terms: List[str] = []
        owl_labels: List[str] = []
        ctime_list = [img_path.stat().st_ctime]

        if ann_path.exists():
            try:
                ctime_list.append(ann_path.stat().st_ctime)
            except Exception:
                pass

        if json_path and json_path.exists():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    doc = json.load(f)
                text = _extract_text(doc)
                vlm_terms: List = _extract_vlm_terms(doc) or []
                owl_labels = _extract_owl_labels(doc) or []
                ctime_list.append(json_path.stat().st_ctime)
            except Exception:
                text = "(failed to read/parse JSON)"

        depth_path = _depth_variant(img_path)
        depth_rel = str(depth_path.relative_to(rel_root)) if depth_path else None

        items.append(Item(
            basename=img_path.stem, 
            image_rel=str(use_path.relative_to(rel_root)),                 
            json_rel=(str(json_path.relative_to(rel_root)) if json_path else None),
            ctime=max(ctime_list),
            text=text,
            vlm_terms=vlm_terms,
            owl_labels=owl_labels,
            depth_rel=depth_rel,

        ))

    items.sort(key=lambda it: it.ctime, reverse=True)
    return items

# -----------------
# Flask app
# -----------------
def create_app(root_dir: Path, scan_interval: float, latest_only: bool, static_dir: Path) -> Flask:
    app = Flask(__name__)
    app.config["ROOT_DIR"] = root_dir
    app.config["SCAN_INTERVAL"] = scan_interval
    app.config["LATEST_ONLY"] = latest_only
    app.config["STATIC_DIR"] = static_dir

    @app.get("/")
    def index():
        return render_template_string(INDEX_HTML, scan_interval=app.config["SCAN_INTERVAL"])

    @app.get("/api/items")
    def api_items():
        root: Path = app.config["ROOT_DIR"]
        latest_only: bool = app.config.get("LATEST_ONLY", False)
        scan_root = root
        current_run = None

        if latest_only:
            last_dir = _latest_run_dir(root)
            if last_dir is not None:
                # assert last_dir.is_symlink(), f"Latest run directory {last_dir} is not a symlink!"
                scan_root = last_dir.resolve()
                current_run = str(last_dir.relative_to(root))

        items = _collect_items(scan_root, rel_root=root)
        payload = [
            {
                "basename": it.basename,
                "image": f"/img/{it.image_rel}",
                "json": (f"/meta/{it.json_rel}" if it.json_rel else None),
                "ctime": it.ctime,
                "vlm_terms": it.vlm_terms or [],
                "owl_labels": it.owl_labels or [],
                "depth": (f"/img/{it.depth_rel}" if it.depth_rel else None),

            }
            for it in items
        ]
        return jsonify({"ok": True, "count": len(payload), "items": payload,
                        "root": str(root), "scan_root": str(scan_root), "current_run": current_run})

    @app.get("/api/puzzles")
    def api_puzzles():
        root: Path = app.config["ROOT_DIR"]
        latest_only: bool = app.config.get("LATEST_ONLY", False)
        scan_root = root
        current_run = None

        if latest_only:
            last_dir = _latest_run_dir(root)
            if last_dir is not None:
                scan_root = last_dir.resolve()
                current_run = str(last_dir.relative_to(root))

        items = _collect_items(scan_root, rel_root=root)
        groups = _group_tiles(items, root=root)

        return jsonify({
            "ok": True,
            "count_groups": len(groups),
            "groups": groups,
            "root": str(root),
            "scan_root": str(scan_root),
            "current_run": current_run
        })

    @app.get("/img/<path:rel>")
    def serve_image(rel: str):
        root: Path = app.config["ROOT_DIR"]
        full = (root / rel).resolve()
        if not str(full).startswith(str(root.resolve())):
            abort(403)
        if not full.exists() or not full.is_file():
            abort(404)
        return send_from_directory(str(full.parent), full.name)

    @app.get("/meta/<path:rel>")
    def serve_json(rel: str):
        root: Path = app.config["ROOT_DIR"]
        full = (root / rel).resolve()
        if not str(full).startswith(str(root.resolve())):
            abort(403)
        if not full.exists() or not full.is_file():
            abort(404)
        return send_from_directory(str(full.parent), full.name, mimetype="application/json")

    @app.get("/static/<path:filename>")
    def serve_static(filename: str):
        static_dir: Path = app.config["STATIC_DIR"]
        full = (static_dir / filename).resolve()
        if not str(full).startswith(str(static_dir.resolve())):
            abort(403)
        if not full.exists() or not full.is_file():
            abort(404)
        return send_from_directory(str(full.parent), full.name)

    return app


# -----------------
# HTML template (inline)
# -----------------

INDEX_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>VLM Ingest Viewer</title>
  <style>
    :root { --bg:#0f172a; --card:#111827; --ink:#e2e8f0; --muted:#9ca3af; --accent:#22d3ee; }
    html,body { margin:0; padding:0; background:var(--bg); color:var(--ink); font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; }
    header { display:flex; gap:12px; align-items:center; padding:14px 18px; border-bottom:1px solid #1f2937; position:sticky; top:0; background:linear-gradient(180deg, rgba(15,23,42,.95), rgba(15,23,42,.75)); backdrop-filter: blur(6px); }
    h1 { margin:0; font-size:35px; letter-spacing:0.3px; }
    .logo { height: 80px; width: auto; border-radius: 10px; margin-right: 10px;}
    .badge { background:#0ea5b7; color:#002227; padding:2px 8px; border-radius:999px; font-weight:700; font-size:12px; }
    .grid{
    display:grid;
    grid-template-columns: 1fr;     
    gap:16px;
    padding:18px;
    width:100%;
    max-width:1600px;              
    margin:0 auto;                 
    }
    .card { background:var(--card); border:1px solid #1f2937; border-radius:16px; overflow:hidden; box-shadow:0 8px 24px rgba(0,0,0,.25); transition: transform .15s ease, box-shadow .15s ease; }
    .card:hover { transform: translateY(-2px); box-shadow:0 12px 28px rgba(0,0,0,.35); }
    .thumb { width:100%; height:180px; object-fit:cover; display:block; background:#0b1220; }
    .body { padding:12px 14px 14px; display:flex; flex-direction:column; gap:8px; }
    .title { display:flex; align-items:center; gap:8px; justify-content:space-between; }
    .basename { font-size:13px; color:var(--muted); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; max-width:70%; }
    .ctime { font-size:12px; color:#7dd3fc; }
    .text { font-size:14px; line-height:1.35; color:#e5e7eb; max-height:72px; overflow:hidden; mask-image: linear-gradient(to bottom, black 70%, transparent 100%); }
    .row { display:flex; gap:8px; align-items:center; }
    .btn { cursor:pointer; border:1px solid #1f2937; background:#0b1325; color:#e2e8f0; padding:8px 10px; border-radius:12px; font-size:13px; }
    .btn:hover { background:#0e162f; }

    /* chips for LLM terms */
    .chips { display:flex; align-items:center; gap:6px; flex-wrap:wrap; }
    .chips-title { color:#9ca3af; font-size:20px; margin-right:4px; }
    .chip { display:inline-block; font-size:20px; padding:2px 8px; border-radius:999px; border:1px solid #1f2937; background:#0b1325; color:#e5e7eb; }

    .footer { color:#9ca3af; font-size:12px; padding:6px 18px 14px; text-align:center; }

    /* Modal */
    .modal { position:fixed; inset:0; display:none; background:rgba(0,0,0,.5); align-items:center; justify-content:center; padding:20px; }
    .modal.open { display:flex; }
    .modal-card { max-width:1100px; width:100%; max-height:90vh; background:#0b1220; border:1px solid #253044; border-radius:18px; overflow:hidden; display:grid; grid-template-columns: 1.1fr 0.9fr; }
    .modal-left { background:#0a0f1b; border-right:1px solid #1e293b; display:flex; align-items:center; justify-content:center; }
    .modal-left img { width:100%; height:100%; object-fit:contain; }
    .modal-right { padding:14px; display:flex; flex-direction:column; gap:10px; }
    .modal-head { display:flex; justify-content:space-between; align-items:center; gap:12px; }
    .modal-title { font-size:14px; color:#93c5fd; }
    .close { cursor:pointer; font-size:22px; line-height:22px; padding:4px 10px; color:#93c5fd; border:1px solid #1f2937; border-radius:10px; }
    pre { margin:0; padding:12px; background:#0b1629; border:1px solid #1f2937; border-radius:12px; color:#c7d2fe; font-size:12px; overflow:auto; max-height:60vh; }
   
    .puzzle-card { background:var(--card); border:1px solid #1f2937; border-radius:16px; overflow:hidden; box-shadow:0 8px 24px rgba(0,0,0,.25); }
    .puzzle-head { display:flex; justify-content:space-between; align-items:center; gap:10px; padding:12px 14px; border-bottom:1px solid #1f2937; }
    .puzzle-title { font-size:14px; color:#93c5fd; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; max-width:75%; }
    .puzzle-grid { display:grid; gap:4px; padding:10px; background:#0b1220; }
    .tile { position:relative; border-radius:10px; overflow:hidden; border:1px solid #111827; background:#0a0f1b; }
    .tile img { width:100%; height:auto; object-fit:cover; display:block; cursor:pointer; }
    .tile-badge {
    position:absolute; left:8px; top:8px;
    font-size:11px; color:#e2e8f0;
    background:rgba(2,6,23,.65);
    border:1px solid rgba(148,163,184,.25);
    padding:2px 8px; border-radius:999px;
    backdrop-filter: blur(4px);
    pointer-events:none;
    }
    .tile-desc{
    padding: 10px 14px 14px;
    color: #e5e7eb;
    font-size: 14px;
    line-height: 1.35;
    max-height: 72px;
    overflow: hidden;
    mask-image: linear-gradient(to bottom, black 70%, transparent 100%);
    }

    .tile { 
    display:flex; 
    flex-direction:column; 
    }

    .tile-media { 
    position:relative; 
    }

    .tile-meta{
    padding: 8px 10px 10px;
    border-top: 1px solid #111827;
    background: rgba(2,6,23,.35);
    }
    .tile-media { aspect-ratio: 16 / 9; }
    .tile-media img { width:100%; height:100%; object-fit:cover; }
    /* --- depth+ann overlay --- */

    /* --- depth+ann overlay --- */
    .tile-overlay { position:relative; aspect-ratio: 16 / 9; }
    .tile-overlay img { position:absolute; inset:0; width:100%; height:100%; object-fit:cover; display:block; }

    /* layering */
    .depth-img { z-index: 1; opacity: 0.95; }
    .ann-img   { z-index: 2; opacity: 1.0; mix-blend-mode: normal; }
    .tile-badge{ z-index: 3; }

    /* only when depth exists */
    .tile-overlay.has-depth .ann-img { opacity: 0.6; }
        
  </style>
</head>
<body>
  <header>
    <img src="/static/sparx_logo.png" alt="Logo" class="logo" />
    <h1>VLM Ingest Viewer</h1>
    <span class="badge" id="count">0</span>
    <div style="margin-left:auto; display:flex; gap:10px; align-items:center;">
      <small style="color:var(--muted)">Auto-refresh <b id="interval"></b> sec</small>
      <button class="btn" id="refreshBtn">Refresh now</button>
    </div>
  </header>

  <main class="grid" id="grid"></main>
  <div class="footer">Serving images + metadata from your ingested folder • VLM on Jetson ♥</div>

  <div class="modal" id="modal">
    <div class="modal-card">
      <div class="modal-left"><img id="modalImg" alt="preview"/></div>
      <div class="modal-right">
        <div class="modal-head">
          <div class="modal-title" id="modalTitle"></div>
          <button class="close" id="closeBtn">×</button>
        </div>
        <div class="row"><small id="modalText" style="color:#e5e7eb"></small></div>
        <pre id="modalJson">{}</pre>
      </div>
    </div>
  </div>

  <script>
    const SCAN_INTERVAL = {{ scan_interval|tojson }};
    let timer = null;

    function fmtTime(ts){
      try { return new Date(ts*1000).toLocaleString(); } catch(e){ return String(ts); }
    }

    // safely collapse newlines and remove </s>
    function cleanText(s){
      return (s || '')
        .replace(/\r?\n/g, ' ')
        .replace(/<\/s>/g, '')
        .trim();
    }

    // build a chip row (LLM only)
    function chipRow(title, arr){
      if(!arr || !arr.length) return '';
      const chips = arr.map(x => `<span class="chip" title="${x}">${x}</span>`).join('');
      return `<div class="chips"><span class="chips-title">${title}</span>${chips}</div>`;
    }

    function renderPuzzles(groups){
    const grid = document.getElementById('grid');
    document.getElementById('count').textContent = groups.length;

    grid.innerHTML = groups.map(g => {
        const cols = Math.max(1, g.cols || 1);
        const style = `grid-template-columns: repeat(${cols}, 1fr);`;

        const firstTile = (g.tiles || [])[0] || {};
        const desc = cleanText(firstTile.text || '');
        const tilesHtml = (g.tiles || []).map(t => `
        <div class="tile">
            <div class="tile-media tile-overlay ${t.depth ? 'has-depth' : ''}">
            <span class="tile-badge">[${t.r},${t.c}]</span>

            ${t.depth ? '<img class="depth-img" src="' + t.depth + '" alt="depth" />' : ''}

            <img class="ann-img" src="${t.image}" alt="[${t.r},${t.c}]"
                onclick="openModal(${encodeURIComponent(JSON.stringify(JSON.stringify(t)))})" />
            </div>

            <div class="tile-meta">
            ${chipRow('VLM', t.vlm_terms || [])}
            </div>
        </div>
        `).join('');

        return `
        <div class="puzzle-card">
            <div class="puzzle-head">
            <div class="puzzle-title" title="${g.group_id}">${g.group_id}</div>
            <div class="ctime">${fmtTime(g.ctime)}</div>
            </div>

            <div class="puzzle-grid" style="${style}">
            ${tilesHtml || '<div style="padding:10px;color:#9ca3af">No tiles</div>'}
            </div>

            ${desc ? `<div class="tile-desc">${desc}</div>` : ''}
        </div>
        `;
    }).join('');
    }


    async function load(){
    try{
        const r = await fetch('/api/puzzles');
        const js = await r.json();
        if(js && js.ok) renderPuzzles(js.groups || []);
    }catch(e){ console.error(e); }
    }
    function start(){
      document.getElementById('interval').textContent = SCAN_INTERVAL;
      load();
      timer = setInterval(load, SCAN_INTERVAL*1000);
    }

    function stop(){ if(timer){ clearInterval(timer); timer = null; } }

    document.getElementById('refreshBtn').addEventListener('click', ()=>{ stop(); load(); start(); });

    async function openModal(serialized){
      const it = JSON.parse(JSON.parse(decodeURIComponent(serialized)));
      document.getElementById('modalImg').src = it.image;
      document.getElementById('modalTitle').textContent = it.basename + ' — ' + fmtTime(it.ctime);

      // LLM chips + caption (no OWL)
      const vlmRow = chipRow('VLM', it.vlm_terms || []);
      const desc = cleanText(it.text || '');
      document.getElementById('modalText').innerHTML =
      vlmRow + (desc ? `<div style="margin-top:8px; color:#e5e7eb; line-height:1.35">${desc}</div>` : '');

      // pretty JSON
      let pretty = '{}';
      try{
        if(it.json){
          const r = await fetch(it.json);
          const js = await r.json();
          pretty = JSON.stringify(js, null, 2);
        }
      }catch(e){ pretty = '(failed to load json)'; }
      document.getElementById('modalJson').textContent = pretty;
      document.getElementById('modal').classList.add('open');
    }

    document.getElementById('closeBtn').addEventListener('click', ()=>{
      document.getElementById('modal').classList.remove('open');
    });
    document.getElementById('modal').addEventListener('click', (e)=>{
      if(e.target.id === 'modal') document.getElementById('modal').classList.remove('open');
    });

    start();
  </script>
</body>
</html>
"""
# -----------------
# Entrypoint
# -----------------

def main():
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"ROOT dir not found: {root}")

    if args.static_dir:
        static_dir = Path(args.static_dir).expanduser().resolve()
    else:
        static_dir = (Path(__file__).parent / "static").resolve()

    static_dir.mkdir(parents=True, exist_ok=True)  # keep it simple

    app = create_app(root, args.scan_interval, args.latest_only, static_dir)

    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()