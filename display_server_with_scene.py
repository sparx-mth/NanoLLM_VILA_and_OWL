#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VLM Ingest DISPLAY Server — Final Debug Version

Updates:
- Forces display of BOTH Scene and Text (no hiding).
- Adds Cache-Busting to prevent browser from showing old data.
- Server prints the exact payload of the first item to verify data integrity.
"""

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
from flask import Flask, jsonify, render_template_string, send_from_directory, abort

# -----------------
# Config & parsing
# -----------------

def parse_args():
    p = argparse.ArgumentParser(description="VLM Ingest DISPLAY Server")
    p.add_argument("--root", required=True, help="Directory holding ingested files")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--scan-interval", type=float, default=2.0)
    p.add_argument("--latest-only", action="store_true")
    p.add_argument("--static", dest="static_dir", default=None)
    return p.parse_args()

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}

@dataclass
class Item:
    basename: str
    image_rel: str
    json_rel: Optional[str]
    ctime: float
    text: str                  # Bullet points
    scene: str                 # Full sentence
    vlm_terms: List[str]

def _clean_str(s):
    if isinstance(s, str):
        return s.replace("</s>", "").strip()
    return ""

def _extract_vlm_terms(doc: Dict) -> List[str]:
    try:
        terms = doc.get("nanoowl", {}).get("prompts", [])
        if isinstance(terms, list):
            return [_clean_str(str(t)) for t in terms if t]
    except Exception:
        pass
    return []

def _extract_text(doc: dict) -> str:
    entries = doc.get("entries")
    if isinstance(entries, list) and len(entries) > 0:
        return _clean_str(entries[0].get("response"))
    return _clean_str(doc.get("response_describe") or doc.get("response"))

def _extract_scene(doc: dict, filename: str) -> str:
    # Priority search for scene
    v = _clean_str(doc.get("scene"))
    if not v: v = _clean_str(doc.get("nanoowl", {}).get("scene"))
    if not v: v = _clean_str(doc.get("nanoowl", {}).get("result", {}).get("scene"))
    
    if v:
        # Keep this log to confirm extraction works
        print(f"[DEBUG] Found SCENE in {filename}: {v[:30]}...")
    return v if v else ""

def _collect_items(root: Path, rel_root: Path) -> List[Item]:
    items: List[Item] = []
    seen_keys = set()
    
    for img_path in root.glob("**/*"):
        if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_EXTS: continue
        if img_path.stem.endswith("_ann"): continue 
        
        # Smart pairing with annotated images
        parent = img_path.parent
        stem = img_path.stem
        ann_path = None
        
        # Check standard annotation folder structure
        ann_folder = parent.with_name(parent.name + "_ann")
        if ann_folder.exists():
            cand = ann_folder / (stem + "_ann.jpg")
            if cand.exists(): ann_path = cand
            else:
                cand = ann_folder / img_path.name
                if cand.exists(): ann_path = cand
        
        if not ann_path:
            cand = img_path.with_name(stem + "_ann.jpg")
            if cand.exists(): ann_path = cand

        use_path = ann_path if (ann_path and ann_path.exists()) else img_path

        if stem in seen_keys: continue
        seen_keys.add(stem)

        json_path = img_path.with_suffix(".json")
        text, scene, vlm_terms = "", "", []
        ctime = img_path.stat().st_ctime

        if json_path.exists():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    doc = json.load(f)
                text = _extract_text(doc)
                scene = _extract_scene(doc, json_path.name)
                vlm_terms = _extract_vlm_terms(doc)
                ctime = max(ctime, json_path.stat().st_ctime)
            except Exception as e:
                print(f"Error reading JSON {json_path}: {e}")

        items.append(Item(
            basename=stem,
            image_rel=str(use_path.relative_to(rel_root)),                 
            json_rel=(str(json_path.relative_to(rel_root)) if json_path.exists() else None),
            ctime=ctime,
            text=text,
            scene=scene,
            vlm_terms=vlm_terms or []
        ))

    items.sort(key=lambda it: it.ctime, reverse=True)
    return items

def create_app(root_dir: Path, scan_interval: float, latest_only: bool,  static_dir: Path) -> Flask:
    app = Flask(__name__)
    app.config["ROOT_DIR"] = root_dir
    app.config["SCAN_INTERVAL"] = scan_interval
    app.config["LATEST_ONLY"] = latest_only
    app.config["STATIC_DIR"] = static_dir

    @app.after_request
    def add_header(r):
        """Force browser to not cache API responses"""
        r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        r.headers["Pragma"] = "no-cache"
        r.headers["Expires"] = "0"
        return r

    @app.get("/")
    def index():
        return render_template_string(INDEX_HTML, scan_interval=app.config["SCAN_INTERVAL"])

    @app.get("/api/items")
    def api_items():
        root = app.config["ROOT_DIR"]
        scan_root = root
        if app.config["LATEST_ONLY"]:
            latest_link = root / "latest"
            if latest_link.exists():
                scan_root = latest_link.resolve() if latest_link.is_symlink() else latest_link

        items = _collect_items(scan_root, rel_root=root)
        
        payload = [
            {
                "basename": it.basename,
                "image": f"/img/{it.image_rel}",
                "json": (f"/meta/{it.json_rel}" if it.json_rel else None),
                "ctime": it.ctime,
                "text": it.text,
                "scene": it.scene,
                "vlm_terms": it.vlm_terms,
            }
            for it in items
        ]
        
        # DEBUG: Print the first item's payload to verify 'scene' is there
        if payload:
            first = payload[0]
            if first['scene']:
                print(f"[API DEBUG] Sending item '{first['basename']}' with SCENE: {first['scene'][:50]}...")
            else:
                print(f"[API DEBUG] Sending item '{first['basename']}' WITHOUT scene.")

        return jsonify({"ok": True, "items": payload})

    @app.get("/img/<path:rel>")
    def serve_image(rel: str): return send_from_directory(str(app.config["ROOT_DIR"]), rel)

    @app.get("/meta/<path:rel>")
    def serve_json(rel: str): return send_from_directory(str(app.config["ROOT_DIR"]), rel, mimetype="application/json")
    
    @app.get("/static/<path:filename>")
    def serve_static(filename: str):
        if app.config["STATIC_DIR"]: return send_from_directory(str(app.config["STATIC_DIR"]), filename)
        abort(404)

    return app

INDEX_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>VLM Viewer v2</title>
  <style>
    :root { --bg:#0f172a; --card:#1e293b; --text:#f1f5f9; --muted:#94a3b8; --accent:#38bdf8; --border:#334155; }
    body { background:var(--bg); color:var(--text); font-family: system-ui, sans-serif; margin:0; }
    
    header { padding:14px 20px; border-bottom:1px solid var(--border); background:rgba(15,23,42,0.95); position:sticky; top:0; z-index:10; display:flex; align-items:center; justify-content:space-between; backdrop-filter:blur(5px); }
    h1 { margin:0; font-size:20px; color: var(--accent); }
    
    .grid { display:grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap:20px; padding:20px; }
    .card { background:var(--card); border:1px solid var(--border); border-radius:12px; overflow:hidden; transition:transform 0.2s; display:flex; flex-direction:column; }
    .card:hover { transform:translateY(-4px); border-color:var(--accent); }
    .thumb { width:100%; height:200px; object-fit:cover; background:#000; cursor:pointer; }
    .body { padding:16px; flex-grow:1; display:flex; flex-direction:column; gap:10px; }
    .meta { font-size:12px; color:var(--muted); display:flex; justify-content:space-between; font-family:monospace; }
    
    /* SCENE BOX */
    .scene-box {
        background: rgba(56, 189, 248, 0.15);
        border-left: 4px solid var(--accent);
        padding: 10px;
        border-radius: 4px;
        font-size: 14px;
        line-height: 1.4;
        color: #e0f2fe;
        display: block; /* Ensure it is not hidden */
        margin-bottom: 6px;
    }

    /* TEXT BOX */
    .text-box {
        background: #0f172a;
        padding: 8px;
        border-radius: 6px;
        font-size: 12px;
        color: var(--muted);
        white-space: pre-wrap;
        border: 1px solid var(--border);
    }

    .chips { display:flex; flex-wrap:wrap; gap:6px; margin-top:auto; }
    .chip { font-size:11px; padding:3px 8px; background:#334155; border-radius:10px; color:#cbd5e1; }
    .links { margin-top:10px; padding-top:10px; border-top:1px solid var(--border); display:flex; gap:10px; }
    a { color:var(--accent); text-decoration:none; font-size:12px; font-weight:bold; }

    /* Modal */
    .modal { position:fixed; inset:0; background:rgba(0,0,0,0.85); display:none; align-items:center; justify-content:center; padding:20px; z-index:100; backdrop-filter:blur(4px); }
    .modal.open { display:flex; }
    .modal-content { background:var(--bg); width:90%; max-width:1200px; height:85vh; border-radius:12px; border:1px solid var(--border); display:grid; grid-template-columns: 1.5fr 1fr; overflow:hidden; }
    .modal-img-container { background:#000; display:flex; align-items:center; justify-content:center; }
    .modal-img-container img { max-width:100%; max-height:100%; }
    .modal-info { padding:24px; overflow-y:auto; border-left:1px solid var(--border); display:flex; flex-direction:column; gap:20px; }
    pre { background:#0f172a; padding:15px; border-radius:8px; overflow:auto; font-size:11px; color:#cbd5e1; border:1px solid var(--border); }
    .big-scene { font-size:18px; line-height:1.5; color:#fff; padding-bottom:15px; border-bottom:1px solid var(--border); }
    .close-btn { position:absolute; top:20px; right:20px; color:#fff; font-size:30px; cursor:pointer; }
  </style>
</head>
<body>
<header>
  <h1>Ingest Viewer <span style="font-size:12px; opacity:0.5;">v3</span></h1>
  <div>
     <span style="font-size:12px; color:var(--muted)">Auto-refresh: <b id="timer">--</b>s</span>
     <button onclick="refresh()" style="background:var(--accent); border:none; border-radius:4px; padding:4px 10px; cursor:pointer;">Refresh</button>
  </div>
</header>
<main class="grid" id="grid"></main>

<div class="modal" id="modal">
    <div class="close-btn" onclick="closeModal()">×</div>
    <div class="modal-content">
        <div class="modal-img-container"><img id="mImg" /></div>
        <div class="modal-info">
            <div id="mScene" class="big-scene"></div>
            <div id="mText" style="white-space:pre-wrap; color:#94a3b8;"></div>
            <div style="margin-top:auto;"><pre id="mJson"></pre></div>
        </div>
    </div>
</div>

<script>
const INTERVAL = {{ scan_interval|tojson }};
let intervalId = null;

function render(items) {
    const grid = document.getElementById('grid');
    grid.innerHTML = items.map(it => {
        // ALWAYS render scene if it exists (no conditional hiding)
        let html = `
        <div class="card" onclick='openModal(${JSON.stringify(it).replace(/'/g, "&#39;")})'>
            <img class="thumb" src="${it.image}" loading="lazy" />
            <div class="body">
                <div class="meta"><span>${it.basename}</span></div>`;
        
        // Scene Section
        if (it.scene && it.scene.length > 0) {
            html += `<div class="scene-box" title="Scene">${it.scene}</div>`;
        }
        
        // Text Section (Objects) - Always show if present
        if (it.text && it.text.length > 0) {
            html += `<div class="text-box" title="Prompts">${it.text}</div>`;
        }

        // Chips
        html += `<div class="chips">
                    ${(it.vlm_terms||[]).map(t => `<span class="chip">${t}</span>`).join('')}
                 </div>
                 <div class="links">
                    ${it.json ? `<a href="${it.json}" target="_blank" onclick="event.stopPropagation()">JSON</a>` : ''}
                    <a href="${it.image}" target="_blank" onclick="event.stopPropagation()">IMAGE</a>
                 </div>
            </div>
        </div>`;
        return html;
    }).join('');
}

async function load() {
    try {
        // Add timestamp to prevent caching
        const res = await fetch('/api/items?t=' + Date.now());
        const data = await res.json();
        if(data.ok) render(data.items);
    } catch(e) { console.error(e); }
}

function openModal(it) {
    document.getElementById('mImg').src = it.image;
    document.getElementById('mScene').textContent = it.scene || "No scene description.";
    document.getElementById('mText').textContent = it.text || "";
    if(it.json) {
        fetch(it.json).then(r=>r.json()).then(j => {
            document.getElementById('mJson').textContent = JSON.stringify(j, null, 2);
        });
    } else { document.getElementById('mJson').textContent = "{}"; }
    document.getElementById('modal').classList.add('open');
}
function closeModal() { document.getElementById('modal').classList.remove('open'); }
function refresh() { clearInterval(intervalId); load(); startTimer(); }
function startTimer() { document.getElementById('timer').textContent = INTERVAL; intervalId = setInterval(load, INTERVAL * 1000); }
document.addEventListener('keydown', e => { if(e.key === 'Escape') closeModal(); });

startTimer();
load();
</script>
</body>
</html>
"""

def main():
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    
    if args.static_dir:
        static = Path(args.static_dir).expanduser().resolve()
        static.mkdir(parents=True, exist_ok=True)
    else: static = None

    app = create_app(root, args.scan_interval, args.latest_only, static)
    print(f"[*] Serving {root} on port {args.port}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)

if __name__ == "__main__":
    main()