import requests
import json
import time
import os
from typing import Optional
from txt_and_image_utils import _update_sidecar_json
def _remap_path(local_path: str, src_root: Optional[str], dst_root: Optional[str]) -> str:
    if not src_root or not dst_root:
        return os.path.abspath(local_path)
    local_path = os.path.abspath(local_path)
    src_root = os.path.abspath(src_root)
    try:
        rel = os.path.relpath(local_path, src_root)
    except ValueError:
        return local_path
    return os.path.join(dst_root, rel)

def prep_vlm(args, jpg_path, pose, json_path):
    img_for_vlm = _remap_path(jpg_path,
                              args.vlm_path_src or None,
                              args.vlm_path_dst or None)
    print(f"[vlm] POST {args.vlm}  image_path={img_for_vlm}")
    vlm_caption = _call_vlm(args.vlm, img_for_vlm, args.vlm_timeout, args.vlm_retries)
    if vlm_caption:
        print(f"[vlm] caption: {vlm_caption[:120]}{'…' if len(vlm_caption) > 120 else ''}")
    else:
        print("[vlm] WARN: no caption returned")
    _update_sidecar_json(json_path, pose, os.path.basename(jpg_path), vlm_text=vlm_caption)


def _call_vlm(endpoint: Optional[str], image_path_for_vlm: str, timeout: float, retries: int) -> Optional[str]:
    if not endpoint:
        return None
    payload = {"image_path": image_path_for_vlm}
    last_err = None
    for _ in range(max(1, retries)):
        try:
            r = requests.post(endpoint, json=payload, timeout=timeout)
            r.raise_for_status()
            try:
                data = r.json()
                if isinstance(data, dict):
                    return data.get("response") or data.get("caption") or json.dumps(data, ensure_ascii=False)
                return json.dumps(data, ensure_ascii=False)
            except ValueError:
                return r.text.strip()
        except Exception as e:
            last_err = e
            time.sleep(0.2)
    print(f"[vlm] WARN: describe failed for {image_path_for_vlm}: {last_err}")
    return None