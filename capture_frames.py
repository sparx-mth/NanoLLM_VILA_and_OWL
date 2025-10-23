#!/usr/bin/env python3
# capture_frames.py
import datetime, sys, glob
import os, json, time, argparse, cv2, re
from pathlib import Path
from typing import Optional, Tuple
import requests
import sys, select, termios, tty
from contextlib import nullcontext

try:
    import Jetson.GPIO as GPIO
except Exception:
    try:
        import RPi.GPIO as GPIO
    except Exception:
        GPIO = None

class HeadlessKeys:
    """Non-blocking single-char reader from stdin (no OpenCV window needed)."""
    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)               # raw-ish mode
        return self
    def __exit__(self, *exc):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)
    def getch(self, timeout_ms=1):
        # returns a single lowercased char like 'c','q' or None if no input
        r, _, _ = select.select([sys.stdin], [], [], timeout_ms/1000.0)
        if r:
            ch = sys.stdin.read(1)
            return ch.lower()
        return None

_POSE_NAME_RES = [
    re.compile(
        r".*?/x(?P<x>-?\d{1,6})y(?P<y>-?\d{1,6})z(?P<z>-?\d{1,6})yaw(?P<yaw>-?\d{1,9})(?:__[^/]+)?\.[A-Za-z0-9]+$"
    ),
    re.compile(
        r".*?_x(?P<x>-?\d+(?:\.\d+)?)_y(?P<y>-?\d+(?:\.\d+)?)_z(?P<z>-?\d+(?:\.\d+)?)_yaw(?P<yaw>-?\d+(?:\.\d+)?)(?:\.[A-Za-z0-9]+)$"
    ),
]

def _fmt_signed(value: float, scale: int, width: int, eps: float) -> str:
    if abs(value) < eps:
        value = 0.0
    n = int(round(value * scale))
    if n < 0:
        return f"-{abs(n):0{width}d}"
    else:
        return f"{n:0{width}d}"

def parse_pose_from_name(fname: str):
    base = os.path.basename(fname)
    for rx in _POSE_NAME_RES:
        m = rx.match(fname) or rx.match(base)
        if not m:
            continue
        gd = m.groupdict()
        # If the captures are integers (mm / microrad), convert to meters / radians
        try:
            # compact-int format → ints
            x_mm   = int(gd["x"])
            y_mm   = int(gd["y"])
            z_mm   = int(gd["z"])
            yaw_ur = int(gd["yaw"])   # microradians
            return {
                "x": x_mm / 1000.0,
                "y": y_mm / 1000.0,
                "z": z_mm / 1000.0,
                "yaw": yaw_ur / 1_000_000.0,
            }
        except ValueError:
            # legacy underscore/float format → floats already in meters/radians
            return {
                "x": float(gd["x"]),
                "y": float(gd["y"]),
                "z": float(gd["z"]),
                "yaw": float(gd["yaw"]),
            }
    return None

def load_angles_map(path: str):
    try:
        with open(path, "r") as f: j = json.load(f)
        return j if isinstance(j, dict) else {}
    except Exception:
        return {}

def list_frames(frames_dir: str):
    exts = ("*.jpg","*.jpeg","*.png","*.bmp")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(frames_dir, ext)))
    return sorted(files)

def get_pose_for_frame(path: str, *, angles_map: dict, from_name: bool):
    if from_name:
        p = parse_pose_from_name(path)
        if p:
            return p
    p = angles_map.get(os.path.basename(path))
    if isinstance(p, dict): return p
    return {"x":0.0,"y":0.0,"z":1.5,"yaw":0.0}

def pose_to_name(pose: dict) -> str:
    x = _fmt_signed(pose["x"],   scale=1000,      width=4, eps=5e-4)     # mm
    y = _fmt_signed(pose["y"],   scale=1000,      width=4, eps=5e-4)     # mm
    z = _fmt_signed(pose["z"],   scale=1000,      width=4, eps=5e-4)     # mm
    yaw = _fmt_signed(pose["yaw"], scale=1_000_000, width=7, eps=5e-7)   # microrad
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d___%H_%M_%S")

    return f"x{x}y{y}z{z}yaw{yaw}__{timestamp}"

def center_crop_frac(img, frac: float):
    """
    Center-crop by a fraction of the original size.
    frac=1.0 → no crop, 0.5 → crop to middle 50% (both width & height).
    """
    frac = max(0.05, min(1.0, float(frac)))  # clamp
    H, W = img.shape[:2]
    cw, ch = int(W * frac), int(H * frac)
    x0 = (W - cw) // 2
    y0 = (H - ch) // 2
    return img[y0:y0+ch, x0:x0+cw], (x0, y0, x0+cw, y0+ch)

def _apply_crop_and_flip(img, crop_frac: float, flip180: bool):
    """Center-crop then optionally rotate 180°."""
    work = img
    crop_box = None
    if crop_frac < 1.0:
        # assumes you already have center_crop_frac(img, frac) -> (cropped, (x1,y1,x2,y2))
        work, crop_box = center_crop_frac(img, crop_frac)
    if flip180:
        work = cv2.rotate(work, cv2.ROTATE_180)
    return work, crop_box



def _save_jpg(path: str, bgr):
    if not cv2.imwrite(path, bgr):
        raise RuntimeError(f"failed to write {path}")

def open_capture(src: str, width=None, height=None, fps=None, fourcc=None):
    if "://" in src:
        cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        try: cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception: pass
        if not cap.isOpened():
            raise RuntimeError(f"failed to open network/file source: {src}")
        return cap

    if src.startswith("/dev/video"):
        cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
        if cap.isOpened():
            if fourcc:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
            if width:  cap.set(cv2.CAP_PROP_FRAME_WIDTH,  int(width))
            if height: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
            if fps:    cap.set(cv2.CAP_PROP_FPS,          float(fps))
            try: cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception: pass
            ok,_ = cap.read()
            if ok: return cap
            cap.release()

        # fallback GStreamer
        gst = (
            f"v4l2src device={src} io-mode=2 do-timestamp=true ! "
            f"image/jpeg,framerate={int(fps) if fps else 30}/1 "
            f"! jpegdec ! videoconvert ! video/x-raw,format=BGR "
            f"! appsink drop=true max-buffers=1 sync=false"
            if (fourcc or "MJPG") == "MJPG" else
            f"v4l2src device={src} io-mode=2 do-timestamp=true ! "
            f"video/x-raw,format=YUY2,framerate={int(fps) if fps else 30}/1 "
            f"! videoconvert ! video/x-raw,format=BGR "
            f"! appsink drop=true max-buffers=1 sync=false"
        )
        cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            return cap
        raise RuntimeError(f"failed to open V4L2 source: {src}")

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open source: {src}")
    return cap

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

def _update_sidecar_json(json_path: str, pose: dict, image_basename: str, vlm_text: Optional[str]):
    obj = {}
    if os.path.isfile(json_path):
        try:
            with open(json_path, "r") as f:
                obj = json.load(f)
        except Exception:
            obj = {}
    obj.setdefault("pose", pose)
    obj.setdefault("image", image_basename)
    if vlm_text:
        obj["vlm_caption"] = vlm_text
        entries = obj.setdefault("entries", [])
        entries.append({
            "timestamp": int(time.time()),
            "prompt": "Describe the image",
            "response": vlm_text
        })
    tmp = json_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, json_path)

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


def manual_interactive(args, cap: cv2.VideoCapture):
    # load the ordered list of poses that we’ll step through
    if not os.path.isfile(args.poses):
        raise ValueError(f"poses file not found: {args.poses}")
    with open(args.poses, "r") as f:
        poses = json.load(f)
    if not isinstance(poses, list) or not poses:
        raise ValueError("poses.json must be a non-empty list of {x,y,z,yaw}")

    print("[interactive] preview on. Press SPACE or 'c' to capture next pose, 'q' to quit.")
    if args.preview:
        cv2.namedWindow("preview", cv2.WINDOW_NORMAL)

    idx = 0
    counter = 1  # used only when --suffix is set
    json_path = None
    # Use HeadlessKeys when preview is off, else keep cv2.waitKey
    with (HeadlessKeys() if not args.preview else nullcontext()):
        while True:
            for _ in range(3):
                cap.grab()
                time.sleep(0.01)

            for _ in range(3):
                cap.grab()
                time.sleep(0.01)

            ok, frame = False, None
            for _ in range(args.retry):
                ok, frame = cap.read()
                if ok and frame is not None:
                    break
                time.sleep(0.01)

            # show or not
            if args.preview:
                cv2.imshow("preview", frame)
                key = cv2.waitKey(1) & 0xFF
                key = chr(key).lower() if key != 255 else None
            else:
                key = HeadlessKeys().getch(1)  # 1ms poll (we're inside context)

            # handle quit
            if key in ('q', '\x1b'):  # q or ESC
                print("[interactive] quit requested.")
                break

            # handle capture
            if key in ('c', ' '):  # c or SPACE
                if idx >= len(poses):
                    print("[interactive] all poses captured. exiting.")
                    break

                pose = poses[idx]
                idx += 1

                base_stem = pose_to_name(pose)
                base_path = os.path.join(args.out, base_stem)
                jpg_path, json_path = (base_path + ".jpg", base_path + ".json")
                if args.suffix:
                    jpg_path, json_path = _unique_name(base_path, True, counter)
                    counter += 1

                # ensure BGR
                if len(frame.shape) == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

                # center-crop if requested (this becomes the image used everywhere)
                full_frame = frame
                crop_box = None
                try:
                    frame, crop_box = _apply_crop_and_flip(full_frame, args.crop_frac, args.flip_180)
                except Exception as e:
                    print(f"[capture] WARN: crop/flip failed ({e}). Using full frame.")
                    frame, crop_box = full_frame, None

                # optional save of original
                if getattr(args, "save_full", False):
                    try:
                        _save_jpg(jpg_path.replace(".jpg", "_full.jpg"), full_frame)
                    except Exception as e:
                        print(f"[capture] WARN: save *_full.jpg failed: {e}")

                # save cropped (or full) frame
                try:
                    _save_jpg(jpg_path, frame)
                except Exception as e:
                    print(f"[capture] ERROR: {e}")
                    continue

                # sidecar (pose + image name)
                _update_sidecar_json(json_path, pose, os.path.basename(jpg_path), vlm_text=None)

                # optional VLM call (unchanged from your code)
                vlm_caption = None
                if args.vlm:
                    prep_vlm(args, jpg_path, pose, json_path)

                print(f"[capture] saved {jpg_path}  ({idx}/{len(poses)})")

    if args.preview:
        cv2.destroyAllWindows()
def interactive_from_folder(args):
    frames = list_frames(args.frames_dir)
    if not frames:
        print(f"[interactive] no images under {args.frames_dir}")
        return
    angles_map = load_angles_map(args.angles_json) if args.angles_json else {}
    idx, counter = 0, 1
    if args.preview:
        cv2.namedWindow("preview", cv2.WINDOW_NORMAL)

    print("[interactive/folder] SPACE/'c' capture • 'n' next • 'p' prev • 'q' quit")
    while True:
        if idx < 0: idx = 0
        if idx >= len(frames):
            if args.loop_frames: idx = 0
            else:
                print("[interactive/folder] reached end.")
                break

        path = frames[idx]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[interactive/folder] WARN unreadable: {path}")
            idx += 1; continue

        pose = get_pose_for_frame(path, angles_map=angles_map, from_name=args.angles_from_name)
        work = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape)==2 else img

        if args.preview:
            cv2.imshow("preview", work)

        key = cv2.waitKey(1) & 0xFF  # still works headless if you export offscreen vars
        if key in (ord('q'), 27): break
        if key == ord('n'): idx += 1; continue
        if key == ord('p'): idx -= 1; continue
        if key in (ord('c'), ord('C'), 32):
            base = os.path.join(args.out, pose_to_name(pose))
            jpg_path, json_path = (base + ".jpg", base + ".json")
            if args.suffix:
                jpg_path, json_path = _unique_name(base, True, counter); counter += 1
            if args.save_full:
                try:
                    _save_jpg(jpg_path.replace(".jpg","_full.jpg"), img)
                except Exception as e: print(f"[capture] WARN saving full: {e}")
            try:
                _save_jpg(jpg_path, work)
                _update_sidecar_json(json_path, pose, os.path.basename(jpg_path), vlm_text=None)
                print(f"[capture] saved {jpg_path}  (src={os.path.basename(path)})")
            except Exception as e:
                print(f"[capture] ERROR: {e}")
            if args.vlm:
                prep_vlm(args, jpg_path, pose, json_path)
            idx += 1

    if args.preview:
        cv2.destroyAllWindows()

def loop_over_folder(args):
    frames = list_frames(args.frames_dir)
    if not frames:
        print(f"[loop] no images under {args.frames_dir}")
        return
    angles_map = load_angles_map(args.angles_json) if args.angles_json else {}
    idx, counter = 0, 1
    print(f"[loop] iterating {len(frames)} frames, sleep={args.loop_sleep}s")
    while True:
        path = frames[idx]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[loop] WARN unreadable: {path}")
        else:
            pose = get_pose_for_frame(path, angles_map=angles_map, from_name=args.angles_from_name)
            work = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape)==2 else img
            base = os.path.join(args.out, pose_to_name(pose))
            jpg_path, json_path = (base + ".jpg", base + ".json")
            if args.suffix:
                jpg_path, json_path = _unique_name(base, True, counter); counter += 1
            try:
                if args.save_full:
                    try: _save_jpg(jpg_path.replace(".jpg","_full.jpg"), img)
                    except Exception as e: print(f"[loop] WARN saving full: {e}")
                _save_jpg(jpg_path, work)
                _update_sidecar_json(json_path, pose, os.path.basename(jpg_path), vlm_text=None)
                print(f"[loop] saved {jpg_path} (src={os.path.basename(path)})")
            except Exception as e:
                print(f"[loop] ERROR: {e}")

            if args.vlm:
                prep_vlm(args, jpg_path, pose, json_path)
        time.sleep(max(0.0, args.loop_sleep))
        idx = (idx + 1) % len(frames)
        if not args.loop_frames and idx == 0:
            break

def timer_capture(args, cap: cv2.VideoCapture):
    # Pose-list mode (original behavior)
    with open(args.poses, "r") as f:
        poses = json.load(f)
    if not isinstance(poses, list) or not poses:
        raise ValueError("poses.json must be a non-empty list of {x,y,z,yaw}")

    for i, pose in enumerate(poses, 1):
        fname = pose_to_name(pose)
        jpg_path = os.path.join(args.out, f"{fname}.jpg")
        json_path = os.path.join(args.out, f"{fname}.json")

        print(f"[capture] {i}/{len(poses)} → {jpg_path}")
        time.sleep(args.sleep)
        for _ in range(3):
            cap.grab()
            time.sleep(0.01)

        ok, frame = False, None
        for _ in range(args.retry):
            ok, frame = cap.read()
            if ok and frame is not None:
                break
            time.sleep(0.01)
        if not ok or frame is None:
            print(f"[capture] WARN: failed to read frame for pose {i}")
            continue

        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        full_frame = frame.copy()
        crop_box = None
        if args.crop_frac < 1.0:
            try:
                frame, crop_box = _apply_crop_and_flip(full_frame, args.crop_frac, args.flip_180)
            except Exception as e:
                print(f"[capture] WARN: crop/flip failed ({e}). Using full frame.")
                frame, crop_box = full_frame, None

        if args.save_full:
            try:
                _save_jpg(jpg_path.replace(".jpg", "_full.jpg"), full_frame)
            except Exception as e:
                print(f"[capture] WARN: failed to save *_full.jpg ({e})")
        # save the (possibly cropped) working image
        try:
            _save_jpg(jpg_path, frame)
        except Exception as e:
            print(f"[capture] ERROR: {e}")
            continue

        _update_sidecar_json(json_path, pose, os.path.basename(jpg_path), vlm_text=None)

        if args.vlm:
            prep_vlm(args, jpg_path, pose, json_path)

def _setup_gpio(pin: int, edge: str, pull: str):
    if GPIO is None:
        raise RuntimeError("GPIO library not available (Jetson.GPIO / RPi.GPIO). Install or disable --gpio-pin.")
    GPIO.setmode(GPIO.BCM)
    pud = GPIO.PUD_UP if pull == "up" else GPIO.PUD_DOWN if pull == "down" else GPIO.PUD_OFF
    GPIO.setup(pin, GPIO.IN, pull_up_down=pud)
    if edge == "rising":
        ev = GPIO.RISING
    elif edge == "falling":
        ev = GPIO.FALLING
    else:
        ev = GPIO.BOTH
    return ev

def gpio_interactive(args, cap: cv2.VideoCapture):
    """
    Step through poses.json. Each GPIO edge triggers the next capture.
    Works headless (no OpenCV window needed).
    """
    # load ordered poses
    if not os.path.isfile(args.poses):
        raise ValueError(f"poses file not found: {args.poses}")
    with open(args.poses, "r") as f:
        poses = json.load(f)
    if not isinstance(poses, list) or not poses:
        raise ValueError("poses.json must be a non-empty list of {x,y,z,yaw}")

    pin = int(args.gpio_pin)
    edge = args.gpio_edge
    pull = args.gpio_pull
    debounce = max(0, int(args.gpio_debounce_ms))

    ev = _setup_gpio(pin, edge, pull)
    print(f"[gpio] listening on BCM {pin} edge={edge} pull={pull} debounce={debounce}ms")
    # Use software debounce by tracking last transition time
    last_ts_ms = 0
    idx = 0
    counter = 1
    prev_value = None
    try:
        while True:
            # Grab a fresh frame so we’re ready when an edge comes
            ok, frame = False, None
            for _ in range(args.retry):
                ok, frame = cap.read()
                if ok and frame is not None:
                    break
                time.sleep(0.01)
            if not ok or frame is None:
                time.sleep(0.01)
                continue

            value = GPIO.input(args.gpio_pin)
            if value != prev_value:
                if value == GPIO.HIGH:
                    value_str = "HIGH"
                    now_ms = int(time.time() * 1000)
                    if now_ms - last_ts_ms < debounce:
                        # ignore bounce
                        continue
                    last_ts_ms = now_ms

                    if idx >= len(poses):
                        print("[gpio] all poses captured. exiting.")
                        break

                    pose = poses[idx]
                    idx += 1

                    # ensure color
                    if len(frame.shape) == 2:
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

                    # center-crop if requested
                    full_frame = frame
                    try:
                        frame, crop_box = _apply_crop_and_flip(full_frame, args.crop_frac, args.flip_180)
                    except Exception as e:
                        print(f"[capture] WARN: crop/flip failed ({e}). Using full frame.")
                        frame, crop_box = full_frame, None

                    # build filepaths
                    base_stem = pose_to_name(pose)
                    base_path = os.path.join(args.out, base_stem)
                    jpg_path, json_path = (base_path + ".jpg", base_path + ".json")
                    if args.suffix:
                        jpg_path, json_path = _unique_name(base_path, True, counter)
                        counter += 1

                    # optional save of original
                    if getattr(args, "save_full", False):
                        try:
                            _save_jpg(jpg_path.replace(".jpg", "_full.jpg"), full_frame)
                        except Exception as e:
                            print(f"[capture] WARN: save *_full.jpg failed: {e}")

                    # save working image
                    try:
                        _save_jpg(jpg_path, frame)
                    except Exception as e:
                        print(f"[capture] ERROR: {e}")
                        continue

                    # sidecar (pose + image)
                    _update_sidecar_json(json_path, pose, os.path.basename(jpg_path), vlm_text=None)

                    # optional VLM call (unchanged)
                    if args.vlm:
                        prep_vlm(args, jpg_path, pose, json_path)

                    print(f"[capture] GPIO-triggered save → {jpg_path}  ({idx}/{len(poses)})")
                else:
                    value_str = "LOW"
                prev_value = value

            # light idle
            time.sleep(0.2)

    finally:
        try:
            GPIO.cleanup()
        except Exception:
            pass

def _unique_name(base_path: str, enable_suffix: bool, counter: int) -> Tuple[str, str]:
    """
    Return (jpg_path, json_path). If enable_suffix, append _0001, _0002...
    """
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d___%H_%M_%S")
    base_path = f"{base_path}___{timestamp}"
    jpg_path = base_path + ".jpg"
    json_path = base_path + ".json"
    if not enable_suffix:
        return jpg_path, json_path
    n = counter
    while True:
        suffix = f"_{n:04d}"
        jp = base_path + suffix + ".jpg"
        jj = base_path + suffix + ".json"
        if not (os.path.exists(jp) or os.path.exists(jj)):
            return jp, jj
        n += 1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="RTSP url or /dev/videoX or file path")
    ap.add_argument("--poses", default="/opt/missions/poses.json", help="poses.json: [{x,y,z,yaw}, ...]")
    ap.add_argument("--out", default="captures", help="output directory")
    ap.add_argument("--sleep", type=float, default=2.0, help="seconds to wait before each grab (pose mode)")
    ap.add_argument("--warmup", type=int, default=10, help="drop N frames on startup")
    ap.add_argument("--retry", type=int, default=5, help="frame grab retries per capture")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--fourcc", default="MJPG", help="MJPG or YUYV for /dev/video cams")
    ap.add_argument("--crop-frac", type=float, default=0.6,
                    help="Center-crop fraction for captured image (1.0=no crop).")
    ap.add_argument("--save-full", action="store_true",
                    help="Also save the full uncropped frame as *_full.jpg for debugging.")
    ap.add_argument("--flip-180", action="store_true",
                    help="Rotate the (cropped) image by 180° before saving.")

    # Read from directory options
    ap.add_argument("--frames-dir", default="", help="If set, operate on images from this folder instead of camera")
    ap.add_argument("--angles-json", default="", help="JSON mapping {filename: {x,y,z,yaw}} for frames-dir")
    ap.add_argument("--angles-from-name", default=True, action="store_true",
                    help="Parse pose from filename like ..._x0.200_y-0.050_z1.500_yaw3.141593.jpg")
    ap.add_argument("--loop-frames", action="store_true", help="Loop over frames-dir without keypress")
    ap.add_argument("--loop-sleep", type=float, default=1.0, help="Sleep between frames in loop-frames mode (s)")

    # Interactive options
    ap.add_argument("--interactive", action="store_true", help="Enable keyboard capture mode (space/c).")
    ap.add_argument("--preview", action="store_true", help="Show OpenCV preview window (required for keypress).")
    ap.add_argument("--manual-pose", default="", help="Pose used in interactive mode as 'x,y,z,yaw' (default 0,0,1.5,0).")
    ap.add_argument("--suffix", action="store_true", help="Append _0001, _0002… to interactive captures to avoid overwrite.")

    # GPIO trigger options
    ap.add_argument("--gpio-pin", type=int, default=18,
                    help="BCM pin number to trigger capture on edge (enables GPIO mode).")
    ap.add_argument("--gpio-edge", choices=["rising","falling","both"], default="rising",
                    help="Which edge to trigger on (default: rising).")
    ap.add_argument("--gpio-pull", choices=["up","down","off"], default="up",
                    help="Internal pull resistor (up/down/off).")
    ap.add_argument("--gpio-debounce-ms", type=int, default=50,
                    help="Debounce window in milliseconds.")

    # VLM options
    ap.add_argument("--vlm", default="", help="VLM describe endpoint, e.g. http://172.16.17.12:8080/describe (empty to disable)")
    ap.add_argument("--vlm-timeout", type=float, default=30.0)
    ap.add_argument("--vlm-retries", type=int, default=2)
    ap.add_argument("--vlm-path-src", default="", help="local root to strip (for path remap), optional")
    ap.add_argument("--vlm-path-dst", default="", help="remote root to prepend (for path remap), optional")

    args = ap.parse_args()

    out_dir = datetime.datetime.now().strftime("%Y_%m_%d___%H_%M_%S")
    args.out = os.path.join(args.out, out_dir)
    os.makedirs(args.out, exist_ok=True)

    cap = open_capture(args.source, args.width, args.height, args.fps, args.fourcc)

    # warmup
    for _ in range(args.warmup):
        cap.read()

    # --- interactive mode: capture-by-key, pose advances from poses.json ---
    # --- mode dispatch ---
    if args.gpio_pin is not None:
        gpio_interactive(args, cap=cap)
    elif args.interactive:
        if args.frames_dir:
            interactive_from_folder(args)  # keyboard + folder
        else:
            manual_interactive(args, cap=cap)  # keyboard + camera
    else:
        if args.frames_dir:
            loop_over_folder(args)  # timed loop over folder
        else:
            timer_capture(args, cap=cap)  # timed poses from camera

    cap.release()
    print("[capture] done.")




if __name__ == "__main__":
    main()
