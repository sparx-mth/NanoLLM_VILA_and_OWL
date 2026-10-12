#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import os
import time
import json
import threading
import argparse
from pathlib import Path

# --- Defaults ---
DEFAULT_ROOT_DIR = "/home/user/jetson-containers/data/R1/"
DEFAULT_INTERVAL = 5          # seconds
DEFAULT_BATCH_SIZE = 20       # images per folder

running = True
latest_frame = None
frame_lock = threading.Lock()

session_folder_path = None
session_count = 0


"""
This script captures images from the camera at regular intervals and organizes them into timestamped folders.
Usage:
python3 auto_capture_interval.py --root /home/user/jetson-containers/data/R1 --interval 5 --batch-size 20 --device 0
"""


# --- Camera Thread ---
def camera_stream_thread(device=0, width=3840, height=2160, mjpg=False):
    global latest_frame, running
    print(f"[Camera] Connecting to {device}...")
    # A device path (e.g. /dev/HD_CAMERA) needs the V4L2 backend; an index works as before.
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2) if isinstance(device, str) else cv2.VideoCapture(device)

    if mjpg:  # many USB cameras only reach HD resolutions in MJPG mode
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    if width and height:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        running = False
        return

    print(f"[Camera] Streaming at {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

    print("[Camera] Streaming...")

    while running:
        ret, frame = cap.read()
        if ret:
            with frame_lock:
                latest_frame = frame
        else:
            time.sleep(0.01)

    cap.release()
    print("[Camera] Released.")


def start_new_session(root_path: Path):
    """Create a new timestamped folder and update 'latest' symlink."""
    global session_folder_path, session_count

    session_count = 0
    current_time = time.strftime("%Y_%m_%d___%H_%M_%S")
    session_folder_path = root_path / current_time
    session_folder_path.mkdir(parents=True, exist_ok=True)

    latest_link = root_path / "latest"
    try:
        if os.path.lexists(str(latest_link)):
            os.remove(str(latest_link))
        os.symlink(str(session_folder_path), str(latest_link))
    except Exception as e:
        print(f"[Symlink] Warning: {e}")

    print(f"\n[*] New session folder: {session_folder_path.name}")


def capture_frame(root_path: Path):
    global latest_frame, session_folder_path, session_count

    # Ensure session exists
    if session_folder_path is None:
        start_new_session(root_path)

    # Copy latest frame
    with frame_lock:
        if latest_frame is None:
            print("[Warning] No frame yet.")
            return False
        frame = latest_frame.copy()

    ts = time.strftime("%Y_%m_%d___%H_%M_%S")
    idx = session_count  # 0..batch_size-1

    img_name = f"frame_{ts}_{idx:03d}.jpg"
    json_name = f"frame_{ts}_{idx:03d}.json"

    img_path = session_folder_path / img_name
    json_path = session_folder_path / json_name

    ok = cv2.imwrite(str(img_path), frame)
    if not ok:
        print("[Error] Failed to write image.")
        return False

    meta = {
        "image": img_name,
        "timestamp": ts,
        "index_in_session": idx,
        "resolution": frame.shape[:2],
        "session": session_folder_path.name
    }
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)

    session_count += 1
    print(f"[Captured] {session_folder_path.name}/{img_name}")

    return True


def main():
    global running, session_folder_path, session_count

    parser = argparse.ArgumentParser(description="Auto interval capture with folder rotation")
    parser.add_argument("--root", default=DEFAULT_ROOT_DIR, help="Root directory for captures")
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL, help="Seconds between captures")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Images per session folder")
    parser.add_argument("--device", default="0",
                        help="Camera device index or path, e.g. 0 or /dev/HD_CAMERA (default: 0)")
    parser.add_argument("--width", type=int, default=640,
                        help="Requested frame width (0 = camera default). 640x360 matches "
                             "docker/da3/camera_info_640x360.yaml's calibration; changing it "
                             "needs a matching recalibration for correct DA3 depth.")
    parser.add_argument("--height", type=int, default=360, help="Requested frame height (0 = camera default)")
    parser.add_argument("--mjpg", action="store_true", default=True,
                        help="Request MJPG from the camera (default: on, matches the drone camera)")
    parser.add_argument("--no-mjpg", dest="mjpg", action="store_false", help="Disable MJPG")

    args = parser.parse_args()
    root_path = Path(args.root)
    root_path.mkdir(parents=True, exist_ok=True)

    # OpenCV only accepts /dev/videoN, not udev symlinks such as /dev/HD_CAMERA
    device = int(args.device) if args.device.isdigit() else os.path.realpath(args.device)
    cam_thread = threading.Thread(
        target=camera_stream_thread,
        kwargs={"device": device, "width": args.width, "height": args.height, "mjpg": args.mjpg},
        daemon=True)
    cam_thread.start()

    time.sleep(2)

    if not running:
        print("Camera failed to open - nothing to capture.")
        return

    print("\n=================================================")
    print(f" Auto Capture every {args.interval}s | batch={args.batch_size} images/folder")
    print(" Creates new timestamped folder every batch and updates 'latest' symlink")
    print(" Stop with Ctrl+C")
    print("=================================================\n")

    try:
        # create first session immediately
        start_new_session(root_path)

        while running:
            ok = capture_frame(root_path)

            # rotate folder after batch-size successful captures
            if ok and session_count >= args.batch_size:
                start_new_session(root_path)

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        running = False
        cam_thread.join()


if __name__ == "__main__":
    main()
