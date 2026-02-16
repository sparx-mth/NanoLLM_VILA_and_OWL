import cv2
import os
import time
import json
import threading
import sys
import select
import termios
import tty
import argparse
from pathlib import Path

# --- Default Configuration ---
DEFAULT_ROOT_DIR = "/home/user/jetson-containers/data/R1/"
DEFAULT_ROWS = 2
DEFAULT_COLS = 2
DEFAULT_WIDTH = 0
DEFAULT_HEIGHT = 0
DEFAULT_OVERLAP = 0.1  # 10%

# Global variables
session_folder_path = None
latest_frame = None
frame_lock = threading.Lock()
running = True
args = None  # Will hold CLI args

# --- Headless Keys Class ---
class HeadlessKeys:
    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self
    def __exit__(self, *exc):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)
    def getch(self, timeout_ms=1):
        r, _, _ = select.select([sys.stdin], [], [], timeout_ms/1000.0)
        if r: return sys.stdin.read(1).lower()
        return None

# --- Camera Thread ---
def camera_stream_thread():
    global latest_frame, running
    print("[Camera] Connecting...")
    cap = cv2.VideoCapture(0)
    
    # Try setting 4K
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        running = False
        return

    print(f"[Camera] Stream started. Resolution set to 4K.")
    while running:
        ret, frame = cap.read()
        if ret:
            with frame_lock: latest_frame = frame
        else:
            time.sleep(0.01)
    cap.release()
    print("[Camera] Released.")

# --- Capture Logic ---
def capture_image():
    global session_folder_path, latest_frame, args

    # 1. Grab frame
    frame_full = None
    with frame_lock:
        if latest_frame is not None:
            frame_full = latest_frame.copy()
    
    if frame_full is None:
        print("\r[Warning] No frame yet!", end="")
        return

    # 2. Session Init
    root_path = Path(args.root)
    latest_link = root_path / "latest"
    
    if session_folder_path is None:
        current_time = time.strftime("%Y_%m_%d___%H_%M_%S")
        session_folder_path = root_path / current_time
        session_folder_path.mkdir(parents=True, exist_ok=True)
        try:
            if os.path.lexists(str(latest_link)): os.remove(str(latest_link))
            os.symlink(str(session_folder_path), str(latest_link))
            print(f"\n[*] NEW SESSION: {current_time}")
        except Exception as e: print(f"\nSymlink error: {e}")

    base_ts = time.strftime("%Y_%m_%d___%H_%M_%S")
    
    # --- Tiling Logic ---
    H, W = frame_full.shape[:2]
    rows = args.rows
    cols = args.cols
    overlap = args.overlap
    
    # Base step (no overlap)
    step_y = H // rows
    step_x = W // cols
    
    # Overlap in pixels
    ov_y = int(step_y * overlap)
    ov_x = int(step_x * overlap)
    
    saved_count = 0
    
    for r in range(rows):
        for c in range(cols):
            # Base coords
            y1 = r * step_y
            y2 = (r + 1) * step_y
            x1 = c * step_x
            x2 = (c + 1) * step_x
            
            # Expand with overlap
            if r > 0: y1 = max(0, y1 - ov_y)
            if r < rows - 1: y2 = min(H, y2 + ov_y)
            if r == rows - 1: y2 = H # Fix edge

            if c > 0: x1 = max(0, x1 - ov_x)
            if c < cols - 1: x2 = min(W, x2 + ov_x)
            if c == cols - 1: x2 = W # Fix edge
            
            # Crop
            tile = frame_full[y1:y2, x1:x2]
            if tile.size == 0: continue

            if args.width > 0 and args.height > 0:
                tile_final = cv2.resize(tile, (args.width, args.height), interpolation=cv2.INTER_AREA)
            else:
                tile_final = tile

            
            #  Save
            tile_name = f"frame_{base_ts}_tile_{r}_{c}.jpg"
            tile_path = session_folder_path / tile_name
            json_path = session_folder_path / f"frame_{base_ts}_tile_{r}_{c}.json"
            
            cv2.imwrite(str(tile_path), tile_final)
            
            meta = {
                "pose": {}, 
                "image": tile_name, 
                "tile_index": [r, c],
                "original_crop": [x1, y1, x2, y2],
                "overlap": overlap,
                "resolution": tile_final.shape[:2]
            }
            with open(json_path, 'w') as f:
                json.dump(meta, f, indent=2)
            
            saved_count += 1

        print(f"\r[Captured] {saved_count} tiles -> {session_folder_path.name}    ", end="")

# --- Main ---
def main():
    global running, args
    
    # Argument Parsing
    parser = argparse.ArgumentParser(description="Headless Tiled Capture Tool")
    parser.add_argument("--root", default=DEFAULT_ROOT_DIR, help="Root directory for captures")
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS, help="Number of tile rows")
    parser.add_argument("--cols", type=int, default=DEFAULT_COLS, help="Number of tile cols")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="Target tile width (resized)")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="Target tile height (resized)")
    parser.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP, help="Overlap fraction (0.0 - 0.5)")
    
    args = parser.parse_args()
    
    # Ensure dir exists
    Path(args.root).mkdir(parents=True, exist_ok=True)

    # Start Camera
    t = threading.Thread(target=camera_stream_thread, daemon=True)
    t.start()
    time.sleep(1.0)

    print("\n" + "="*50)
    print(f" CAPTURE TOOL | {args.rows}x{args.cols} Grid | {int(args.overlap*100)}% Overlap")
    print(f" Target Size: {args.width}x{args.height}")
    print(f" Saving to: {args.root}")
    print(" [SPACE]/[ENTER] to capture, [q] to quit")
    print("="*50 + "\n")

    try:
        with HeadlessKeys() as k:
            while running:
                key = k.getch()
                if key:
                    if key in [' ', 'c', '\n', '\r']: capture_image()
                    elif key == 'q':
                        print("\nQuitting...")
                        running = False
                        break
                time.sleep(0.01)
    except KeyboardInterrupt: print("\nInterrupted.")
    finally: running = False; t.join()

if __name__ == "__main__":
    main()
