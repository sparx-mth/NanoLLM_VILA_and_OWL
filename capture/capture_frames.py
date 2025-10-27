#!/usr/bin/env python3
# capture_frames.py
import datetime
import os,  argparse, cv2

try:
    import Jetson.GPIO as GPIO
except Exception:
    try:
        import RPi.GPIO as GPIO
    except Exception:
        GPIO = None

from gpio_helper import gpio_interactive
from manual_and_timer_capture import timer_capture, manual_interactive
from handle_folder_reading import loop_over_folder, interactive_from_folder

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
    ap.add_argument("--gpio-pin", type=int, default=None,
                    help="BCM pin number to trigger capture on edge (enables GPIO mode).")
    ap.add_argument("--gpio-edge", choices=["rising","falling","both"], default="rising",
                    help="Which edge to trigger on (default: rising).")
    ap.add_argument("--gpio-first-frame", type=int, default=None, help="BIT from arduino when loop started")
    ap.add_argument("--gpio-pull", choices=["up","down","off"], default="up",
                    help="Internal pull resistor (up/down/off).")
    ap.add_argument("--gpio-debounce-ms", type=int, default=50,
                    help="Debounce window in milliseconds.")

    ap.add_argument("--pin-mode", choices=["BOARD","BCM"], default="BOARD",
                    help="Pin numbering scheme (default BOARD = physical pin numbers)")

    ap.add_argument("--idx-pull", choices=["up", "down", "none"], default="up", help="Pull for index pins.")
    ap.add_argument("--idx-debounce-ms", type=int, default=10, help="Debounce window for index pins.")
    ap.add_argument("--idx-active-low", action="store_true", help="Invert logic on index pins (LOW=1).")

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
