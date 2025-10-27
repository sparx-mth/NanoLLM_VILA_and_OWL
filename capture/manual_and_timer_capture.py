import os, json, time, argparse, cv2
from txt_and_image_utils import (_update_sidecar_json, _apply_crop_and_flip,
                                 pose_to_name, _save_jpg, _unique_name)
from vlm_helper import prep_vlm
from contextlib import nullcontext
import sys, termios, tty, select

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