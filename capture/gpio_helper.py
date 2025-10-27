import os, json, time, argparse, cv2
import Jetson.GPIO as GPIO

from txt_and_image_utils import _update_sidecar_json, _apply_crop_and_flip, pose_to_name, _save_jpg, _unique_name
from vlm_helper import prep_vlm

def init_gpio(args):
    # 1) Select numbering scheme
    if args.pin_mode.upper() == "BOARD":
        GPIO.setmode(GPIO.BOARD)
        mode_label = "BOARD"
    else:
        GPIO.setmode(GPIO.BCM)
        mode_label = "BCM"
    GPIO.setwarnings(False)

    # 2) Setup trigger pin
    GPIO.setup(args.gpio_pin, GPIO.IN)
    GPIO.setup(args.gpio_first_frame, GPIO.IN)

def gpio_interactive(args, cap: cv2.VideoCapture):
    """
    Step through poses.json. Each GPIO edge triggers the next capture.
    Works headless (no OpenCV window needed).
    """

    if GPIO is None:
        raise RuntimeError("GPIO library not available on this device")

    init_gpio(args)

    # load ordered poses
    if not os.path.isfile(args.poses):
        raise ValueError(f"poses file not found: {args.poses}")
    with open(args.poses, "r") as f:
        poses = json.load(f)
    if not isinstance(poses, list) or not poses:
        raise ValueError("poses.json must be a non-empty list of {x,y,z,yaw}")
    debounce = max(0, int(args.gpio_debounce_ms))

    print(f"[gpio] listening on {args.pin_mode} {args.gpio_pin} (edge={args.gpio_edge})")
    # Use software debounce by tracking last transition time
    idx = 0
    last_ms = 0
    counter = 1
    prev = None
    is_initialized = False
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

            val = GPIO.input(args.gpio_pin)
            first_frame_val = GPIO.input(args.gpio_first_frame)

            if val != prev:
                # if not is_initialized:
                #     if first_frame_val == GPIO.HIGH:
                #         idx = 0
                #         is_initialized = True
                #     else:
                #         continue

                prev = val
                if val == GPIO.HIGH:  # rising
                    now = int(time.time() * 1000)
                    if now - last_ms < debounce:
                        continue
                    last_ms = now

                    if idx >= len(poses):
                        print("[gpio] all poses captured. exiting.")
                        break

                    # if idx >= len(poses):
                    #     idx = min(idx, len(poses) - 1)
                    # else:
                    #     if idx >= len(poses):
                    #         print("[gpio] all poses captured. exiting.")
                    #         break

                    pose = poses[idx]
                    # If you want sequential fallback when no index pins:
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

            # light idle
            time.sleep(0.2)

    finally:
        try:
            GPIO.cleanup()
        except Exception:
            pass