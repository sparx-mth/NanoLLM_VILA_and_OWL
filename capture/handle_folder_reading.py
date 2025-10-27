import os, time, cv2
from txt_and_image_utils import  (list_frames, get_pose_for_frame,
                                  _save_jpg, _update_sidecar_json,
                                  pose_to_name, _unique_name,
                                  load_angles_map)

from vlm_helper import prep_vlm

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