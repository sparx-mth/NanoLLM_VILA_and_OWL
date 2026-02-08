import torch
import time
import PIL.Image, PIL.ImageDraw, PIL.ImageFont
import numpy as np
from torchvision.ops import nms
import os
from nanoowl.owl_predictor import OwlPredictor

# --- CONFIG ---
ENGINE_PATH = "data/owl_image_encoder_b4_fp16.engine"
LABELS = [
    "a couch", "a chair", "a plant", "a pot", "a book",
    "a screen", "a desk", "a window", "an office chair",
    "a coffee table", "a rug", "a cabinet", "a lamp"
]
OVERLAP = 0.1
THRESHOLD = 0.15

predictor = OwlPredictor(
    image_encoder_engine=ENGINE_PATH,
    image_encoder_engine_max_batch_size=4
)


def apply_class_nms(detections, iou_threshold=0.1):
    if not detections:
        return []

    final_keep = []
    # Get unique labels (couch, chair, etc.)
    unique_labels = set([d['label'] for d in detections])

    for label in unique_labels:
        # Filter detections for just this label
        cls_dets = [d for d in detections if d['label'] == label]

        boxes = torch.tensor([d['bbox'] for d in cls_dets]).cuda()
        scores = torch.tensor([d['score'] for d in cls_dets]).cuda()

        # Only suppress boxes of the SAME label
        keep_indices = nms(boxes, scores, iou_threshold)
        final_keep.extend([cls_dets[i] for i in keep_indices])

    return final_keep


def get_tiles(img):
    w, h = img.size
    mid_x, mid_y = w // 2, h // 2
    off_x, off_y = int(mid_x * OVERLAP), int(mid_y * OVERLAP)
    quad_coords = [
        (0, 0, mid_x + off_x, mid_y + off_y),  # TL
        (mid_x - off_x, 0, w, mid_y + off_y),  # TR
        (0, mid_y - off_y, mid_x + off_x, h),  # BL
        (mid_x - off_x, mid_y - off_y, w, h)  # BR
    ]
    return [img.crop(q) for q in quad_coords], quad_coords


def run_comparison(img_path):
    full_img = PIL.Image.open(img_path).convert("RGB")
    W, H = full_img.size
    text_enc = predictor.encode_text(LABELS)

    # --- TEST 1: WITHOUT TILING (Squashed to 768x768) ---
    t0 = time.perf_counter()
    out_single = predictor.predict(image=full_img, text=LABELS, text_encodings=text_enc, threshold=THRESHOLD)
    dt_single = (time.perf_counter() - t0) * 1000

    # Calculate stats for Single
    single_count = len(out_single.labels)
    single_avg_conf = out_single.scores.mean().item() if single_count > 0 else 0

    # --- TEST 2: WITH TILING (Batch of 4) ---
    tiles, tile_offsets = get_tiles(full_img)
    t1 = time.perf_counter()
    batch_output = predictor.predict(image=tiles, text=LABELS, text_encodings=text_enc, threshold=THRESHOLD)
    dt_batch = (time.perf_counter() - t1) * 1000

    # Stitching & Stats for Tiled
    final_detections = []
    for i in range(len(tiles)):
        ox, oy, _, _ = tile_offsets[i]
        mask = (batch_output.input_indices == i)
        tile_labels, tile_scores, tile_boxes = batch_output.labels[mask], batch_output.scores[mask], batch_output.boxes[
            mask]
        for j in range(len(tile_labels)):
            bx1, by1, bx2, by2 = tile_boxes[j].tolist()
            final_detections.append({
                "label": LABELS[int(tile_labels[j])],
                "score": float(tile_scores[j]),
                "bbox": [bx1 + ox, by1 + oy, bx2 + ox, by2 + oy]
            })

    clean_detections = apply_class_nms(final_detections, iou_threshold=0.8)
    tiled_count = len(clean_detections)
    tiled_avg_conf = sum(d['score'] for d in clean_detections) / tiled_count if tiled_count > 0 else 0

    # --- PRINT COMPARISON TABLE ---
    print(f"\n{'=' * 50}")
    print(f"IMAGE: {os.path.basename(img_path)} ({W}x{H})")
    print(f"{'Metric':<20} | {'Without Tiling':<15} | {'With Tiles (Batch)':<15}")
    print(f"{'-' * 50}")
    print(f"{'Objects Detected':<20} | {single_count:<15} | {tiled_count:<15}")
    print(f"{'Avg Confidence':<20} | {single_avg_conf * 100:>13.1f}% | {tiled_avg_conf * 100:>13.1f}%")
    print(f"{'Inference Time':<20} | {dt_single:>13.1f}ms | {dt_batch:>13.1f}ms")
    print(f"{'=' * 50}")

    # --- SAVE LABELED IMAGE ---
    draw = PIL.ImageDraw.Draw(full_img)
    for det in final_detections:
        b = det['bbox']
        draw.rectangle(b, outline="red", width=3)
        draw.text((b[0], b[1] - 10), f"{det['label']} {det['score']:.2f}", fill="red")

    save_path = f"output_{os.path.basename(img_path)}"
    full_img.save(save_path)
    print(f"Saved result to: {save_path}")


# Run on your images
images = ["nanoowl/ExpoTLV/test_1024.jpg", "nanoowl/ExpoTLV/test_1500.jpg", "nanoowl/ExpoTLV/test_hd.jpg"]
for img in images:
    if os.path.exists(img):
        run_comparison(img)