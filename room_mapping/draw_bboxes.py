#!/usr/bin/env python3
"""Draw bounding boxes from a JSON file onto an image."""
import json, sys
from PIL import Image, ImageDraw, ImageFont

img_path = sys.argv[1] if len(sys.argv) > 1 else "/home/nadavc/PycharmProjects/TheAgency_workspace/NanoLLM_VILA_and_OWL/room_mapping/data/img_1.png"
json_path = sys.argv[2] if len(sys.argv) > 2 else "/home/nadavc/PycharmProjects/TheAgency_workspace/NanoLLM_VILA_and_OWL/room_mapping/data/3.json"

with open(json_path) as f:
    data = json.load(f)

img = Image.open(img_path).convert("RGB")
draw = ImageDraw.Draw(img)
try: font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
except: font = ImageFont.load_default()

for obj in data["objects"]:
    x1, y1, x2, y2 = obj["bbox"]
    label = f'{obj["label"]} {obj["depth_m"]:.2f}m'
    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
    draw.text((x1, y1 - 30), label, fill="red", font=font)

out = img_path.rsplit(".", 1)[0] + "_bboxes.png"
img.save(out)
print(f"Saved: {out}")