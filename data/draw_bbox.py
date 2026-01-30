#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from PIL import Image

from utils.drawer import DashedImageDraw


def _ensure_bbox_list(value):
    if not value:
        return []
    if isinstance(value[0], (list, tuple)) and len(value[0]) == 4:
        return [list(b) for b in value]
    if len(value) == 4:
        return [list(value)]
    return []


def _draw_bboxes(draw, bboxes, color, width):
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        draw.dashed_rectangle([(x1, y1), (x2, y2)], dash=(5, 5), outline=color, width=width)


def main():
    parser = argparse.ArgumentParser(
        description="Draw bounding boxes from a condition JSON onto an image."
    )
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument("--cond", required=True, help="Path to the condition JSON.")
    parser.add_argument("--output", default=None, help="Output image path.")
    parser.add_argument("--width", type=int, default=4, help="Line width.")
    args = parser.parse_args()

    image_path = Path(args.image)
    cond_path = Path(args.cond)
    if not image_path.exists():
        raise SystemExit(f"Image not found: {image_path}")
    if not cond_path.exists():
        raise SystemExit(f"Condition JSON not found: {cond_path}")

    cond = json.loads(cond_path.read_text(encoding="utf-8"))

    image = Image.open(image_path).convert("RGB")
    drawer = DashedImageDraw(image)

    colors = ["blue", "red", "green", "orange", "purple"]

    bbox_s = cond.get("bbox_s")
    bbox_o = cond.get("bbox_o")
    bbox_a = cond.get("bbox_a")
    bbox = cond.get("bbox")

    if bbox_s and bbox_o:
        _draw_bboxes(drawer, _ensure_bbox_list(bbox_s), colors[0], args.width)
        _draw_bboxes(drawer, _ensure_bbox_list(bbox_o), colors[1], args.width)
        if bbox_a:
            _draw_bboxes(drawer, _ensure_bbox_list(bbox_a), colors[2], args.width)
    elif bbox:
        _draw_bboxes(drawer, _ensure_bbox_list(bbox), colors[0], args.width)
    else:
        raise SystemExit("No bbox, bbox_s/bbox_o found in condition JSON.")

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = image_path.with_name(f"{image_path.stem}_cond1_bbox{image_path.suffix}")
    image.save(output_path)
    print(output_path)


if __name__ == "__main__":
    main()
