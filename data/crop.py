#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from PIL import Image


def _is_bbox(value):
    return (
        isinstance(value, list)
        and len(value) == 4
        and all(isinstance(v, (int, float)) for v in value)
    )


def _adjust_bbox(bbox, x_offset, scale=None):
    x1, y1, x2, y2 = bbox
    x1 -= x_offset
    x2 -= x_offset
    if scale is not None:
        x1 = int(round(x1 * scale))
        x2 = int(round(x2 * scale))
        y1 = int(round(y1 * scale))
        y2 = int(round(y2 * scale))
    return [x1, y1, x2, y2]


def _update_condition(cond, x_offset, scale=None):
    updated = dict(cond)
    for key, value in cond.items():
        if _is_bbox(value):
            updated[key] = _adjust_bbox(value, x_offset, scale)
        elif (
            isinstance(value, list)
            and value
            and all(_is_bbox(item) for item in value)
        ):
            updated[key] = [_adjust_bbox(item, x_offset, scale) for item in value]
    return updated


def _process_cond(path, x_offset, scale=None):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    updated = _update_condition(data, x_offset, scale)
    Path(path).write_text(json.dumps(updated, indent=4), encoding="utf-8")
    return updated


def main():
    parser = argparse.ArgumentParser(
        description="Crop an image horizontally and update condition JSON bboxes."
    )
    parser.add_argument("--image", default="data/test.jpg", help="Image path to crop.")
    parser.add_argument("--left", type=int, default=112, help="Pixels to crop from left.")
    parser.add_argument("--right", type=int, default=202, help="Pixels to crop from right.")
    parser.add_argument(
        "--output-image",
        default=None,
        help="Output image path (default: overwrite input).",
    )
    parser.add_argument(
        "--cond",
        action="append",
        default=["condition/test_condition.json"],
        help="Condition JSON(s) to update with crop only (can repeat).",
    )
    parser.add_argument(
        "--cond-512",
        action="append",
        default=["condition/test_condition_512.json"],
        help="Condition JSON(s) to update with crop + scale to target size.",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=512,
        help="Target size for scaled condition files.",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise SystemExit(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    left = args.left
    right = args.right
    if left + right >= width:
        raise SystemExit("Crop exceeds image width.")

    cropped = image.crop((left, 0, width - right, height))
    output_path = Path(args.output_image) if args.output_image else image_path
    cropped.save(output_path)

    cropped_width, cropped_height = cropped.size
    scale = args.target / cropped_height

    for path in args.cond:
        if Path(path).exists():
            _process_cond(path, left, scale=None)
        else:
            print(f"Skip missing condition file: {path}")

    for path in args.cond_512:
        if Path(path).exists():
            _process_cond(path, left, scale=scale)
        else:
            print(f"Skip missing condition file: {path}")

    print(f"Saved cropped image to: {output_path}")
    print(f"Cropped size: {cropped_width}x{cropped_height}")
    print(f"Scale for --cond-512: {scale:.6f}")


if __name__ == "__main__":
    main()
