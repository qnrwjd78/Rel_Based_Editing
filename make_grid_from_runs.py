#!/usr/bin/env python3
"""
Build per-key image grids across multiple experiment runs.

Typical usage:
  python make_grid_from_runs.py --input /tmp/test_run_methods_full_smartcompose --output /tmp/grids

By default it looks for files like:
  cond_*_attn_map_*_*.png
in each run directory (e.g. 1/, 2/, 3/, ...), and for each filename ("key")
creates a grid image that contains the corresponding images from all runs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont


def _try_int(s: str) -> Optional[int]:
    try:
        return int(s)
    except Exception:
        return None


def _list_run_dirs(base: Path) -> List[Path]:
    dirs = [p for p in base.iterdir() if p.is_dir()]
    numeric = [p for p in dirs if _try_int(p.name) is not None]
    # If any numeric run directories exist, assume those are the experiment runs.
    if numeric:
        return sorted(numeric, key=lambda p: _try_int(p.name) or 0)
    return []


def _read_run_label(run_dir: Path) -> str:
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        return run_dir.name
    try:
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
        erase = data.get("erase_method")
        compose = data.get("compose_method")
        parts = []
        if erase:
            parts.append(f"{erase}, ")
        if compose:
            parts.append(f"{compose}")
        if parts:
            return f"{run_dir.name} ({', '.join(parts)})"
    except Exception:
        pass
    return run_dir.name


def _load_font(size: int) -> ImageFont.ImageFont:
    # Degrade gracefully; PIL always has a default bitmap font.
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def _draw_centered_text(img: Image.Image, text: str, *, y: int, font: ImageFont.ImageFont) -> None:
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    x = max(0, (img.width - text_w) // 2)
    draw.text((x, y), text, fill=(0, 0, 0), font=font)


def _make_placeholder(size: Tuple[int, int]) -> Image.Image:
    img = Image.new("RGB", size, (245, 245, 245))
    draw = ImageDraw.Draw(img)
    draw.rectangle([(0, 0), (size[0] - 1, size[1] - 1)], outline=(210, 210, 210), width=2)
    return img


def _fit_to_tile(img: Image.Image, tile_size: Tuple[int, int]) -> Image.Image:
    # Preserve aspect ratio; pad with white to tile.
    src = img.convert("RGB")
    src.thumbnail(tile_size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", tile_size, (255, 255, 255))
    x = (tile_size[0] - src.width) // 2
    y = (tile_size[1] - src.height) // 2
    canvas.paste(src, (x, y))
    return canvas


def _grid_dims(n: int, cols: Optional[int]) -> Tuple[int, int]:
    if n <= 0:
        return 0, 0
    if cols and cols > 0:
        c = min(cols, n)
        r = int(math.ceil(n / c))
        return r, c
    c = int(math.ceil(math.sqrt(n)))
    r = int(math.ceil(n / c))
    return r, c


def build_grid(
    *,
    key: str,
    images_by_run: List[Tuple[str, Optional[Path]]],  # (label, image_path)
    out_path: Path,
    cols: Optional[int],
    tile_size: Tuple[int, int],
    title: bool,
    labels: bool,
) -> None:
    n = len(images_by_run)
    rows, cols_eff = _grid_dims(n, cols)
    if rows == 0 or cols_eff == 0:
        return

    title_h = 56 if title else 0
    label_h = 36 if labels else 0
    pad = 6
    cell_w, cell_h = tile_size[0], tile_size[1] + label_h

    grid_w = cols_eff * cell_w + (cols_eff + 1) * pad
    grid_h = title_h + rows * cell_h + (rows + 1) * pad
    canvas = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))

    font_title = _load_font(20)
    font_label = _load_font(14)

    if title:
        # simple title bar
        draw = ImageDraw.Draw(canvas)
        draw.rectangle([(0, 0), (grid_w, title_h)], fill=(255, 255, 255))
        _draw_centered_text(canvas, key, y=12, font=font_title)

    for idx, (run_label, img_path) in enumerate(images_by_run):
        r = idx // cols_eff
        c = idx % cols_eff
        x0 = pad + c * (cell_w + pad)
        y0 = title_h + pad + r * (cell_h + pad)

        if img_path and img_path.exists():
            img = Image.open(img_path)
            tile = _fit_to_tile(img, tile_size)
        else:
            tile = _make_placeholder(tile_size)

        canvas.paste(tile, (x0, y0))

        if labels:
            draw = ImageDraw.Draw(canvas)
            y_text = y0 + tile_size[1] + 6
            # truncate long labels
            text = run_label
            if len(text) > 40:
                text = text[:37] + "..."
            draw.text((x0 + 4, y_text), text, fill=(0, 0, 0), font=font_label)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Base output directory containing run folders (e.g. 1/,2/,3/...).")
    ap.add_argument("--output", default=None, help="Where to write grids. Default: <input>/grids")
    ap.add_argument("--pattern", default="cond_*_attn_map_*_*.png", help="Glob for images inside each run dir.")
    ap.add_argument("--cols", type=int, default=6, help="Number of columns in the grid.")
    ap.add_argument("--tile", default="256x256", help="Tile size WxH, e.g. 256x256")
    ap.add_argument("--no_title", action="store_true", help="Do not draw the grid title.")
    ap.add_argument("--no_labels", action="store_true", help="Do not draw per-run labels.")
    ap.add_argument("--max_keys", type=int, default=0, help="Limit number of keys (0 = no limit).")
    args = ap.parse_args()

    base = Path(args.input)
    if not base.exists():
        raise SystemExit(f"Input not found: {base}")

    out_dir = Path(args.output) if args.output else (base / "grids")
    try:
        w_s, h_s = args.tile.lower().split("x", 1)
        tile_size = (int(w_s), int(h_s))
    except Exception:
        raise SystemExit("--tile must be like 256x256")

    run_dirs = _list_run_dirs(base)
    # Allow a single-run directory (no subfolders) for convenience.
    if not run_dirs:
        run_dirs = [base]

    run_labels = {d: _read_run_label(d) for d in run_dirs}

    # Collect keys (filenames) across runs
    keys: Dict[str, List[Tuple[str, Optional[Path]]]] = {}
    for d in run_dirs:
        for p in sorted(d.glob(args.pattern)):
            if not p.is_file():
                continue
            key = p.name
            keys.setdefault(key, [])

    # Initialize all keys with all runs in a stable order.
    all_keys = sorted(keys.keys())
    if args.max_keys and args.max_keys > 0:
        all_keys = all_keys[: args.max_keys]

    for key in all_keys:
        per_run: List[Tuple[str, Optional[Path]]] = []
        for d in run_dirs:
            label = run_labels[d]
            img_path = d / key
            per_run.append((label, img_path if img_path.exists() else None))

        out_path = out_dir / key.replace(".png", "_grid.png")
        build_grid(
            key=key.replace(".png", ""),
            images_by_run=per_run,
            out_path=out_path,
            cols=args.cols,
            tile_size=tile_size,
            title=not args.no_title,
            labels=not args.no_labels,
        )

    print(f"Wrote {len(all_keys)} grids to: {out_dir}")


if __name__ == "__main__":
    main()
