import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


def _extract_so(data: Dict) -> Tuple[Optional[str], Optional[str]]:
    s = data.get("s_word") or data.get("s") or data.get("subject")
    o = data.get("o_word") or data.get("o") or data.get("object")
    return s, o


def _collect_condition_files(base_dir: str) -> List[str]:
    exact = []
    for root, _, files in os.walk(base_dir):
        for name in files:
            if name == "condition.json":
                exact.append(os.path.join(root, name))
    if exact:
        return sorted(exact)

    if os.path.isdir(base_dir):
        return sorted(
            os.path.join(base_dir, f)
            for f in os.listdir(base_dir)
            if f.endswith(".json")
        )
    return []


def _load_image(image_path: str) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    return np.array(image)


def _save_mask(mask: np.ndarray, out_path: str) -> None:
    mask_img = (mask.astype(np.uint8) * 255)
    Image.fromarray(mask_img).save(out_path)


def _resolve_image_path(data: Dict, image_root: str) -> Optional[str]:
    if "img_path" in data:
        return data["img_path"]
    fname = data.get("fname")
    if not fname:
        return None
    return os.path.join(image_root, fname)


def _select_best_mask(masks: np.ndarray, scores: np.ndarray) -> np.ndarray:
    if masks.ndim == 2:
        return masks
    if scores is None or scores.size == 0:
        return masks[0]
    best_idx = int(np.argmax(scores))
    return masks[best_idx]


def _load_sam2_predictor(model_root: str, model_id: Optional[str], model_cfg: Optional[str], checkpoint: Optional[str]):
    sam2_root = os.path.join(model_root, "sam2")
    if sam2_root not in sys.path:
        sys.path.insert(0, sam2_root)

    from sam2.sam2_image_predictor import SAM2ImagePredictor
    if model_id:
        return SAM2ImagePredictor.from_pretrained(model_id)

    if not model_cfg or not checkpoint:
        raise ValueError("Provide either --model_id or both --model_cfg and --checkpoint.")

    from sam2.build_sam import build_sam2
    sam2_model = build_sam2(model_cfg, checkpoint)
    return SAM2ImagePredictor(sam2_model)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--condition_dir",
        default="./condition",
        help="Directory to search for condition JSON files",
    )
    parser.add_argument(
        "--condition_file",
        default=None,
        help="Specific condition JSON file to read",
    )
    parser.add_argument(
        "--image_root",
        default="./data/",
        help="Base directory for condition fname images",
    )
    parser.add_argument(
        "--output_dir",
        default="./outputs/sam2",
        help="Directory to save SAM2 masks",
    )
    parser.add_argument(
        "--model_root",
        default="./model",
        help="Root directory containing the sam2 repo",
    )
    parser.add_argument(
        "--model_id",
        default=None,
        help="SAM2 HF model id (e.g., facebook/sam2-hiera-large)",
    )
    parser.add_argument(
        "--model_cfg",
        default=None,
        help="SAM2 model config path for build_sam2 (e.g., sam2_hiera_l.yaml)",
    )
    parser.add_argument(
        "--checkpoint",
        default="model/sam2/checkpoints/sam2.1_hiera_large.pth",
        help="SAM2 checkpoint path for build_sam2",
    )
    args = parser.parse_args()

    files: List[str] = []
    if args.condition_file:
        files = [args.condition_file]
    else:
        files = _collect_condition_files(args.condition_dir)

    if not files:
        print("No condition JSON files found.")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    predictor = _load_sam2_predictor(args.model_root, args.model_id, args.model_cfg, args.checkpoint)

    for path in files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            print(f"[ERROR] {path}: {exc}")
            continue

        s, o = _extract_so(data)
        image_path = _resolve_image_path(data, args.image_root)
        if not image_path or not os.path.exists(image_path):
            print(f"[WARN] image not found for {path}: {image_path}")
            continue

        image = _load_image(image_path)
        predictor.set_image(image)

        bbox_s = data.get("bbox_s_pre")
        bbox_o = data.get("bbox_o_pre")

        base = os.path.splitext(os.path.basename(path))[0]

        if bbox_s:
            masks, scores, _ = predictor.predict(box=np.array(bbox_s, dtype=np.float32), multimask_output=True)
            best = _select_best_mask(masks, scores)
            out_name = f"{base}_s_{s or 's'}_mask.png"
            _save_mask(best, os.path.join(args.output_dir, out_name))
        else:
            print(f"[WARN] bbox_s missing in {path}")

        if bbox_o:
            masks, scores, _ = predictor.predict(box=np.array(bbox_o, dtype=np.float32), multimask_output=True)
            best = _select_best_mask(masks, scores)
            out_name = f"{base}_o_{o or 'o'}_mask.png"
            _save_mask(best, os.path.join(args.output_dir, out_name))
        else:
            print(f"[WARN] bbox_o missing in {path}")


if __name__ == "__main__":
    main()
