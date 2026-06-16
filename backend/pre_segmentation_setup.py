"""
Pre-segmentation ROI and ignore-region masks.

Stored under outputs/<job_id>/setup/ separately from post-segmentation corrections.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

ROI_MASK_FILENAME = "roi_mask.png"
IGNORE_MASK_FILENAME = "ignore_mask.png"
METADATA_FILENAME = "setup_metadata.json"


def get_setup_dir(output_dir) -> Path:
    path = Path(output_dir) / "setup"
    path.mkdir(parents=True, exist_ok=True)
    return path


def roi_mask_path(setup_dir: Path) -> Path:
    return Path(setup_dir) / ROI_MASK_FILENAME


def ignore_mask_path(setup_dir: Path) -> Path:
    return Path(setup_dir) / IGNORE_MASK_FILENAME


def setup_metadata_path(setup_dir: Path) -> Path:
    return Path(setup_dir) / METADATA_FILENAME


def _load_mask_png(path: Path, shape: Tuple[int, int]) -> np.ndarray:
    h, w = shape
    if not path.exists():
        return np.zeros((h, w), dtype=bool)
    img = np.array(Image.open(path).convert("L"))
    if img.shape[:2] != (h, w):
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
    return img > 127


def save_mask_png(path: Path, mask: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.asarray(mask).astype(bool).astype(np.uint8) * 255)).save(path)


def load_setup_metadata(output_dir) -> dict:
    setup_dir = get_setup_dir(output_dir)
    path = setup_metadata_path(setup_dir)
    default = {
        "setup_completed": False,
        "roi_defined": False,
        "ignore_defined": False,
        "preview_frame_indices": {"first": 0, "middle": 0, "last": 0},
        "updated_at": None,
    }
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    default.update(data)
    return default


def save_setup_metadata(output_dir, metadata: dict) -> dict:
    setup_dir = get_setup_dir(output_dir)
    path = setup_metadata_path(setup_dir)
    existing = load_setup_metadata(output_dir)
    existing.update(metadata)
    existing["updated_at"] = datetime.now(timezone.utc).isoformat()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)
    return existing


def load_setup_masks(output_dir, shape: Tuple[int, ...]) -> dict:
    """Load ROI and ignore masks. ROI defaults to full image when undefined."""
    h, w = shape[:2]
    setup_dir = get_setup_dir(output_dir)
    roi_path = roi_mask_path(setup_dir)
    ignore_path = ignore_mask_path(setup_dir)

    ignore_mask = _load_mask_png(ignore_path, (h, w))
    has_ignore = bool(np.any(ignore_mask))

    if roi_path.exists():
        roi_mask = _load_mask_png(roi_path, (h, w))
        has_roi = bool(np.any(roi_mask)) and not bool(np.all(roi_mask))
    else:
        roi_mask = np.ones((h, w), dtype=bool)
        has_roi = False

    allowed_mask = roi_mask & ~ignore_mask
    return {
        "roi_mask": roi_mask,
        "ignore_mask": ignore_mask,
        "allowed_mask": allowed_mask,
        "has_roi": has_roi,
        "has_ignore": has_ignore,
    }


def save_setup_mask(output_dir, mask_type: str, mask: np.ndarray, shape: Tuple[int, ...]) -> dict:
    setup_dir = get_setup_dir(output_dir)
    h, w = shape[:2]
    mask_bool = np.asarray(mask).astype(bool)
    if mask_bool.shape[:2] != (h, w):
        mask_bool = cv2.resize(
            mask_bool.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
        ).astype(bool)

    if mask_type == "roi":
        save_mask_png(roi_mask_path(setup_dir), mask_bool)
        meta_key = "roi_defined"
        defined = bool(np.any(mask_bool)) and not bool(np.all(mask_bool))
    elif mask_type == "ignore":
        save_mask_png(ignore_mask_path(setup_dir), mask_bool)
        meta_key = "ignore_defined"
        defined = bool(np.any(mask_bool))
    else:
        raise ValueError(f"Unknown setup mask type: {mask_type}")

    return save_setup_metadata(
        output_dir,
        {
            meta_key: defined,
            "image_width": w,
            "image_height": h,
        },
    )


def apply_pre_segmentation_mask(mask: np.ndarray, allowed_mask: np.ndarray) -> np.ndarray:
    """Zero segmentation outside allowed (ROI minus ignore) regions."""
    result = np.asarray(mask).astype(bool) & np.asarray(allowed_mask).astype(bool)
    return result.astype(np.uint8)


def compute_roi_bbox(roi_mask: np.ndarray, padding: int = 4) -> Optional[Tuple[int, int, int, int]]:
    """
    Return (r0, c0, r1, c1) crop bounds for a non-full ROI, else None.
    """
    roi_bool = np.asarray(roi_mask).astype(bool)
    if not np.any(roi_bool) or np.all(roi_bool):
        return None

    rows = np.any(roi_bool, axis=1)
    cols = np.any(roi_bool, axis=0)
    r0, r1 = int(np.argmax(rows)), int(len(rows) - np.argmax(rows[::-1]))
    c0, c1 = int(np.argmax(cols)), int(len(cols) - np.argmax(cols[::-1]))

    h, w = roi_bool.shape
    r0 = max(0, r0 - padding)
    c0 = max(0, c0 - padding)
    r1 = min(h, r1 + padding)
    c1 = min(w, c1 + padding)
    if r1 <= r0 or c1 <= c0:
        return None
    return r0, c0, r1, c1


def crop_image(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return image[r0:r1, c0:c1]


def embed_crop_mask(crop_mask: np.ndarray, bbox: Tuple[int, int, int, int], full_shape: Tuple[int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    h, w = full_shape
    full = np.zeros((h, w), dtype=np.uint8)
    full[r0:r1, c0:c1] = np.asarray(crop_mask).astype(np.uint8)
    return full


def load_setup_context(output_dir, shape: Tuple[int, ...]) -> dict:
    masks = load_setup_masks(output_dir, shape)
    bbox = compute_roi_bbox(masks["roi_mask"]) if masks["has_roi"] else None
    return {
        **masks,
        "roi_bbox": bbox,
    }


def preview_frame_indices(total_frames: int) -> dict:
    if total_frames <= 0:
        return {"first": 0, "middle": 0, "last": 0}
    return {
        "first": 0,
        "middle": total_frames // 2,
        "last": max(0, total_frames - 1),
    }
