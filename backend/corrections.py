"""
Segmentation correction layer — raster mask edits painted on auto segmentation.
"""

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from annotations import render_difference_map


def get_corrections_dir(output_dir):
    path = Path(output_dir) / "corrections"
    path.mkdir(parents=True, exist_ok=True)
    return path


def correction_add_path(corrections_dir, frame_index):
    return Path(corrections_dir) / f"frame_{frame_index:06d}_add.png"


def correction_remove_path(corrections_dir, frame_index):
    return Path(corrections_dir) / f"frame_{frame_index:06d}_remove.png"


def correction_static_path(corrections_dir, frame_index):
    return Path(corrections_dir) / f"frame_{frame_index:06d}_static.png"


def global_static_correction_path(corrections_dir):
    return Path(corrections_dir) / "global_static.png"


def empty_correction_masks(shape):
    h, w = shape[:2]
    empty = np.zeros((h, w), dtype=bool)
    return {
        "add_mask": empty.copy(),
        "remove_mask": empty.copy(),
        "static_mask": empty.copy(),
    }


def _load_mask_png(path, shape):
    h, w = shape[:2]
    if not path.exists():
        return np.zeros((h, w), dtype=bool)
    img = np.array(Image.open(path).convert("L"))
    if img.shape[:2] != (h, w):
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
    return img > 127


def load_global_static_mask(output_dir, shape):
    corrections_dir = get_corrections_dir(output_dir)
    return _load_mask_png(global_static_correction_path(corrections_dir), shape)


def load_frame_correction_masks(output_dir, frame_index, shape):
    """Load per-frame correction masks only (no global static)."""
    corrections_dir = get_corrections_dir(output_dir)
    return {
        "add_mask": _load_mask_png(correction_add_path(corrections_dir, frame_index), shape),
        "remove_mask": _load_mask_png(correction_remove_path(corrections_dir, frame_index), shape),
        "static_mask": _load_mask_png(correction_static_path(corrections_dir, frame_index), shape),
    }


def load_correction_masks(output_dir, frame_index, shape, include_global_static=True):
    """Load frame corrections; optionally merge explicit global static mask."""
    masks = load_frame_correction_masks(output_dir, frame_index, shape)
    if include_global_static:
        global_static = load_global_static_mask(output_dir, shape)
        if np.any(global_static):
            masks = dict(masks)
            masks["static_mask"] = masks["static_mask"] | global_static
    return masks


def save_correction_masks(output_dir, frame_index, correction_masks):
    corrections_dir = get_corrections_dir(output_dir)
    for key, filename_key in (
        ("add_mask", "add"),
        ("remove_mask", "remove"),
        ("static_mask", "static"),
    ):
        mask = np.asarray(correction_masks[key]).astype(bool)
        path = Path(corrections_dir) / f"frame_{frame_index:06d}_{filename_key}.png"
        Image.fromarray((mask.astype(np.uint8) * 255)).save(path)
    return load_correction_masks(output_dir, frame_index, mask.shape)


def has_corrections(correction_masks):
    for key in ("add_mask", "remove_mask", "static_mask"):
        if np.any(correction_masks.get(key, False)):
            return True
    return False


def compose_mask_with_corrections(raw_mask, correction_masks):
    """auto OR add, minus remove/static correction layers."""
    mask = np.asarray(raw_mask).astype(bool).copy()
    add_mask = np.asarray(correction_masks["add_mask"]).astype(bool)
    remove_mask = np.asarray(correction_masks["remove_mask"]).astype(bool)
    static_mask = np.asarray(correction_masks["static_mask"]).astype(bool)
    mask |= add_mask
    mask &= ~remove_mask
    mask &= ~static_mask
    return mask.astype(np.uint8)


def enforce_hard_corrections(mask, correction_masks):
    """Re-apply hard correction masks after post-processing."""
    return compose_mask_with_corrections(mask, correction_masks)


def compose_full_guided_mask(raw_mask, ann_masks=None, correction_masks=None):
    """Combine auto mask with vector annotations and raster corrections."""
    from annotations import compose_hard_guided_mask

    mask = compose_hard_guided_mask(raw_mask, ann_masks) if ann_masks else np.asarray(raw_mask).astype(np.uint8)
    if correction_masks is not None and has_corrections(correction_masks):
        mask = compose_mask_with_corrections(mask, correction_masks)
    return mask.astype(np.uint8)


def enforce_full_guided_mask(mask, ann_masks=None, correction_masks=None):
    """Re-apply annotations and corrections after cleanup."""
    from annotations import enforce_hard_guided_mask

    result = enforce_hard_guided_mask(mask, ann_masks) if ann_masks else np.asarray(mask).astype(np.uint8)
    if correction_masks is not None and has_corrections(correction_masks):
        result = enforce_hard_corrections(result, correction_masks)
    return result.astype(np.uint8)


def get_hard_foreground_protect_mask(ann_masks=None, correction_masks=None):
    """Pixels that must survive morphology/temporal/bridge cleanup."""
    protect = np.zeros((1, 1), dtype=bool)
    if ann_masks is not None:
        protect = np.asarray(ann_masks.get("object_mask", protect)).astype(bool)
    if correction_masks is not None:
        protect |= np.asarray(correction_masks.get("add_mask", False)).astype(bool)
    return protect


def render_correction_overlay(frame_rgb, auto_mask, correction_masks, alpha_auto=0.35):
    """Original + green auto overlay + cyan/magenta/blue correction layers."""
    overlay = np.asarray(frame_rgb).copy()
    h, w = overlay.shape[:2]
    auto_bool = np.asarray(auto_mask).astype(bool)

    def blend(mask, color, alpha=0.5):
        nonlocal overlay
        if mask is None or not np.any(mask):
            return
        m = np.asarray(mask).astype(bool)
        overlay[m] = (
            (1 - alpha) * overlay[m].astype(np.float32) + alpha * np.array(color, dtype=np.float32)
        ).astype(np.uint8)

    blend(auto_bool, [40, 220, 120], alpha=alpha_auto)
    blend(correction_masks.get("add_mask"), [0, 220, 255], alpha=0.55)
    blend(correction_masks.get("remove_mask"), [255, 0, 200], alpha=0.55)
    blend(correction_masks.get("static_mask"), [60, 120, 255], alpha=0.55)
    return overlay


def render_correction_difference_map(auto_mask, final_mask, correction_masks):
    """Green = added by corrections, red = removed by corrections."""
    auto_bool = np.asarray(auto_mask).astype(bool)
    final_bool = np.asarray(final_mask).astype(bool)
    add_mask = np.asarray(correction_masks["add_mask"]).astype(bool)
    remove_mask = np.asarray(correction_masks["remove_mask"]).astype(bool)
    static_mask = np.asarray(correction_masks["static_mask"]).astype(bool)

    h, w = auto_bool.shape
    diff = np.zeros((h, w, 3), dtype=np.uint8)
    added = (final_bool & ~auto_bool) & (add_mask | (add_mask & final_bool))
    added = final_bool & ~auto_bool & add_mask
    removed = auto_bool & ~final_bool & (remove_mask | static_mask)
    diff[added] = [40, 220, 120]
    diff[removed] = [220, 60, 60]
    return diff


def compute_correction_debug_metadata(auto_mask, final_mask, correction_masks):
    auto_bool = np.asarray(auto_mask).astype(bool)
    final_bool = np.asarray(final_mask).astype(bool)
    add_mask = np.asarray(correction_masks["add_mask"]).astype(bool)
    remove_mask = np.asarray(correction_masks["remove_mask"]).astype(bool)
    static_mask = np.asarray(correction_masks["static_mask"]).astype(bool)

    add_pixels = int(add_mask.sum())
    remove_pixels = int(remove_mask.sum())
    static_pixels = int(static_mask.sum())
    pixels_added = int((add_mask & final_bool & ~auto_bool).sum())
    pixels_removed = int((auto_bool & ~final_bool & (remove_mask | static_mask)).sum())

    if add_pixels > 0:
        overlap = int((add_mask & final_bool).sum())
        corrections_survived = overlap > 0
        correction_overlap_fraction = float(overlap / add_pixels)
    else:
        corrections_survived = True
        correction_overlap_fraction = 1.0

    return {
        "correction_pixels_added": add_pixels,
        "correction_pixels_removed": remove_pixels,
        "correction_static_pixels": static_pixels,
        "pixels_added_by_corrections": pixels_added,
        "pixels_removed_by_corrections": pixels_removed,
        "corrections_survived": corrections_survived,
        "correction_overlap_fraction": round(correction_overlap_fraction, 4),
        "corrections_not_applied": add_pixels > 0 and not corrections_survived,
    }


def get_correction_debug_dir(output_dir):
    path = Path(output_dir) / "correction_debug"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_correction_debug_frame_dir(output_dir, frame_index):
    path = get_correction_debug_dir(output_dir) / f"frame_{frame_index:06d}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_correction_frame_debug(output_dir, frame_index, frame_rgb, auto_mask, final_mask, correction_masks):
    debug_dir = get_correction_debug_frame_dir(output_dir, frame_index)
    auto_bool = np.asarray(auto_mask).astype(bool)
    final_u8 = np.asarray(final_mask).astype(np.uint8)

    for key, fname in (
        ("add_mask", "add_mask.png"),
        ("remove_mask", "remove_mask.png"),
        ("static_mask", "static_background_mask.png"),
    ):
        Image.fromarray((np.asarray(correction_masks[key]).astype(np.uint8) * 255)).save(
            debug_dir / fname
        )

    Image.fromarray((auto_bool.astype(np.uint8) * 255)).save(debug_dir / "auto_mask.png")
    Image.fromarray((final_u8 * 255).astype(np.uint8)).save(debug_dir / "final_guided_mask.png")

    overlay = render_correction_overlay(frame_rgb, auto_bool, correction_masks)
    Image.fromarray(overlay).save(debug_dir / "correction_overlay.png")

    diff_img = render_correction_difference_map(auto_bool, final_u8, correction_masks)
    Image.fromarray(diff_img).save(debug_dir / "difference_map.png")

    meta = compute_correction_debug_metadata(auto_mask, final_u8, correction_masks)
    meta["frame_index"] = frame_index
    with open(debug_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    return meta


def load_correction_debug_metadata(output_dir, frame_index):
    path = get_correction_debug_frame_dir(output_dir, frame_index) / "metadata.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def reset_frame_corrections(output_dir, frame_index, shape):
    empty = empty_correction_masks(shape)
    save_correction_masks(output_dir, frame_index, empty)
    return empty


def save_global_static_mask(output_dir, static_mask, shape):
    corrections_dir = get_corrections_dir(output_dir)
    mask = np.asarray(static_mask).astype(bool)
    if mask.shape[:2] != shape[:2]:
        mask = cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    Image.fromarray((mask.astype(np.uint8) * 255)).save(global_static_correction_path(corrections_dir))
    return mask


def reset_global_static_mask(output_dir, shape):
    return save_global_static_mask(output_dir, np.zeros(shape[:2], dtype=bool), shape)


def reset_all_corrections(output_dir, total_frames, shape):
    for i in range(total_frames):
        reset_frame_corrections(output_dir, i, shape)
    reset_global_static_mask(output_dir, shape)
