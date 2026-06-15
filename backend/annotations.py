import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ANNOTATION_TOOL_VERSION = "1.1"
DEFAULT_POINT_RADIUS = 12


def get_job_previews_dir(output_dir):
    path = Path(output_dir) / "previews"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_difference_maps_dir(output_dir):
    path = Path(output_dir) / "difference_maps"
    path.mkdir(parents=True, exist_ok=True)
    return path


def job_config_path(annotations_dir):
    return Path(annotations_dir) / "job_config.json"


def load_job_config(annotations_dir, default_mode="keyframes"):
    path = job_config_path(annotations_dir)
    default = {
        "annotation_mode": default_mode,
        "tool_version": ANNOTATION_TOOL_VERSION,
        "propagation_window": 10,
        "use_temporal_propagation": True,
        "repair_disconnected_tubes": False,
        "max_bridge_gap_px": 15,
        "max_bridge_angle_degrees": 45,
        "use_temporal_continuity": True,
        "temporal_memory_frames": 3,
        "temporal_persistence_weight": 0.5,
        "max_allowed_area_drop_fraction": 0.15,
        "recover_missing_middle_frames": True,
        "min_temporal_component_persistence": 2,
        "allow_tip_growth": True,
        "min_bridge_intensity_percentile": 60,
        "max_endpoints_per_frame": 40,
        "max_bridge_pair_checks": 80,
        "max_bridges_per_frame": 5,
        "frame_time_budget_sec": 25.0,
        "branch_node_merge_radius_px": 6,
        "branch_node_temporal_smoothing": True,
        "branch_node_max_tracking_distance_px": 10,
    }
    return load_json_annotation(path, lambda: default)


def save_job_config(annotations_dir, config):
    save_json_annotation(job_config_path(annotations_dir), config)


def global_ignore_mask_path(annotations_dir):
    return Path(annotations_dir) / "global_ignore_mask.png"


def get_annotations_dir(output_dir):
    ann_dir = Path(output_dir) / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    (ann_dir / "previews").mkdir(exist_ok=True)
    return ann_dir


def frame_annotation_path(annotations_dir, frame_index):
    return Path(annotations_dir) / f"frame_{frame_index:06d}.json"


def static_background_path(annotations_dir):
    return Path(annotations_dir) / "static_background.json"


def empty_frame_annotation(frame_index, image_width, image_height):
    return {
        "frame_index": frame_index,
        "image_width": image_width,
        "image_height": image_height,
        "is_keyframe": False,
        "accepted_preview": False,
        "preview_status": "none",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool_version": ANNOTATION_TOOL_VERSION,
        "object_points": [],
        "background_points": [],
        "static_background_points": [],
        "object_brush_strokes": [],
        "background_brush_strokes": [],
        "static_background_brush_strokes": [],
        "bounding_boxes": [],
    }


def empty_static_background_annotation(image_width=0, image_height=0):
    return {
        "image_width": image_width,
        "image_height": image_height,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool_version": ANNOTATION_TOOL_VERSION,
        "static_background_points": [],
        "static_background_brush_strokes": [],
        "bounding_boxes": [],
    }


def load_json_annotation(path, default_factory):
    path = Path(path)
    if not path.exists():
        return default_factory()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def save_json_annotation(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data["timestamp"] = datetime.now(timezone.utc).isoformat()
    data["tool_version"] = ANNOTATION_TOOL_VERSION
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_frame_annotation(annotations_dir, frame_index, image_width=0, image_height=0):
    path = frame_annotation_path(annotations_dir, frame_index)
    return load_json_annotation(
        path,
        lambda: empty_frame_annotation(frame_index, image_width, image_height),
    )


def save_frame_annotation(annotations_dir, data):
    frame_index = int(data["frame_index"])
    save_json_annotation(frame_annotation_path(annotations_dir, frame_index), data)


def load_static_background(annotations_dir, image_width=0, image_height=0):
    return load_json_annotation(
        static_background_path(annotations_dir),
        lambda: empty_static_background_annotation(image_width, image_height),
    )


def save_static_background(annotations_dir, data):
    save_json_annotation(static_background_path(annotations_dir), data)


def rasterize_points(points, shape, radius=DEFAULT_POINT_RADIUS):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    h, w = mask.shape
    for pt in points or []:
        if len(pt) < 2:
            continue
        x, y = int(round(pt[0])), int(round(pt[1]))
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(mask, (x, y), radius, 1, -1)
    return mask.astype(bool)


def rasterize_brush_strokes(strokes, shape):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    h, w = mask.shape
    for stroke in strokes or []:
        pts = stroke.get("points", [])
        size = max(1, int(stroke.get("size", 12)))
        for i, pt in enumerate(pts):
            if len(pt) < 2:
                continue
            x, y = int(round(pt[0])), int(round(pt[1]))
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(mask, (x, y), max(1, size // 2), 1, -1)
            if i > 0 and len(pts[i - 1]) >= 2:
                x0, y0 = int(round(pts[i - 1][0])), int(round(pts[i - 1][1]))
                cv2.line(mask, (x0, y0), (x, y), 1, size)
    return mask.astype(bool)


def rasterize_bounding_boxes(boxes, shape):
    if not boxes:
        return None
    mask = np.zeros(shape[:2], dtype=np.uint8)
    h, w = mask.shape
    for box in boxes:
        x1 = int(round(min(box["x1"], box["x2"])))
        y1 = int(round(min(box["y1"], box["y2"])))
        x2 = int(round(max(box["x1"], box["x2"])))
        y2 = int(round(max(box["y1"], box["y2"])))
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        if x2 > x1 and y2 > y1:
            cv2.rectangle(mask, (x1, y1), (x2, y2), 1, -1)
    return mask.astype(bool)


def build_static_ignore_mask(frame_ann, static_ann, shape, point_radius=DEFAULT_POINT_RADIUS):
    ignore = np.zeros(shape[:2], dtype=bool)
    for source in (static_ann, frame_ann):
        if not source:
            continue
        ignore |= rasterize_points(source.get("static_background_points", []), shape, point_radius)
        ignore |= rasterize_brush_strokes(source.get("static_background_brush_strokes", []), shape)
    return ignore


def build_object_annotation_mask(frame_ann, shape, point_radius=DEFAULT_POINT_RADIUS):
    """Hard foreground from object points and brush strokes."""
    obj = rasterize_points(frame_ann.get("object_points", []), shape, point_radius)
    obj |= rasterize_brush_strokes(frame_ann.get("object_brush_strokes", []), shape)
    return obj.astype(bool)


def build_background_annotation_mask(frame_ann, shape, point_radius=DEFAULT_POINT_RADIUS):
    """Hard exclusion from per-frame background annotations."""
    bg = rasterize_points(frame_ann.get("background_points", []), shape, point_radius)
    bg |= rasterize_brush_strokes(frame_ann.get("background_brush_strokes", []), shape)
    return bg.astype(bool)


def build_static_background_annotation_mask(frame_ann, static_ann, shape, point_radius=DEFAULT_POINT_RADIUS):
    return build_static_ignore_mask(frame_ann, static_ann, shape, point_radius)


def get_roi_constrain_mask(frame_ann, shape):
    return rasterize_bounding_boxes(frame_ann.get("bounding_boxes", []), shape)


def compute_guided_annotation_masks(frame_ann, static_ann, shape, point_radius=DEFAULT_POINT_RADIUS):
    return {
        "object_mask": build_object_annotation_mask(frame_ann, shape, point_radius),
        "background_mask": build_background_annotation_mask(frame_ann, shape, point_radius),
        "static_mask": build_static_background_annotation_mask(frame_ann, static_ann, shape, point_radius),
        "roi_constrain": get_roi_constrain_mask(frame_ann, shape),
    }


def compose_hard_guided_mask(raw_mask, ann_masks):
    """auto OR object, minus background/static, optional ROI constrain."""
    mask = np.asarray(raw_mask).astype(bool).copy()
    object_mask = np.asarray(ann_masks["object_mask"]).astype(bool)
    background_mask = np.asarray(ann_masks["background_mask"]).astype(bool)
    static_mask = np.asarray(ann_masks["static_mask"]).astype(bool)
    roi = ann_masks.get("roi_constrain")

    mask |= object_mask
    mask &= ~background_mask
    mask &= ~static_mask
    if roi is not None and np.any(roi):
        mask &= np.asarray(roi).astype(bool)
    return mask.astype(np.uint8)


def enforce_hard_guided_mask(mask, ann_masks):
    """Re-apply hard object foreground and exclusion masks after post-processing."""
    result = np.asarray(mask).astype(bool).copy()
    object_mask = np.asarray(ann_masks["object_mask"]).astype(bool)
    background_mask = np.asarray(ann_masks["background_mask"]).astype(bool)
    static_mask = np.asarray(ann_masks["static_mask"]).astype(bool)

    result |= object_mask
    result &= ~background_mask
    result &= ~static_mask
    roi = ann_masks.get("roi_constrain")
    if roi is not None and np.any(roi):
        result &= np.asarray(roi).astype(bool)
    return result.astype(np.uint8)


def compute_guided_debug_metadata(auto_mask, final_mask, ann_masks):
    auto_bool = np.asarray(auto_mask).astype(bool)
    final_bool = np.asarray(final_mask).astype(bool)
    object_mask = np.asarray(ann_masks["object_mask"]).astype(bool)
    background_mask = np.asarray(ann_masks["background_mask"]).astype(bool)
    static_mask = np.asarray(ann_masks["static_mask"]).astype(bool)

    object_pixels = int(object_mask.sum())
    background_pixels = int(background_mask.sum())
    static_pixels = int(static_mask.sum())
    pixels_added_by_object = int((object_mask & final_bool & ~auto_bool).sum())
    pixels_removed_by_exclusion = int((auto_bool & ~final_bool & (background_mask | static_mask)).sum())

    if object_pixels > 0:
        overlap = int((object_mask & final_bool).sum())
        annotations_survived = overlap > 0
        annotation_overlap_fraction = float(overlap / object_pixels)
    else:
        annotations_survived = True
        annotation_overlap_fraction = 1.0

    return {
        "object_annotation_pixels": object_pixels,
        "background_annotation_pixels": background_pixels,
        "static_background_pixels": static_pixels,
        "pixels_added_by_object": pixels_added_by_object,
        "pixels_removed_by_exclusion": pixels_removed_by_exclusion,
        "annotations_survived": annotations_survived,
        "annotation_overlap_fraction": round(annotation_overlap_fraction, 4),
        "object_annotations_not_applied": object_pixels > 0 and not annotations_survived,
    }


def get_guided_debug_dir(output_dir):
    path = Path(output_dir) / "guided_debug"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_guided_debug_frame_dir(output_dir, frame_index):
    path = get_guided_debug_dir(output_dir) / f"frame_{frame_index:06d}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_guided_frame_debug(
    output_dir,
    frame_index,
    frame_rgb,
    auto_mask,
    final_mask,
    ann_masks,
    extra_metadata=None,
):
    from sam2_pipeline import make_overlay, skeleton_and_branchpoints

    debug_dir = get_guided_debug_frame_dir(output_dir, frame_index)
    auto_bool = np.asarray(auto_mask).astype(bool)
    final_u8 = np.asarray(final_mask).astype(np.uint8)

    Image.fromarray((np.asarray(ann_masks["object_mask"]).astype(np.uint8) * 255)).save(
        debug_dir / "object_annotation_mask.png"
    )
    Image.fromarray((np.asarray(ann_masks["background_mask"]).astype(np.uint8) * 255)).save(
        debug_dir / "background_annotation_mask.png"
    )
    Image.fromarray((np.asarray(ann_masks["static_mask"]).astype(np.uint8) * 255)).save(
        debug_dir / "static_background_mask.png"
    )
    exclusion = (
        np.asarray(ann_masks["background_mask"]).astype(bool)
        | np.asarray(ann_masks["static_mask"]).astype(bool)
    )
    Image.fromarray((exclusion.astype(np.uint8) * 255)).save(debug_dir / "exclusion_mask.png")
    Image.fromarray((auto_bool.astype(np.uint8) * 255)).save(debug_dir / "auto_mask.png")
    Image.fromarray((final_u8 * 255).astype(np.uint8)).save(debug_dir / "final_guided_mask.png")

    skel, branches, _, _, _ = skeleton_and_branchpoints(final_u8)
    overlay = make_overlay(frame_rgb, final_u8, skel, branches)
    Image.fromarray(overlay).save(debug_dir / "final_guided_overlay.png")

    diff_img = render_difference_map(frame_rgb, auto_bool, final_u8)
    Image.fromarray(diff_img).save(debug_dir / "difference_map.png")

    meta = compute_guided_debug_metadata(auto_mask, final_u8, ann_masks)
    if extra_metadata:
        meta.update(extra_metadata)
    meta["frame_index"] = frame_index
    with open(debug_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    return meta


def reapply_hard_guided_annotations_to_masks(
    masks,
    frames,
    annotations_dir,
    static_ann,
    output_dir=None,
    auto_masks=None,
    save_debug=False,
    extra_metadata_by_frame=None,
):
    """Re-apply hard annotations and corrections to every frame mask after temporal/post steps."""
    from corrections import (
        enforce_full_guided_mask,
        load_correction_masks,
        save_correction_frame_debug,
    )

    annotations_dir = Path(annotations_dir)
    image_h, image_w = frames[0].shape[:2]
    corrected = []
    metadata_list = []

    for i, (frame, mask) in enumerate(zip(frames, masks)):
        ann = load_frame_annotation(annotations_dir, i, image_w, image_h)
        ann_masks = compute_guided_annotation_masks(ann, static_ann, frame.shape)
        correction_masks = (
            load_correction_masks(output_dir, i, frame.shape) if output_dir is not None else None
        )
        final_mask = enforce_full_guided_mask(mask, ann_masks, correction_masks)
        corrected.append(final_mask)

        meta = None
        if save_debug and output_dir is not None:
            auto = auto_masks[i] if auto_masks is not None else mask
            extra = (extra_metadata_by_frame or {}).get(i, {})
            meta = save_guided_frame_debug(
                output_dir, i, frame, auto, final_mask, ann_masks, extra_metadata=extra
            )
            if correction_masks is not None:
                save_correction_frame_debug(
                    output_dir, i, frame, auto, final_mask, correction_masks
                )
        metadata_list.append(meta)

    return corrected, metadata_list


def apply_guided_postprocess(raw_mask, frame_ann, static_ann, shape, point_radius=DEFAULT_POINT_RADIUS, correction_masks=None):
    ann_masks = compute_guided_annotation_masks(frame_ann, static_ann, shape, point_radius)
    from corrections import compose_full_guided_mask

    return compose_full_guided_mask(raw_mask, ann_masks, correction_masks)


def render_annotation_overlay(image_rgb, frame_ann, static_ann=None):
    overlay = np.asarray(image_rgb).copy()
    h, w = overlay.shape[:2]

    def blend(mask, color, alpha=0.45):
        nonlocal overlay
        if mask is None or not np.any(mask):
            return
        m = mask.astype(bool)
        overlay[m] = (
            (1 - alpha) * overlay[m].astype(np.float32) + alpha * np.array(color, dtype=np.float32)
        ).astype(np.uint8)

    static_mask = build_static_ignore_mask(frame_ann, static_ann or {}, (h, w))
    bg_mask = rasterize_points(frame_ann.get("background_points", []), (h, w))
    bg_mask |= rasterize_brush_strokes(frame_ann.get("background_brush_strokes", []), (h, w))
    obj_mask = rasterize_points(frame_ann.get("object_points", []), (h, w))
    obj_mask |= rasterize_brush_strokes(frame_ann.get("object_brush_strokes", []), (h, w))

    blend(static_mask, [220, 60, 60])
    blend(bg_mask, [60, 120, 255])
    blend(obj_mask, [40, 220, 120])

    for box in frame_ann.get("bounding_boxes", []):
        x1 = int(round(min(box["x1"], box["x2"])))
        y1 = int(round(min(box["y1"], box["y2"])))
        x2 = int(round(max(box["x1"], box["x2"])))
        y2 = int(round(max(box["y1"], box["y2"])))
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 220, 60), 2)

    for pts_key, color in (
        ("object_points", (40, 220, 120)),
        ("background_points", (60, 120, 255)),
        ("static_background_points", (220, 60, 60)),
    ):
        for pt in frame_ann.get(pts_key, []):
            if len(pt) >= 2:
                cv2.circle(overlay, (int(pt[0]), int(pt[1])), 5, color, -1)

    if static_ann:
        for pt in static_ann.get("static_background_points", []):
            if len(pt) >= 2:
                cv2.circle(overlay, (int(pt[0]), int(pt[1])), 5, (220, 60, 60), -1)

    return overlay


def save_annotation_preview(annotations_dir, frame_index, image_rgb, frame_ann, static_ann=None):
    preview = render_annotation_overlay(image_rgb, frame_ann, static_ann)
    preview_path = Path(annotations_dir) / "previews" / f"frame_{frame_index:06d}.png"
    Image.fromarray(preview).save(preview_path)
    return preview_path


def apply_static_background_to_all_frames(annotations_dir, total_frames, image_width, image_height):
    static_ann = load_static_background(annotations_dir, image_width, image_height)
    for i in range(total_frames):
        frame_ann = load_frame_annotation(annotations_dir, i, image_width, image_height)
        frame_ann["static_background_points"] = list(static_ann.get("static_background_points", []))
        frame_ann["static_background_brush_strokes"] = list(
            static_ann.get("static_background_brush_strokes", [])
        )
        save_frame_annotation(annotations_dir, frame_ann)
    return total_frames


def annotation_has_guidance(frame_ann):
    if not frame_ann:
        return False
    return any(
        frame_ann.get(key)
        for key in (
            "object_points",
            "background_points",
            "static_background_points",
            "object_brush_strokes",
            "background_brush_strokes",
            "static_background_brush_strokes",
            "bounding_boxes",
        )
    )


def list_keyframes(annotations_dir, total_frames, image_width=0, image_height=0):
    keyframes = []
    for i in range(total_frames):
        ann = load_frame_annotation(annotations_dir, i, image_width, image_height)
        if ann.get("is_keyframe"):
            keyframes.append(
                {
                    "frame_index": i,
                    "is_keyframe": True,
                    "has_guidance": annotation_has_guidance(ann),
                }
            )
    return keyframes


def suggest_keyframes(annotations_dir, total_frames, image_width, image_height, strategy="endpoints", every_n=10):
    if total_frames <= 0:
        return []
    if strategy == "every_n":
        indices = list(range(0, total_frames, max(1, every_n)))
    else:
        indices = sorted(
            {
                0,
                max(0, total_frames // 2),
                max(0, total_frames - 1),
            }
        )

    marked = []
    for i in indices:
        ann = load_frame_annotation(annotations_dir, i, image_width, image_height)
        ann["is_keyframe"] = True
        save_frame_annotation(annotations_dir, ann)
        marked.append(i)
    return marked


def find_nearest_keyframe_with_guidance(frame_index, annotations_dir, total_frames, image_width, image_height):
    candidates = []
    for i in range(total_frames):
        ann = load_frame_annotation(annotations_dir, i, image_width, image_height)
        if ann.get("is_keyframe") and annotation_has_guidance(ann):
            candidates.append(i)
    if not candidates:
        return None, None
    nearest = min(candidates, key=lambda i: abs(i - frame_index))
    return nearest, load_frame_annotation(annotations_dir, nearest, image_width, image_height)


def resolve_guidance_annotation(
    frame_index,
    annotations_dir,
    annotation_mode,
    total_frames,
    image_width,
    image_height,
):
    own = load_frame_annotation(annotations_dir, frame_index, image_width, image_height)
    static_ann = load_static_background(annotations_dir, image_width, image_height)

    if annotation_has_guidance(own):
        return own, static_ann, "frame"

    nearest_idx, nearest_ann = find_nearest_keyframe_with_guidance(
        frame_index, annotations_dir, total_frames, image_width, image_height
    )
    if nearest_ann is not None:
        return nearest_ann, static_ann, f"propagated_keyframe_{nearest_idx}"

    return None, static_ann, "auto_only"


def render_simple_mask_overlay(image_rgb, mask, color=(0, 255, 0), alpha=0.45):
    overlay = np.asarray(image_rgb).copy()
    mask_bool = np.asarray(mask).astype(bool)
    if not np.any(mask_bool):
        return overlay
    overlay[mask_bool] = (
        (1 - alpha) * overlay[mask_bool].astype(np.float32)
        + alpha * np.array(color, dtype=np.float32)
    ).astype(np.uint8)
    return overlay


def render_difference_map(image_rgb, auto_mask, guided_mask):
    image = np.asarray(image_rgb).copy()
    auto_bool = np.asarray(auto_mask).astype(bool)
    guided_bool = np.asarray(guided_mask).astype(bool)
    added = guided_bool & ~auto_bool
    removed = auto_bool & ~guided_bool

    overlay = image.astype(np.float32)
    overlay[added] = 0.55 * overlay[added] + 0.45 * np.array([40, 220, 120], dtype=np.float32)
    overlay[removed] = 0.55 * overlay[removed] + 0.45 * np.array([220, 60, 60], dtype=np.float32)
    return overlay.astype(np.uint8)


def save_global_ignore_mask(annotations_dir, static_ann, shape):
    ignore = build_static_ignore_mask({}, static_ann, shape)
    path = global_ignore_mask_path(annotations_dir)
    Image.fromarray((ignore.astype(np.uint8) * 255)).save(path)
    return path


def reset_frame_annotation(annotations_dir, frame_index, image_width, image_height):
    ann = empty_frame_annotation(frame_index, image_width, image_height)
    save_frame_annotation(annotations_dir, ann)
    return ann


def reset_all_annotations(annotations_dir, total_frames, image_width, image_height):
    for i in range(total_frames):
        reset_frame_annotation(annotations_dir, i, image_width, image_height)
    static_ann = empty_static_background_annotation(image_width, image_height)
    save_static_background(annotations_dir, static_ann)
    return total_frames


def mask_metrics_summary(metrics_dict):
    return {
        "mask_area_px": int(metrics_dict.get("hyphal_area_px", 0)),
        "mask_area_um2": float(metrics_dict.get("hyphal_area_um2", 0)),
        "skeleton_length_um": float(metrics_dict.get("hyphal_length_um", 0)),
        "tip_count": int(metrics_dict.get("tip_count", 0)),
        "branch_points": int(metrics_dict.get("branch_points", 0)),
        "raw_junction_pixel_count": int(metrics_dict.get("raw_junction_pixel_count", 0)),
        "normalized_branch_point_count": int(
            metrics_dict.get("normalized_branch_point_count", metrics_dict.get("branch_points", 0))
        ),
    }
