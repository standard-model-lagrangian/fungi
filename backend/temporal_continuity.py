"""
Temporal continuity post-processing for hyphal segmentation.
"""

import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage.morphology import binary_dilation, disk, remove_small_objects

from annotations import build_static_ignore_mask, render_difference_map

# Safety limits (override via config keys when present)
DEFAULT_MAX_ENDPOINTS = 40
DEFAULT_MAX_PAIR_CHECKS = 80
DEFAULT_MAX_BRIDGES_PER_FRAME = 5
DEFAULT_FRAME_TIME_BUDGET_SEC = 25.0


def default_temporal_config():
    return {
        "use_temporal_continuity": True,
        "temporal_memory_frames": 3,
        "temporal_persistence_weight": 0.5,
        "max_allowed_area_drop_fraction": 0.15,
        "recover_missing_middle_frames": True,
        "min_temporal_component_persistence": 2,
        "allow_tip_growth": True,
        "repair_disconnected_tubes": False,
        "max_bridge_gap_px": 15,
        "max_bridge_angle_degrees": 45,
        "min_bridge_intensity_percentile": 60,
        "max_endpoints_per_frame": DEFAULT_MAX_ENDPOINTS,
        "max_bridge_pair_checks": DEFAULT_MAX_PAIR_CHECKS,
        "max_bridges_per_frame": DEFAULT_MAX_BRIDGES_PER_FRAME,
        "frame_time_budget_sec": DEFAULT_FRAME_TIME_BUDGET_SEC,
    }


def merge_temporal_config(config):
    merged = default_temporal_config()
    merged.update(config or {})
    return merged


def get_temporal_debug_log_path(output_dir):
    return Path(output_dir) / "temporal_debug_log.txt"


def get_temporal_masks_dir(output_dir):
    path = Path(output_dir) / "temporal_masks"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_temporal_overlays_dir(output_dir):
    path = Path(output_dir) / "temporal_overlays"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_temporal_difference_maps_dir(output_dir):
    path = Path(output_dir) / "temporal_difference_maps"
    path.mkdir(parents=True, exist_ok=True)
    return path


def temporal_metadata_path(diff_dir, frame_index):
    return Path(diff_dir) / f"frame_{frame_index:06d}_meta.json"


class TemporalDebugLogger:
    def __init__(self, output_dir):
        self.path = get_temporal_debug_log_path(output_dir)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as handle:
            handle.write(f"=== temporal continuity log {datetime.now(timezone.utc).isoformat()} ===\n")

    def log(self, message):
        line = f"{datetime.now(timezone.utc).isoformat()} {message}"
        print(f"[Temporal] {message}")
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def timed_step(self, frame_index, step_name):
        return _TimedStep(self, frame_index, step_name)


class _TimedStep:
    def __init__(self, logger, frame_index, step_name):
        self.logger = logger
        self.frame_index = frame_index
        self.step_name = step_name
        self.start = None

    def __enter__(self):
        self.start = time.perf_counter()
        self.logger.log(f"frame {self.frame_index + 1}: start {self.step_name}")
        return self

    def __exit__(self, exc_type, exc, tb):
        elapsed = time.perf_counter() - self.start
        if exc_type is None:
            self.logger.log(
                f"frame {self.frame_index + 1}: done {self.step_name} ({elapsed:.3f}s)"
            )
        else:
            self.logger.log(
                f"frame {self.frame_index + 1}: error {self.step_name} ({elapsed:.3f}s) {exc}"
            )
        return False


def _tube_intensity_mask(frame_rgb, percentile=72):
    gray = cv2.cvtColor(np.asarray(frame_rgb), cv2.COLOR_RGB2GRAY).astype(np.float32)
    thresh = float(np.percentile(gray, percentile))
    return gray >= thresh * 0.82


def _low_intensity_mask(frame_rgb, percentile=28):
    gray = cv2.cvtColor(np.asarray(frame_rgb), cv2.COLOR_RGB2GRAY).astype(np.float32)
    thresh = float(np.percentile(gray, percentile))
    return gray <= thresh * 1.08


def _endpoint_coords(skel_bool):
    skel_u8 = skel_bool.astype(np.uint8)
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    neigh = cv2.filter2D(skel_u8, -1, kernel)
    endpoints = skel_bool & (neigh == 1)
    return [(int(c), int(r)) for r, c in np.argwhere(endpoints)]


def _skeleton_direction_at(skel_bool, x, y, radius=4):
    h, w = skel_bool.shape
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    patch = skel_bool[y0:y1, x0:x1]
    if not np.any(patch):
        return None
    pts = np.argwhere(patch)
    cy, cx = pts.mean(axis=0)
    return np.array([cx - (x - x0), cy - (y - y0)], dtype=np.float32)


def _angle_between(v1, v2):
    if v1 is None or v2 is None:
        return 180.0
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-3 or n2 < 1e-3:
        return 180.0
    cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def _path_intensity_supported(intensity, static_ignore, x1, y1, x2, y2, samples=8):
    for t in np.linspace(0, 1, samples):
        x = int(round(x1 + (x2 - x1) * t))
        y = int(round(y1 + (y2 - y1) * t))
        if y < 0 or x < 0 or y >= intensity.shape[0] or x >= intensity.shape[1]:
            return False
        if static_ignore[y, x] or not intensity[y, x]:
            return False
    return True


def _tip_boundary_mask_fast(mask_bool, radius=4):
    """Fast tip/boundary protection without skeletonization."""
    dilated = binary_dilation(mask_bool, disk(radius))
    eroded = binary_dilation(mask_bool, disk(1))
    return (dilated & ~eroded).astype(bool)


def repair_disconnected_tubes(
    mask,
    static_ignore,
    intensity,
    max_bridge_gap_px=15,
    max_bridge_angle_degrees=45,
    max_endpoints=DEFAULT_MAX_ENDPOINTS,
    max_pair_checks=DEFAULT_MAX_PAIR_CHECKS,
    max_bridges_per_frame=DEFAULT_MAX_BRIDGES_PER_FRAME,
    logger=None,
    frame_index=0,
    protected_object_mask=None,
):
    from skimage.morphology import skeletonize

    mask_bool = np.asarray(mask).astype(bool).copy()
    bridge_debug = np.zeros(mask_bool.shape, dtype=np.uint8)
    bridges = 0
    pair_checks = 0

    if logger:
        with logger.timed_step(frame_index, "bridge_repair_skeletonize"):
            skel = skeletonize(mask_bool)
    else:
        skel = skeletonize(mask_bool)

    endpoints = _endpoint_coords(skel)
    if logger:
        logger.log(
            f"frame {frame_index + 1}: endpoints={len(endpoints)} "
            f"(cap={max_endpoints}, pair_checks_cap={max_pair_checks})"
        )

    if len(endpoints) > max_endpoints:
        if logger:
            logger.log(f"frame {frame_index + 1}: skip bridge repair (too many endpoints)")
        return mask_bool.astype(np.uint8), bridges, bridge_debug

    endpoints = endpoints[:max_endpoints]

    for i, (x1, y1) in enumerate(endpoints):
        if bridges >= max_bridges_per_frame or pair_checks >= max_pair_checks:
            break
        dir1 = _skeleton_direction_at(skel, x1, y1)
        best_j = None
        best_gap = float(max_bridge_gap_px) + 1.0

        for j in range(i + 1, len(endpoints)):
            if pair_checks >= max_pair_checks or bridges >= max_bridges_per_frame:
                break
            pair_checks += 1
            x2, y2 = endpoints[j]
            gap = float(np.hypot(x2 - x1, y2 - y1))
            if gap < 2 or gap > max_bridge_gap_px:
                continue
            if gap >= best_gap:
                continue

            dir2 = _skeleton_direction_at(skel, x2, y2)
            gap_vec = np.array([x2 - x1, y2 - y1], dtype=np.float32)
            angle1 = _angle_between(dir1, gap_vec)
            angle2 = _angle_between(dir2, -gap_vec)
            if min(angle1, angle2) > max_bridge_angle_degrees:
                continue
            if not _path_intensity_supported(intensity, static_ignore, x1, y1, x2, y2):
                continue
            best_j = j
            best_gap = gap

        if best_j is None:
            continue

        x2, y2 = endpoints[best_j]
        mask_work = mask_bool.astype(np.uint8)
        cv2.line(mask_work, (x1, y1), (x2, y2), 1, thickness=2)
        mask_bool = mask_work.astype(bool)
        cv2.line(bridge_debug, (x1, y1), (x2, y2), 255, 1)
        bridges += 1

    if bridges > 0:
        mask_bool = binary_dilation(mask_bool, disk(2))

    if protected_object_mask is not None and np.any(protected_object_mask):
        mask_bool |= _as_bool_mask(protected_object_mask)

    if logger:
        logger.log(
            f"frame {frame_index + 1}: bridges_added={bridges} pair_checks={pair_checks}"
        )

    return mask_bool.astype(np.uint8), bridges, bridge_debug


def _as_bool_mask(mask):
    """Normalize any mask array (uint8/bool) to bool for bitwise ops."""
    return np.asarray(mask).astype(bool)


def _nearby_mask_stack(masks, frame_index, memory, corrected):
    nearby = []
    for dt in range(-memory, memory + 1):
        if dt == 0:
            continue
        j = frame_index + dt
        if 0 <= j < len(masks):
            source = corrected if j < len(corrected) and corrected[j] is not None else masks
            nearby.append(_as_bool_mask(source[j]))
    return nearby


def _persistent_regions(nearby, min_persistence):
    if not nearby:
        return None
    counts = np.sum(np.stack(nearby, axis=0), axis=0)
    return counts >= min_persistence


def render_temporal_difference_map(image_rgb, before_mask, after_mask):
    return render_difference_map(image_rgb, before_mask, after_mask)


def _frame_timed_out(frame_start, budget_sec):
    return (time.perf_counter() - frame_start) >= budget_sec


def apply_temporal_continuity_pass(
    masks,
    frames,
    static_ann,
    config,
    progress_callback=None,
    job_id=None,
    logger=None,
    hard_ann_masks_list=None,
    correction_masks_list=None,
    output_dir=None,
):
    config = merge_temporal_config(config)
    if not config["use_temporal_continuity"] or len(masks) == 0:
        return list(masks), [], []

    if logger:
        logger.log("loading static ignore mask")
    static_ignore = build_static_ignore_mask({}, static_ann, frames[0].shape)
    setup_allowed = np.ones(frames[0].shape[:2], dtype=bool)
    if output_dir is not None:
        from pre_segmentation_setup import load_setup_masks

        setup_masks = load_setup_masks(output_dir, frames[0].shape)
        static_ignore = static_ignore | setup_masks["ignore_mask"]
        setup_allowed = setup_masks["allowed_mask"]

    memory = max(1, int(config["temporal_memory_frames"]))
    min_persist = max(1, int(config["min_temporal_component_persistence"]))
    persistence_w = float(config["temporal_persistence_weight"])
    max_drop = float(config["max_allowed_area_drop_fraction"])
    intensity_pct = int(config["min_bridge_intensity_percentile"])
    frame_budget = float(config.get("frame_time_budget_sec", DEFAULT_FRAME_TIME_BUDGET_SEC))

    corrected = [None] * len(masks)
    metadata_list = []
    bridge_debug_list = []

    cumulative_traversed = np.zeros(frames[0].shape[:2], dtype=bool)
    calc_progress_percent = None
    if progress_callback:
        from sam2_pipeline import calc_progress_percent as _calc_progress_percent
        calc_progress_percent = _calc_progress_percent

    if logger:
        logger.log(f"processing {len(frames)} frames (repair={config['repair_disconnected_tubes']})")

    for t, frame in enumerate(frames):
        frame_start = time.perf_counter()

        if progress_callback and calc_progress_percent is not None:
            progress_callback(
                stage="temporal",
                current_frame_index=t,
                total_frames=len(frames),
                current_frame_preview_url=f"/api/jobs/{job_id}/preview/{t}" if job_id else None,
                progress_percent=calc_progress_percent("temporal", t, len(frames)),
            )

        step_logger = logger
        ann_masks = None
        correction_masks = None
        object_protect = None
        if hard_ann_masks_list is not None and t < len(hard_ann_masks_list):
            ann_masks = hard_ann_masks_list[t]
            object_protect = _as_bool_mask(ann_masks["object_mask"])
        if correction_masks_list is not None and t < len(correction_masks_list):
            correction_masks = correction_masks_list[t]
            add_protect = _as_bool_mask(correction_masks["add_mask"])
            object_protect = add_protect if object_protect is None else (object_protect | add_protect)

        if step_logger:
            step_logger.log(f"frame {t + 1}/{len(frames)}: begin")

        with (step_logger.timed_step(t, "load_mask") if step_logger else _null_context()):
            curr = _as_bool_mask(masks[t])
            curr &= ~static_ignore
            curr &= setup_allowed
            before_area = int(curr.sum())

        with (step_logger.timed_step(t, "intensity_maps") if step_logger else _null_context()):
            intensity_support = _tube_intensity_mask(frame, percentile=intensity_pct)
            low_intensity = _low_intensity_mask(frame)

        with (step_logger.timed_step(t, "nearby_persistent") if step_logger else _null_context()):
            nearby = _nearby_mask_stack(masks, t, memory, corrected)
            persistent = _persistent_regions(nearby, min_persist)
            if persistent is None:
                persistent = np.zeros(curr.shape, dtype=bool)

        result = curr.copy()
        recovered = np.zeros(curr.shape, dtype=bool)
        removed = np.zeros(curr.shape, dtype=bool)
        middle_applied = False
        repair_skipped = False
        repair_skip_reason = None

        prev_m = None
        if t > 0:
            if corrected[t - 1] is not None:
                prev_m = _as_bool_mask(corrected[t - 1]) & ~static_ignore
            else:
                prev_m = _as_bool_mask(masks[t - 1]) & ~static_ignore

        next_m = None
        if t + 1 < len(masks):
            next_m = _as_bool_mask(masks[t + 1]) & ~static_ignore

        if not _frame_timed_out(frame_start, frame_budget):
            with (step_logger.timed_step(t, "middle_frame_recovery") if step_logger else _null_context()):
                if config["recover_missing_middle_frames"] and prev_m is not None and next_m is not None:
                    mid_recover = prev_m & next_m & ~result & intensity_support & ~static_ignore
                    mid_recover = _as_bool_mask(
                        remove_small_objects(mid_recover, min_size=20)
                    )
                    if np.any(mid_recover):
                        middle_applied = True
                    recovered |= mid_recover
                    result |= mid_recover

        if not _frame_timed_out(frame_start, frame_budget):
            with (step_logger.timed_step(t, "cumulative_traversed_prior") if step_logger else _null_context()):
                if persistence_w > 0 and prev_m is not None:
                    sudden_loss = prev_m & ~result & ~static_ignore
                    keep_loss = sudden_loss & (intensity_support | cumulative_traversed) & ~low_intensity
                    keep_loss = _as_bool_mask(
                        remove_small_objects(keep_loss, min_size=16)
                    )
                    recovered |= keep_loss & ~curr
                    if persistence_w >= 0.5:
                        result |= keep_loss
                    else:
                        result |= keep_loss & persistent

                path_keep = cumulative_traversed & ~result & ~static_ignore & intensity_support
                path_keep = _as_bool_mask(
                    remove_small_objects(path_keep, min_size=12)
                )
                recovered |= path_keep & ~curr
                result |= path_keep

                if nearby:
                    support = persistent & intensity_support & ~static_ignore
                    support = _as_bool_mask(
                        remove_small_objects(support, min_size=18)
                    )
                    recovered |= support & ~result
                    result |= support & binary_dilation(result, disk(2))

        if not _frame_timed_out(frame_start, frame_budget):
            with (step_logger.timed_step(t, "flicker_removal") if step_logger else _null_context()):
                if nearby:
                    isolated = result.copy()
                    for nb in nearby:
                        isolated &= ~binary_dilation(nb, disk(2))
                    noise = isolated & ~static_ignore
                    noise = _as_bool_mask(
                        remove_small_objects(noise, min_size=28)
                    )
                    if config["allow_tip_growth"]:
                        tips = _tip_boundary_mask_fast(result)
                        noise &= ~tips
                    if object_protect is not None:
                        noise &= ~object_protect
                    removed |= noise & curr
                    result &= ~noise

        if not _frame_timed_out(frame_start, frame_budget):
            after_area = int(result.sum())
            if before_area > 0 and (before_area - after_area) / before_area > max_drop and prev_m is not None:
                candidates = prev_m & ~result & intensity_support & ~static_ignore
                candidates = _as_bool_mask(
                    remove_small_objects(candidates, min_size=8)
                )
                if np.any(candidates):
                    recovered |= candidates
                    result |= candidates

        bridge_debug = np.zeros(frame.shape[:2], dtype=np.uint8)
        bridges = 0
        if config["repair_disconnected_tubes"]:
            if _frame_timed_out(frame_start, frame_budget):
                repair_skipped = True
                repair_skip_reason = "frame_time_budget_exceeded"
            else:
                try:
                    with (step_logger.timed_step(t, "bridge_repair") if step_logger else _null_context()):
                        result, bridges, bridge_debug = repair_disconnected_tubes(
                            result.astype(np.uint8),
                            static_ignore,
                            intensity_support,
                            max_bridge_gap_px=int(config["max_bridge_gap_px"]),
                            max_bridge_angle_degrees=float(config["max_bridge_angle_degrees"]),
                            max_endpoints=int(config.get("max_endpoints_per_frame", DEFAULT_MAX_ENDPOINTS)),
                            max_pair_checks=int(config.get("max_bridge_pair_checks", DEFAULT_MAX_PAIR_CHECKS)),
                            max_bridges_per_frame=int(config.get("max_bridges_per_frame", DEFAULT_MAX_BRIDGES_PER_FRAME)),
                            logger=step_logger,
                            frame_index=t,
                            protected_object_mask=object_protect,
                        )
                    result = result.astype(bool)
                except Exception as exc:
                    repair_skipped = True
                    repair_skip_reason = f"bridge_repair_error: {exc}"
                    if step_logger:
                        step_logger.log(f"frame {t + 1}: bridge repair failed, continuing ({exc})")

        result = _as_bool_mask(result)
        result &= ~static_ignore
        result &= setup_allowed
        if ann_masks is not None or correction_masks is not None:
            from corrections import enforce_full_guided_mask

            result = _as_bool_mask(
                enforce_full_guided_mask(result.astype(np.uint8), ann_masks, correction_masks)
            )
        cumulative_traversed |= result

        elapsed = time.perf_counter() - frame_start
        meta = {
            "frame_index": t,
            "pixels_recovered": int(recovered.sum()),
            "pixels_removed_flicker": int(removed.sum()),
            "area_before": before_area,
            "area_after": int(result.sum()),
            "middle_frame_recovery_applied": middle_applied,
            "bridges_added": bridges,
            "tube_repair_applied": bridges > 0,
            "repair_skipped": repair_skipped,
            "repair_skip_reason": repair_skip_reason,
            "frame_elapsed_sec": round(elapsed, 3),
        }
        if np.any(bridge_debug):
            meta["bridge_debug_available"] = True

        corrected[t] = result
        metadata_list.append(meta)
        bridge_debug_list.append(bridge_debug)

        if step_logger:
            step_logger.log(
                f"frame {t + 1}: complete elapsed={elapsed:.3f}s "
                f"recovered={meta['pixels_recovered']} removed={meta['pixels_removed_flicker']}"
            )

    corrected_uint8 = [np.asarray(m).astype(np.uint8) for m in corrected]
    return corrected_uint8, metadata_list, bridge_debug_list


class _null_context:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def save_temporal_outputs(
    frames,
    before_masks,
    after_masks,
    metadata_list,
    output_dir,
    logger=None,
):
    output_dir = Path(output_dir)
    masks_dir = get_temporal_masks_dir(output_dir)
    diff_dir = get_temporal_difference_maps_dir(output_dir)

    if logger:
        logger.log("saving temporal masks and difference maps")

    for i, (frame, before, after, meta) in enumerate(
        zip(frames, before_masks, after_masks, metadata_list)
    ):
        if logger:
            with logger.timed_step(i, "save_outputs"):
                mask = np.asarray(after).astype(np.uint8)
                Image.fromarray((mask * 255).astype(np.uint8)).save(masks_dir / f"mask_{i:04d}.png")
                diff_img = render_temporal_difference_map(frame, before, mask)
                Image.fromarray(diff_img).save(diff_dir / f"frame_{i:06d}.png")
                with open(temporal_metadata_path(diff_dir, i), "w", encoding="utf-8") as handle:
                    json.dump(meta, handle, indent=2)
        else:
            mask = np.asarray(after).astype(np.uint8)
            Image.fromarray((mask * 255).astype(np.uint8)).save(masks_dir / f"mask_{i:04d}.png")
            diff_img = render_temporal_difference_map(frame, before, mask)
            Image.fromarray(diff_img).save(diff_dir / f"frame_{i:06d}.png")
            with open(temporal_metadata_path(diff_dir, i), "w", encoding="utf-8") as handle:
                json.dump(meta, handle, indent=2)

    summary_path = diff_dir / "temporal_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump({"frames": metadata_list}, handle, indent=2)

    if logger:
        logger.log(f"saved temporal outputs to {masks_dir} and {diff_dir}")


def load_temporal_metadata(output_dir, frame_index):
    diff_dir = get_temporal_difference_maps_dir(output_dir)
    path = temporal_metadata_path(diff_dir, frame_index)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def run_temporal_continuity_pipeline(
    frames,
    masks,
    output_dir,
    static_ann,
    config,
    job_id=None,
    progress_callback=None,
    annotations_dir=None,
    image_width=0,
    image_height=0,
):
    """
    Apply temporal continuity with logging, safety limits, and fallback metadata.
    Returns (masks, info_dict).
    """
    config = merge_temporal_config(config)
    output_dir = Path(output_dir)
    logger = TemporalDebugLogger(output_dir)
    info = {
        "success": True,
        "fallback": False,
        "warning": None,
        "log_path": str(logger.path),
    }

    if not config["use_temporal_continuity"]:
        logger.log("temporal continuity disabled in config")
        return list(masks), info

    before_masks = [np.asarray(m).astype(np.uint8) for m in masks]
    logger.log(f"loaded {len(before_masks)} masks for temporal pass")

    hard_ann_masks_list = None
    correction_masks_list = None
    if annotations_dir is not None:
        from annotations import compute_guided_annotation_masks, load_frame_annotation, load_static_background
        from corrections import load_correction_masks

        ann_dir = Path(annotations_dir)
        static = load_static_background(ann_dir, image_width, image_height)
        hard_ann_masks_list = []
        correction_masks_list = []
        for i, frame in enumerate(frames):
            ann = load_frame_annotation(ann_dir, i, image_width, image_height)
            hard_ann_masks_list.append(
                compute_guided_annotation_masks(ann, static, frame.shape)
            )
            correction_masks_list.append(
                load_correction_masks(output_dir, i, frame.shape)
            )
        logger.log(f"loaded hard annotation and correction masks for {len(hard_ann_masks_list)} frames")

    try:
        corrected, metadata, bridge_debug_list = apply_temporal_continuity_pass(
            before_masks,
            frames,
            static_ann,
            config,
            progress_callback=progress_callback,
            job_id=job_id,
            logger=logger,
            hard_ann_masks_list=hard_ann_masks_list,
            correction_masks_list=correction_masks_list,
            output_dir=output_dir,
        )
        save_temporal_outputs(
            frames,
            before_masks,
            corrected,
            metadata,
            output_dir,
            logger=logger,
        )
        logger.log("temporal continuity completed successfully")
        return corrected, info
    except Exception as exc:
        tb = traceback.format_exc()
        warning = f"Temporal continuity failed: {exc}"
        logger.log(warning)
        logger.log(tb)
        info.update({
            "success": False,
            "fallback": True,
            "warning": warning,
        })
        return before_masks, info

