import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage.morphology import binary_dilation, disk

from annotations import (
    annotation_has_guidance,
    apply_guided_postprocess,
    build_static_ignore_mask,
    compute_guided_annotation_masks,
    compute_guided_debug_metadata,
    enforce_hard_guided_mask,
    get_annotations_dir,
    load_frame_annotation,
    load_job_config,
    load_static_background,
    render_difference_map,
    render_simple_mask_overlay,
    save_guided_frame_debug,
)
from corrections import (
    enforce_full_guided_mask,
    get_hard_foreground_protect_mask,
    load_correction_masks,
    save_correction_frame_debug,
)
from temporal_continuity import repair_disconnected_tubes


def default_propagation_config():
    return {
        "propagation_window": 10,
        "use_temporal_propagation": True,
        "repair_disconnected_tubes": True,
        "max_bridge_gap_px": 15,
        "max_bridge_angle_degrees": 45,
    }


def merge_propagation_config(config):
    merged = default_propagation_config()
    merged.update(config or {})
    return merged


def get_propagation_debug_dir(output_dir):
    path = Path(output_dir) / "propagation_debug"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_guided_masks_dir(output_dir):
    path = Path(output_dir) / "guided_masks"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_guided_overlays_dir(output_dir):
    path = Path(output_dir) / "guided_overlays"
    path.mkdir(parents=True, exist_ok=True)
    return path


def propagation_metadata_path(debug_dir, frame_index):
    return Path(debug_dir) / f"frame_{frame_index:06d}_meta.json"


def list_guidance_keyframes(annotations_dir, total_frames, image_width, image_height):
    keyframes = []
    for i in range(total_frames):
        ann = load_frame_annotation(annotations_dir, i, image_width, image_height)
        if annotation_has_guidance(ann):
            keyframes.append(i)
    return sorted(keyframes)


def _tube_intensity_mask(frame_rgb, percentile=72):
    gray = cv2.cvtColor(np.asarray(frame_rgb), cv2.COLOR_RGB2GRAY).astype(np.float32)
    thresh = float(np.percentile(gray, percentile))
    return gray >= thresh * 0.82


def _merge_propagated_object_masks(
    frame_index,
    frame_ann_masks,
    keyframe_ann_masks,
    keyframes,
    propagation_window,
):
    """Union local object annotations with dilated object masks from nearby keyframes."""
    merged = np.asarray(frame_ann_masks["object_mask"]).astype(bool).copy()
    for k in keyframes:
        dist = abs(k - frame_index)
        if dist == 0 or dist > propagation_window:
            continue
        k_obj = np.asarray(keyframe_ann_masks[k]["object_mask"]).astype(bool)
        if not np.any(k_obj):
            continue
        dilate_r = max(1, int(round(dist * 0.8)))
        merged |= binary_dilation(k_obj, disk(dilate_r))
    frame_ann_masks = dict(frame_ann_masks)
    frame_ann_masks["object_mask"] = merged
    return frame_ann_masks


def temporal_propagate_mask(
    frame_index,
    frame_rgb,
    auto_mask,
    auto_masks,
    keyframe_guided_masks,
    keyframes,
    propagation_window,
    static_ann,
    frame_ann_masks,
    keyframe_ann_masks,
    correction_masks=None,
):
    static_ignore = build_static_ignore_mask({}, static_ann, frame_rgb.shape)
    result = np.asarray(auto_mask).astype(bool).copy()
    result &= ~static_ignore

    influencers = []
    for k in keyframes:
        dist = abs(k - frame_index)
        if dist == 0 or dist > propagation_window:
            continue
        weight = 1.0 - (dist / max(propagation_window, 1))
        influencers.append((k, weight))

    effective_ann_masks = _merge_propagated_object_masks(
        frame_index,
        frame_ann_masks,
        keyframe_ann_masks,
        keyframes,
        propagation_window,
    )

    if not influencers:
        result = enforce_full_guided_mask(result, effective_ann_masks, correction_masks)
        return result.astype(np.uint8), {
            "frame_index": frame_index,
            "directly_annotated": False,
            "source_keyframe": None,
            "propagation_distance": None,
            "influencing_keyframes": [],
            "mode": "auto_only",
            "tube_repair_applied": False,
            "bridged_gaps": 0,
        }

    intensity = _tube_intensity_mask(frame_rgb)
    auto_bool = np.asarray(auto_mask).astype(bool)
    object_hard = np.asarray(effective_ann_masks["object_mask"]).astype(bool)
    if correction_masks is not None:
        object_hard |= np.asarray(correction_masks["add_mask"]).astype(bool)

    for k, weight in influencers:
        dist = abs(k - frame_index)
        k_guided = np.asarray(keyframe_guided_masks[k]).astype(bool)
        k_auto = np.asarray(auto_masks[k]).astype(bool)
        add_prior = k_guided & ~k_auto
        remove_prior = k_auto & ~k_guided

        dilate_r = max(1, int(round(dist * 0.6)))
        selem = disk(dilate_r)

        add_prop = binary_dilation(add_prior, selem)
        add_candidates = add_prop & intensity
        add_candidates |= add_prop & object_hard
        add_candidates |= add_prop & binary_dilation(auto_bool, disk(3))
        if weight >= 0.35:
            result |= add_candidates
        else:
            result |= add_candidates & binary_dilation(result, disk(4))

        remove_prop = binary_dilation(remove_prior, selem)
        remove_prop &= ~object_hard
        if weight >= 0.45:
            result &= ~remove_prop

    result = enforce_full_guided_mask(result, effective_ann_masks, correction_masks)

    nearest_k, nearest_dist = min(
        ((k, abs(k - frame_index)) for k, _ in influencers),
        key=lambda item: item[1],
    )

    return result.astype(np.uint8), {
        "frame_index": frame_index,
        "directly_annotated": False,
        "source_keyframe": nearest_k,
        "propagation_distance": nearest_dist,
        "influencing_keyframes": [k for k, _ in influencers],
        "mode": "propagated",
        "tube_repair_applied": False,
        "bridged_gaps": 0,
    }


def build_guided_masks_with_propagation(
    frames,
    auto_masks,
    output_dir,
    progress_callback=None,
    job_id=None,
):
    from sam2_pipeline import calc_progress_percent, make_overlay, skeleton_and_branchpoints

    output_dir = Path(output_dir)
    annotations_dir = get_annotations_dir(output_dir)
    config = merge_propagation_config(load_job_config(annotations_dir))
    image_w, image_h = frames[0].shape[1], frames[0].shape[0]
    static_ann = load_static_background(annotations_dir, image_w, image_h)
    static_ignore_global = build_static_ignore_mask({}, static_ann, frames[0].shape)

    total_frames = len(frames)
    keyframes = list_guidance_keyframes(
        annotations_dir,
        total_frames,
        image_w,
        image_h,
    )

    frame_ann_masks_by_index = {}
    keyframe_ann_masks = {}
    correction_masks_by_index = {}
    for i in range(total_frames):
        ann = load_frame_annotation(annotations_dir, i, image_w, image_h)
        ann_masks = compute_guided_annotation_masks(ann, static_ann, frames[i].shape)
        frame_ann_masks_by_index[i] = ann_masks
        correction_masks_by_index[i] = load_correction_masks(output_dir, i, frames[i].shape)
        if i in keyframes:
            keyframe_ann_masks[i] = ann_masks

    keyframe_guided = {}
    for k in keyframes:
        ann = load_frame_annotation(annotations_dir, k, image_w, image_h)
        corr = correction_masks_by_index[k]
        mask = apply_guided_postprocess(
            auto_masks[k], ann, static_ann, frames[k].shape, correction_masks=corr
        )
        mask = enforce_full_guided_mask(mask, keyframe_ann_masks[k], corr)
        keyframe_guided[k] = mask

    guided_masks = []
    metadata_list = []
    bridge_debug_list = []
    debug_dir = get_propagation_debug_dir(output_dir)
    guided_masks_dir = get_guided_masks_dir(output_dir)
    guided_overlays_dir = get_guided_overlays_dir(output_dir)

    for i, fr in enumerate(frames):
        if progress_callback:
            progress_callback(
                stage="segmenting",
                current_frame_index=i,
                total_frames=total_frames,
                current_frame_preview_url=f"/api/jobs/{job_id}/preview/{i}" if job_id else None,
                progress_percent=calc_progress_percent("segmenting", i, total_frames),
            )

        ann_masks = frame_ann_masks_by_index[i]
        correction_masks = correction_masks_by_index[i]

        if i in keyframe_guided:
            mask = np.asarray(keyframe_guided[i]).astype(np.uint8)
            meta = {
                "frame_index": i,
                "directly_annotated": True,
                "source_keyframe": i,
                "propagation_distance": 0,
                "influencing_keyframes": [i],
                "mode": "keyframe",
                "tube_repair_applied": False,
                "bridged_gaps": 0,
            }
        elif config["use_temporal_propagation"] and keyframes:
            mask, meta = temporal_propagate_mask(
                i,
                fr,
                auto_masks[i],
                auto_masks,
                keyframe_guided,
                keyframes,
                int(config["propagation_window"]),
                static_ann,
                ann_masks,
                keyframe_ann_masks,
                correction_masks,
            )
        else:
            auto_base = (np.asarray(auto_masks[i]).astype(bool) & ~static_ignore_global).astype(np.uint8)
            mask = enforce_full_guided_mask(auto_base, ann_masks, correction_masks)
            meta = {
                "frame_index": i,
                "directly_annotated": False,
                "source_keyframe": None,
                "propagation_distance": None,
                "influencing_keyframes": [],
                "mode": "auto_only",
                "tube_repair_applied": False,
                "bridged_gaps": 0,
            }

        bridge_debug = np.zeros(fr.shape[:2], dtype=np.uint8)
        if config["repair_disconnected_tubes"]:
            static_ignore = build_static_ignore_mask({}, static_ann, fr.shape)
            intensity = _tube_intensity_mask(
                fr, percentile=int(config.get("min_bridge_intensity_percentile", 60))
            )
            object_protect = get_hard_foreground_protect_mask(ann_masks, correction_masks)
            mask, bridges, bridge_debug = repair_disconnected_tubes(
                mask,
                static_ignore,
                intensity,
                max_bridge_gap_px=int(config["max_bridge_gap_px"]),
                max_bridge_angle_degrees=float(config["max_bridge_angle_degrees"]),
                max_endpoints=int(config.get("max_endpoints_per_frame", 40)),
                max_pair_checks=int(config.get("max_bridge_pair_checks", 80)),
                max_bridges_per_frame=int(config.get("max_bridges_per_frame", 5)),
                frame_index=i,
                protected_object_mask=object_protect,
            )
            mask = enforce_full_guided_mask(mask, ann_masks, correction_masks)
            meta["tube_repair_applied"] = bridges > 0
            meta["bridged_gaps"] = bridges
            if np.any(object_protect):
                meta["object_annotation_protected"] = True

        mask = enforce_full_guided_mask(mask, ann_masks, correction_masks)
        debug_meta = compute_guided_debug_metadata(auto_masks[i], mask, ann_masks)
        meta.update(debug_meta)
        save_guided_frame_debug(output_dir, i, fr, auto_masks[i], mask, ann_masks, extra_metadata=meta)
        save_correction_frame_debug(output_dir, i, fr, auto_masks[i], mask, correction_masks)

        guided_masks.append(mask)
        metadata_list.append(meta)
        bridge_debug_list.append(bridge_debug)

    for i, (fr, mask, meta, bridge_debug) in enumerate(
        zip(frames, guided_masks, metadata_list, bridge_debug_list)
    ):
        Image.fromarray((mask.astype(np.uint8) * 255)).save(
            guided_masks_dir / f"mask_{i:04d}.png"
        )

        diff_img = render_difference_map(fr, auto_masks[i], mask)
        Image.fromarray(diff_img).save(debug_dir / f"frame_{i:06d}_diff.png")

        if np.any(bridge_debug):
            overlay_dbg = cv2.cvtColor(fr.copy(), cv2.COLOR_RGB2BGR)
            overlay_dbg[bridge_debug > 0] = (0, 255, 255)
            Image.fromarray(cv2.cvtColor(overlay_dbg, cv2.COLOR_BGR2RGB)).save(
                debug_dir / f"frame_{i:06d}_bridges.png"
            )

        with open(propagation_metadata_path(debug_dir, i), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    for i, (fr, mask) in enumerate(zip(frames, guided_masks)):
        if progress_callback:
            progress_callback(
                stage="tracking",
                current_frame_index=i,
                total_frames=total_frames,
                current_frame_preview_url=f"/api/jobs/{job_id}/preview/{i}" if job_id else None,
                progress_percent=calc_progress_percent("tracking", i, total_frames),
            )
        skel, branches, _, _, _ = skeleton_and_branchpoints(mask)
        overlay = make_overlay(fr, mask, skel, branches)
        Image.fromarray(overlay).save(guided_overlays_dir / f"overlay_{i:04d}.png")

        prop_viz = render_simple_mask_overlay(fr, mask, color=(40, 220, 120), alpha=0.35)
        meta = metadata_list[i]
        if meta.get("source_keyframe") is not None and not meta.get("directly_annotated"):
            label = f"KF {meta['source_keyframe'] + 1} d={meta['propagation_distance']}"
            cv2.putText(
                prop_viz,
                label,
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        Image.fromarray(prop_viz).save(debug_dir / f"frame_{i:06d}_propagation.png")

    with open(debug_dir / "propagation_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "keyframes": keyframes,
                "config": config,
                "frames": metadata_list,
            },
            f,
            indent=2,
        )

    return guided_masks, metadata_list, config


def load_propagation_metadata(output_dir, frame_index):
    debug_dir = get_propagation_debug_dir(output_dir)
    path = propagation_metadata_path(debug_dir, frame_index)
    if not path.exists():
        guided_meta = Path(output_dir) / "guided_debug" / f"frame_{frame_index:06d}" / "metadata.json"
        if guided_meta.exists():
            with open(guided_meta, "r", encoding="utf-8") as handle:
                return json.load(handle)
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_guided_debug_metadata(output_dir, frame_index):
    path = Path(output_dir) / "guided_debug" / f"frame_{frame_index:06d}" / "metadata.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(f)
