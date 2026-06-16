"""
Segmentation presets, workflow flags, and validation for fungal vs bacterial pipelines.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

PRESET_FUNGAL_HYPHAE = "fungal_hyphae"
PRESET_MIXED = "mixed_fungi_bacteria"
PRESET_BACTERIA = "bacteria"

# Alias for UI
TARGET_FUNGAL = PRESET_FUNGAL_HYPHAE
TARGET_BACTERIA = PRESET_BACTERIA
TARGET_MIXED = PRESET_MIXED

SKELETON_MODE_HYPHAE_ONLY = "hyphae_only"
SKELETON_MODE_ALL_OBJECTS = "all_objects"

SEGMENTATION_PRESETS: Dict[str, Dict[str, Any]] = {
    PRESET_FUNGAL_HYPHAE: {
        "label": "Fungal Hyphae",
        "target_object_type": PRESET_FUNGAL_HYPHAE,
        "min_object_size_px": 40,
        "hole_fill_area": 200,
        "dilation_radius": 8,
        "min_branch_length_px": 10,
        "skeletonization_mode": SKELETON_MODE_HYPHAE_ONLY,
        "skeleton_min_object_area_px": 40,
        "use_temporal_continuity": True,
        "repair_disconnected_tubes": True,
        "hyphae_min_length_px": 12,
        "hyphae_min_aspect_ratio": 2.5,
        "bacteria_max_length_px": 15,
        "bacteria_max_area_px2": 200,
        "max_component_count_threshold": 5000,
        "enable_bacterial_tracking": False,
        "max_bacteria_displacement_px": 20,
        "max_track_gap_frames": 2,
        "min_track_length_frames": 2,
        "trajectory_tail_frames": 20,
        "generate_trajectory_overlay_video": True,
        "generate_heatmaps": True,
        "max_objects_per_frame_for_tracking": 5000,
    },
    PRESET_MIXED: {
        "label": "Mixed Fungi + Bacteria",
        "target_object_type": PRESET_MIXED,
        "min_object_size_px": 1,
        "hole_fill_area": 5,
        "dilation_radius": 2,
        "min_branch_length_px": 3,
        "skeletonization_mode": SKELETON_MODE_HYPHAE_ONLY,
        "skeleton_min_object_area_px": 15,
        "use_temporal_continuity": True,
        "repair_disconnected_tubes": False,
        "hyphae_min_length_px": 8,
        "hyphae_min_aspect_ratio": 2.0,
        "bacteria_max_length_px": 12,
        "bacteria_max_area_px2": 120,
        "max_component_count_threshold": 5000,
        "enable_bacterial_tracking": True,
        "max_bacteria_displacement_px": 20,
        "max_track_gap_frames": 2,
        "min_track_length_frames": 2,
        "trajectory_tail_frames": 20,
        "generate_trajectory_overlay_video": True,
        "generate_heatmaps": True,
        "max_objects_per_frame_for_tracking": 5000,
    },
    PRESET_BACTERIA: {
        "label": "Bacteria",
        "target_object_type": PRESET_BACTERIA,
        "min_object_size_px": 1,
        "hole_fill_area": 0,
        "dilation_radius": 1,
        "min_branch_length_px": 1,
        "skeletonization_mode": SKELETON_MODE_HYPHAE_ONLY,
        "skeleton_min_object_area_px": 1,
        "use_temporal_continuity": False,
        "repair_disconnected_tubes": False,
        "hyphae_min_length_px": 9999,
        "hyphae_min_aspect_ratio": 99.0,
        "bacteria_max_length_px": 9999,
        "bacteria_max_area_px2": 1_000_000,
        "max_component_count_threshold": 8000,
        "enable_bacterial_tracking": True,
        "max_bacteria_displacement_px": 20,
        "max_track_gap_frames": 2,
        "min_track_length_frames": 2,
        "trajectory_tail_frames": 20,
        "generate_trajectory_overlay_video": True,
        "generate_heatmaps": True,
        "max_objects_per_frame_for_tracking": 5000,
    },
}

WORKFLOW_FLAGS: Dict[str, Dict[str, bool]] = {
    PRESET_FUNGAL_HYPHAE: {
        "enable_skeletonization": True,
        "enable_branch_detection": True,
        "enable_tip_detection": True,
        "enable_tube_repair": True,
        "enable_traversed_persistence": True,
        "enable_hyphal_growth_metrics": True,
        "enable_bacterial_metrics": False,
        "enable_fungal_processing": True,
        "enable_bacterial_processing": False,
    },
    PRESET_MIXED: {
        "enable_skeletonization": True,
        "enable_branch_detection": True,
        "enable_tip_detection": True,
        "enable_tube_repair": False,
        "enable_traversed_persistence": True,
        "enable_hyphal_growth_metrics": True,
        "enable_bacterial_metrics": True,
        "enable_fungal_processing": True,
        "enable_bacterial_processing": True,
    },
    PRESET_BACTERIA: {
        "enable_skeletonization": False,
        "enable_branch_detection": False,
        "enable_tip_detection": False,
        "enable_tube_repair": False,
        "enable_traversed_persistence": False,
        "enable_hyphal_growth_metrics": False,
        "enable_bacterial_metrics": True,
        "enable_fungal_processing": False,
        "enable_bacterial_processing": True,
    },
}

PARAM_LIMITS = {
    "min_object_size_px": {"min": 1, "max": 10000},
    "hole_fill_area": {"min": 0, "max": 100000},
    "dilation_radius": {"min": 0, "max": 100},
    "min_branch_length_px": {"min": 1, "max": 500},
    "pixel_size_um": {"min": 0.001, "max": 1000.0},
    "frame_interval_min": {"min": 0.001, "max": 10000.0},
    "hyphae_min_length_px": {"min": 1, "max": 5000},
    "hyphae_min_aspect_ratio": {"min": 1.0, "max": 100.0},
    "bacteria_max_length_px": {"min": 1, "max": 5000},
    "bacteria_max_area_px2": {"min": 1, "max": 1_000_000},
    "max_component_count_threshold": {"min": 100, "max": 100000},
    "max_bacteria_displacement_px": {"min": 1, "max": 500},
    "max_track_gap_frames": {"min": 0, "max": 50},
    "min_track_length_frames": {"min": 1, "max": 1000},
    "trajectory_tail_frames": {"min": 1, "max": 500},
    "max_objects_per_frame_for_tracking": {"min": 100, "max": 50000},
}

SMALL_OBJECT_WARNING_THRESHOLD_PX = 5


def normalize_preset_name(preset: Optional[str]) -> str:
    if preset in SEGMENTATION_PRESETS:
        return preset
    return PRESET_FUNGAL_HYPHAE


def preset_values(preset: Optional[str]) -> Dict[str, Any]:
    return dict(SEGMENTATION_PRESETS[normalize_preset_name(preset)])


def workflow_flags(preset: Optional[str]) -> Dict[str, bool]:
    return dict(WORKFLOW_FLAGS[normalize_preset_name(preset)])


def get_workflow_config(params: Dict[str, Any]) -> Dict[str, Any]:
    preset = normalize_preset_name(params.get("target_object_type") or params.get("segmentation_preset"))
    flags = workflow_flags(preset)
    preset_cfg = preset_values(preset)
    return {
        "target_object_type": preset,
        "segmentation_preset": preset,
        **flags,
        "classification": {
            "hyphae_min_length_px": int(params.get("hyphae_min_length_px", preset_cfg["hyphae_min_length_px"])),
            "hyphae_min_aspect_ratio": float(
                params.get("hyphae_min_aspect_ratio", preset_cfg["hyphae_min_aspect_ratio"])
            ),
            "bacteria_max_length_px": int(params.get("bacteria_max_length_px", preset_cfg["bacteria_max_length_px"])),
            "bacteria_max_area_px2": int(params.get("bacteria_max_area_px2", preset_cfg["bacteria_max_area_px2"])),
            "max_component_count_threshold": int(
                params.get("max_component_count_threshold", preset_cfg["max_component_count_threshold"])
            ),
        },
    }


def clamp_param(name: str, value: float) -> float:
    limits = PARAM_LIMITS.get(name, {"min": 0, "max": 1e9})
    return max(limits["min"], min(limits["max"], value))


def validate_segmentation_params(params: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(params or {})
    target = normalize_preset_name(
        out.get("target_object_type") or out.get("segmentation_preset")
    )
    out["target_object_type"] = target
    out["segmentation_preset"] = target
    preset_cfg = preset_values(target)

    out["min_object_size_px"] = int(clamp_param("min_object_size_px", int(out.get("min_object_size_px", 40))))
    out["hole_fill_area"] = int(clamp_param("hole_fill_area", int(out.get("hole_fill_area", 200))))
    out["dilation_radius"] = int(clamp_param("dilation_radius", int(out.get("dilation_radius", 8))))
    out["min_branch_length_px"] = int(
        clamp_param("min_branch_length_px", int(out.get("min_branch_length_px", 8)))
    )
    out["pixel_size_um"] = float(clamp_param("pixel_size_um", float(out.get("pixel_size_um", 1.0))))
    out["frame_interval_min"] = float(
        clamp_param("frame_interval_min", float(out.get("frame_interval_min", 1.0)))
    )
    out["hyphae_min_length_px"] = int(
        clamp_param("hyphae_min_length_px", int(out.get("hyphae_min_length_px", preset_cfg["hyphae_min_length_px"])))
    )
    out["hyphae_min_aspect_ratio"] = float(
        clamp_param(
            "hyphae_min_aspect_ratio",
            float(out.get("hyphae_min_aspect_ratio", preset_cfg["hyphae_min_aspect_ratio"])),
        )
    )
    out["bacteria_max_length_px"] = int(
        clamp_param(
            "bacteria_max_length_px",
            int(out.get("bacteria_max_length_px", preset_cfg["bacteria_max_length_px"])),
        )
    )
    out["bacteria_max_area_px2"] = int(
        clamp_param(
            "bacteria_max_area_px2",
            int(out.get("bacteria_max_area_px2", preset_cfg["bacteria_max_area_px2"])),
        )
    )
    out["max_component_count_threshold"] = int(
        clamp_param(
            "max_component_count_threshold",
            int(out.get("max_component_count_threshold", preset_cfg["max_component_count_threshold"])),
        )
    )
    out["enable_bacterial_tracking"] = bool(
        out.get("enable_bacterial_tracking", preset_cfg.get("enable_bacterial_tracking", False))
    )
    out["max_bacteria_displacement_px"] = int(
        clamp_param(
            "max_bacteria_displacement_px",
            int(out.get("max_bacteria_displacement_px", preset_cfg.get("max_bacteria_displacement_px", 20))),
        )
    )
    out["max_track_gap_frames"] = int(
        clamp_param(
            "max_track_gap_frames",
            int(out.get("max_track_gap_frames", preset_cfg.get("max_track_gap_frames", 2))),
        )
    )
    out["min_track_length_frames"] = int(
        clamp_param(
            "min_track_length_frames",
            int(out.get("min_track_length_frames", preset_cfg.get("min_track_length_frames", 2))),
        )
    )
    out["trajectory_tail_frames"] = int(
        clamp_param(
            "trajectory_tail_frames",
            int(out.get("trajectory_tail_frames", preset_cfg.get("trajectory_tail_frames", 20))),
        )
    )
    out["generate_trajectory_overlay_video"] = bool(
        out.get(
            "generate_trajectory_overlay_video",
            preset_cfg.get("generate_trajectory_overlay_video", True),
        )
    )
    out["generate_heatmaps"] = bool(out.get("generate_heatmaps", preset_cfg.get("generate_heatmaps", True)))
    out["max_objects_per_frame_for_tracking"] = int(
        clamp_param(
            "max_objects_per_frame_for_tracking",
            int(
                out.get(
                    "max_objects_per_frame_for_tracking",
                    preset_cfg.get("max_objects_per_frame_for_tracking", 5000),
                )
            ),
        )
    )
    out["skeletonization_mode"] = SKELETON_MODE_HYPHAE_ONLY
    out["skeleton_min_object_area_px"] = int(
        max(1, int(out.get("skeleton_min_object_area_px", preset_cfg["skeleton_min_object_area_px"])))
    )
    for key in (
        "hyphae_min_length_px",
        "hyphae_min_aspect_ratio",
        "bacteria_max_length_px",
        "bacteria_max_area_px2",
        "max_component_count_threshold",
    ):
        if key not in out:
            out[key] = preset_cfg[key]
    return out


def small_object_warning(min_object_size_px: int) -> Optional[str]:
    if int(min_object_size_px) < SMALL_OBJECT_WARNING_THRESHOLD_PX:
        return "Very small object sizes may increase noise and processing time."
    return None


def cleanup_profile(preset: str) -> str:
    preset = normalize_preset_name(preset)
    if preset == PRESET_BACTERIA:
        return "bacterial"
    if preset == PRESET_MIXED:
        return "mixed"
    return "fungal"


def temporal_min_size(base: int, min_object_size_px: int) -> int:
    return max(1, int(round(base * max(1, min_object_size_px) / 40)))


def list_presets_for_api() -> Dict[str, Any]:
    return {
        "target_object_types": [
            {
                "id": key,
                "label": val["label"],
                "values": {k: v for k, v in val.items() if k not in ("label", "target_object_type")},
                "workflow": WORKFLOW_FLAGS[key],
            }
            for key, val in SEGMENTATION_PRESETS.items()
        ],
        "presets": [
            {"id": key, "label": val["label"], "values": {k: v for k, v in val.items() if k != "label"}}
            for key, val in SEGMENTATION_PRESETS.items()
        ],
        "limits": PARAM_LIMITS,
        "small_object_warning_threshold_px": SMALL_OBJECT_WARNING_THRESHOLD_PX,
    }
