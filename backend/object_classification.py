"""
Split segmentation masks into fungal (hyphal) and bacterial object layers.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from skimage.measure import label, regionprops

from segmentation_config import PRESET_BACTERIA, PRESET_FUNGAL_HYPHAE, PRESET_MIXED, normalize_preset_name


def default_classification_config(preset: str) -> Dict[str, float]:
    preset = normalize_preset_name(preset)
    if preset == PRESET_BACTERIA:
        return {
            "hyphae_min_length_px": 9999,
            "hyphae_min_aspect_ratio": 99.0,
            "bacteria_max_length_px": 9999,
            "bacteria_max_area_px2": 1e9,
            "max_component_count_threshold": 8000,
        }
    if preset == PRESET_MIXED:
        return {
            "hyphae_min_length_px": 8,
            "hyphae_min_aspect_ratio": 2.0,
            "bacteria_max_length_px": 12,
            "bacteria_max_area_px2": 120,
            "max_component_count_threshold": 5000,
        }
    return {
        "hyphae_min_length_px": 12,
        "hyphae_min_aspect_ratio": 2.5,
        "bacteria_max_length_px": 15,
        "bacteria_max_area_px2": 200,
        "max_component_count_threshold": 5000,
    }


def _region_aspect(region) -> float:
    minor = max(float(getattr(region, "minor_axis_length", 1.0)), 1.0)
    return float(getattr(region, "major_axis_length", 0.0)) / minor


def _is_hyphal_candidate(region, cfg: Dict) -> bool:
    major = float(getattr(region, "major_axis_length", 0.0))
    aspect = _region_aspect(region)
    ecc = float(getattr(region, "eccentricity", 0.0))
    min_len = float(cfg["hyphae_min_length_px"])
    min_aspect = float(cfg["hyphae_min_aspect_ratio"])
    return major >= min_len and (aspect >= min_aspect or ecc >= 0.75)


def _is_bacterial_candidate(region, cfg: Dict) -> bool:
    major = float(getattr(region, "major_axis_length", 0.0))
    area = float(region.area)
    aspect = _region_aspect(region)
    return (
        area <= float(cfg["bacteria_max_area_px2"])
        and major <= float(cfg["bacteria_max_length_px"])
        and aspect < float(cfg["hyphae_min_aspect_ratio"])
    )


def classify_mask_layers(
    mask_bool: np.ndarray,
    target_object_type: str,
    classification_config: Dict,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Return (fungi_mask, bacteria_mask, stats) as bool arrays in original shape.
    """
    preset = normalize_preset_name(target_object_type)
    mask_bool = np.asarray(mask_bool).astype(bool)
    h, w = mask_bool.shape
    fungi = np.zeros((h, w), dtype=bool)
    bacteria = np.zeros((h, w), dtype=bool)

    if not mask_bool.any():
        stats = {
            "total_objects": 0,
            "fungal_objects": 0,
            "bacterial_objects": 0,
            "skeletonized_objects": 0,
            "force_disable_skeleton": False,
        }
        return fungi, bacteria, stats

    if preset == PRESET_FUNGAL_HYPHAE:
        return mask_bool.copy(), bacteria, {
            "total_objects": int(label(mask_bool).max()),
            "fungal_objects": int(label(mask_bool).max()),
            "bacterial_objects": 0,
            "skeletonized_objects": int(label(mask_bool).max()),
            "force_disable_skeleton": False,
        }

    if preset == PRESET_BACTERIA:
        total = int(label(mask_bool).max())
        return fungi, mask_bool.copy(), {
            "total_objects": total,
            "fungal_objects": 0,
            "bacterial_objects": total,
            "skeletonized_objects": 0,
            "force_disable_skeleton": True,
        }

    cfg = {**default_classification_config(preset), **(classification_config or {})}
    labeled = label(mask_bool)
    regions = list(regionprops(labeled))
    total = len(regions)
    fungal_labels = set()
    bacterial_labels = set()

    for region in regions:
        rid = region.label
        if _is_hyphal_candidate(region, cfg):
            fungal_labels.add(rid)
        elif _is_bacterial_candidate(region, cfg):
            bacterial_labels.add(rid)

    unassigned = [r.label for r in regions if r.label not in fungal_labels and r.label not in bacterial_labels]
    for rid in unassigned:
        region = next(r for r in regions if r.label == rid)
        if _region_aspect(region) >= float(cfg["hyphae_min_aspect_ratio"]):
            fungal_labels.add(rid)
        else:
            bacterial_labels.add(rid)

    for rid in fungal_labels:
        fungi[labeled == rid] = True
    for rid in bacterial_labels:
        bacteria[labeled == rid] = True

    force_disable = total > int(cfg.get("max_component_count_threshold", 5000))
    skeletonized = len(fungal_labels) if not force_disable else 0

    return fungi, bacteria, {
        "total_objects": total,
        "fungal_objects": len(fungal_labels),
        "bacterial_objects": len(bacterial_labels),
        "skeletonized_objects": skeletonized,
        "force_disable_skeleton": force_disable,
    }
