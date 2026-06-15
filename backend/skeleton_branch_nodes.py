"""
Normalize skeleton junction pixels into stable branch nodes.

Raw degree>=3 junction pixels inside thick Y-junctions are clustered (8-connectivity),
optionally merged by centroid distance, snapped to the skeleton, and temporally smoothed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image
from skimage.measure import label, regionprops

Node = Tuple[int, int]

BRANCH_NODE_MERGE_RADIUS_PX = 6
BRANCH_NODE_TEMPORAL_SMOOTHING = True
BRANCH_NODE_MAX_TRACKING_DISTANCE_PX = 10


def snap_to_nearest_skeleton_pixel(
    y: int,
    x: int,
    skel_bool: np.ndarray,
    prefer_points: Optional[np.ndarray] = None,
    search_radius: int = 20,
) -> Node:
    """Snap a coordinate to the nearest skeleton pixel."""
    if prefer_points is not None and len(prefer_points) > 0:
        dists = (prefer_points[:, 0] - y) ** 2 + (prefer_points[:, 1] - x) ** 2
        idx = int(np.argmin(dists))
        return int(prefer_points[idx, 0]), int(prefer_points[idx, 1])

    h, w = skel_bool.shape
    y0, y1 = max(0, y - search_radius), min(h, y + search_radius + 1)
    x0, x1 = max(0, x - search_radius), min(w, x + search_radius + 1)
    window = skel_bool[y0:y1, x0:x1]
    if not window.any():
        yy = int(np.clip(y, 0, h - 1))
        xx = int(np.clip(x, 0, w - 1))
        return yy, xx

    pts = np.argwhere(window)
    dists = (pts[:, 0] + y0 - y) ** 2 + (pts[:, 1] + x0 - x) ** 2
    idx = int(np.argmin(dists))
    return int(pts[idx, 0] + y0), int(pts[idx, 1] + x0)


def cluster_junction_components(
    raw_branch: np.ndarray,
    skel_bool: np.ndarray,
) -> Tuple[List[Node], np.ndarray, int]:
    """
    Label 8-connected junction pixels and return one canonical node per component.
    """
    raw_bool = raw_branch.astype(bool)
    raw_count = int(raw_bool.sum())
    if raw_count == 0:
        return [], np.zeros_like(raw_bool, dtype=np.int32), 0

    cluster_labels = label(raw_bool, connectivity=2)
    nodes: List[Node] = []
    for region in regionprops(cluster_labels):
        cluster_mask = cluster_labels == region.label
        pts = np.argwhere(cluster_mask)
        if len(pts) == 0:
            continue
        cy, cx = region.centroid
        dists = (pts[:, 0] - cy) ** 2 + (pts[:, 1] - cx) ** 2
        idx = int(np.argmin(dists))
        y, x = int(pts[idx, 0]), int(pts[idx, 1])
        y, x = snap_to_nearest_skeleton_pixel(y, x, skel_bool, prefer_points=pts)
        nodes.append((y, x))

    return nodes, cluster_labels, raw_count


def merge_nodes_by_radius(
    nodes: Sequence[Node],
    skel_bool: np.ndarray,
    merge_radius_px: int,
) -> List[Node]:
    """Merge branch nodes whose coordinates are within merge_radius_px."""
    if len(nodes) <= 1:
        return list(nodes)

    n = len(nodes)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    radius_sq = merge_radius_px * merge_radius_px
    for i in range(n):
        y0, x0 = nodes[i]
        for j in range(i + 1, n):
            y1, x1 = nodes[j]
            dy, dx = y0 - y1, x0 - x1
            if dy * dy + dx * dx <= radius_sq:
                union(i, j)

    groups: dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    merged: List[Node] = []
    for indices in groups.values():
        ys = [nodes[i][0] for i in indices]
        xs = [nodes[i][1] for i in indices]
        cy = float(np.mean(ys))
        cx = float(np.mean(xs))
        group_pts = np.array([(nodes[i][0], nodes[i][1]) for i in indices], dtype=np.int32)
        y, x = snap_to_nearest_skeleton_pixel(
            int(round(cy)), int(round(cx)), skel_bool, prefer_points=group_pts
        )
        merged.append((y, x))

    return merged


def stabilize_branch_nodes(
    current_nodes: Sequence[Node],
    prev_nodes: Optional[Sequence[Node]],
    skel_bool: np.ndarray,
    max_tracking_distance_px: int = BRANCH_NODE_MAX_TRACKING_DISTANCE_PX,
    smoothing: bool = BRANCH_NODE_TEMPORAL_SMOOTHING,
) -> List[Node]:
    """Match nodes to the previous frame and smooth small movements."""
    if not smoothing or not prev_nodes:
        return list(current_nodes)

    prev = list(prev_nodes)
    used_prev: set[int] = set()
    stabilized: List[Node] = []
    max_dist_sq = max_tracking_distance_px * max_tracking_distance_px

    for cy, cx in current_nodes:
        best_pi: Optional[int] = None
        best_d2 = max_dist_sq + 1
        for pi, (py, px) in enumerate(prev):
            if pi in used_prev:
                continue
            d2 = (cy - py) ** 2 + (cx - px) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_pi = pi

        if best_pi is not None and best_d2 <= max_dist_sq:
            used_prev.add(best_pi)
            py, px = prev[best_pi]
            sy = int(round(0.65 * py + 0.35 * cy))
            sx = int(round(0.65 * px + 0.35 * cx))
            sy, sx = snap_to_nearest_skeleton_pixel(sy, sx, skel_bool)
            stabilized.append((sy, sx))
        else:
            stabilized.append((cy, cx))

    return stabilized


def nodes_to_mask(shape: Tuple[int, int], nodes: Iterable[Node]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    h, w = shape
    for y, x in nodes:
        if 0 <= y < h and 0 <= x < w:
            mask[y, x] = 1
    return mask


def normalize_branch_nodes(
    raw_branch: np.ndarray,
    skel_bool: np.ndarray,
    prev_nodes: Optional[Sequence[Node]] = None,
    merge_radius_px: int = BRANCH_NODE_MERGE_RADIUS_PX,
    temporal_smoothing: bool = BRANCH_NODE_TEMPORAL_SMOOTHING,
    max_tracking_distance_px: int = BRANCH_NODE_MAX_TRACKING_DISTANCE_PX,
) -> dict:
    """
    Cluster raw junction pixels, merge nearby clusters, and optionally stabilize temporally.
    """
    nodes, cluster_labels, raw_count = cluster_junction_components(raw_branch, skel_bool)
    nodes = merge_nodes_by_radius(nodes, skel_bool, merge_radius_px)
    nodes = stabilize_branch_nodes(
        nodes,
        prev_nodes,
        skel_bool,
        max_tracking_distance_px=max_tracking_distance_px,
        smoothing=temporal_smoothing,
    )

    normalized_count = len(nodes)
    branches_mask = nodes_to_mask(skel_bool.shape, nodes)

    return {
        "raw_junction_pixel_count": raw_count,
        "normalized_branch_point_count": normalized_count,
        "normalized_nodes": nodes,
        "branches_mask": branches_mask,
        "cluster_labels": cluster_labels,
    }


def render_skeleton_overlay(
    skel_bool: np.ndarray,
    nodes: Sequence[Node],
    raw_branch: Optional[np.ndarray] = None,
    debug_raw_junctions: bool = False,
    node_radius: int = 3,
) -> np.ndarray:
    """White skeleton on black with yellow normalized branch nodes."""
    img = np.zeros((*skel_bool.shape, 3), dtype=np.uint8)
    img[skel_bool.astype(bool)] = [255, 255, 255]

    if debug_raw_junctions and raw_branch is not None:
        raw_bool = raw_branch.astype(bool)
        img[raw_bool] = [255, 0, 255]

    for x, y in ((x, y) for y, x in nodes):
        cv2.circle(img, (x, y), node_radius, (255, 255, 0), -1)

    return img


def save_skeleton_debug(
    output_dir: Path,
    frame_index: int,
    skel_bool: np.ndarray,
    raw_branch: np.ndarray,
    cluster_labels: np.ndarray,
    nodes: Sequence[Node],
    overlay: np.ndarray,
    metadata: Optional[dict] = None,
) -> Path:
    """Persist per-frame skeleton branch debug artefacts."""
    debug_dir = Path(output_dir) / "skeleton_debug" / f"frame_{frame_index:06d}"
    debug_dir.mkdir(parents=True, exist_ok=True)

    raw_u8 = (raw_branch.astype(np.uint8) * 255)
    Image.fromarray(raw_u8).save(debug_dir / "raw_junctions.png")

    if cluster_labels is not None and int(cluster_labels.max()) > 0:
        clusters_vis = ((cluster_labels % 20) + 1) * (cluster_labels > 0)
        clusters_vis = (clusters_vis.astype(np.float32) / max(1, clusters_vis.max()) * 255).astype(np.uint8)
    else:
        clusters_vis = np.zeros_like(raw_u8)
    Image.fromarray(clusters_vis).save(debug_dir / "clusters.png")

    nodes_mask = nodes_to_mask(skel_bool.shape, nodes)
    Image.fromarray((nodes_mask * 255).astype(np.uint8)).save(debug_dir / "normalized_nodes.png")
    Image.fromarray(overlay).save(debug_dir / "skeleton_overlay.png")

    meta = {
        "frame": frame_index,
        "raw_junction_pixel_count": int(raw_branch.sum()),
        "normalized_branch_point_count": len(nodes),
        "normalized_nodes": [[int(y), int(x)] for y, x in nodes],
    }
    if metadata:
        meta.update(metadata)
    (debug_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return debug_dir


def load_prev_normalized_nodes(output_dir: Path, frame_index: int) -> Optional[List[Node]]:
    if frame_index <= 0:
        return None
    meta_path = (
        Path(output_dir) / "skeleton_debug" / f"frame_{frame_index - 1:06d}" / "metadata.json"
    )
    if not meta_path.exists():
        return None
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        nodes = data.get("normalized_nodes", [])
        return [(int(p[0]), int(p[1])) for p in nodes]
    except Exception:
        return None
