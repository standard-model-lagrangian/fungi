"""
Centroid-based bacterial object tracking, trajectory metrics, overlays, and heatmaps.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy.optimize import linear_sum_assignment
from skimage.measure import label, regionprops

DEFAULT_MAX_OBJECTS_PER_FRAME = 5000


def default_bacterial_tracking_config() -> Dict[str, Any]:
    return {
        "enable_bacterial_tracking": True,
        "max_bacteria_displacement_px": 20,
        "max_track_gap_frames": 2,
        "min_track_length_frames": 2,
        "trajectory_tail_frames": 20,
        "generate_tracking_label_overlay": True,
        "generate_trajectory_overlay_video": True,
        "generate_heatmaps": True,
        "max_objects_per_frame_for_tracking": DEFAULT_MAX_OBJECTS_PER_FRAME,
    }


def merge_bacterial_tracking_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    merged = default_bacterial_tracking_config()
    merged.update(config or {})
    return merged


def get_bacterial_tracking_dir(output_dir: Path) -> Path:
    path = Path(output_dir) / "bacteria_tracking"
    path.mkdir(parents=True, exist_ok=True)
    return path


def detect_bacterial_objects(
    mask_bool: np.ndarray,
    frame_index: int,
    frame_rgb: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    mask_bool = np.asarray(mask_bool).astype(bool)
    intensity = None
    if frame_rgb is not None and np.asarray(frame_rgb).size > 0:
        intensity = cv2.cvtColor(np.asarray(frame_rgb), cv2.COLOR_RGB2GRAY)
    labeled = label(mask_bool)
    detections: List[Dict[str, Any]] = []
    for region in regionprops(labeled, intensity_image=intensity):
        minr, minc, maxr, maxc = region.bbox
        area = float(region.area)
        eq_diam = float(math.sqrt(max(area, 1.0) * 4.0 / math.pi))
        cy, cx = region.centroid
        detections.append(
            {
                "frame_index": int(frame_index),
                "object_id_temp": int(region.label),
                "centroid_x_px": float(cx),
                "centroid_y_px": float(cy),
                "area_px": int(area),
                "equivalent_diameter_px": eq_diam,
                "bbox_x": int(minc),
                "bbox_y": int(minr),
                "bbox_width": int(maxc - minc),
                "bbox_height": int(maxr - minr),
                "mean_intensity": float(region.mean_intensity) if intensity is not None else np.nan,
            }
        )
    return detections


def _filter_detections_by_count(
    detections: List[Dict[str, Any]],
    max_objects: int,
) -> Tuple[List[Dict[str, Any]], bool, str]:
    if len(detections) <= max_objects:
        return detections, False, ""
    sorted_dets = sorted(detections, key=lambda d: d["area_px"], reverse=True)[:max_objects]
    warning = (
        f"frame {detections[0]['frame_index']}: {len(detections)} objects detected; "
        f"kept largest {max_objects} for tracking"
    )
    return sorted_dets, True, warning


def track_bacterial_detections(
    detections_by_frame: Dict[int, List[Dict[str, Any]]],
    max_displacement_px: float = 20.0,
    max_gap_frames: int = 2,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Nearest-neighbour centroid tracking with Hungarian assignment and gap tolerance.
    """
    warnings: List[str] = []
    frames = sorted(detections_by_frame.keys())
    if not frames:
        return pd.DataFrame(), warnings

    next_track_id = 1
    active: Dict[int, Dict[str, Any]] = {}
    linked_rows: List[Dict[str, Any]] = []

    for frame_idx in frames:
        detections = list(detections_by_frame.get(frame_idx, []))
        eligible_ids = [
            tid
            for tid, state in active.items()
            if frame_idx - int(state["last_frame"]) <= int(max_gap_frames)
        ]
        ended_ids = [
            tid
            for tid, state in active.items()
            if frame_idx - int(state["last_frame"]) > int(max_gap_frames)
        ]
        for tid in ended_ids:
            del active[tid]

        if not detections:
            continue

        if not eligible_ids:
            for det in detections:
                row = dict(det)
                row["track_id"] = next_track_id
                linked_rows.append(row)
                active[next_track_id] = {"last_frame": frame_idx, "last_det": det}
                next_track_id += 1
            continue

        prev_states = [active[tid] for tid in eligible_ids]
        cost = np.zeros((len(prev_states), len(detections)), dtype=float)
        for i, state in enumerate(prev_states):
            px = state["last_det"]["centroid_x_px"]
            py = state["last_det"]["centroid_y_px"]
            gap = max(1, frame_idx - int(state["last_frame"]))
            for j, det in enumerate(detections):
                dx = px - det["centroid_x_px"]
                dy = py - det["centroid_y_px"]
                dist = math.sqrt(dx * dx + dy * dy)
                cost[i, j] = dist if dist <= max_displacement_px * gap else 1e9

        rows, cols = linear_sum_assignment(cost)
        assigned_det = set()
        for r, c in zip(rows, cols):
            if cost[r, c] >= 1e8:
                continue
            tid = eligible_ids[r]
            det = detections[c]
            row = dict(det)
            row["track_id"] = tid
            linked_rows.append(row)
            active[tid] = {"last_frame": frame_idx, "last_det": det}
            assigned_det.add(c)

        for j, det in enumerate(detections):
            if j in assigned_det:
                continue
            row = dict(det)
            row["track_id"] = next_track_id
            linked_rows.append(row)
            active[next_track_id] = {"last_frame": frame_idx, "last_det": det}
            next_track_id += 1

    tracks_df = pd.DataFrame(linked_rows)
    if not tracks_df.empty:
        tracks_df = tracks_df.sort_values(["track_id", "frame_index"]).reset_index(drop=True)
    return tracks_df, warnings


def _step_speeds_um_per_min(
    track_df: pd.DataFrame,
    pixel_size_um: float,
    frame_interval_min: float,
) -> Tuple[float, float, float, float, float, float, float]:
    if len(track_df) < 2 or frame_interval_min <= 0:
        return 0.0, 0.0, 0.0, 0.0, np.nan, 0.0, 0.0

    ordered = track_df.sort_values("frame_index")
    xs = ordered["centroid_x_px"].to_numpy(dtype=float)
    ys = ordered["centroid_y_px"].to_numpy(dtype=float)
    frames = ordered["frame_index"].to_numpy(dtype=int)

    total_path_px = 0.0
    step_speeds: List[float] = []
    for i in range(1, len(xs)):
        dx = xs[i] - xs[i - 1]
        dy = ys[i] - ys[i - 1]
        dist_px = math.sqrt(dx * dx + dy * dy)
        total_path_px += dist_px
        dt_frames = max(1, int(frames[i] - frames[i - 1]))
        dt_min = dt_frames * frame_interval_min
        if dt_min > 0:
            step_speeds.append((dist_px * pixel_size_um) / dt_min)

    start_x, start_y = xs[0], ys[0]
    end_x, end_y = xs[-1], ys[-1]
    net_px = math.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2)
    duration_frames = max(1, int(frames[-1] - frames[0]))
    duration_min = duration_frames * frame_interval_min
    total_path_um = total_path_px * pixel_size_um
    net_um = net_px * pixel_size_um
    mean_speed = total_path_um / duration_min if duration_min > 0 else 0.0
    max_speed = max(step_speeds) if step_speeds else 0.0
    angle = math.degrees(math.atan2(end_y - start_y, end_x - start_x))
    return total_path_px, net_px, mean_speed, max_speed, angle, total_path_um, net_um


def compute_track_summaries(
    tracks_df: pd.DataFrame,
    pixel_size_um: float,
    frame_interval_min: float,
    min_track_length_frames: int = 2,
) -> pd.DataFrame:
    if tracks_df.empty:
        return pd.DataFrame()

    summaries = []
    for track_id, group in tracks_df.groupby("track_id"):
        group = group.sort_values("frame_index")
        duration_frames = int(group["frame_index"].max() - group["frame_index"].min()) + 1
        if duration_frames < int(min_track_length_frames):
            continue
        (
            total_path_px,
            net_px,
            mean_speed,
            max_speed,
            angle,
            total_path_um,
            net_um,
        ) = _step_speeds_um_per_min(group, pixel_size_um, frame_interval_min)
        straightness = (net_px / total_path_px) if total_path_px > 0 else np.nan
        summaries.append(
            {
                "track_id": int(track_id),
                "start_frame": int(group["frame_index"].min()),
                "end_frame": int(group["frame_index"].max()),
                "duration_frames": duration_frames,
                "duration_min": duration_frames * frame_interval_min,
                "start_x_px": float(group.iloc[0]["centroid_x_px"]),
                "start_y_px": float(group.iloc[0]["centroid_y_px"]),
                "end_x_px": float(group.iloc[-1]["centroid_x_px"]),
                "end_y_px": float(group.iloc[-1]["centroid_y_px"]),
                "total_path_length_px": total_path_px,
                "total_path_length_um": total_path_um,
                "net_displacement_px": net_px,
                "net_displacement_um": net_um,
                "mean_speed_um_per_min": mean_speed,
                "max_speed_um_per_min": max_speed,
                "direction_angle_degrees": angle,
                "straightness_index": straightness,
                "mean_area_px": float(group["area_px"].mean()),
                "median_area_px": float(group["area_px"].median()),
            }
        )
    return pd.DataFrame(summaries)


def compute_bacterial_frame_metrics(
    detections_df: pd.DataFrame,
    tracks_df: pd.DataFrame,
    frame_interval_min: float,
    pixel_size_um: float,
    total_frames: int,
) -> pd.DataFrame:
    rows = []
    for frame_idx in range(total_frames):
        dets = detections_df[detections_df["frame_index"] == frame_idx] if not detections_df.empty else pd.DataFrame()
        frame_tracks = tracks_df[tracks_df["frame_index"] == frame_idx] if not tracks_df.empty else pd.DataFrame()
        count = int(len(dets))
        total_area = int(dets["area_px"].sum()) if count else 0
        mean_area = float(total_area / count) if count else 0.0
        active_tracks = int(frame_tracks["track_id"].nunique()) if not frame_tracks.empty else 0

        mean_speed = np.nan
        if not frame_tracks.empty and frame_idx > 0:
            speeds = []
            for track_id, grp in tracks_df.groupby("track_id"):
                grp = grp.sort_values("frame_index")
                prev = grp[grp["frame_index"] < frame_idx].tail(1)
                curr = grp[grp["frame_index"] == frame_idx]
                if prev.empty or curr.empty:
                    continue
                dx = float(curr.iloc[0]["centroid_x_px"] - prev.iloc[0]["centroid_x_px"])
                dy = float(curr.iloc[0]["centroid_y_px"] - prev.iloc[0]["centroid_y_px"])
                dist_px = math.sqrt(dx * dx + dy * dy)
                gap = max(1, int(frame_idx - int(prev.iloc[0]["frame_index"])))
                dt_min = gap * frame_interval_min
                if dt_min > 0:
                    speeds.append((dist_px * pixel_size_um) / dt_min)
            if speeds:
                mean_speed = float(np.mean(speeds))

        rows.append(
            {
                "frame_index": frame_idx,
                "time_min": frame_idx * frame_interval_min,
                "bacteria_count": count,
                "total_bacteria_area_px": total_area,
                "mean_bacteria_area_px": mean_area,
                "tracks_active": active_tracks,
                "mean_speed_um_per_min": mean_speed,
            }
        )
    return pd.DataFrame(rows)


def _track_color(track_id: int) -> Tuple[int, int, int]:
    hue = int((track_id * 47) % 180)
    hsv = np.uint8([[[hue, 220, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[2]), int(bgr[1]), int(bgr[0])


def bacterial_label(track_id: int) -> str:
    return f"Bacteria {int(track_id)}"


def prepare_tracks_export_df(tracks_df: pd.DataFrame) -> pd.DataFrame:
    if tracks_df.empty:
        return pd.DataFrame(
            columns=[
                "frame_index",
                "track_id",
                "label",
                "centroid_x_px",
                "centroid_y_px",
                "bbox_x",
                "bbox_y",
                "bbox_width",
                "bbox_height",
                "area_px",
            ]
        )
    out = tracks_df.copy()
    out["label"] = out["track_id"].apply(bacterial_label)
    cols = [
        "frame_index",
        "track_id",
        "label",
        "centroid_x_px",
        "centroid_y_px",
        "bbox_x",
        "bbox_y",
        "bbox_width",
        "bbox_height",
        "area_px",
    ]
    return out[cols]


def tracking_label_overlay_filename(frame_index: int) -> str:
    return f"frame_{frame_index + 1:06d}.png"


def resolve_tracking_label_overlay_path(out_dir: Path, frame_index: int) -> Path:
    label_dir = out_dir / "overlay_frames"
    primary = label_dir / tracking_label_overlay_filename(frame_index)
    if primary.exists():
        return primary
    legacy = label_dir / f"frame_{frame_index:06d}.png"
    if legacy.exists():
        return legacy
    return primary


def render_tracking_label_overlay_frame(
    base_image: np.ndarray,
    tracks_df: pd.DataFrame,
    frame_index: int,
) -> np.ndarray:
    overlay = np.asarray(base_image).copy()
    if tracks_df.empty:
        return overlay

    frame_rows = tracks_df[tracks_df["frame_index"] == frame_index]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1
    for _, row in frame_rows.iterrows():
        track_id = int(row["track_id"])
        color = _track_color(track_id)
        x = int(row["bbox_x"])
        y = int(row["bbox_y"])
        w = max(1, int(row["bbox_width"]))
        h = max(1, int(row["bbox_height"]))
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2, cv2.LINE_AA)

        label = bacterial_label(track_id)
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        label_x = x
        label_y = max(text_h + 6, y - 6)
        pad = 3
        cv2.rectangle(
            overlay,
            (label_x, label_y - text_h - pad),
            (label_x + text_w + pad * 2, label_y + baseline + pad),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            overlay,
            label,
            (label_x + pad, label_y),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
    return overlay


def render_trajectory_overlay_frame(
    base_image: np.ndarray,
    tracks_df: pd.DataFrame,
    frame_index: int,
    tail_frames: int = 20,
) -> np.ndarray:
    overlay = np.asarray(base_image).copy()
    if tracks_df.empty:
        return overlay

    for track_id, group in tracks_df.groupby("track_id"):
        group = group.sort_values("frame_index")
        hist = group[group["frame_index"] <= frame_index].tail(max(1, int(tail_frames) + 1))
        if hist.empty:
            continue
        color = _track_color(int(track_id))
        pts = [
            (int(round(row["centroid_x_px"])), int(round(row["centroid_y_px"])))
            for _, row in hist.iterrows()
        ]
        if len(pts) >= 2:
            cv2.polylines(overlay, [np.array(pts, dtype=np.int32)], False, color, 2, cv2.LINE_AA)
        curr = hist[hist["frame_index"] == frame_index]
        if not curr.empty:
            cx = int(round(curr.iloc[0]["centroid_x_px"]))
            cy = int(round(curr.iloc[0]["centroid_y_px"]))
            cv2.circle(overlay, (cx, cy), 4, color, -1, lineType=cv2.LINE_AA)
            cv2.circle(overlay, (cx, cy), 5, (255, 255, 255), 1, lineType=cv2.LINE_AA)
    return overlay


def _accumulate_heatmap(
    values: List[Tuple[float, float, float]],
    shape: Tuple[int, int],
    bin_size: int = 8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = shape
    grid_h = max(1, int(math.ceil(h / bin_size)))
    grid_w = max(1, int(math.ceil(w / bin_size)))
    count = np.zeros((grid_h, grid_w), dtype=np.float64)
    speed_sum = np.zeros((grid_h, grid_w), dtype=np.float64)
    speed_weight = np.zeros((grid_h, grid_w), dtype=np.float64)
    dir_x = np.zeros((grid_h, grid_w), dtype=np.float64)
    dir_y = np.zeros((grid_h, grid_w), dtype=np.float64)

    for x, y, weight in values:
        gx = min(grid_w - 1, int(x // bin_size))
        gy = min(grid_h - 1, int(y // bin_size))
        count[gy, gx] += 1.0
        if weight >= 0:
            speed_sum[gy, gx] += weight
            speed_weight[gy, gx] += 1.0

    speed_avg = np.divide(speed_sum, speed_weight, out=np.zeros_like(speed_sum), where=speed_weight > 0)
    return count, speed_avg, (grid_h, grid_w)


def generate_heatmaps(
    detections_df: pd.DataFrame,
    tracks_df: pd.DataFrame,
    track_summary_df: pd.DataFrame,
    image_shape: Tuple[int, int],
    out_dir: Path,
    pixel_size_um: float,
    frame_interval_min: float,
) -> Dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h, w = image_shape
    paths: Dict[str, str] = {}
    bin_size = max(4, min(h, w) // 64)

    occ_vals = [
        (d["centroid_x_px"], d["centroid_y_px"], 1.0)
        for _, d in detections_df.iterrows()
    ] if not detections_df.empty else []
    traj_vals = [
        (r["centroid_x_px"], r["centroid_y_px"], 1.0)
        for _, r in tracks_df.iterrows()
    ] if not tracks_df.empty else []
    speed_vals = []
    if not tracks_df.empty:
        for track_id, grp in tracks_df.groupby("track_id"):
            grp = grp.sort_values("frame_index")
            for i in range(1, len(grp)):
                prev = grp.iloc[i - 1]
                curr = grp.iloc[i]
                dx = curr["centroid_x_px"] - prev["centroid_x_px"]
                dy = curr["centroid_y_px"] - prev["centroid_y_px"]
                dist_px = math.sqrt(dx * dx + dy * dy)
                gap = max(1, int(curr["frame_index"] - prev["frame_index"]))
                dt_min = gap * frame_interval_min
                speed = (dist_px * pixel_size_um) / dt_min if dt_min > 0 else 0.0
                speed_vals.append((curr["centroid_x_px"], curr["centroid_y_px"], speed))

    occ_count, _, grid_shape = _accumulate_heatmap(occ_vals, (h, w), bin_size)
    traj_count, _, _ = _accumulate_heatmap(traj_vals, (h, w), bin_size)
    _, speed_map, _ = _accumulate_heatmap(speed_vals, (h, w), bin_size)

    def _save_map(data: np.ndarray, filename: str, title: str, cmap: str = "magma"):
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(data, origin="upper", cmap=cmap, aspect="auto")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046)
        path = out_dir / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths[filename.replace(".png", "")] = str(path)

    _save_map(occ_count, "bacterial_occupancy_heatmap.png", "Bacterial Occupancy")
    _save_map(traj_count, "bacterial_trajectory_density_heatmap.png", "Trajectory Density")
    _save_map(speed_map, "bacterial_speed_heatmap.png", "Mean Speed (µm/min)")

    if not track_summary_df.empty:
        dir_grid = np.zeros(grid_shape, dtype=np.float64)
        for _, row in track_summary_df.iterrows():
            gx = min(grid_shape[1] - 1, int(row["end_x_px"] // bin_size))
            gy = min(grid_shape[0] - 1, int(row["end_y_px"] // bin_size))
            dir_grid[gy, gx] += 1.0
        _save_map(dir_grid, "bacterial_direction_heatmap.png", "Track Endpoints Density")

    return paths


def run_bacterial_tracking_pipeline(
    frames: List[np.ndarray],
    bacteria_masks: List[np.ndarray],
    output_dir: Path,
    pixel_size_um: float,
    frame_interval_min: float,
    config: Optional[Dict[str, Any]] = None,
    perf_logger=None,
    base_overlays: Optional[List[np.ndarray]] = None,
) -> Dict[str, Any]:
    cfg = merge_bacterial_tracking_config(config)
    out_dir = get_bacterial_tracking_dir(output_dir)
    overlay_frames_dir = out_dir / "bacterial_trajectory_overlay_frames"
    overlay_frames_dir.mkdir(parents=True, exist_ok=True)

    metadata: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": cfg,
        "warnings": [],
        "skipped": False,
        "skip_reason": None,
    }
    result_paths: Dict[str, str] = {"bacteria_tracking_dir": str(out_dir)}

    if not cfg.get("enable_bacterial_tracking", True):
        metadata["skipped"] = True
        metadata["skip_reason"] = "disabled"
        _write_metadata(out_dir, metadata)
        return result_paths

    total_frames = len(frames)
    if total_frames == 0:
        metadata["skipped"] = True
        metadata["skip_reason"] = "no_frames"
        _write_metadata(out_dir, metadata)
        return result_paths

    detections_by_frame: Dict[int, List[Dict[str, Any]]] = {}
    all_detections: List[Dict[str, Any]] = []
    max_per_frame = int(cfg["max_objects_per_frame_for_tracking"])

    detect_ctx = perf_logger.timed("bacterial_detection") if perf_logger else _null_context()
    with detect_ctx:
        for i, (frame, mask) in enumerate(zip(frames, bacteria_masks)):
            mask_bool = np.asarray(mask).astype(bool)
            dets = detect_bacterial_objects(mask_bool, i, frame_rgb=frame)
            if len(dets) > max_per_frame:
                dets, filtered, warning = _filter_detections_by_count(dets, max_per_frame)
                if filtered:
                    metadata["warnings"].append(warning)
                    if perf_logger:
                        perf_logger.log(f"[warning] {warning}")
            detections_by_frame[i] = dets
            all_detections.extend(dets)

    detections_df = pd.DataFrame(all_detections)
    detections_csv = out_dir / "bacterial_detections.csv"
    detections_df.to_csv(detections_csv, index=False)
    result_paths["bacterial_detections_csv"] = str(detections_csv)

    if not all_detections:
        _write_metadata(out_dir, metadata)
        return result_paths

    track_ctx = perf_logger.timed("bacterial_tracking") if perf_logger else _null_context()
    with track_ctx:
        tracks_df, track_warnings = track_bacterial_detections(
            detections_by_frame,
            max_displacement_px=float(cfg["max_bacteria_displacement_px"]),
            max_gap_frames=int(cfg["max_track_gap_frames"]),
        )
    metadata["warnings"].extend(track_warnings)

    tracks_csv = out_dir / "bacterial_tracks.csv"
    prepare_tracks_export_df(tracks_df).to_csv(tracks_csv, index=False)
    result_paths["bacterial_tracks_csv"] = str(tracks_csv)

    summary_df = compute_track_summaries(
        tracks_df,
        pixel_size_um=pixel_size_um,
        frame_interval_min=frame_interval_min,
        min_track_length_frames=int(cfg["min_track_length_frames"]),
    )
    summary_csv = out_dir / "bacterial_track_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    result_paths["bacterial_track_summary_csv"] = str(summary_csv)

    frame_metrics_df = compute_bacterial_frame_metrics(
        detections_df,
        tracks_df,
        frame_interval_min=frame_interval_min,
        pixel_size_um=pixel_size_um,
        total_frames=total_frames,
    )
    frame_metrics_csv = out_dir / "bacterial_frame_metrics.csv"
    frame_metrics_df.to_csv(frame_metrics_csv, index=False)
    result_paths["bacterial_frame_metrics_csv"] = str(frame_metrics_csv)

    label_overlay_frames_dir = out_dir / "overlay_frames"
    label_overlay_frames_dir.mkdir(parents=True, exist_ok=True)
    label_overlay_video_frames: List[np.ndarray] = []
    if cfg.get("generate_tracking_label_overlay", True):
        label_ctx = perf_logger.timed("bacterial_tracking_label_overlay") if perf_logger else _null_context()
        with label_ctx:
            for i, frame in enumerate(frames):
                base = base_overlays[i] if base_overlays is not None and i < len(base_overlays) else frame
                rendered = render_tracking_label_overlay_frame(base, tracks_df, i)
                label_overlay_video_frames.append(rendered)
                Image.fromarray(rendered).save(
                    label_overlay_frames_dir / tracking_label_overlay_filename(i)
                )

            if label_overlay_video_frames:
                h, w = label_overlay_video_frames[0].shape[:2]
                label_video_path = out_dir / "bacteria_tracking_overlay.mp4"
                writer = cv2.VideoWriter(
                    str(label_video_path),
                    cv2.VideoWriter_fourcc(*"avc1"),
                    max(1, int(60 / max(frame_interval_min, 1))),
                    (w, h),
                )
                for fr in label_overlay_video_frames:
                    writer.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
                writer.release()
                result_paths["bacteria_tracking_overlay_video"] = str(label_video_path)

    tail = int(cfg["trajectory_tail_frames"])
    overlay_video_frames: List[np.ndarray] = []
    if cfg.get("generate_trajectory_overlay_video", True):
        overlay_ctx = perf_logger.timed("bacterial_trajectory_overlay") if perf_logger else _null_context()
        with overlay_ctx:
            for i, frame in enumerate(frames):
                base = base_overlays[i] if base_overlays is not None and i < len(base_overlays) else frame
                rendered = render_trajectory_overlay_frame(base, tracks_df, i, tail_frames=tail)
                overlay_video_frames.append(rendered)
                Image.fromarray(rendered).save(overlay_frames_dir / f"overlay_{i:04d}.png")

            if overlay_video_frames:
                h, w = overlay_video_frames[0].shape[:2]
                video_path = out_dir / "bacterial_trajectory_overlay.mp4"
                writer = cv2.VideoWriter(
                    str(video_path),
                    cv2.VideoWriter_fourcc(*"avc1"),
                    max(1, int(60 / max(frame_interval_min, 1))),
                    (w, h),
                )
                for fr in overlay_video_frames:
                    writer.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
                writer.release()
                result_paths["bacterial_trajectory_video"] = str(video_path)

    if cfg.get("generate_heatmaps", True):
        heat_ctx = perf_logger.timed("bacterial_heatmaps") if perf_logger else _null_context()
        with heat_ctx:
            heatmap_paths = generate_heatmaps(
                detections_df,
                tracks_df,
                summary_df,
                image_shape=frames[0].shape[:2],
                out_dir=out_dir,
                pixel_size_um=pixel_size_um,
                frame_interval_min=frame_interval_min,
            )
            result_paths.update(heatmap_paths)

    metadata.update(
        {
            "total_detections": int(len(detections_df)),
            "total_track_points": int(len(tracks_df)),
            "unique_tracks": int(tracks_df["track_id"].nunique()) if not tracks_df.empty else 0,
            "summary_tracks": int(len(summary_df)),
            "outputs": result_paths,
        }
    )
    _write_metadata(out_dir, metadata)
    result_paths["bacterial_tracking_metadata"] = str(out_dir / "bacterial_tracking_metadata.json")
    return result_paths


def _write_metadata(out_dir: Path, metadata: Dict[str, Any]) -> None:
    path = out_dir / "bacterial_tracking_metadata.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def load_bacterial_tracking_metadata(output_dir: Path) -> Dict[str, Any]:
    path = get_bacterial_tracking_dir(output_dir) / "bacterial_tracking_metadata.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class _null_context:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False
