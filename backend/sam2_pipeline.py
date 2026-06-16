import os
import sys
import subprocess
import importlib.util
import re
import traceback
from contextlib import nullcontext
from datetime import datetime
from typing import List
from pathlib import Path
import urllib.request
import shutil
import math

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

def pip_install(args):
    print(f"\n[INSTALL] pip install {args}\n")
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + args)

def ensure_package(import_name, pip_args):
    if importlib.util.find_spec(import_name) is None:
        pip_install(pip_args)

BACKEND_DIR = Path(__file__).resolve().parent

def check_cellsam_installed():
    try:
        import cellSAM
        return True
    except ImportError:
        return False

def ensure_dependencies():
    base = [
        ("numpy", ["numpy"]),
        ("cv2", ["opencv-python"]),
        ("PIL", ["pillow"]),
        ("pandas", ["pandas"]),
        ("skimage", ["scikit-image"]),
        ("tifffile", ["tifffile"]),
        ("scipy", ["scipy"]),
        ("matplotlib", ["matplotlib"]),
        ("skan", ["skan"]),
        ("fastapi", ["fastapi", "uvicorn", "python-multipart"]),
    ]
    for imp, pip_args in base:
        ensure_package(imp, pip_args)

    if importlib.util.find_spec("torch") is None:
        if sys.platform == "darwin":
            pip_install(["torch", "torchvision"])
        else:
            try:
                pip_install(["torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu128"])
            except Exception as e:
                print(f"[WARN] Failed to install CUDA-enabled torch, trying standard torch: {e}")
                pip_install(["torch", "torchvision"])

    if not check_cellsam_installed():
        try:
            pip_install(["git+https://github.com/vanvalenlab/cellSAM.git"])
        except Exception as e:
            print("[WARN] CellSAM install failed:", e)

    if importlib.util.find_spec("sam2") is None:
        repo_dir = BACKEND_DIR / "sam2"
        if not repo_dir.exists():
            subprocess.check_call(["git", "clone", "https://github.com/facebookresearch/sam2.git", str(repo_dir)])
        try:
            pip_install(["-e", str(repo_dir)])
        except Exception as e:
            print("[WARN] SAM2 install failed:", e)

    ckpt_dir = BACKEND_DIR / "sam2_checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / "sam2.1_hiera_large.pt"
    if not ckpt_path.exists():
        url = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
        print(f"[DOWNLOAD] {url}")
        urllib.request.urlretrieve(url, ckpt_path)

ensure_dependencies()

import numpy as np
import cv2
import pandas as pd
import tifffile
from PIL import Image
from scipy.optimize import linear_sum_assignment
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import (
    skeletonize,
    remove_small_objects,
    remove_small_holes,
    binary_closing,
    binary_dilation,
    disk,
)
from scipy.ndimage import binary_fill_holes
from skimage.util import img_as_ubyte

try:
    from skan import Skeleton, summarize
    SKAN_AVAILABLE = True
except Exception as e:
    Skeleton = None
    summarize = None
    SKAN_AVAILABLE = False

import torch

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except Exception as e:
    SAM2ImagePredictor = None
    build_sam2 = None

OUT_DIR = BACKEND_DIR / "fungal_sam2_demo_output"
OUT_DIR.mkdir(exist_ok=True)

OUTPUTS_ROOT = BACKEND_DIR / "outputs"
OUTPUTS_ROOT.mkdir(exist_ok=True)

_UNSAFE_CHARS = re.compile(r"[^\w\-.]+")

def sanitize_stem(filename):
    stem = Path(filename).stem
    stem = _UNSAFE_CHARS.sub("_", stem)
    stem = re.sub(r"_+", "_", stem).strip("_.")
    return stem or "upload"

def sanitize_filename(filename):
    path = Path(filename)
    safe_stem = sanitize_stem(filename)
    suffix = path.suffix.lower()
    return f"{safe_stem}{suffix}" if suffix else safe_stem

def create_job_output_dir(original_filename):
    stem = sanitize_stem(original_filename)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_name = f"{stem}_{timestamp}"
    candidate = OUTPUTS_ROOT / base_name
    if not candidate.exists():
        candidate.mkdir(parents=True)
        return candidate

    counter = 2
    while True:
        candidate = OUTPUTS_ROOT / f"{base_name}_{counter}"
        if not candidate.exists():
            candidate.mkdir(parents=True)
            return candidate
        counter += 1

def ensure_job_subdirs(output_dir):
    subdirs = {}
    for name in [
        "original",
        "frames",
        "overlays",
        "masks",
        "skeletons",
        "experimental",
        "results",
        "annotations",
        "previews",
        "difference_maps",
        "guided_masks",
        "guided_overlays",
        "propagation_debug",
        "temporal_masks",
        "temporal_overlays",
        "temporal_difference_maps",
        "corrections",
        "skeleton_debug",
        "setup",
        "fungi_masks",
        "bacteria_masks",
        "fungi_overlays",
        "bacteria_overlays",
        "bacteria_tracking",
    ]:
        path = Path(output_dir) / name
        path.mkdir(parents=True, exist_ok=True)
        subdirs[name] = path
    (Path(output_dir) / "annotations" / "previews").mkdir(parents=True, exist_ok=True)
    return subdirs

SAM2_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_CKPT = BACKEND_DIR / "sam2_checkpoints" / "sam2.1_hiera_large.pt"

MIN_OBJECT_SIZE_PX = 40
PIXEL_SIZE_UM = 1.0
FRAME_INTERVAL_MIN = 1.0
MIN_SKAN_BRANCH_LENGTH_PX = 8
BRANCH_CLUSTER_RADIUS_PX = 8

class CellSAMWrapper:
    def __init__(self, deepcell_token=None):
        self.available = False
        self.segment_func = None

        if deepcell_token:
            os.environ["DEEPCELL_ACCESS_TOKEN"] = deepcell_token

        if not os.environ.get("DEEPCELL_ACCESS_TOKEN"):
            print("[WARN] DEEPCELL_ACCESS_TOKEN not set; CellSAM weight download will fail if not cached.")

        try:
            from cellSAM import segment_cellular_image
            self.segment_func = segment_cellular_image
            self.available = True
            print("[CellSAM] Loaded wrapper successfully.")
        except Exception as e:
            print("[WARN] Could not import cellSAM:", e)

class SAM2Wrapper:
    def __init__(self):
        self.available = False
        self.predictor = None

        if build_sam2 is None or SAM2ImagePredictor is None:
            print("[WARN] SAM2 unavailable; zero-shot threshold mode still works.")
            return

        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[SAM2] Loading on {device}")
            # Patch for Mac (mps) to load the config from the repo path if needed
            cfg_path = "sam2.1_hiera_l.yaml"
            model = build_sam2(cfg_path, str(SAM2_CKPT), device=device)
            self.predictor = SAM2ImagePredictor(model)
            self.available = True
        except Exception as e:
            print("[WARN] Could not initialise SAM2:", e)
            self.available = False

    def predict(self, image_rgb, points=None, labels_=None, box=None):
        if not self.available:
            raise RuntimeError("SAM2 is not available.")

        points_np = None if not points else np.array(points, dtype=np.float32)
        labels_np = None if labels_ is None or len(labels_) == 0 else np.array(labels_, dtype=np.int32)
        box_np = None if box is None else np.array(box, dtype=np.float32)

        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        with torch.inference_mode():
            if device_type == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    self.predictor.set_image(image_rgb)
                    masks, scores, _ = self.predictor.predict(
                        point_coords=points_np,
                        point_labels=labels_np,
                        box=box_np,
                        multimask_output=True,
                    )
            else:
                self.predictor.set_image(image_rgb)
                masks, scores, _ = self.predictor.predict(
                    point_coords=points_np,
                    point_labels=labels_np,
                    box=box_np,
                    multimask_output=True,
                )

        best = masks[int(np.argmax(scores))]
        return best.astype(np.uint8)

# (Reusing processing functions from the original script)
def normalize_to_uint8(frame):
    arr = np.asarray(frame)
    if arr.ndim == 3 and arr.shape[-1] > 3:
        arr = arr[..., :3]

    if arr.dtype == np.uint8:
        out = arr
    else:
        arr = arr.astype(np.float32)
        lo, hi = np.percentile(arr, [1, 99.8])
        if hi <= lo:
            hi = arr.max() if arr.max() > lo else lo + 1
        out = np.clip((arr - lo) / (hi - lo), 0, 1)
        out = (out * 255).astype(np.uint8)

    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)
    elif out.shape[-1] == 4:
        out = out[..., :3]
    return out

def save_frame_preview(frame, preview_dir, index):
    preview_dir.mkdir(parents=True, exist_ok=True)
    preview_path = preview_dir / f"frame_{index:04d}.jpg"
    Image.fromarray(frame).save(preview_path, "JPEG", quality=85)
    return preview_path

def calc_progress_percent(stage, current_frame_index, total_frames):
    if total_frames <= 0:
        return 0
    ratio = (current_frame_index + 1) / total_frames
    if stage == "extracting_frames":
        return min(15, int(ratio * 15))
    if stage == "segmenting":
        return min(80, int(15 + ratio * 65))
    if stage == "temporal_overlays":
        return min(83, int(80 + ratio * 3))
    if stage == "temporal":
        return min(88, int(83 + ratio * 5))
    if stage == "tracking":
        return min(99, int(88 + ratio * 11))
    if stage == "finished":
        return 100
    return 0

def load_video_or_tiff(path, preview_dir=None, on_frame=None):
    path = Path(path)
    suffix = path.suffix.lower()
    frames = []

    def add_frame(frame):
        normalized = normalize_to_uint8(frame)
        index = len(frames)
        frames.append(normalized)
        if preview_dir is not None:
            save_frame_preview(normalized, preview_dir, index)
        if on_frame is not None:
            on_frame(index)
        return normalized

    if suffix in [".tif", ".tiff"]:
        data = tifffile.imread(str(path))
        data = np.asarray(data)

        if data.ndim == 2:
            add_frame(data)
        elif data.ndim == 3:
            if data.shape[-1] in [3, 4]:
                add_frame(data)
            else:
                for i in range(data.shape[0]):
                    add_frame(data[i])
        elif data.ndim == 4:
            if data.shape[-1] in [3, 4]:
                for i in range(data.shape[0]):
                    add_frame(data[i])
            else:
                for i in range(data.shape[0]):
                    add_frame(np.max(data[i], axis=0))
        else:
            raise ValueError(f"Unsupported TIFF shape: {data.shape}")
    else:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError("Could not open video file.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            add_frame(frame)
        cap.release()

    if not frames:
        raise ValueError("No frames loaded.")
    return frames

def load_frames_from_dir(frames_dir):
    frames_dir = Path(frames_dir)
    paths = sorted(frames_dir.glob("frame_*.jpg"))
    if not paths:
        raise ValueError("No extracted frames found.")
    frames = []
    for path in paths:
        frames.append(np.array(Image.open(path).convert("RGB")))
    return frames

def get_frame_count(frames_dir):
    return len(list(Path(frames_dir).glob("frame_*.jpg")))

def extract_frames_for_job(
    file_path,
    output_dir,
    job_id=None,
    progress_callback=None,
):
    output_dir = Path(output_dir)
    dirs = ensure_job_subdirs(output_dir)
    frames_dir = dirs["frames"]

    def report(stage, current_frame_index, total_frames):
        if progress_callback is None:
            return
        preview_url = None
        if job_id:
            preview_url = f"/api/jobs/{job_id}/frames/{current_frame_index}"
        progress_callback(
            stage=stage,
            current_frame_index=current_frame_index,
            total_frames=total_frames,
            current_frame_preview_url=preview_url,
            progress_percent=calc_progress_percent(stage, current_frame_index, total_frames),
        )

    def on_frame_extracted(index):
        report("extracting_frames", index, index + 1)

    load_video_or_tiff(Path(file_path), preview_dir=frames_dir, on_frame=on_frame_extracted)
    total_frames = get_frame_count(frames_dir)
    if total_frames > 0:
        report("extracting_frames", total_frames - 1, total_frames)
    return total_frames

def _segment_frame(fr, cellsam, dilation_radius, min_object_size_px, hole_fill_area, segmentation_preset="fungal_hyphae"):
    if cellsam.available:
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            raw_mask, _, _ = cellsam.segment_func(fr, device=device)
            return clean_mask(
                raw_mask,
                radius_px=dilation_radius,
                min_size=min_object_size_px,
                hole_fill_area=hole_fill_area,
                segmentation_preset=segmentation_preset,
            )
        except Exception as e:
            print(f"[WARN] CellSAM failed, falling back to zero-shot: {e}")
            traceback.print_exc()
    raw_mask = zero_shot_fluorescence_mask(fr)
    return clean_mask(
        raw_mask,
        radius_px=dilation_radius,
        min_size=min_object_size_px,
        hole_fill_area=hole_fill_area,
        segmentation_preset=segmentation_preset,
    )


def _segment_frame_with_setup(
    fr,
    cellsam,
    dilation_radius,
    min_object_size_px,
    hole_fill_area,
    setup_context=None,
    segmentation_preset="fungal_hyphae",
):
    from pre_segmentation_setup import apply_pre_segmentation_mask, crop_image, embed_crop_mask

    if setup_context is None:
        return _segment_frame(
            fr, cellsam, dilation_radius, min_object_size_px, hole_fill_area, segmentation_preset
        )

    bbox = setup_context.get("roi_bbox")
    allowed_mask = setup_context["allowed_mask"]

    if bbox is not None:
        crop = crop_image(fr, bbox)
        mask_crop = _segment_frame(
            crop,
            cellsam,
            dilation_radius,
            min_object_size_px,
            hole_fill_area,
            segmentation_preset,
        )
        mask = embed_crop_mask(mask_crop, bbox, fr.shape[:2])
    else:
        mask = _segment_frame(
            fr, cellsam, dilation_radius, min_object_size_px, hole_fill_area, segmentation_preset
        )

    return apply_pre_segmentation_mask(mask, allowed_mask)

def _save_mask_pngs(masks, output_dir, masks_subdir):
    masks_dir = Path(output_dir) / masks_subdir
    masks_dir.mkdir(parents=True, exist_ok=True)
    for i, mask in enumerate(masks):
        Image.fromarray((np.asarray(mask).astype(np.uint8) * 255)).save(
            masks_dir / f"mask_{i:04d}.png"
        )


def _save_lightweight_overlays(
    frames,
    masks,
    output_dir,
    overlays_subdir,
    job_id=None,
    progress_callback=None,
):
    """Save mask-only overlays quickly (no skeletonization)."""
    overlays_dir = Path(output_dir) / overlays_subdir
    overlays_dir.mkdir(parents=True, exist_ok=True)
    total_frames = len(frames)

    for i, (img, mask) in enumerate(zip(frames, masks)):
        if progress_callback:
            progress_callback(
                stage="temporal_overlays",
                current_frame_index=i,
                total_frames=total_frames,
                current_frame_preview_url=f"/api/jobs/{job_id}/preview/{i}" if job_id else None,
                progress_percent=calc_progress_percent("temporal_overlays", i, total_frames),
            )
        overlay = make_overlay(img, mask, skel=None, branches=None)
        Image.fromarray(overlay).save(overlays_dir / f"overlay_{i:04d}.png")


def _save_overlays_with_progress(
    frames,
    masks,
    output_dir,
    overlays_subdir,
    job_id=None,
    progress_callback=None,
    progress_stage="temporal",
    enable_skeletonization=True,
):
    overlays_dir = Path(output_dir) / overlays_subdir
    overlays_dir.mkdir(parents=True, exist_ok=True)
    total_frames = len(frames)

    for i, (img, mask) in enumerate(zip(frames, masks)):
        if progress_callback:
            progress_callback(
                stage=progress_stage,
                current_frame_index=i,
                total_frames=total_frames,
                current_frame_preview_url=f"/api/jobs/{job_id}/preview/{i}" if job_id else None,
                progress_percent=calc_progress_percent(progress_stage, i, total_frames),
            )
        if enable_skeletonization:
            skel, branches, _, _, _ = skeleton_and_branchpoints(mask, enable_analysis=True)
            overlay = make_overlay(img, mask, skel, branches)
        else:
            overlay = make_overlay(img, mask, skel=None, branches=None)
        Image.fromarray(overlay).save(overlays_dir / f"overlay_{i:04d}.png")


def _empty_branch_meta(mask_shape):
    skel_bool = np.zeros(mask_shape, dtype=bool)
    return {
        "raw_junction_pixel_count": 0,
        "normalized_branch_point_count": 0,
        "normalized_nodes": None,
        "raw_branch": skel_bool,
        "cluster_labels": np.zeros(mask_shape, dtype=np.int32),
        "skel_bool": skel_bool,
    }


def _empty_skel_outputs(mask):
    empty = np.zeros_like(mask, dtype=np.uint8)
    return empty, empty, empty, pd.DataFrame(), _empty_branch_meta(mask.shape)


def bacterial_frame_metrics(mask, frame_idx, pixel_size_um=1.0, frame_interval_min=1.0):
    mask_bool = np.asarray(mask).astype(bool)
    labeled = label(mask_bool)
    regions = [r for r in regionprops(labeled) if r.area > 0]
    areas = [int(r.area) for r in regions]
    count = len(areas)
    total_area = int(sum(areas))
    mean_area = float(total_area / count) if count else 0.0
    h, w = mask_bool.shape
    density = float(count / (h * w)) if h * w else 0.0
    px_um2 = pixel_size_um * pixel_size_um
    return {
        "frame": frame_idx,
        "time_min": frame_idx * frame_interval_min,
        "object_count": count,
        "mean_area_px": mean_area,
        "total_area_px": total_area,
        "object_density_per_px2": density,
        "mean_area_um2": mean_area * px_um2,
        "total_area_um2": total_area * px_um2,
    }


def bacterial_object_props(mask, frame_idx, min_object_size_px=1):
    return object_props(mask, frame_idx, min_object_size_px=max(1, int(min_object_size_px)))


def bacterial_growth_rates(metrics_df):
    if len(metrics_df) < 2:
        metrics_df["object_count_growth_per_min"] = np.nan
        metrics_df["total_area_growth_um2_per_min"] = np.nan
        return metrics_df
    metrics_df = metrics_df.sort_values("frame").copy()
    dt = metrics_df["time_min"].diff()
    metrics_df["object_count_growth_per_min"] = metrics_df["object_count"].diff() / dt
    metrics_df["total_area_growth_um2_per_min"] = metrics_df["total_area_um2"].diff() / dt
    return metrics_df


def _empty_fungal_metrics(frame_idx, frame_interval_min):
    return {
        "frame": frame_idx,
        "time_min": frame_idx * frame_interval_min,
        "hyphal_area_px": 0,
        "hyphal_area_um2": 0.0,
        "hyphal_length_px": 0.0,
        "hyphal_length_um": 0.0,
        "branch_points": 0,
        "raw_junction_pixel_count": 0,
        "normalized_branch_point_count": 0,
        "tip_count": 0,
        "skan_graph_branches": np.nan,
        "skan_terminal_branches": np.nan,
        "skan_junction_to_junction_branches": np.nan,
        "small_object_count": 0,
        "small_object_area_px": 0,
        "hyphal_object_count": 0,
        "hyphal_mask_area_px": 0,
    }


def make_workflow_overlay(image, fungi_mask, bacteria_mask, fungi_skel, fungi_branches, workflow):
    from annotations import render_simple_mask_overlay

    overlay = image.copy()
    fungi_mask = np.asarray(fungi_mask).astype(bool)
    bacteria_mask = np.asarray(bacteria_mask).astype(bool)
    if workflow.get("enable_fungal_processing") and fungi_mask.any():
        skel = fungi_skel if workflow.get("enable_skeletonization") else None
        branches = fungi_branches if workflow.get("enable_branch_detection") else None
        overlay = make_overlay(overlay, fungi_mask, skel, branches)
    if workflow.get("enable_bacterial_processing") and bacteria_mask.any():
        overlay = render_simple_mask_overlay(overlay, bacteria_mask, color=(0, 255, 255), alpha=0.45)
    if not fungi_mask.any() and not bacteria_mask.any():
        combined = fungi_mask | bacteria_mask
        if combined.any():
            overlay = make_overlay(overlay, combined, None, None)
    return overlay


def _finalize_job_outputs(
    frames,
    masks,
    output_dir,
    pixel_size_um,
    frame_interval_min,
    min_object_size_px,
    min_branch_length_px,
    job_id=None,
    progress_callback=None,
    annotations_dir=None,
    static_ann=None,
    save_annotation_previews=False,
    masks_subdir="masks",
    overlays_subdir="overlays",
    perf_logger=None,
):
    from annotations import (
        load_frame_annotation,
        save_annotation_preview,
        get_annotations_dir,
        load_job_config,
    )
    from object_classification import classify_mask_layers
    from performance_logger import PerformanceLogger
    from segmentation_config import get_workflow_config, PRESET_BACTERIA, PRESET_FUNGAL_HYPHAE, PRESET_MIXED
    from skeleton_branch_nodes import render_skeleton_overlay, save_skeleton_debug

    output_dir = Path(output_dir)
    dirs = ensure_job_subdirs(output_dir)
    masks_dir = dirs[masks_subdir] if masks_subdir in dirs else Path(output_dir) / masks_subdir
    masks_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir = dirs[overlays_subdir] if overlays_subdir in dirs else Path(output_dir) / overlays_subdir
    overlays_dir.mkdir(parents=True, exist_ok=True)
    skeletons_dir = dirs["skeletons"]
    experimental_dir = dirs["experimental"]
    results_dir = dirs["results"]
    fungi_masks_dir = dirs["fungi_masks"]
    bacteria_masks_dir = dirs["bacteria_masks"]
    fungi_overlays_dir = dirs["fungi_overlays"]
    bacteria_overlays_dir = dirs["bacteria_overlays"]
    total_frames = len(frames)

    if annotations_dir is None:
        annotations_dir = get_annotations_dir(output_dir)
    if static_ann is None:
        from annotations import load_static_background
        static_ann = load_static_background(annotations_dir)

    def report(stage, current_frame_index, total):
        if progress_callback is None:
            return
        preview_url = f"/api/jobs/{job_id}/preview/{current_frame_index}" if job_id else None
        progress_callback(
            stage=stage,
            current_frame_index=current_frame_index,
            total_frames=total,
            current_frame_preview_url=preview_url,
            progress_percent=calc_progress_percent(stage, current_frame_index, total),
        )

    job_cfg = load_job_config(annotations_dir)
    workflow = get_workflow_config(job_cfg)
    classification_cfg = workflow["classification"]
    target_type = workflow["target_object_type"]

    branch_merge_radius = int(job_cfg.get("branch_node_merge_radius_px", 6))
    branch_temporal_smoothing = bool(job_cfg.get("branch_node_temporal_smoothing", True))
    branch_max_track = int(job_cfg.get("branch_node_max_tracking_distance_px", 10))
    skeletonization_mode = job_cfg.get("skeletonization_mode", "hyphae_only")
    skeleton_min_object_area_px = int(job_cfg.get("skeleton_min_object_area_px", 40))

    perf = perf_logger or PerformanceLogger(output_dir)
    perf.log(
        f"[workflow] target_object_type={target_type} "
        f"skeleton={workflow['enable_skeletonization']} "
        f"branch={workflow['enable_branch_detection']} "
        f"temporal_traversed={workflow['enable_traversed_persistence']}"
    )

    fungi_metrics = []
    bacteria_metrics = []
    fungi_props = []
    bacteria_props = []
    legacy_metrics = []
    legacy_props = []
    overlay_frames = []
    bacteria_masks_per_frame: List[np.ndarray] = []
    prev_normalized_nodes = None
    summary_counts = {
        "total_objects": 0,
        "fungal_objects": 0,
        "bacterial_objects": 0,
        "skeletonized_objects": 0,
    }

    for i, (img, mask) in enumerate(zip(frames, masks)):
        report("tracking", i, total_frames)
        mask_bool = np.asarray(mask).astype(bool)

        with perf.timed("object_classification"):
            fungi_mask, bacteria_mask, cls_stats = classify_mask_layers(
                mask_bool, target_type, classification_cfg
            )

        summary_counts["total_objects"] += cls_stats["total_objects"]
        summary_counts["fungal_objects"] += cls_stats["fungal_objects"]
        summary_counts["bacterial_objects"] += cls_stats["bacterial_objects"]
        summary_counts["skeletonized_objects"] += cls_stats["skeletonized_objects"]
        perf.log_counts(
            i,
            cls_stats["total_objects"],
            cls_stats["fungal_objects"],
            cls_stats["bacterial_objects"],
            cls_stats["skeletonized_objects"],
        )

        skel = np.zeros_like(mask, dtype=np.uint8)
        branches = np.zeros_like(mask, dtype=np.uint8)
        branch_meta = _empty_branch_meta(mask.shape)
        fungi_skel = skel
        fungi_branches = branches

        run_skeleton = (
            workflow["enable_skeletonization"]
            and workflow["enable_fungal_processing"]
            and fungi_mask.any()
            and not cls_stats.get("force_disable_skeleton", False)
        )

        if workflow["enable_fungal_processing"]:
            if fungi_mask.any():
                if run_skeleton:
                    with perf.timed("skeletonization"):
                        met, skel, branches, branch_meta = frame_metrics(
                            fungi_mask,
                            i,
                            pixel_size_um=pixel_size_um,
                            frame_interval_min=frame_interval_min,
                            min_skan_branch_length_px=min_branch_length_px,
                            prev_normalized_nodes=prev_normalized_nodes,
                            merge_radius_px=branch_merge_radius,
                            temporal_smoothing=branch_temporal_smoothing,
                            max_tracking_distance_px=branch_max_track,
                            skeletonization_mode=skeletonization_mode,
                            skeleton_min_object_area_px=skeleton_min_object_area_px,
                            enable_skeletonization=True,
                            enable_branch_detection=workflow["enable_branch_detection"],
                            enable_tip_detection=workflow["enable_tip_detection"],
                        )
                    if workflow["enable_branch_detection"]:
                        with perf.timed("branch_detection"):
                            prev_normalized_nodes = branch_meta["normalized_nodes"]
                    fungi_skel, fungi_branches = skel, branches
                else:
                    met = _empty_fungal_metrics(i, frame_interval_min)
                    met["hyphal_area_px"] = int(fungi_mask.sum())
                    met["hyphal_area_um2"] = met["hyphal_area_px"] * pixel_size_um * pixel_size_um
                    met["hyphal_mask_area_px"] = met["hyphal_area_px"]
                    met["hyphal_object_count"] = cls_stats["fungal_objects"]
                fungi_metrics.append(met)
                fungi_props.extend(object_props(fungi_mask, i, min_object_size_px=min_object_size_px))
                if target_type == PRESET_FUNGAL_HYPHAE:
                    legacy_metrics.append(met)
                    legacy_props.extend(object_props(fungi_mask, i, min_object_size_px=min_object_size_px))
            else:
                met = _empty_fungal_metrics(i, frame_interval_min)
                fungi_metrics.append(met)
                if target_type == PRESET_FUNGAL_HYPHAE:
                    legacy_metrics.append(met)

        if workflow["enable_bacterial_processing"]:
            if bacteria_mask.any():
                bmet = bacterial_frame_metrics(
                    bacteria_mask, i, pixel_size_um=pixel_size_um, frame_interval_min=frame_interval_min
                )
                bacteria_metrics.append(bmet)
                bacteria_props.extend(bacterial_object_props(bacteria_mask, i, min_object_size_px=1))
            elif target_type == PRESET_BACTERIA:
                bacteria_metrics.append(
                    bacterial_frame_metrics(
                        mask_bool, i, pixel_size_um=pixel_size_um, frame_interval_min=frame_interval_min
                    )
                )
            elif target_type != PRESET_FUNGAL_HYPHAE:
                bacteria_metrics.append(
                    bacterial_frame_metrics(
                        bacteria_mask, i, pixel_size_um=pixel_size_um, frame_interval_min=frame_interval_min
                    )
                )

        if workflow["enable_bacterial_processing"]:
            bacteria_masks_per_frame.append(bacteria_mask.copy())

        tifffile.imwrite(str(masks_dir / f"mask_{i:04d}.tif"), mask_bool.astype(np.uint8) * 255)
        Image.fromarray((mask_bool.astype(np.uint8) * 255)).save(masks_dir / f"mask_{i:04d}.png")

        if workflow["enable_fungal_processing"]:
            fungi_u8 = (fungi_mask.astype(np.uint8) * 255)
            tifffile.imwrite(str(fungi_masks_dir / f"mask_{i:04d}.tif"), fungi_u8)
            Image.fromarray(fungi_u8).save(fungi_masks_dir / f"mask_{i:04d}.png")
        if workflow["enable_bacterial_processing"]:
            bacteria_u8 = (bacteria_mask.astype(np.uint8) * 255)
            tifffile.imwrite(str(bacteria_masks_dir / f"mask_{i:04d}.tif"), bacteria_u8)
            Image.fromarray(bacteria_u8).save(bacteria_masks_dir / f"mask_{i:04d}.png")

        if run_skeleton:
            tifffile.imwrite(str(skeletons_dir / f"skeleton_{i:04d}.tif"), skel.astype(np.uint8) * 255)
            tifffile.imwrite(
                str(experimental_dir / f"branches_{i:04d}.tif"),
                branches.astype(np.uint8) * 255,
            )
            skel_bool = branch_meta["skel_bool"]
            skeleton_overlay = render_skeleton_overlay(
                skel_bool,
                branch_meta["normalized_nodes"],
                raw_branch=branch_meta["raw_branch"],
                debug_raw_junctions=False,
            )
            Image.fromarray(skeleton_overlay).save(skeletons_dir / f"skeleton_{i:04d}.png")
            save_skeleton_debug(
                output_dir,
                i,
                skel_bool,
                branch_meta["raw_branch"],
                branch_meta["cluster_labels"],
                branch_meta["normalized_nodes"],
                skeleton_overlay,
            )

        overlay = make_workflow_overlay(
            img, fungi_mask, bacteria_mask, fungi_skel, fungi_branches, workflow
        )
        Image.fromarray(overlay).save(overlays_dir / f"overlay_{i:04d}.png")
        overlay_frames.append(overlay)

        if workflow["enable_fungal_processing"] and fungi_mask.any():
            fungi_overlay = make_workflow_overlay(
                img, fungi_mask, np.zeros_like(fungi_mask), fungi_skel, fungi_branches, workflow
            )
            Image.fromarray(fungi_overlay).save(fungi_overlays_dir / f"overlay_{i:04d}.png")
        if workflow["enable_bacterial_processing"] and bacteria_mask.any():
            from annotations import render_simple_mask_overlay
            bacteria_overlay = render_simple_mask_overlay(img, bacteria_mask, color=(0, 255, 255), alpha=0.45)
            Image.fromarray(bacteria_overlay).save(bacteria_overlays_dir / f"overlay_{i:04d}.png")

        if save_annotation_previews:
            frame_ann = load_frame_annotation(annotations_dir, i, img.shape[1], img.shape[0])
            save_annotation_preview(annotations_dir, i, img, frame_ann, static_ann)

    perf.log_counts(
        None,
        summary_counts["total_objects"],
        summary_counts["fungal_objects"],
        summary_counts["bacterial_objects"],
        summary_counts["skeletonized_objects"],
        extra="job_total",
    )

    result_paths = {"output_dir": str(output_dir)}

    if workflow["enable_fungal_processing"] and fungi_metrics:
        fungi_df = growth_rates(pd.DataFrame(fungi_metrics))
        with perf.timed("tracking"):
            fungi_tracks = track_objects(fungi_props)
        fungi_metrics_csv = Path(output_dir) / "fungi_metrics.csv"
        fungi_tracks_csv = results_dir / "fungi_object_tracks.csv"
        fungi_df.to_csv(fungi_metrics_csv, index=False)
        pd.DataFrame(fungi_tracks).to_csv(fungi_tracks_csv, index=False)
        result_paths["fungi_metrics_csv"] = str(fungi_metrics_csv)
        result_paths["fungi_tracks_csv"] = str(fungi_tracks_csv)
        if target_type in (PRESET_FUNGAL_HYPHAE, PRESET_MIXED):
            legacy_df = fungi_df
            legacy_tracks = fungi_tracks
            metrics_csv_path = results_dir / "hyphal_metrics.csv"
            tracks_csv_path = results_dir / "object_tracks.csv"
            metrics_json_path = results_dir / "hyphal_metrics.json"
            legacy_df.to_csv(metrics_csv_path, index=False)
            pd.DataFrame(legacy_tracks).to_csv(tracks_csv_path, index=False)
            legacy_df.to_json(metrics_json_path, orient="records", indent=2)
            result_paths["metrics_csv"] = str(metrics_csv_path)
            result_paths["tracks_csv"] = str(tracks_csv_path)
            result_paths["metrics_json"] = str(metrics_json_path)

    if workflow["enable_bacterial_processing"] and bacteria_metrics:
        bacteria_df = bacterial_growth_rates(pd.DataFrame(bacteria_metrics))
        bacteria_metrics_csv = Path(output_dir) / "bacteria_metrics.csv"
        bacteria_df.to_csv(bacteria_metrics_csv, index=False)
        result_paths["bacteria_metrics_csv"] = str(bacteria_metrics_csv)
        if target_type == PRESET_BACTERIA or (
            target_type == PRESET_MIXED and not result_paths.get("metrics_csv")
        ):
            result_paths["metrics_csv"] = str(bacteria_metrics_csv)

        from bacterial_tracking import merge_bacterial_tracking_config, run_bacterial_tracking_pipeline

        tracking_cfg = merge_bacterial_tracking_config(job_cfg)
        if tracking_cfg.get("enable_bacterial_tracking", True) and bacteria_masks_per_frame:
            tracking_paths = run_bacterial_tracking_pipeline(
                frames,
                bacteria_masks_per_frame,
                output_dir,
                pixel_size_um=pixel_size_um,
                frame_interval_min=frame_interval_min,
                config=tracking_cfg,
                perf_logger=perf,
                base_overlays=overlay_frames if overlay_frames else None,
            )
            result_paths.update(tracking_paths)
            if tracking_paths.get("bacterial_tracks_csv"):
                result_paths["bacteria_tracks_csv"] = tracking_paths["bacterial_tracks_csv"]
                if target_type == PRESET_BACTERIA:
                    result_paths["tracks_csv"] = tracking_paths["bacterial_tracks_csv"]
        else:
            with perf.timed("tracking"):
                bacteria_tracks = track_objects(bacteria_props, max_dist=40)
            bacteria_tracks_csv = results_dir / "bacteria_object_tracks.csv"
            pd.DataFrame(bacteria_tracks).to_csv(bacteria_tracks_csv, index=False)
            result_paths["bacteria_tracks_csv"] = str(bacteria_tracks_csv)
            if target_type == PRESET_BACTERIA:
                result_paths["tracks_csv"] = str(bacteria_tracks_csv)

    if overlay_frames:
        with perf.timed("video_generation"):
            h, w = overlay_frames[0].shape[:2]
            video_path = results_dir / "segmentation_overlay.mp4"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"avc1"),
                max(1, int(60 / max(frame_interval_min, 1))),
                (w, h),
            )
            for fr in overlay_frames:
                writer.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
            writer.release()
            result_paths["video"] = str(video_path)

    perf.finish()
    result_paths["performance_log"] = str(perf.path)
    report("finished", total_frames - 1, total_frames)
    return result_paths

def zero_shot_fluorescence_mask(image_rgb):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    try:
        thr = threshold_otsu(blur)
    except Exception:
        thr = np.percentile(blur, 90)
    mask = blur > thr
    return mask.astype(np.uint8)

def reconstruct_hypha_tube(mask, radius_px=6, nearby_margin_px=10, min_size=40, hole_fill_area=200):
    mask = np.asarray(mask).astype(bool)
    min_size = max(1, int(min_size))
    working = binary_dilation(mask, disk(2))
    working = binary_closing(working, disk(3))
    if hole_fill_area > 0:
        working = remove_small_holes(working, area_threshold=hole_fill_area)
    skel = skeletonize(working)
    skel = binary_dilation(skel, disk(1))
    skel = binary_closing(skel, disk(3))
    skel = skeletonize(skel)
    tube = binary_dilation(skel, disk(max(1, int(radius_px))))
    nearby = binary_dilation(working, disk(max(1, int(nearby_margin_px))))
    tube = tube & nearby
    tube = binary_closing(tube, disk(2))
    if hole_fill_area > 0:
        tube = remove_small_holes(
            tube, area_threshold=max(hole_fill_area, int(hole_fill_area * 2.5))
        )
    tube = remove_small_objects(tube, min_size=min_size)
    return tube.astype(np.uint8)


def clean_mask(
    mask,
    radius_px=8,
    nearby_margin_px=10,
    min_size=40,
    hole_fill_area=200,
    segmentation_preset="fungal_hyphae",
):
    from segmentation_config import cleanup_profile

    min_size = max(1, int(min_size))
    profile = cleanup_profile(segmentation_preset)
    mask = np.asarray(mask).astype(bool)
    mask = remove_small_objects(mask, min_size=min_size)

    if profile == "bacterial":
        if hole_fill_area > 0:
            mask = remove_small_holes(mask, area_threshold=hole_fill_area)
        if radius_px > 0:
            mask = binary_closing(mask, disk(max(1, int(radius_px))))
        return mask.astype(np.uint8)

    if profile == "mixed":
        mask = binary_dilation(mask, disk(max(1, min(2, int(radius_px)))))
        mask = binary_closing(mask, disk(max(1, min(3, int(radius_px)))))
        if hole_fill_area > 0:
            mask = remove_small_holes(mask, area_threshold=hole_fill_area)
        return reconstruct_hypha_tube(
            mask,
            radius_px=max(1, min(int(radius_px), 4)),
            nearby_margin_px=max(4, int(nearby_margin_px) // 2),
            min_size=min_size,
            hole_fill_area=hole_fill_area,
        )

    mask = binary_dilation(mask, disk(2))
    mask = binary_closing(mask, disk(3))
    if hole_fill_area > 0:
        mask = remove_small_holes(mask, area_threshold=hole_fill_area)
    mask = reconstruct_hypha_tube(
        mask,
        radius_px=radius_px,
        nearby_margin_px=nearby_margin_px,
        min_size=min_size,
        hole_fill_area=hole_fill_area,
    )
    return mask.astype(np.uint8)


def _is_hyphal_component(region, min_hyphal_area=40):
    if region.area < max(1, int(min_hyphal_area)):
        return False
    eccentricity = float(getattr(region, "eccentricity", 0.0))
    major = float(getattr(region, "major_axis_length", 0.0))
    minor = max(float(getattr(region, "minor_axis_length", 1.0)), 1.0)
    aspect = major / minor
    return eccentricity >= 0.7 or aspect >= 2.5 or major >= 12.0


def classify_mask_objects(mask_bool, min_hyphal_area=40):
    """Separate small/isolated objects from hyphal-like structures."""
    labeled = label(mask_bool.astype(bool))
    small_object_count = 0
    small_object_area_px = 0
    hyphal_object_count = 0
    hyphal_mask_area_px = 0
    for region in regionprops(labeled):
        if _is_hyphal_component(region, min_hyphal_area=min_hyphal_area):
            hyphal_object_count += 1
            hyphal_mask_area_px += int(region.area)
        else:
            small_object_count += 1
            small_object_area_px += int(region.area)
    return {
        "small_object_count": small_object_count,
        "small_object_area_px": small_object_area_px,
        "hyphal_object_count": hyphal_object_count,
        "hyphal_mask_area_px": hyphal_mask_area_px,
    }


def skeleton_target_mask(
    mask_bool,
    skeletonization_mode="hyphae_only",
    skeleton_min_object_area_px=40,
):
    from segmentation_config import SKELETON_MODE_ALL_OBJECTS

    mask_bool = mask_bool.astype(bool)
    if skeletonization_mode == SKELETON_MODE_ALL_OBJECTS:
        return mask_bool

    labeled = label(mask_bool)
    target = np.zeros_like(mask_bool, dtype=bool)
    for region in regionprops(labeled):
        if region.area < max(1, int(skeleton_min_object_area_px)):
            continue
        if _is_hyphal_component(region, min_hyphal_area=skeleton_min_object_area_px):
            target[labeled == region.label] = True
    return target

def _neighbor_count(skel_bool):
    skel_u8 = skel_bool.astype(np.uint8)
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    return cv2.filter2D(skel_u8, -1, kernel)

def _remove_short_skeleton_spurs_with_skan(skel_bool, min_branch_length_px=MIN_SKAN_BRANCH_LENGTH_PX):
    if not SKAN_AVAILABLE or skel_bool.sum() == 0:
        return skel_bool
    try:
        sk = Skeleton(skel_bool.astype(bool))
        df = summarize(sk)
        pruned = skel_bool.copy()
        for branch_idx, row in df.iterrows():
            branch_type = int(row.get("branch-type", -1))
            branch_distance = float(row.get("branch-distance", row.get("euclidean-distance", 0)))
            if branch_type == 1 and branch_distance < min_branch_length_px:
                coords = sk.path_coordinates(branch_idx)
                rr = np.clip(coords[:, 0].astype(int), 0, pruned.shape[0] - 1)
                cc = np.clip(coords[:, 1].astype(int), 0, pruned.shape[1] - 1)
                pruned[rr, cc] = False
        pruned = skeletonize(pruned)
        return pruned.astype(bool)
    except Exception as e:
        return skel_bool

def _cluster_tip_pixels(endpoint_mask, min_cluster_size=1):
    """Cluster degree-1 tip pixels; tips are not merged with branch nodes."""
    lab = label(endpoint_mask.astype(bool), connectivity=2)
    clustered = np.zeros_like(endpoint_mask, dtype=np.uint8)
    count = 0
    for r in regionprops(lab):
        if r.area < min_cluster_size:
            continue
        cy, cx = r.centroid
        clustered[int(round(cy)), int(round(cx))] = 1
        count += 1
    return clustered, count

def skeleton_and_branchpoints(
    mask,
    min_skan_branch_length_px=8,
    prev_normalized_nodes=None,
    merge_radius_px=None,
    temporal_smoothing=None,
    max_tracking_distance_px=None,
    skeletonization_mode="hyphae_only",
    skeleton_min_object_area_px=40,
    enable_analysis=True,
    enable_branch_detection=True,
    enable_tip_detection=True,
):
    from skeleton_branch_nodes import (
        BRANCH_NODE_MAX_TRACKING_DISTANCE_PX,
        BRANCH_NODE_MERGE_RADIUS_PX,
        BRANCH_NODE_TEMPORAL_SMOOTHING,
        normalize_branch_nodes,
    )

    mask_bool = np.asarray(mask).astype(bool)
    if not enable_analysis or not mask_bool.any():
        return _empty_skel_outputs(mask_bool)

    if merge_radius_px is None:
        merge_radius_px = BRANCH_NODE_MERGE_RADIUS_PX
    if temporal_smoothing is None:
        temporal_smoothing = BRANCH_NODE_TEMPORAL_SMOOTHING
    if max_tracking_distance_px is None:
        max_tracking_distance_px = BRANCH_NODE_MAX_TRACKING_DISTANCE_PX

    skel_target = skeleton_target_mask(
        mask_bool,
        skeletonization_mode=skeletonization_mode,
        skeleton_min_object_area_px=skeleton_min_object_area_px,
    )
    skel_bool = skeletonize(skel_target) if np.any(skel_target) else np.zeros_like(mask_bool, dtype=bool)
    skel_bool = _remove_short_skeleton_spurs_with_skan(skel_bool, min_branch_length_px=min_skan_branch_length_px)
    skel_u8 = skel_bool.astype(np.uint8)

    skan_summary_df = pd.DataFrame()
    if SKAN_AVAILABLE and skel_bool.sum() > 0:
        try:
            sk = Skeleton(skel_bool)
            skan_summary_df = summarize(sk)
        except Exception:
            pass

    if not enable_branch_detection:
        endpoints_u8 = np.zeros_like(skel_u8, dtype=np.uint8)
        branch_meta = {
            "raw_junction_pixel_count": 0,
            "normalized_branch_point_count": 0,
            "normalized_nodes": prev_normalized_nodes,
            "raw_branch": np.zeros_like(mask_bool, dtype=bool),
            "cluster_labels": np.zeros(mask_bool.shape, dtype=np.int32),
            "skel_bool": skel_bool,
        }
        return skel_u8, np.zeros_like(skel_u8), endpoints_u8, skan_summary_df, branch_meta

    neigh = _neighbor_count(skel_bool)
    raw_branch = skel_bool & (neigh >= 3)
    branch_info = normalize_branch_nodes(
        raw_branch,
        skel_bool,
        prev_nodes=prev_normalized_nodes,
        merge_radius_px=merge_radius_px,
        temporal_smoothing=temporal_smoothing,
        max_tracking_distance_px=max_tracking_distance_px,
    )
    branches_u8 = branch_info["branches_mask"].astype(np.uint8)

    if enable_tip_detection:
        raw_endpoints = skel_bool & (neigh == 1)
        endpoints_u8, _ = _cluster_tip_pixels(raw_endpoints)
    else:
        endpoints_u8 = np.zeros_like(skel_u8, dtype=np.uint8)

    branch_meta = {
        "raw_junction_pixel_count": branch_info["raw_junction_pixel_count"],
        "normalized_branch_point_count": branch_info["normalized_branch_point_count"],
        "normalized_nodes": branch_info["normalized_nodes"],
        "raw_branch": raw_branch,
        "cluster_labels": branch_info["cluster_labels"],
        "skel_bool": skel_bool,
    }

    return skel_u8, branches_u8, endpoints_u8.astype(np.uint8), skan_summary_df, branch_meta

def frame_metrics(
    mask,
    frame_idx,
    pixel_size_um=1.0,
    frame_interval_min=1.0,
    min_skan_branch_length_px=8,
    prev_normalized_nodes=None,
    merge_radius_px=None,
    temporal_smoothing=None,
    max_tracking_distance_px=None,
    skeletonization_mode="hyphae_only",
    skeleton_min_object_area_px=40,
    enable_skeletonization=True,
    enable_branch_detection=True,
    enable_tip_detection=True,
):
    skel, branches, endpoints, skan_df, branch_meta = skeleton_and_branchpoints(
        mask,
        min_skan_branch_length_px,
        prev_normalized_nodes=prev_normalized_nodes,
        merge_radius_px=merge_radius_px,
        temporal_smoothing=temporal_smoothing,
        max_tracking_distance_px=max_tracking_distance_px,
        skeletonization_mode=skeletonization_mode,
        skeleton_min_object_area_px=skeleton_min_object_area_px,
        enable_analysis=enable_skeletonization,
        enable_branch_detection=enable_branch_detection,
        enable_tip_detection=enable_tip_detection,
    )
    area_px = int(mask.sum())
    object_stats = classify_mask_objects(
        mask.astype(bool),
        min_hyphal_area=skeleton_min_object_area_px,
    )

    if SKAN_AVAILABLE and skan_df is not None and len(skan_df) > 0 and "branch-distance" in skan_df.columns:
        length_px = float(skan_df["branch-distance"].sum())
        skan_branch_count = int(len(skan_df))
        terminal_branch_count = int((skan_df["branch-type"] == 1).sum()) if "branch-type" in skan_df.columns else np.nan
        junction_to_junction_count = int((skan_df["branch-type"] == 2).sum()) if "branch-type" in skan_df.columns else np.nan
    else:
        length_px = float(skel.sum())
        skan_branch_count = np.nan
        terminal_branch_count = np.nan
        junction_to_junction_count = np.nan

    branch_count = int(branch_meta["normalized_branch_point_count"])
    tip_count = int(endpoints.sum())

    return {
        "frame": frame_idx,
        "time_min": frame_idx * frame_interval_min,
        "hyphal_area_px": area_px,
        "hyphal_area_um2": area_px * pixel_size_um * pixel_size_um,
        "hyphal_length_px": length_px,
        "hyphal_length_um": length_px * pixel_size_um,
        "branch_points": branch_count,
        "raw_junction_pixel_count": int(branch_meta["raw_junction_pixel_count"]),
        "normalized_branch_point_count": branch_count,
        "tip_count": tip_count,
        "skan_graph_branches": skan_branch_count,
        "skan_terminal_branches": terminal_branch_count,
        "skan_junction_to_junction_branches": junction_to_junction_count,
        "small_object_count": int(object_stats["small_object_count"]),
        "small_object_area_px": int(object_stats["small_object_area_px"]),
        "hyphal_object_count": int(object_stats["hyphal_object_count"]),
        "hyphal_mask_area_px": int(object_stats["hyphal_mask_area_px"]),
    }, skel, branches, branch_meta

def object_props(mask, frame_idx, min_object_size_px=40):
    lab = label(mask)
    props = []
    for r in regionprops(lab):
        if r.area < min_object_size_px:
            continue
        y, x = r.centroid
        props.append({
            "frame": frame_idx,
            "label": int(r.label),
            "centroid_x": float(x),
            "centroid_y": float(y),
            "area_px": int(r.area),
            "bbox_minr": int(r.bbox[0]),
            "bbox_minc": int(r.bbox[1]),
            "bbox_maxr": int(r.bbox[2]),
            "bbox_maxc": int(r.bbox[3]),
        })
    return props

def track_objects(all_props, max_dist=80):
    tracks = []
    next_id = 1
    active = {}
    frames = sorted(set(p["frame"] for p in all_props))
    by_frame = {f: [p for p in all_props if p["frame"] == f] for f in frames}

    for f in frames:
        detections = by_frame[f]
        if not active:
            for d in detections:
                d["track_id"] = next_id
                active[next_id] = d
                next_id += 1
                tracks.append(d)
            continue

        active_ids = list(active.keys())
        prev = [active[i] for i in active_ids]

        if len(detections) == 0:
            active = {}
            continue

        cost = np.zeros((len(prev), len(detections)), dtype=float)
        for i, p in enumerate(prev):
            for j, d in enumerate(detections):
                dx = p["centroid_x"] - d["centroid_x"]
                dy = p["centroid_y"] - d["centroid_y"]
                cost[i, j] = math.sqrt(dx * dx + dy * dy)

        rows, cols = linear_sum_assignment(cost)
        assigned_det = set()
        new_active = {}

        for r, c in zip(rows, cols):
            if cost[r, c] <= max_dist:
                tid = active_ids[r]
                detections[c]["track_id"] = tid
                new_active[tid] = detections[c]
                assigned_det.add(c)
                tracks.append(detections[c])

        for j, d in enumerate(detections):
            if j not in assigned_det:
                d["track_id"] = next_id
                new_active[next_id] = d
                next_id += 1
                tracks.append(d)

        active = new_active

    return tracks

def growth_rates(metrics_df):
    if len(metrics_df) < 2:
        metrics_df["length_growth_um_per_min"] = np.nan
        metrics_df["area_growth_um2_per_min"] = np.nan
        metrics_df["branch_growth_per_min"] = np.nan
        return metrics_df

    metrics_df = metrics_df.sort_values("frame").copy()
    dt = metrics_df["time_min"].diff()
    metrics_df["length_growth_um_per_min"] = metrics_df["hyphal_length_um"].diff() / dt
    metrics_df["area_growth_um2_per_min"] = metrics_df["hyphal_area_um2"].diff() / dt
    metrics_df["branch_growth_per_min"] = metrics_df["branch_points"].diff() / dt
    return metrics_df

def make_overlay(image, mask, skel=None, branches=None):
    overlay = image.copy()
    mask_bool = mask.astype(bool)
    overlay[mask_bool] = (0.55 * overlay[mask_bool] + 0.45 * np.array([0, 255, 0])).astype(np.uint8)

    if skel is not None:
        overlay[skel.astype(bool)] = np.array([255, 0, 0], dtype=np.uint8)

    if branches is not None:
        ys, xs = np.where(branches.astype(bool))
        for x, y in zip(xs, ys):
            cv2.circle(overlay, (x, y), 3, (255, 255, 0), -1)

    return overlay

def process_file(
    file_path: Path,
    pixel_size_um: float = 1.0,
    frame_interval_min: float = 1.0,
    min_object_size_px: int = 40,
    dilation_radius: int = 8,
    hole_fill_area: int = 200,
    min_branch_length_px: int = 8,
    deepcell_token: str = None,
    job_id: str = None,
    output_dir: Path = None,
    progress_callback=None,
    frames_preloaded: bool = False,
    segmentation_preset: str = "fungal_hyphae",
    skeletonization_mode: str = "hyphae_only",
    skeleton_min_object_area_px: int = 40,
):
    """
    Main entrypoint for the API.
    Loads frames, runs CellSAM segmentation (or zero-shot threshold fallback), and saves outputs.
    """
    if output_dir is not None:
        output_dir = Path(output_dir)
        dirs = ensure_job_subdirs(output_dir)
        frames_dir = dirs["frames"]
    else:
        output_dir = OUT_DIR
        frames_dir = None
        for d in [OUT_DIR / "masks", OUT_DIR / "overlays", OUT_DIR / "skeletons", OUT_DIR / "experimental"]:
            d.mkdir(exist_ok=True)

    preview_dir = frames_dir

    def report_progress(stage, current_frame_index, total_frames):
        if progress_callback is None:
            return
        preview_url = None
        if job_id and preview_dir is not None:
            preview_url = f"/api/jobs/{job_id}/preview/{current_frame_index}"
        progress_callback(
            stage=stage,
            current_frame_index=current_frame_index,
            total_frames=total_frames,
            current_frame_preview_url=preview_url,
            progress_percent=calc_progress_percent(stage, current_frame_index, total_frames),
        )

    if frames_preloaded and frames_dir is not None and get_frame_count(frames_dir) > 0:
        frames = load_frames_from_dir(frames_dir)
        total_frames = len(frames)
    else:
        def on_frame_extracted(index):
            report_progress("extracting_frames", index, index + 1)

        frames = load_video_or_tiff(file_path, preview_dir=preview_dir, on_frame=on_frame_extracted)
        total_frames = len(frames)
        if total_frames > 0:
            report_progress("extracting_frames", total_frames - 1, total_frames)

    masks = [np.zeros(frames[0].shape[:2], dtype=np.uint8) for _ in frames]
    cellsam = CellSAMWrapper(deepcell_token)

    from pre_segmentation_setup import apply_pre_segmentation_mask, load_setup_context
    from performance_logger import PerformanceLogger

    setup_context = load_setup_context(output_dir, frames[0].shape) if output_dir is not None else None
    seg_perf = PerformanceLogger(output_dir) if output_dir is not None else None

    with (seg_perf.timed("segmentation") if seg_perf else nullcontext()):
        for i, fr in enumerate(frames):
            report_progress("segmenting", i, total_frames)
            if cellsam.available:
                print(f"[CellSAM] Segmenting frame {i + 1}/{len(frames)}")
            else:
                print(f"[Zero-shot] Segmenting frame {i + 1}/{len(frames)}")
            masks[i] = _segment_frame_with_setup(
                fr,
                cellsam,
                dilation_radius,
                min_object_size_px,
                hole_fill_area,
                setup_context=setup_context,
                segmentation_preset=segmentation_preset,
            )

    if setup_context is not None:
        allowed_mask = setup_context["allowed_mask"]
        masks = [apply_pre_segmentation_mask(m, allowed_mask) for m in masks]

    from annotations import get_annotations_dir, load_job_config, load_static_background, save_job_config
    from temporal_continuity import merge_temporal_config, run_temporal_continuity_pipeline
    from performance_logger import PerformanceLogger
    from segmentation_config import get_workflow_config, preset_values, normalize_preset_name

    annotations_dir = get_annotations_dir(output_dir)
    static_ann = load_static_background(annotations_dir, frames[0].shape[1], frames[0].shape[0])
    job_cfg = load_job_config(annotations_dir)
    preset = normalize_preset_name(segmentation_preset or job_cfg.get("target_object_type"))
    job_cfg["target_object_type"] = preset
    job_cfg["segmentation_preset"] = preset
    job_cfg["skeletonization_mode"] = skeletonization_mode
    job_cfg["skeleton_min_object_area_px"] = skeleton_min_object_area_px
    job_cfg["min_object_size_px"] = min_object_size_px
    save_job_config(annotations_dir, job_cfg)
    workflow = get_workflow_config(job_cfg)
    perf = PerformanceLogger(output_dir)

    temporal_config = merge_temporal_config(job_cfg)
    temporal_config["min_object_size_px"] = min_object_size_px
    temporal_config["repair_disconnected_tubes"] = (
        bool(temporal_config.get("repair_disconnected_tubes"))
        and workflow["enable_tube_repair"]
    )
    temporal_config["enable_traversed_persistence"] = workflow["enable_traversed_persistence"]
    if not workflow["enable_fungal_processing"]:
        temporal_config["allow_tip_growth"] = False

    working_masks = masks
    masks_subdir = "masks"
    overlays_subdir = "overlays"
    temporal_warning = None

    if temporal_config["use_temporal_continuity"]:
        _save_mask_pngs(masks, output_dir, "masks")
        _save_lightweight_overlays(
            frames,
            masks,
            output_dir,
            "overlays",
            job_id=job_id,
            progress_callback=progress_callback,
        )
        with perf.timed("temporal_continuity"):
            working_masks, temporal_info = run_temporal_continuity_pipeline(
                frames,
                masks,
                output_dir,
                static_ann,
                temporal_config,
                job_id=job_id,
                progress_callback=progress_callback,
            )
        if temporal_info.get("fallback"):
            temporal_warning = temporal_info.get("warning")
            working_masks = masks
            masks_subdir = "masks"
            overlays_subdir = "overlays"
        else:
            masks_subdir = "temporal_masks"
            overlays_subdir = "temporal_overlays"

    results = _finalize_job_outputs(
        frames,
        working_masks,
        output_dir,
        pixel_size_um,
        frame_interval_min,
        min_object_size_px,
        min_branch_length_px,
        job_id=job_id,
        progress_callback=progress_callback,
        annotations_dir=annotations_dir,
        static_ann=static_ann,
        masks_subdir=masks_subdir,
        overlays_subdir=overlays_subdir,
        perf_logger=perf,
    )
    if temporal_warning:
        results["temporal_warning"] = temporal_warning
    return results


def process_file_guided(
    file_path: Path,
    pixel_size_um: float = 1.0,
    frame_interval_min: float = 1.0,
    min_object_size_px: int = 40,
    dilation_radius: int = 8,
    hole_fill_area: int = 200,
    min_branch_length_px: int = 8,
    deepcell_token: str = None,
    job_id: str = None,
    output_dir: Path = None,
    progress_callback=None,
    annotation_mode: str = "keyframes",
):
    from annotations import get_annotations_dir, load_static_background, save_global_ignore_mask
    from guided_propagation import build_guided_masks_with_propagation

    output_dir = Path(output_dir)
    dirs = ensure_job_subdirs(output_dir)
    frames_dir = dirs["frames"]
    masks_dir = dirs["masks"]
    annotations_dir = get_annotations_dir(output_dir)

    frames = load_frames_from_dir(frames_dir)
    total_frames = len(frames)
    image_h, image_w = frames[0].shape[:2]
    static_ann = load_static_background(annotations_dir, image_w, image_h)
    save_global_ignore_mask(annotations_dir, static_ann, (image_h, image_w))

    def report_progress(stage, current_frame_index, total):
        if progress_callback is None:
            return
        preview_url = f"/api/jobs/{job_id}/preview/{current_frame_index}" if job_id else None
        progress_callback(
            stage=stage,
            current_frame_index=current_frame_index,
            total_frames=total,
            current_frame_preview_url=preview_url,
            progress_percent=calc_progress_percent(stage, current_frame_index, total),
        )

    from pre_segmentation_setup import apply_pre_segmentation_mask, load_setup_context

    setup_context = load_setup_context(output_dir, frames[0].shape)

    from annotations import load_job_config

    job_cfg = load_job_config(annotations_dir)
    segmentation_preset = job_cfg.get("segmentation_preset", "fungal_hyphae")

    auto_masks = []
    cellsam = None
    for i, fr in enumerate(frames):
        mask_path = masks_dir / f"mask_{i:04d}.png"
        if mask_path.exists():
            auto_masks.append((np.array(Image.open(mask_path)) > 127).astype(np.uint8))
        else:
            if cellsam is None:
                cellsam = CellSAMWrapper(deepcell_token)
            auto_masks.append(
                _segment_frame_with_setup(
                    fr,
                    cellsam,
                    dilation_radius,
                    min_object_size_px,
                    hole_fill_area,
                    setup_context=setup_context,
                    segmentation_preset=segmentation_preset,
                )
            )

    auto_masks = [
        apply_pre_segmentation_mask(m, setup_context["allowed_mask"]) for m in auto_masks
    ]

    guided_masks, _, _ = build_guided_masks_with_propagation(
        frames,
        auto_masks,
        output_dir,
        progress_callback=progress_callback,
        job_id=job_id,
    )
    guided_masks = [
        apply_pre_segmentation_mask(m, setup_context["allowed_mask"]) for m in guided_masks
    ]

    from annotations import load_job_config
    from temporal_continuity import merge_temporal_config, run_temporal_continuity_pipeline
    from performance_logger import PerformanceLogger
    from segmentation_config import get_workflow_config

    job_cfg = load_job_config(annotations_dir)
    workflow = get_workflow_config(job_cfg)
    perf = PerformanceLogger(output_dir)

    temporal_config = merge_temporal_config(job_cfg)
    temporal_config["min_object_size_px"] = min_object_size_px
    temporal_config["repair_disconnected_tubes"] = (
        bool(temporal_config.get("repair_disconnected_tubes")) and workflow["enable_tube_repair"]
    )
    temporal_config["enable_traversed_persistence"] = workflow["enable_traversed_persistence"]
    if not workflow["enable_fungal_processing"]:
        temporal_config["allow_tip_growth"] = False
    working_masks = guided_masks
    masks_subdir = "guided_masks"
    overlays_subdir = "guided_overlays"
    temporal_warning = None

    if temporal_config["use_temporal_continuity"]:
        with perf.timed("temporal_continuity"):
            working_masks, temporal_info = run_temporal_continuity_pipeline(
                frames,
                guided_masks,
                output_dir,
                static_ann,
                temporal_config,
                job_id=job_id,
                progress_callback=progress_callback,
                annotations_dir=annotations_dir,
                image_width=image_w,
                image_height=image_h,
            )
        if temporal_info.get("fallback"):
            temporal_warning = temporal_info.get("warning")
            working_masks = guided_masks
            masks_subdir = "guided_masks"
            overlays_subdir = "guided_overlays"
        else:
            masks_subdir = "temporal_masks"
            overlays_subdir = "temporal_overlays"

    from annotations import reapply_hard_guided_annotations_to_masks

    working_masks, _ = reapply_hard_guided_annotations_to_masks(
        working_masks,
        frames,
        annotations_dir,
        static_ann,
        output_dir=output_dir,
        auto_masks=auto_masks,
        save_debug=True,
    )

    results = _finalize_job_outputs(
        frames,
        working_masks,
        output_dir,
        pixel_size_um,
        frame_interval_min,
        min_object_size_px,
        min_branch_length_px,
        job_id=job_id,
        progress_callback=progress_callback,
        annotations_dir=annotations_dir,
        static_ann=static_ann,
        save_annotation_previews=True,
        masks_subdir=masks_subdir,
        overlays_subdir=overlays_subdir,
        perf_logger=perf,
    )
    if temporal_warning:
        results["temporal_warning"] = temporal_warning
    return results


def preview_frame_segmentation(
    output_dir,
    frame_index,
    pixel_size_um=1.0,
    frame_interval_min=1.0,
    min_object_size_px=40,
    dilation_radius=8,
    hole_fill_area=200,
    min_branch_length_px=8,
    deepcell_token=None,
    frame_ann=None,
    job_id=None,
):
    from annotations import (
        apply_guided_postprocess,
        get_annotations_dir,
        get_difference_maps_dir,
        get_job_previews_dir,
        load_frame_annotation,
        load_static_background,
        mask_metrics_summary,
        render_difference_map,
        render_simple_mask_overlay,
        save_frame_annotation,
    )

    output_dir = Path(output_dir)
    dirs = ensure_job_subdirs(output_dir)
    frames_dir = dirs["frames"]
    annotations_dir = get_annotations_dir(output_dir)
    previews_dir = get_job_previews_dir(output_dir)
    diff_dir = get_difference_maps_dir(output_dir)

    frame_path = frames_dir / f"frame_{frame_index:04d}.jpg"
    if not frame_path.exists():
        raise ValueError(f"Frame {frame_index} not found")

    fr = np.array(Image.open(frame_path).convert("RGB"))
    h, w = fr.shape[:2]

    if frame_ann is not None:
        frame_ann = dict(frame_ann)
        frame_ann["frame_index"] = frame_index
        frame_ann["image_width"] = w
        frame_ann["image_height"] = h
        save_frame_annotation(annotations_dir, frame_ann)
    else:
        frame_ann = load_frame_annotation(annotations_dir, frame_index, w, h)

    static_ann = load_static_background(annotations_dir, w, h)
    from annotations import load_job_config

    job_cfg = load_job_config(annotations_dir)
    segmentation_preset = job_cfg.get("segmentation_preset", "fungal_hyphae")
    skeletonization_mode = job_cfg.get("skeletonization_mode", "hyphae_only")
    skeleton_min_object_area_px = int(job_cfg.get("skeleton_min_object_area_px", 40))
    from corrections import load_correction_masks

    mask_path = dirs["masks"] / f"mask_{frame_index:04d}.png"
    if mask_path.exists():
        auto_mask = (np.array(Image.open(mask_path)) > 127).astype(np.uint8)
    else:
        cellsam = CellSAMWrapper(deepcell_token)
        auto_mask = _segment_frame(
            fr, cellsam, dilation_radius, min_object_size_px, hole_fill_area, segmentation_preset
        )

    correction_masks = load_correction_masks(output_dir, frame_index, fr.shape)
    guided_mask = apply_guided_postprocess(
        auto_mask, frame_ann, static_ann, fr.shape, correction_masks=correction_masks
    )

    auto_overlay = render_simple_mask_overlay(fr, auto_mask, color=(0, 180, 255))
    guided_overlay = render_simple_mask_overlay(fr, guided_mask, color=(40, 220, 120))
    diff_overlay = render_difference_map(fr, auto_mask, guided_mask)

    stem = f"frame_{frame_index:06d}"
    auto_path = previews_dir / f"{stem}_auto.png"
    guided_path = previews_dir / f"{stem}_guided.png"
    diff_path = diff_dir / f"{stem}.png"
    Image.fromarray(auto_overlay).save(auto_path)
    Image.fromarray(guided_overlay).save(guided_path)
    Image.fromarray(diff_overlay).save(diff_path)

    auto_metrics, _, _, _ = frame_metrics(
        auto_mask,
        frame_index,
        pixel_size_um,
        frame_interval_min,
        min_branch_length_px,
        skeletonization_mode=skeletonization_mode,
        skeleton_min_object_area_px=skeleton_min_object_area_px,
    )
    guided_metrics, _, _, _ = frame_metrics(
        guided_mask,
        frame_index,
        pixel_size_um,
        frame_interval_min,
        min_branch_length_px,
        skeletonization_mode=skeletonization_mode,
        skeleton_min_object_area_px=skeleton_min_object_area_px,
    )

    frame_ann["preview_status"] = "pending"
    save_frame_annotation(annotations_dir, frame_ann)

    prefix = f"/api/jobs/{job_id}" if job_id else ""
    return {
        "frame_index": frame_index,
        "guidance_source": "current_frame",
        "original_url": f"{prefix}/frames/{frame_index}" if job_id else "",
        "auto_overlay_url": f"{prefix}/overlays/{frame_index}" if job_id else "",
        "guided_overlay_url": f"{prefix}/previews/{frame_index}/guided" if job_id else "",
        "difference_map_url": f"{prefix}/difference_maps/{frame_index}" if job_id else "",
        "auto_metrics": mask_metrics_summary(auto_metrics),
        "guided_metrics": mask_metrics_summary(guided_metrics),
    }
