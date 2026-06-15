import os
import sys
import subprocess
import importlib.util
from pathlib import Path
import urllib.request
import shutil
import math

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
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
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

        device_type = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
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

def load_video_or_tiff(path):
    path = Path(path)
    suffix = path.suffix.lower()
    frames = []

    if suffix in [".tif", ".tiff"]:
        data = tifffile.imread(str(path))
        data = np.asarray(data)

        if data.ndim == 2:
            frames = [normalize_to_uint8(data)]
        elif data.ndim == 3:
            if data.shape[-1] in [3, 4]:
                frames = [normalize_to_uint8(data)]
            else:
                frames = [normalize_to_uint8(data[i]) for i in range(data.shape[0])]
        elif data.ndim == 4:
            if data.shape[-1] in [3, 4]:
                frames = [normalize_to_uint8(data[i]) for i in range(data.shape[0])]
            else:
                frames = [normalize_to_uint8(np.max(data[i], axis=0)) for i in range(data.shape[0])]
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
            frames.append(normalize_to_uint8(frame))
        cap.release()

    if not frames:
        raise ValueError("No frames loaded.")
    return frames

def zero_shot_fluorescence_mask(image_rgb):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    try:
        thr = threshold_otsu(blur)
    except Exception:
        thr = np.percentile(blur, 90)
    mask = blur > thr
    return mask.astype(np.uint8)

def reconstruct_hypha_tube(mask, radius_px=6, nearby_margin_px=10, min_size=40):
    mask = np.asarray(mask).astype(bool)
    working = binary_dilation(mask, disk(2))
    working = binary_closing(working, disk(3))
    working = remove_small_holes(working, area_threshold=200)
    skel = skeletonize(working)
    skel = binary_dilation(skel, disk(1))
    skel = binary_closing(skel, disk(3))
    skel = skeletonize(skel)
    tube = binary_dilation(skel, disk(radius_px))
    nearby = binary_dilation(working, disk(nearby_margin_px))
    tube = tube & nearby
    tube = binary_closing(tube, disk(2))
    tube = remove_small_holes(tube, area_threshold=500)
    tube = remove_small_objects(tube, min_size=min_size)
    return tube.astype(np.uint8)

def clean_mask(mask, radius_px=8, nearby_margin_px=10, min_size=40):
    mask = np.asarray(mask).astype(bool)
    mask = remove_small_objects(mask, min_size=min_size)
    mask = binary_dilation(mask, disk(2))
    mask = binary_closing(mask, disk(3))
    mask = remove_small_holes(mask, area_threshold=200)
    mask = reconstruct_hypha_tube(mask, radius_px=radius_px, nearby_margin_px=nearby_margin_px, min_size=min_size)
    return mask.astype(np.uint8)

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

def _cluster_branch_pixels(branch_mask, min_cluster_size=1):
    lab = label(branch_mask.astype(bool))
    clustered = np.zeros_like(branch_mask, dtype=np.uint8)
    count = 0
    for r in regionprops(lab):
        if r.area < min_cluster_size:
            continue
        cy, cx = r.centroid
        clustered[int(round(cy)), int(round(cx))] = 1
        count += 1
    return clustered, count

def skeleton_and_branchpoints(mask, min_skan_branch_length_px=8):
    mask_bool = mask.astype(bool)
    skel_bool = skeletonize(mask_bool)
    skel_bool = _remove_short_skeleton_spurs_with_skan(skel_bool, min_branch_length_px=min_skan_branch_length_px)
    skel_u8 = skel_bool.astype(np.uint8)
    
    skan_summary_df = pd.DataFrame()
    if SKAN_AVAILABLE and skel_bool.sum() > 0:
        try:
            sk = Skeleton(skel_bool)
            skan_summary_df = summarize(sk)
        except Exception:
            pass

    neigh = _neighbor_count(skel_bool)
    raw_branch = skel_bool & (neigh >= 3)
    branches_u8, branch_count = _cluster_branch_pixels(raw_branch)
    raw_endpoints = skel_bool & (neigh == 1)
    endpoints_u8, tip_count = _cluster_branch_pixels(raw_endpoints)

    return skel_u8, branches_u8.astype(np.uint8), endpoints_u8.astype(np.uint8), skan_summary_df

def frame_metrics(mask, frame_idx, pixel_size_um=1.0, frame_interval_min=1.0, min_skan_branch_length_px=8):
    skel, branches, endpoints, skan_df = skeleton_and_branchpoints(mask, min_skan_branch_length_px)
    area_px = int(mask.sum())

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

    branch_count = int(branches.sum())
    tip_count = int(endpoints.sum())

    return {
        "frame": frame_idx,
        "time_min": frame_idx * frame_interval_min,
        "hyphal_area_px": area_px,
        "hyphal_area_um2": area_px * pixel_size_um * pixel_size_um,
        "hyphal_length_px": length_px,
        "hyphal_length_um": length_px * pixel_size_um,
        "branch_points": branch_count,
        "tip_count": tip_count,
        "skan_graph_branches": skan_branch_count,
        "skan_terminal_branches": terminal_branch_count,
        "skan_junction_to_junction_branches": junction_to_junction_count,
    }, skel, branches

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
    deepcell_token: str = None
):
    """
    Main entrypoint for the API.
    Loads frames, runs CellSAM segmentation (or zero-shot threshold fallback), and saves outputs.
    """
    frames = load_video_or_tiff(file_path)
    masks = [np.zeros(frames[0].shape[:2], dtype=np.uint8) for _ in frames]
    
    cellsam = CellSAMWrapper(deepcell_token)
    
    for i, fr in enumerate(frames):
        if cellsam.available:
            try:
                print(f"[CellSAM] Segmenting frame {i + 1}/{len(frames)}")
                device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
                raw_mask, _, _ = cellsam.segment_func(fr, device=device)
                masks[i] = clean_mask(raw_mask, radius_px=dilation_radius, min_size=min_object_size_px)
            except Exception as e:
                print(f"[WARN] CellSAM failed on frame {i}, falling back to zero-shot: {e}")
                raw_mask = zero_shot_fluorescence_mask(fr)
                masks[i] = clean_mask(raw_mask, radius_px=dilation_radius, min_size=min_object_size_px)
        else:
            print(f"[Zero-shot] Segmenting frame {i + 1}/{len(frames)}")
            raw_mask = zero_shot_fluorescence_mask(fr)
            masks[i] = clean_mask(raw_mask, radius_px=dilation_radius, min_size=min_object_size_px)

    masks_dir = OUT_DIR / "masks"
    overlays_dir = OUT_DIR / "overlays"
    skeletons_dir = OUT_DIR / "skeletons"
    for d in [masks_dir, overlays_dir, skeletons_dir]:
        d.mkdir(exist_ok=True)

    metrics = []
    props = []
    overlay_frames = []

    for i, (img, mask) in enumerate(zip(frames, masks)):
        met, skel, branches = frame_metrics(
            mask, 
            i, 
            pixel_size_um=pixel_size_um, 
            frame_interval_min=frame_interval_min
        )
        metrics.append(met)
        props.extend(object_props(mask, i, min_object_size_px=min_object_size_px))

        tifffile.imwrite(str(masks_dir / f"mask_{i:04d}.tif"), mask.astype(np.uint8) * 255)
        tifffile.imwrite(str(skeletons_dir / f"skeleton_{i:04d}.tif"), skel.astype(np.uint8) * 255)

        overlay = make_overlay(img, mask, skel, branches)
        Image.fromarray(overlay).save(overlays_dir / f"overlay_{i:04d}.png")
        overlay_frames.append(overlay)

    metrics_df = growth_rates(pd.DataFrame(metrics))
    tracks = track_objects(props)
    tracks_df = pd.DataFrame(tracks)

    metrics_df.to_csv(OUT_DIR / "hyphal_metrics.csv", index=False)
    tracks_df.to_csv(OUT_DIR / "object_tracks.csv", index=False)

    h, w = overlay_frames[0].shape[:2]
    video_path = OUT_DIR / "segmentation_overlay.mp4"
    
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"avc1"), 
        max(1, int(60 / max(frame_interval_min, 1))),
        (w, h),
    )
    for fr in overlay_frames:
        writer.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
    writer.release()

    return {
        "metrics_csv": str(OUT_DIR / "hyphal_metrics.csv"),
        "tracks_csv": str(OUT_DIR / "object_tracks.csv"),
        "video": str(video_path)
    }
