# fungal_sam2_demo.py
# Demo: fungi timelapse segmentation with SAM2 + zero-shot threshold fallback
# Outputs masks, overlays, skeletons, per-frame metrics, and simple object tracking.

import os
import sys
import subprocess
import importlib.util
from pathlib import Path
import urllib.request
import shutil
import math
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# =========================
# AUTO-INSTALL SECTION
# =========================

def pip_install(args):
    print(f"\n[INSTALL] pip install {args}\n")
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + args)

def ensure_package(import_name, pip_args):
    if importlib.util.find_spec(import_name) is None:
        pip_install(pip_args)

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
    ]
    for imp, pip_args in base:
        ensure_package(imp, pip_args)

    # Torch install: assume user has CUDA-capable environment.
    # If this fails, install PyTorch manually from https://pytorch.org/get-started/locally/
    if importlib.util.find_spec("torch") is None:
        pip_install(["torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu128"])

    # SAM2 local install
    if importlib.util.find_spec("sam2") is None:
        repo_dir = Path.cwd() / "sam2"
        if not repo_dir.exists():
            subprocess.check_call(["git", "clone", "https://github.com/facebookresearch/sam2.git", str(repo_dir)])
        pip_install(["-e", str(repo_dir)])

    # Download default checkpoint if missing
    ckpt_dir = Path.cwd() / "sam2_checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / "sam2.1_hiera_large.pt"
    if not ckpt_path.exists():
        url = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
        print(f"[DOWNLOAD] {url}")
        urllib.request.urlretrieve(url, ckpt_path)

ensure_dependencies()

# =========================
# IMPORTS AFTER INSTALL
# =========================

import numpy as np
import cv2
import pandas as pd
import tifffile
from PIL import Image, ImageTk
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
    print("[WARN] skan import failed:", e)
    Skeleton = None
    summarize = None
    SKAN_AVAILABLE = False

import torch

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except Exception as e:
    print("SAM2 import failed:", e)
    SAM2ImagePredictor = None
    build_sam2 = None


# =========================
# CONFIG
# =========================

OUT_DIR = Path.cwd() / "fungal_sam2_demo_output"
OUT_DIR.mkdir(exist_ok=True)

SAM2_REPO = Path.cwd() / "sam2"
SAM2_CKPT = Path.cwd() / "sam2_checkpoints" / "sam2.1_hiera_large.pt"
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"

MAX_DISPLAY_W = 1100
MAX_DISPLAY_H = 800

MIN_OBJECT_SIZE_PX = 40
PIXEL_SIZE_UM = 1.0      # change this if known
FRAME_INTERVAL_MIN = 1.0 # change this if known

# Skan skeleton graph settings
MIN_SKAN_BRANCH_LENGTH_PX = 8      # remove tiny skeleton spurs shorter than this
BRANCH_CLUSTER_RADIUS_PX = 8       # merge neighbouring branch-node pixels into one branch point


# =========================
# VIDEO / TIFF LOADING
# =========================

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
            # either T,H,W or H,W,C
            if data.shape[-1] in [3, 4]:
                frames = [normalize_to_uint8(data)]
            else:
                frames = [normalize_to_uint8(data[i]) for i in range(data.shape[0])]
        elif data.ndim == 4:
            # T,H,W,C or T,Z,H,W: take max projection if needed
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


# =========================
# SAM2
# =========================

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
            model = build_sam2(SAM2_CFG, str(SAM2_CKPT), device=device)
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
        return clean_mask(best)


# =========================
# SEGMENTATION + ANALYSIS
# =========================

def zero_shot_fluorescence_mask(image_rgb):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    try:
        thr = threshold_otsu(blur)
    except Exception:
        thr = np.percentile(blur, 90)

    mask = blur > thr
    return clean_mask(mask)
from skimage.morphology import medial_axis

def reconstruct_hypha_tube(mask, radius_px=6, nearby_margin_px=10):
    mask = np.asarray(mask).astype(bool)

    # Make broken puncta join locally before centreline extraction
    working = binary_dilation(mask, disk(2))
    working = binary_closing(working, disk(3))
    working = remove_small_holes(working, area_threshold=200)

    # Get centreline
    skel = skeletonize(working)

    # Bridge small centreline gaps
    skel = binary_dilation(skel, disk(1))
    skel = binary_closing(skel, disk(3))
    skel = skeletonize(skel)

    # Rebuild tube around centreline
    tube = binary_dilation(skel, disk(radius_px))

    # Keep tube near original/working signal to avoid filling spaces between branches
    nearby = binary_dilation(working, disk(nearby_margin_px))
    tube = tube & nearby

    # Smooth tube
    tube = binary_closing(tube, disk(2))
    tube = remove_small_holes(tube, area_threshold=500)
    tube = remove_small_objects(tube, min_size=MIN_OBJECT_SIZE_PX)

    return tube.astype(np.uint8)
def clean_mask(mask):
    mask = np.asarray(mask).astype(bool)

    mask = remove_small_objects(mask, min_size=MIN_OBJECT_SIZE_PX)

    # More aggressive local connection before tube reconstruction
    mask = binary_dilation(mask, disk(2))
    mask = binary_closing(mask, disk(3))
    mask = remove_small_holes(mask, area_threshold=200)

    # Main fix
    mask = reconstruct_hypha_tube(
        mask,
        radius_px=8,
        nearby_margin_px=10
    )

    return mask.astype(np.uint8)

def _neighbor_count(skel_bool):
    """
    Count 8-connected neighbours for each skeleton pixel.
    Used for robust endpoint / branch-node visualisation.
    """
    skel_u8 = skel_bool.astype(np.uint8)
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    return cv2.filter2D(skel_u8, -1, kernel)


def _remove_short_skeleton_spurs_with_skan(skel_bool, min_branch_length_px=MIN_SKAN_BRANCH_LENGTH_PX):
    """
    Uses skan's skeleton graph to delete short terminal branches/spurs.
    This prevents rough mask edges from producing many false yellow dots.
    """
    if not SKAN_AVAILABLE:
        return skel_bool

    if skel_bool.sum() == 0:
        return skel_bool

    try:
        sk = Skeleton(skel_bool.astype(bool))
        df = summarize(sk)

        # Keep a copy and remove only very short endpoint-to-junction branches.
        pruned = skel_bool.copy()

        for branch_idx, row in df.iterrows():
            branch_type = int(row.get("branch-type", -1))
            branch_distance = float(row.get("branch-distance", row.get("euclidean-distance", 0)))

            # In skan, branch-type usually means:
            # 0 endpoint-endpoint, 1 endpoint-junction, 2 junction-junction, 3 isolated cycle.
            # The noisy false branches are usually short endpoint-junction branches.
            if branch_type == 1 and branch_distance < min_branch_length_px:
                coords = sk.path_coordinates(branch_idx)
                rr = np.clip(coords[:, 0].astype(int), 0, pruned.shape[0] - 1)
                cc = np.clip(coords[:, 1].astype(int), 0, pruned.shape[1] - 1)
                pruned[rr, cc] = False

        # Re-skeletonise after pruning to tidy topology.
        pruned = skeletonize(pruned)
        return pruned.astype(bool)

    except Exception as e:
        print("[WARN] skan pruning failed, using raw skeleton:", e)
        return skel_bool


def _cluster_branch_pixels(branch_mask, min_cluster_size=1):
    """
    Branch/junction areas can contain several neighbouring skeleton pixels.
    This merges each connected branch-node cluster into a single visual dot.
    """
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


def skeleton_and_branchpoints(mask):
    """
    Skan-backed skeletonisation.

    Returns:
      skel_u8: skeleton mask
      branches_u8: one-pixel branch-point markers
      endpoints_u8: one-pixel endpoint markers
      skan_summary_df: per-branch graph summary, or empty DataFrame
    """
    mask_bool = mask.astype(bool)

    # Skeletonise
    skel_bool = skeletonize(mask_bool)

    # Remove tiny false terminal spurs using skan graph paths
    skel_bool = _remove_short_skeleton_spurs_with_skan(
        skel_bool,
        min_branch_length_px=MIN_SKAN_BRANCH_LENGTH_PX
    )

    skel_u8 = skel_bool.astype(np.uint8)

    # Get graph-level branch summary from skan
    skan_summary_df = pd.DataFrame()
    if SKAN_AVAILABLE and skel_bool.sum() > 0:
        try:
            sk = Skeleton(skel_bool)
            skan_summary_df = summarize(sk)
        except Exception as e:
            print("[WARN] skan summarize failed:", e)

    # Visual branch/end points from topology after pruning
    neigh = _neighbor_count(skel_bool)

    # branch pixels are skeleton pixels with >=3 neighbours
    raw_branch = skel_bool & (neigh >= 3)
    branches_u8, branch_count = _cluster_branch_pixels(raw_branch)

    # endpoints are skeleton pixels with exactly 1 neighbour
    raw_endpoints = skel_bool & (neigh == 1)
    endpoints_u8, tip_count = _cluster_branch_pixels(raw_endpoints)

    return skel_u8, branches_u8.astype(np.uint8), endpoints_u8.astype(np.uint8), skan_summary_df


def frame_metrics(mask, frame_idx):
    skel, branches, endpoints, skan_df = skeleton_and_branchpoints(mask)

    area_px = int(mask.sum())

    # Prefer skan branch-distance sum for length if available.
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
        "time_min": frame_idx * FRAME_INTERVAL_MIN,
        "hyphal_area_px": area_px,
        "hyphal_area_um2": area_px * PIXEL_SIZE_UM * PIXEL_SIZE_UM,
        "hyphal_length_px": length_px,
        "hyphal_length_um": length_px * PIXEL_SIZE_UM,
        "branch_points": branch_count,
        "tip_count": tip_count,
        "skan_graph_branches": skan_branch_count,
        "skan_terminal_branches": terminal_branch_count,
        "skan_junction_to_junction_branches": junction_to_junction_count,
    }, skel, branches

def object_props(mask, frame_idx):
    lab = label(mask)
    props = []
    for r in regionprops(lab):
        if r.area < MIN_OBJECT_SIZE_PX:
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

    # green mask
    overlay[mask_bool] = (0.55 * overlay[mask_bool] + 0.45 * np.array([0, 255, 0])).astype(np.uint8)

    if skel is not None:
        overlay[skel.astype(bool)] = np.array([255, 0, 0], dtype=np.uint8)

    if branches is not None:
        ys, xs = np.where(branches.astype(bool))
        for x, y in zip(xs, ys):
            cv2.circle(overlay, (x, y), 3, (255, 255, 0), -1)

    return overlay

def save_outputs(frames, masks):
    masks_dir = OUT_DIR / "masks"
    overlays_dir = OUT_DIR / "overlays"
    skeletons_dir = OUT_DIR / "skeletons"
    for d in [masks_dir, overlays_dir, skeletons_dir]:
        d.mkdir(exist_ok=True)

    metrics = []
    props = []
    overlay_frames = []

    for i, (img, mask) in enumerate(zip(frames, masks)):
        met, skel, branches = frame_metrics(mask, i)
        metrics.append(met)
        props.extend(object_props(mask, i))

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

    # Save overlay video
    h, w = overlay_frames[0].shape[:2]
    video_path = OUT_DIR / "segmentation_overlay.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(1, int(60 / max(FRAME_INTERVAL_MIN, 1))),
        (w, h),
    )
    for fr in overlay_frames:
        writer.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
    writer.release()

    print(f"\n[DONE] Output folder: {OUT_DIR}")
    return metrics_df, tracks_df


# =========================
# TKINTER UI
# =========================

class FungalSegmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fungal SAM2 Segmentation Demo")

        self.frames = []
        self.masks = []
        self.current_idx = 0
        self.display_scale = 1.0

        self.points = []
        self.point_labels = []
        self.box_start = None
        self.box = None
        self.mode = tk.StringVar(value="click_fg")

        self.sam = SAM2Wrapper()

        self.build_ui()

    def build_ui(self):
        top = ttk.Frame(self.root)
        top.pack(fill="x", padx=8, pady=8)

        ttk.Button(top, text="Load MP4/TIFF", command=self.load_file).pack(side="left", padx=4)
        ttk.Button(top, text="Zero-shot all frames", command=self.segment_all_zero_shot).pack(side="left", padx=4)
        ttk.Button(top, text="SAM2 segment current", command=self.segment_current_sam2).pack(side="left", padx=4)
        ttk.Button(top, text="Propagate current mask naively", command=self.propagate_current_mask).pack(side="left", padx=4)
        ttk.Button(top, text="Save outputs", command=self.save).pack(side="left", padx=4)

        ttk.Label(top, text="Mode:").pack(side="left", padx=(20, 4))
        ttk.Radiobutton(top, text="FG click", variable=self.mode, value="click_fg").pack(side="left")
        ttk.Radiobutton(top, text="BG click", variable=self.mode, value="click_bg").pack(side="left")
        ttk.Radiobutton(top, text="Box", variable=self.mode, value="box").pack(side="left")

        nav = ttk.Frame(self.root)
        nav.pack(fill="x", padx=8)

        ttk.Button(nav, text="< Prev", command=self.prev_frame).pack(side="left", padx=4)
        ttk.Button(nav, text="Next >", command=self.next_frame).pack(side="left", padx=4)
        ttk.Button(nav, text="Clear prompts", command=self.clear_prompts).pack(side="left", padx=4)

        self.status = ttk.Label(nav, text="No file loaded")
        self.status.pack(side="left", padx=12)

        self.canvas = tk.Canvas(self.root, bg="black", width=900, height=650)
        self.canvas.pack(fill="both", expand=True, padx=8, pady=8)

        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

    def load_file(self):
        path = filedialog.askopenfilename(
            filetypes=[
                ("Video/TIFF", "*.mp4 *.avi *.mov *.mkv *.tif *.tiff"),
                ("All files", "*.*"),
            ]
        )
        if not path:
            return

        try:
            self.frames = load_video_or_tiff(path)
            self.masks = [np.zeros(self.frames[0].shape[:2], dtype=np.uint8) for _ in self.frames]
            self.current_idx = 0
            self.clear_prompts()
            self.show_frame()
        except Exception as e:
            messagebox.showerror("Load error", str(e))

    def current_image(self):
        return self.frames[self.current_idx]

    def to_image_coords(self, x, y):
        return int(x / self.display_scale), int(y / self.display_scale)

    def on_mouse_down(self, event):
        if not self.frames:
            return

        x, y = self.to_image_coords(event.x, event.y)

        if self.mode.get() == "box":
            self.box_start = (x, y)
            self.box = None
        else:
            label_val = 1 if self.mode.get() == "click_fg" else 0
            self.points.append([x, y])
            self.point_labels.append(label_val)
            self.show_frame()

    def on_mouse_drag(self, event):
        if self.mode.get() != "box" or self.box_start is None:
            return
        x, y = self.to_image_coords(event.x, event.y)
        x0, y0 = self.box_start
        self.box = [min(x0, x), min(y0, y), max(x0, x), max(y0, y)]
        self.show_frame()

    def on_mouse_up(self, event):
        if self.mode.get() == "box":
            x, y = self.to_image_coords(event.x, event.y)
            x0, y0 = self.box_start
            self.box = [min(x0, x), min(y0, y), max(x0, x), max(y0, y)]
            self.box_start = None
            self.show_frame()

    def clear_prompts(self):
        self.points = []
        self.point_labels = []
        self.box = None
        self.box_start = None
        if self.frames:
            self.show_frame()

    def show_frame(self):
        img = self.current_image().copy()
        mask = self.masks[self.current_idx]

        if mask.sum() > 0:
            skel, branches, _, _ = skeleton_and_branchpoints(mask)
            img = make_overlay(img, mask, skel, branches)

        h, w = img.shape[:2]
        self.display_scale = min(MAX_DISPLAY_W / w, MAX_DISPLAY_H / h, 1.0)
        disp = cv2.resize(img, (int(w * self.display_scale), int(h * self.display_scale)))

        pil = Image.fromarray(disp)
        self.tk_img = ImageTk.PhotoImage(pil)

        self.canvas.config(width=disp.shape[1], height=disp.shape[0])
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)

        # draw prompts
        for (x, y), labv in zip(self.points, self.point_labels):
            sx, sy = x * self.display_scale, y * self.display_scale
            color = "lime" if labv == 1 else "red"
            self.canvas.create_oval(sx - 5, sy - 5, sx + 5, sy + 5, outline=color, width=2)

        if self.box is not None:
            x1, y1, x2, y2 = [v * self.display_scale for v in self.box]
            self.canvas.create_rectangle(x1, y1, x2, y2, outline="yellow", width=2)

        self.status.config(
            text=f"Frame {self.current_idx + 1}/{len(self.frames)} | "
                 f"Prompts: {len(self.points)} pts | Box: {self.box is not None} | "
                 f"SAM2: {'ON' if self.sam.available else 'OFF'}"
        )

    def prev_frame(self):
        if not self.frames:
            return
        self.current_idx = max(0, self.current_idx - 1)
        self.clear_prompts()
        self.show_frame()

    def next_frame(self):
        if not self.frames:
            return
        self.current_idx = min(len(self.frames) - 1, self.current_idx + 1)
        self.clear_prompts()
        self.show_frame()

    def segment_current_sam2(self):
        if not self.frames:
            return

        if not self.sam.available:
            messagebox.showwarning("SAM2 unavailable", "SAM2 failed to load. Use zero-shot threshold mode.")
            return

        if not self.points and self.box is None:
            messagebox.showinfo("Need prompt", "Add at least one foreground click or draw a box.")
            return

        try:
            mask = self.sam.predict(
                self.current_image(),
                points=self.points,
                labels_=self.point_labels,
                box=self.box,
            )
            self.masks[self.current_idx] = mask
            self.show_frame()
        except Exception as e:
            messagebox.showerror("SAM2 error", str(e))

    def segment_all_zero_shot(self):
        if not self.frames:
            return

        for i, fr in enumerate(self.frames):
            self.masks[i] = zero_shot_fluorescence_mask(fr)
            print(f"[Zero-shot] segmented frame {i + 1}/{len(self.frames)}")

        self.show_frame()
        messagebox.showinfo("Done", "Zero-shot threshold segmentation complete.")

    def propagate_current_mask(self):
        """
        Simple demo propagation:
        Uses previous frame mask as a prior and combines with fluorescence thresholding.
        This is not true SAM2 video propagation, but works for a quick fungi-only demo.
        """
        if not self.frames:
            return

        if self.masks[self.current_idx].sum() == 0:
            messagebox.showinfo("No current mask", "Create a mask on the current frame first.")
            return

        prev_mask = self.masks[self.current_idx].astype(bool)

        for i in range(self.current_idx + 1, len(self.frames)):
            z = zero_shot_fluorescence_mask(self.frames[i]).astype(bool)
            dil = cv2.dilate(prev_mask.astype(np.uint8), np.ones((11, 11), np.uint8), iterations=1).astype(bool)
            combined = z & dil
            if combined.sum() < MIN_OBJECT_SIZE_PX:
                combined = z
            self.masks[i] = clean_mask(combined)
            prev_mask = self.masks[i].astype(bool)
            print(f"[Propagate] frame {i + 1}/{len(self.frames)}")

        self.show_frame()
        messagebox.showinfo("Done", "Naive propagation complete.")

    def save(self):
        if not self.frames:
            return

        try:
            metrics_df, tracks_df = save_outputs(self.frames, self.masks)
            msg = (
                f"Saved to:\n{OUT_DIR}\n\n"
                f"Metrics rows: {len(metrics_df)}\n"
                f"Track rows: {len(tracks_df)}\n\n"
                f"Key file: hyphal_metrics.csv"
            )
            messagebox.showinfo("Saved", msg)
        except Exception as e:
            messagebox.showerror("Save error", str(e))


def main():
    root = tk.Tk()
    app = FungalSegmentationApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()