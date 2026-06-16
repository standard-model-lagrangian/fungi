from sam2_pipeline import (
    process_file,
    process_file_guided,
    extract_frames_for_job,
    preview_frame_segmentation,
    get_frame_count,
    OUT_DIR,
    BACKEND_DIR,
    create_job_output_dir,
    ensure_job_subdirs,
    sanitize_filename,
)
from annotations import (
    get_annotations_dir,
    get_job_previews_dir,
    get_difference_maps_dir,
    load_frame_annotation,
    save_frame_annotation,
    load_static_background,
    save_static_background,
    save_annotation_preview,
    apply_static_background_to_all_frames,
    render_annotation_overlay,
    list_keyframes,
    suggest_keyframes,
    load_job_config,
    save_job_config,
    reset_frame_annotation,
    reset_all_annotations,
    annotation_has_guidance,
)
from guided_propagation import (
    default_propagation_config,
    merge_propagation_config,
    load_propagation_metadata,
    load_guided_debug_metadata,
)
from temporal_continuity import (
    merge_temporal_config,
    load_temporal_metadata,
)
from pre_segmentation_setup import (
    get_setup_dir,
    roi_mask_path,
    ignore_mask_path,
    load_setup_metadata,
    save_setup_metadata,
    save_setup_mask,
    preview_frame_indices,
)
from segmentation_config import (
    list_presets_for_api,
    preset_values,
    small_object_warning,
    validate_segmentation_params,
    get_workflow_config,
    workflow_flags,
    normalize_preset_name,
    PRESET_BACTERIA,
    PRESET_MIXED,
)
from bacterial_tracking import (
    get_bacterial_tracking_dir,
    load_bacterial_tracking_metadata,
    merge_bacterial_tracking_config,
    resolve_tracking_label_overlay_path,
)
from corrections import (
    load_correction_masks,
    load_frame_correction_masks,
    load_global_static_mask,
    save_correction_masks,
    save_global_static_mask,
    reset_frame_corrections,
    reset_all_corrections,
    render_correction_overlay,
    render_correction_difference_map,
    compute_correction_debug_metadata,
    load_correction_debug_metadata,
    correction_add_path,
    correction_remove_path,
    correction_static_path,
    global_static_correction_path,
    get_corrections_dir,
)
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Form, Body
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import uuid
import pandas as pd
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional

app = FastAPI(title="Fungi SAM2 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

jobs = {}
latest_completed_job_id = None


class BrushStrokeModel(BaseModel):
    points: List[List[float]] = Field(default_factory=list)
    size: int = 12


class BoundingBoxModel(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class FrameAnnotationModel(BaseModel):
    frame_index: int
    image_width: int
    image_height: int
    is_keyframe: bool = False
    accepted_preview: bool = False
    preview_status: str = "none"
    object_points: List[List[float]] = Field(default_factory=list)
    background_points: List[List[float]] = Field(default_factory=list)
    static_background_points: List[List[float]] = Field(default_factory=list)
    object_brush_strokes: List[BrushStrokeModel] = Field(default_factory=list)
    background_brush_strokes: List[BrushStrokeModel] = Field(default_factory=list)
    static_background_brush_strokes: List[BrushStrokeModel] = Field(default_factory=list)
    bounding_boxes: List[BoundingBoxModel] = Field(default_factory=list)
    timestamp: Optional[str] = None
    tool_version: Optional[str] = None


class AnnotationModeModel(BaseModel):
    annotation_mode: str = "keyframes"


class KeyframeSuggestModel(BaseModel):
    strategy: str = "endpoints"
    every_n: int = 10


class PropagationSettingsModel(BaseModel):
    propagation_window: int = 10
    use_temporal_propagation: bool = True
    repair_disconnected_tubes: bool = True
    max_bridge_gap_px: int = 15
    max_bridge_angle_degrees: float = 45


class TemporalContinuitySettingsModel(BaseModel):
    use_temporal_continuity: bool = True
    temporal_memory_frames: int = 3
    temporal_persistence_weight: float = 0.5
    max_allowed_area_drop_fraction: float = 0.15
    recover_missing_middle_frames: bool = True
    min_temporal_component_persistence: int = 2
    allow_tip_growth: bool = True
    repair_disconnected_tubes: bool = False
    max_bridge_gap_px: int = 15
    max_bridge_angle_degrees: float = 45
    min_bridge_intensity_percentile: int = 60
    max_endpoints_per_frame: int = 40
    max_bridge_pair_checks: int = 80
    max_bridges_per_frame: int = 5
    frame_time_budget_sec: float = 25.0
    branch_node_merge_radius_px: int = 6
    branch_node_temporal_smoothing: bool = True
    branch_node_max_tracking_distance_px: int = 10


class SegmentationParamsModel(BaseModel):
    pixel_size_um: float = Field(1.0, ge=0.001)
    frame_interval_min: float = Field(1.0, ge=0.001)
    min_object_size_px: int = Field(40, ge=1)
    dilation_radius: int = Field(8, ge=0)
    hole_fill_area: int = Field(200, ge=0)
    min_branch_length_px: int = Field(10, ge=1)
    deepcell_token: Optional[str] = None
    target_object_type: str = "fungal_hyphae"
    segmentation_preset: str = "fungal_hyphae"
    skeletonization_mode: str = "hyphae_only"
    skeleton_min_object_area_px: Optional[int] = Field(None, ge=1)
    hyphae_min_length_px: int = Field(12, ge=1)
    hyphae_min_aspect_ratio: float = Field(2.5, ge=1.0)
    bacteria_max_length_px: int = Field(15, ge=1)
    bacteria_max_area_px2: int = Field(200, ge=1)
    max_component_count_threshold: int = Field(5000, ge=100)
    enable_bacterial_tracking: bool = True
    max_bacteria_displacement_px: int = Field(20, ge=1)
    max_track_gap_frames: int = Field(2, ge=0)
    min_track_length_frames: int = Field(2, ge=1)
    trajectory_tail_frames: int = Field(20, ge=1)
    generate_trajectory_overlay_video: bool = True
    generate_heatmaps: bool = True
    max_objects_per_frame_for_tracking: int = Field(5000, ge=100)


class StaticBackgroundModel(BaseModel):
    image_width: int = 0
    image_height: int = 0
    static_background_points: List[List[float]] = Field(default_factory=list)
    static_background_brush_strokes: List[BrushStrokeModel] = Field(default_factory=list)
    bounding_boxes: List[BoundingBoxModel] = Field(default_factory=list)
    timestamp: Optional[str] = None
    tool_version: Optional[str] = None


def _get_job(job_id):
    if job_id not in jobs:
        return None
    return jobs[job_id]


def _job_output_dir(job_id):
    return Path(jobs[job_id]["output_dir"])


def _get_latest_media_paths():
    if latest_completed_job_id is None or latest_completed_job_id not in jobs:
        return None
    job = jobs[latest_completed_job_id]
    if job.get("status") != "completed" or "results" not in job:
        return None
    return job["results"]


def _frame_image_path(job_id, frame_index):
    return _job_output_dir(job_id) / "frames" / f"frame_{frame_index:04d}.jpg"


def _overlay_image_path(job_id, frame_index):
    return _job_output_dir(job_id) / "overlays" / f"overlay_{frame_index:04d}.png"


def _mask_image_path(job_id, frame_index):
    output_dir = _job_output_dir(job_id)
    png_path = output_dir / "masks" / f"mask_{frame_index:04d}.png"
    if png_path.exists():
        return png_path
    return output_dir / "masks" / f"mask_{frame_index:04d}.tif"


def _build_frame_asset_entry(job_id, frame_index, output_dir):
    output_dir = Path(output_dir)
    overlays_dir = output_dir / "overlays"
    masks_dir = output_dir / "masks"
    previews_dir = get_job_previews_dir(output_dir)
    diff_dir = get_difference_maps_dir(output_dir)

    guided_overlay_dir = output_dir / "guided_overlays"
    propagation_debug_dir = output_dir / "propagation_debug"
    temporal_overlay_dir = output_dir / "temporal_overlays"
    temporal_diff_dir = output_dir / "temporal_difference_maps"

    overlay_path = overlays_dir / f"overlay_{frame_index:04d}.png"
    mask_png = masks_dir / f"mask_{frame_index:04d}.png"
    mask_tif = masks_dir / f"mask_{frame_index:04d}.tif"
    guided_overlay_path = guided_overlay_dir / f"overlay_{frame_index:04d}.png"
    temporal_overlay_path = temporal_overlay_dir / f"overlay_{frame_index:04d}.png"
    temporal_diff_path = temporal_diff_dir / f"frame_{frame_index:06d}.png"
    guided_path = previews_dir / f"frame_{frame_index:06d}_guided.png"
    diff_path = diff_dir / f"frame_{frame_index:06d}.png"
    propagation_diff = propagation_debug_dir / f"frame_{frame_index:06d}_diff.png"
    propagation_view = propagation_debug_dir / f"frame_{frame_index:06d}_propagation.png"
    guided_debug_dir = output_dir / "guided_debug" / f"frame_{frame_index:06d}"
    guided_object_mask = guided_debug_dir / "object_annotation_mask.png"
    guided_exclusion_mask = guided_debug_dir / "exclusion_mask.png"
    guided_final_mask = guided_debug_dir / "final_guided_mask.png"
    guided_debug_diff = guided_debug_dir / "difference_map.png"
    correction_debug_dir = output_dir / "correction_debug" / f"frame_{frame_index:06d}"
    correction_overlay_path = correction_debug_dir / "correction_overlay.png"
    correction_diff_path = correction_debug_dir / "difference_map.png"
    corrections_dir = output_dir / "corrections"
    correction_add = corrections_dir / f"frame_{frame_index:06d}_add.png"
    trajectory_overlay_path = (
        output_dir
        / "bacteria_tracking"
        / "bacterial_trajectory_overlay_frames"
        / f"overlay_{frame_index:04d}.png"
    )
    tracking_label_overlay_path = resolve_tracking_label_overlay_path(
        output_dir / "bacteria_tracking",
        frame_index,
    )
    skeleton_tif = output_dir / "skeletons" / f"skeleton_{frame_index:04d}.tif"
    skeleton_png = output_dir / "skeletons" / f"skeleton_{frame_index:04d}.png"
    frame_path = output_dir / "frames" / f"frame_{frame_index:04d}.jpg"

    prefix = f"/api/jobs/{job_id}"
    return {
        "frame_index": frame_index,
        "original_frame_url": f"{prefix}/frames/{frame_index}" if frame_path.exists() else None,
        "auto_overlay_url": f"{prefix}/overlays/{frame_index}" if overlay_path.exists() else None,
        "auto_mask_url": f"{prefix}/masks/{frame_index}" if (mask_png.exists() or mask_tif.exists()) else None,
        "skeleton_frame_url": f"{prefix}/skeletons/{frame_index}"
        if (skeleton_tif.exists() or skeleton_png.exists())
        else None,
        "temporal_overlay_url": f"{prefix}/temporal_overlays/{frame_index}"
        if temporal_overlay_path.exists()
        else None,
        "temporal_difference_map_url": f"{prefix}/temporal_difference_maps/{frame_index}"
        if temporal_diff_path.exists()
        else None,
        "guided_propagated_overlay_url": f"{prefix}/guided_overlays/{frame_index}"
        if guided_overlay_path.exists()
        else None,
        "guided_preview_url": f"{prefix}/previews/{frame_index}/guided" if guided_path.exists() else None,
        "difference_map_url": f"{prefix}/guided_debug/{frame_index}/difference_map"
        if guided_debug_diff.exists()
        else (
            f"{prefix}/temporal_difference_maps/{frame_index}"
            if temporal_diff_path.exists()
            else (
                f"{prefix}/propagation_debug/{frame_index}/diff"
                if propagation_diff.exists()
                else (f"{prefix}/difference_maps/{frame_index}" if diff_path.exists() else None)
            )
        ),
        "propagation_debug_url": f"{prefix}/propagation_debug/{frame_index}/view"
        if propagation_view.exists()
        else None,
        "guided_object_annotation_mask_url": f"{prefix}/guided_debug/{frame_index}/object_mask"
        if guided_object_mask.exists()
        else None,
        "guided_exclusion_mask_url": f"{prefix}/guided_debug/{frame_index}/exclusion_mask"
        if guided_exclusion_mask.exists()
        else None,
        "guided_final_mask_url": f"{prefix}/guided_debug/{frame_index}/final_mask"
        if guided_final_mask.exists()
        else None,
        "guided_debug_difference_map_url": f"{prefix}/guided_debug/{frame_index}/difference_map"
        if guided_debug_diff.exists()
        else None,
        "correction_overlay_url": f"{prefix}/corrections/{frame_index}/overlay"
        if correction_overlay_path.exists()
        else None,
        "correction_difference_map_url": f"{prefix}/corrections/{frame_index}/difference_map"
        if correction_diff_path.exists()
        else None,
        "correction_add_mask_url": f"{prefix}/corrections/{frame_index}/add"
        if correction_add.exists()
        else None,
        "bacterial_trajectory_overlay_url": f"{prefix}/bacterial-tracking/trajectory/{frame_index}"
        if trajectory_overlay_path.exists()
        else None,
        "bacterial_tracking_overlay_url": f"{prefix}/bacterial-tracking/overlay/{frame_index}"
        if tracking_label_overlay_path.exists()
        else None,
        "final_guided_overlay_url": f"{prefix}/guided_overlays/{frame_index}"
        if guided_overlay_path.exists()
        else (
            f"{prefix}/temporal_overlays/{frame_index}"
            if temporal_overlay_path.exists()
            else None
        ),
    }


def _annotation_to_dict(model):
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


@app.post("/api/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    pixel_size_um: float = Form(1.0),
    frame_interval_min: float = Form(1.0),
    min_object_size_px: int = Form(40),
    dilation_radius: int = Form(8),
    hole_fill_area: int = Form(200),
    min_branch_length_px: int = Form(8),
    deepcell_token: str = Form(None),
):
    job_id = str(uuid.uuid4())
    upload_dir = BACKEND_DIR / "uploads"
    upload_dir.mkdir(exist_ok=True)

    output_dir = create_job_output_dir(file.filename)
    subdirs = ensure_job_subdirs(output_dir)
    safe_original_name = sanitize_filename(file.filename)

    file_path = upload_dir / f"{job_id}_{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    shutil.copy2(file_path, subdirs["original"] / safe_original_name)

    jobs[job_id] = {
        "status": "setup",
        "file": str(file_path),
        "output_dir": str(output_dir),
        "stage": "extracting_frames",
        "current_frame_index": 0,
        "total_frames": 0,
        "current_frame_preview_url": None,
        "progress_percent": 0,
        "params": validate_segmentation_params(
            {
                "pixel_size_um": pixel_size_um,
                "frame_interval_min": frame_interval_min,
                "min_object_size_px": min_object_size_px,
                "dilation_radius": dilation_radius,
                "hole_fill_area": hole_fill_area,
                "min_branch_length_px": min_branch_length_px,
                "deepcell_token": deepcell_token,
                "segmentation_preset": "fungal_hyphae",
                "target_object_type": "fungal_hyphae",
                "hyphae_min_length_px": preset_values("fungal_hyphae")["hyphae_min_length_px"],
                "hyphae_min_aspect_ratio": preset_values("fungal_hyphae")["hyphae_min_aspect_ratio"],
                "bacteria_max_length_px": preset_values("fungal_hyphae")["bacteria_max_length_px"],
                "bacteria_max_area_px2": preset_values("fungal_hyphae")["bacteria_max_area_px2"],
                "max_component_count_threshold": preset_values("fungal_hyphae")[
                    "max_component_count_threshold"
                ],
                "skeletonization_mode": "hyphae_only",
                "skeleton_min_object_area_px": preset_values("fungal_hyphae")[
                    "skeleton_min_object_area_px"
                ],
            }
        ),
    }

    def upload_pipeline_task(jid, path, job_output_dir):
        def update_progress(**kwargs):
            jobs[jid].update(kwargs)

        try:
            ann_dir = get_annotations_dir(job_output_dir)
            save_job_config(ann_dir, load_job_config(ann_dir))
            extract_frames_for_job(
                Path(path),
                job_output_dir,
                job_id=jid,
                progress_callback=update_progress,
            )
            total = get_frame_count(Path(job_output_dir) / "frames")
            preview_indices = preview_frame_indices(total)
            save_setup_metadata(
                job_output_dir,
                {
                    "preview_frame_indices": preview_indices,
                    "image_width": 0,
                    "image_height": 0,
                },
            )
            jobs[jid]["status"] = "setup"
            jobs[jid]["stage"] = "ready"
            jobs[jid]["total_frames"] = total
            jobs[jid]["progress_percent"] = 0
        except Exception as e:
            jobs[jid]["status"] = "failed"
            jobs[jid]["stage"] = "error"
            jobs[jid]["error"] = str(e)

    background_tasks.add_task(upload_pipeline_task, job_id, file_path, output_dir)

    return {"job_id": job_id, "status": "setup"}


def _job_params(job):
    return validate_segmentation_params(job.get("params", {}))


def _sync_job_params_to_config(job_id, params):
    ann_dir = get_annotations_dir(jobs[job_id]["output_dir"])
    config = load_job_config(ann_dir)
    config.update(
        {
            "target_object_type": params["target_object_type"],
            "segmentation_preset": params["segmentation_preset"],
            "skeletonization_mode": params["skeletonization_mode"],
            "skeleton_min_object_area_px": params["skeleton_min_object_area_px"],
            "min_object_size_px": params["min_object_size_px"],
            "hyphae_min_length_px": params["hyphae_min_length_px"],
            "hyphae_min_aspect_ratio": params["hyphae_min_aspect_ratio"],
            "bacteria_max_length_px": params["bacteria_max_length_px"],
            "bacteria_max_area_px2": params["bacteria_max_area_px2"],
            "max_component_count_threshold": params["max_component_count_threshold"],
            "enable_bacterial_tracking": params.get("enable_bacterial_tracking", True),
            "max_bacteria_displacement_px": params.get("max_bacteria_displacement_px", 20),
            "max_track_gap_frames": params.get("max_track_gap_frames", 2),
            "min_track_length_frames": params.get("min_track_length_frames", 2),
            "trajectory_tail_frames": params.get("trajectory_tail_frames", 20),
            "generate_trajectory_overlay_video": params.get("generate_trajectory_overlay_video", True),
            "generate_heatmaps": params.get("generate_heatmaps", True),
            "max_objects_per_frame_for_tracking": params.get(
                "max_objects_per_frame_for_tracking", 5000
            ),
        }
    )
    save_job_config(ann_dir, config)


def _run_segmentation(jid, guided=False, review_on_complete=True):
    global latest_completed_job_id
    job = jobs[jid]
    params = _job_params(job)
    ann_dir = get_annotations_dir(job["output_dir"])
    config = load_job_config(ann_dir)
    annotation_mode = config.get("annotation_mode", job.get("annotation_mode", "keyframes"))

    def update_progress(**kwargs):
        jobs[jid].update(kwargs)

    try:
        if guided:
            results = process_file_guided(
                Path(job["file"]),
                pixel_size_um=params["pixel_size_um"],
                frame_interval_min=params["frame_interval_min"],
                min_object_size_px=params["min_object_size_px"],
                dilation_radius=params["dilation_radius"],
                hole_fill_area=params["hole_fill_area"],
                min_branch_length_px=params["min_branch_length_px"],
                deepcell_token=params.get("deepcell_token"),
                job_id=jid,
                output_dir=Path(job["output_dir"]),
                progress_callback=update_progress,
                annotation_mode=annotation_mode,
            )
        else:
            results = process_file(
                Path(job["file"]),
                pixel_size_um=params["pixel_size_um"],
                frame_interval_min=params["frame_interval_min"],
                min_object_size_px=params["min_object_size_px"],
                dilation_radius=params["dilation_radius"],
                hole_fill_area=params["hole_fill_area"],
                min_branch_length_px=params["min_branch_length_px"],
                deepcell_token=params.get("deepcell_token"),
                job_id=jid,
                output_dir=Path(job["output_dir"]),
                progress_callback=update_progress,
                frames_preloaded=True,
                segmentation_preset=params["segmentation_preset"],
                skeletonization_mode=params["skeletonization_mode"],
                skeleton_min_object_area_px=params["skeleton_min_object_area_px"],
            )
    except Exception as e:
        jobs[jid]["status"] = "failed"
        jobs[jid]["stage"] = "error"
        jobs[jid]["error"] = str(e)
        raise

    jobs[jid]["status"] = "review" if review_on_complete else "completed"
    jobs[jid]["stage"] = "review" if review_on_complete else "finished"
    jobs[jid]["progress_percent"] = 100
    jobs[jid]["results"] = results
    jobs[jid]["mode"] = "guided" if guided else "auto"
    jobs[jid]["temporal_warning"] = results.get("temporal_warning")
    latest_completed_job_id = jid


@app.post("/api/jobs/{job_id}/annotation-mode")
async def set_annotation_mode(job_id: str, body: AnnotationModeModel):
    job = _get_job(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    if body.annotation_mode not in ("keyframes", "every_frame"):
        return JSONResponse({"error": "Invalid annotation mode"}, status_code=400)

    ann_dir = get_annotations_dir(job["output_dir"])
    config = load_job_config(ann_dir)
    config["annotation_mode"] = body.annotation_mode
    save_job_config(ann_dir, config)
    jobs[job_id]["annotation_mode"] = body.annotation_mode
    return {"annotation_mode": body.annotation_mode}


@app.get("/api/jobs/{job_id}/annotation-mode")
async def get_annotation_mode(job_id: str):
    job = _get_job(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    ann_dir = get_annotations_dir(job["output_dir"])
    config = load_job_config(ann_dir)
    return {"annotation_mode": config.get("annotation_mode", "keyframes")}


@app.get("/api/jobs/{job_id}/propagation-settings")
async def get_propagation_settings(job_id: str):
    job = _get_job(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    ann_dir = get_annotations_dir(job["output_dir"])
    return merge_propagation_config(load_job_config(ann_dir))


@app.put("/api/jobs/{job_id}/propagation-settings")
async def put_propagation_settings(job_id: str, body: PropagationSettingsModel):
    job = _get_job(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    ann_dir = get_annotations_dir(job["output_dir"])
    config = load_job_config(ann_dir)
    config.update(body.model_dump() if hasattr(body, "model_dump") else body.dict())
    save_job_config(ann_dir, config)
    return merge_propagation_config(config)


@app.get("/api/jobs/{job_id}/temporal-settings")
async def get_temporal_settings(job_id: str):
    job = _get_job(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    ann_dir = get_annotations_dir(job["output_dir"])
    return merge_temporal_config(load_job_config(ann_dir))


@app.put("/api/jobs/{job_id}/temporal-settings")
async def put_temporal_settings(job_id: str, body: TemporalContinuitySettingsModel):
    job = _get_job(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    ann_dir = get_annotations_dir(job["output_dir"])
    config = load_job_config(ann_dir)
    config.update(body.model_dump() if hasattr(body, "model_dump") else body.dict())
    save_job_config(ann_dir, config)
    return merge_temporal_config(config)


@app.get("/api/segmentation-presets")
async def get_segmentation_presets():
    return list_presets_for_api()


@app.get("/api/jobs/{job_id}/params")
async def get_segmentation_params(job_id: str):
    job = _get_job(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    params = _job_params(job)
    workflow = get_workflow_config(params)
    return {
        **params,
        "workflow": workflow_flags(params["target_object_type"]),
        "small_object_warning": small_object_warning(params["min_object_size_px"]),
    }


@app.put("/api/jobs/{job_id}/params")
async def put_segmentation_params(job_id: str, body: SegmentationParamsModel):
    job = _get_job(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    raw = body.model_dump() if hasattr(body, "model_dump") else body.dict()
    if raw.get("target_object_type"):
        raw["segmentation_preset"] = raw["target_object_type"]
    elif raw.get("segmentation_preset"):
        raw["target_object_type"] = raw["segmentation_preset"]
    if raw.get("skeleton_min_object_area_px") is None:
        raw["skeleton_min_object_area_px"] = preset_values(raw.get("target_object_type"))[
            "skeleton_min_object_area_px"
        ]
    params = validate_segmentation_params({**jobs[job_id].get("params", {}), **raw})
    preset_cfg = preset_values(params["target_object_type"])
    for key in (
        "hyphae_min_length_px",
        "hyphae_min_aspect_ratio",
        "bacteria_max_length_px",
        "bacteria_max_area_px2",
        "max_component_count_threshold",
    ):
        if key not in raw:
            params[key] = preset_cfg[key]
    jobs[job_id]["params"] = params
    _sync_job_params_to_config(job_id, params)
    return {
        **params,
        "workflow": workflow_flags(params["target_object_type"]),
        "small_object_warning": small_object_warning(params["min_object_size_px"]),
    }


@app.get("/api/jobs/{job_id}/setup/info")
async def get_setup_info(job_id: str):
    job = _get_job(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)

    output_dir = Path(job["output_dir"])
    frames_dir = output_dir / "frames"
    total = get_frame_count(frames_dir)
    image_width = 0
    image_height = 0
    if total > 0:
        from PIL import Image

        with Image.open(frames_dir / "frame_0000.jpg") as img:
            image_width, image_height = img.size

    indices = preview_frame_indices(total)
    metadata = load_setup_metadata(output_dir)
    metadata["preview_frame_indices"] = indices
    metadata["image_width"] = image_width
    metadata["image_height"] = image_height
    metadata["total_frames"] = total
    metadata["params"] = _job_params(job)
    metadata["small_object_warning"] = small_object_warning(metadata["params"]["min_object_size_px"])

    prefix = f"/api/jobs/{job_id}"
    metadata["preview_frames"] = {
        key: {
            "frame_index": idx,
            "frame_url": f"{prefix}/frames/{idx}",
        }
        for key, idx in indices.items()
    }
    return metadata


@app.get("/api/jobs/{job_id}/setup/metadata")
async def get_setup_metadata_endpoint(job_id: str):
    job = _get_job(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    return load_setup_metadata(job["output_dir"])


@app.put("/api/jobs/{job_id}/setup/metadata")
async def put_setup_metadata(job_id: str, body: dict = Body(...)):
    job = _get_job(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    return save_setup_metadata(job["output_dir"], body)


def _setup_mask_response(path: Path, width: int, height: int):
    import io

    import numpy as np
    from PIL import Image

    if path.exists():
        return FileResponse(path, media_type="image/png")
    blank = np.zeros((max(1, height), max(1, width)), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(blank).save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


@app.get("/api/jobs/{job_id}/setup/roi-mask")
async def get_setup_roi_mask(job_id: str):
    job = _get_job(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    frames_info = await list_frames(job_id)
    setup_dir = get_setup_dir(job["output_dir"])
    path = roi_mask_path(setup_dir)
    if path.exists():
        return FileResponse(path, media_type="image/png")
    import io

    import numpy as np
    from PIL import Image

    h, w = frames_info["image_height"], frames_info["image_width"]
    blank = np.ones((max(1, h), max(1, w)), dtype=np.uint8) * 255
    buf = io.BytesIO()
    Image.fromarray(blank).save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


@app.get("/api/jobs/{job_id}/setup/ignore-mask")
async def get_setup_ignore_mask(job_id: str):
    job = _get_job(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    frames_info = await list_frames(job_id)
    setup_dir = get_setup_dir(job["output_dir"])
    return _setup_mask_response(
        ignore_mask_path(setup_dir),
        frames_info["image_width"],
        frames_info["image_height"],
    )


@app.put("/api/jobs/{job_id}/setup/roi-mask")
async def put_setup_roi_mask(job_id: str, roi_mask: UploadFile = File(...)):
    import io

    import numpy as np
    from PIL import Image

    job = _get_job(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    frames_info = await list_frames(job_id)
    h, w = frames_info["image_height"], frames_info["image_width"]
    data = await roi_mask.read()
    img = np.array(Image.open(io.BytesIO(data)).convert("L"))
    meta = save_setup_mask(job["output_dir"], "roi", img > 127, (h, w))
    return meta


@app.put("/api/jobs/{job_id}/setup/ignore-mask")
async def put_setup_ignore_mask(job_id: str, ignore_mask: UploadFile = File(...)):
    import io

    import numpy as np
    from PIL import Image

    job = _get_job(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    frames_info = await list_frames(job_id)
    h, w = frames_info["image_height"], frames_info["image_width"]
    data = await ignore_mask.read()
    img = np.array(Image.open(io.BytesIO(data)).convert("L"))
    meta = save_setup_mask(job["output_dir"], "ignore", img > 127, (h, w))
    return meta


@app.get("/api/jobs/{job_id}/temporal/{frame_index}")
async def get_temporal_metadata(job_id: str, frame_index: int):
    job = _get_job(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    meta = load_temporal_metadata(job["output_dir"], frame_index)
    if meta is None:
        return {
            "frame_index": frame_index,
            "pixels_recovered": 0,
            "pixels_removed_flicker": 0,
            "area_before": None,
            "area_after": None,
            "middle_frame_recovery_applied": False,
            "bridges_added": 0,
        }
    return meta


@app.get("/api/jobs/{job_id}/temporal_overlays/{frame_index}")
async def get_temporal_overlay_frame(job_id: str, frame_index: int):
    job = _get_job(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    path = Path(job["output_dir"]) / "temporal_overlays" / f"overlay_{frame_index:04d}.png"
    if not path.exists():
        return JSONResponse({"error": "Temporal overlay not found"}, status_code=404)
    return FileResponse(path, media_type="image/png")


@app.get("/api/jobs/{job_id}/temporal_difference_maps/{frame_index}")
async def get_temporal_difference_map(job_id: str, frame_index: int):
    job = _get_job(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    path = Path(job["output_dir"]) / "temporal_difference_maps" / f"frame_{frame_index:06d}.png"
    if not path.exists():
        return JSONResponse({"error": "Temporal difference map not found"}, status_code=404)
    return FileResponse(path, media_type="image/png")


@app.get("/api/jobs/{job_id}/propagation/{frame_index}")
async def get_propagation_metadata(job_id: str, frame_index: int):
    job = _get_job(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    meta = load_propagation_metadata(job["output_dir"], frame_index)
    if meta is None:
        ann_dir = get_annotations_dir(job["output_dir"])
        frames_info = await list_frames(job_id)
        ann = load_frame_annotation(
            ann_dir,
            frame_index,
            frames_info["image_width"],
            frames_info["image_height"],
        )
        if annotation_has_guidance(ann):
            return {
                "frame_index": frame_index,
                "directly_annotated": True,
                "source_keyframe": frame_index,
                "propagation_distance": 0,
                "mode": "keyframe_pending",
                "tube_repair_applied": False,
                "bridged_gaps": 0,
            }
        return {
            "frame_index": frame_index,
            "directly_annotated": False,
            "source_keyframe": None,
            "propagation_distance": None,
            "mode": "auto_only",
            "tube_repair_applied": False,
            "bridged_gaps": 0,
        }
    return meta


@app.get("/api/jobs/{job_id}/guided_overlays/{frame_index}")
async def get_guided_overlay_frame(job_id: str, frame_index: int):
    job = _get_job(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    path = Path(job["output_dir"]) / "guided_overlays" / f"overlay_{frame_index:04d}.png"
    if not path.exists():
        return JSONResponse({"error": "Guided overlay not found"}, status_code=404)
    return FileResponse(path, media_type="image/png")


@app.get("/api/jobs/{job_id}/propagation_debug/{frame_index}/view")
async def get_propagation_debug_view(job_id: str, frame_index: int):
    job = _get_job(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    path = Path(job["output_dir"]) / "propagation_debug" / f"frame_{frame_index:06d}_propagation.png"
    if not path.exists():
        return JSONResponse({"error": "Propagation debug view not found"}, status_code=404)
    return FileResponse(path, media_type="image/png")


@app.get("/api/jobs/{job_id}/propagation_debug/{frame_index}/diff")
async def get_propagation_debug_diff(job_id: str, frame_index: int):
    job = _get_job(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    path = Path(job["output_dir"]) / "propagation_debug" / f"frame_{frame_index:06d}_diff.png"
    if not path.exists():
        return JSONResponse({"error": "Propagation difference map not found"}, status_code=404)
    return FileResponse(path, media_type="image/png")


def _guided_debug_frame_dir(job, frame_index):
    return Path(job["output_dir"]) / "guided_debug" / f"frame_{frame_index:06d}"


@app.get("/api/jobs/{job_id}/guided_debug/{frame_index}/object_mask")
async def get_guided_debug_object_mask(job_id: str, frame_index: int):
    job = _get_job(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    path = _guided_debug_frame_dir(job, frame_index) / "object_annotation_mask.png"
    if not path.exists():
        return JSONResponse({"error": "Object annotation mask not found"}, status_code=404)
    return FileResponse(path, media_type="image/png")


@app.get("/api/jobs/{job_id}/guided_debug/{frame_index}/exclusion_mask")
async def get_guided_debug_exclusion_mask(job_id: str, frame_index: int):
    job = _get_job(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    path = _guided_debug_frame_dir(job, frame_index) / "exclusion_mask.png"
    if not path.exists():
        return JSONResponse({"error": "Exclusion mask not found"}, status_code=404)
    return FileResponse(path, media_type="image/png")


@app.get("/api/jobs/{job_id}/guided_debug/{frame_index}/final_mask")
async def get_guided_debug_final_mask(job_id: str, frame_index: int):
    job = _get_job(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    path = _guided_debug_frame_dir(job, frame_index) / "final_guided_mask.png"
    if not path.exists():
        return JSONResponse({"error": "Final guided mask not found"}, status_code=404)
    return FileResponse(path, media_type="image/png")


@app.get("/api/jobs/{job_id}/guided_debug/{frame_index}/difference_map")
async def get_guided_debug_difference_map(job_id: str, frame_index: int):
    job = _get_job(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    path = _guided_debug_frame_dir(job, frame_index) / "difference_map.png"
    if not path.exists():
        return JSONResponse({"error": "Guided difference map not found"}, status_code=404)
    return FileResponse(path, media_type="image/png")


@app.get("/api/jobs/{job_id}/guided_debug/{frame_index}/metadata")
async def get_guided_debug_metadata(job_id: str, frame_index: int):
    job = _get_job(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    meta = load_guided_debug_metadata(job["output_dir"], frame_index)
    if meta is None:
        return JSONResponse({"error": "Guided debug metadata not found"}, status_code=404)
    return meta


def _corrections_frame_dir(job, frame_index):
    return get_corrections_dir(job["output_dir"])


def _load_frame_and_auto_mask(job, frame_index):
    output_dir = Path(job["output_dir"])
    frame_path = output_dir / "frames" / f"frame_{frame_index:04d}.jpg"
    if not frame_path.exists():
        return None, None, None
    from PIL import Image
    import numpy as np

    frame = np.array(Image.open(frame_path).convert("RGB"))
    mask_path = output_dir / "masks" / f"mask_{frame_index:04d}.png"
    if mask_path.exists():
        auto_mask = (np.array(Image.open(mask_path)) > 127).astype(np.uint8)
    else:
        auto_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    return frame, auto_mask, frame.shape


@app.get("/api/jobs/{job_id}/corrections/{frame_index}/add")
async def get_correction_add_mask(job_id: str, frame_index: int):
    job = _get_job(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    path = correction_add_path(get_corrections_dir(job["output_dir"]), frame_index)
    if not path.exists():
        return JSONResponse({"error": "Add correction mask not found"}, status_code=404)
    return FileResponse(path, media_type="image/png")


@app.get("/api/jobs/{job_id}/corrections/{frame_index}/remove")
async def get_correction_remove_mask(job_id: str, frame_index: int):
    job = _get_job(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    path = correction_remove_path(get_corrections_dir(job["output_dir"]), frame_index)
    if not path.exists():
        return JSONResponse({"error": "Remove correction mask not found"}, status_code=404)
    return FileResponse(path, media_type="image/png")


@app.get("/api/jobs/{job_id}/corrections/{frame_index}/static")
async def get_correction_static_mask(job_id: str, frame_index: int):
    job = _get_job(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    path = correction_static_path(get_corrections_dir(job["output_dir"]), frame_index)
    if not path.exists():
        return JSONResponse({"error": "Static correction mask not found"}, status_code=404)
    return FileResponse(path, media_type="image/png")


@app.put("/api/jobs/{job_id}/corrections/{frame_index}")
async def save_corrections(
    job_id: str,
    frame_index: int,
    add_mask: UploadFile = File(None),
    remove_mask: UploadFile = File(None),
    static_mask: UploadFile = File(None),
):
    from PIL import Image
    import numpy as np
    import io

    job = _get_job(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)

    frame, _, shape = _load_frame_and_auto_mask(job, frame_index)
    if frame is None:
        return JSONResponse({"error": "Frame not found"}, status_code=404)

    existing = load_frame_correction_masks(job["output_dir"], frame_index, shape)
    corrections_dir = get_corrections_dir(job["output_dir"])

    async def read_mask(upload, key, path_fn):
        if upload is None:
            return existing[key]
        data = await upload.read()
        if not data:
            return existing[key]
        img = np.array(Image.open(io.BytesIO(data)).convert("L"))
        h, w = shape[:2]
        if img.shape[:2] != (h, w):
            import cv2
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
        return img > 127

    updated = {
        "add_mask": await read_mask(add_mask, "add_mask", correction_add_path),
        "remove_mask": await read_mask(remove_mask, "remove_mask", correction_remove_path),
        "static_mask": await read_mask(static_mask, "static_mask", correction_static_path),
    }
    save_correction_masks(job["output_dir"], frame_index, updated)
    return {"frame_index": frame_index, "saved": True}


@app.get("/api/jobs/{job_id}/corrections/global-static")
async def get_global_static_correction(job_id: str):
    job = _get_job(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    frame, _, shape = _load_frame_and_auto_mask(job, 0)
    if frame is None:
        return JSONResponse({"error": "Frame not found"}, status_code=404)
    path = global_static_correction_path(get_corrections_dir(job["output_dir"]))
    if not path.exists():
        return JSONResponse({"error": "Global static correction not found"}, status_code=404)
    return FileResponse(path, media_type="image/png")


@app.post("/api/jobs/{job_id}/corrections/apply-global-static")
async def apply_global_static_correction(job_id: str, frame_index: int = Form(...)):
    job = _get_job(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    frame, _, shape = _load_frame_and_auto_mask(job, frame_index)
    if frame is None:
        return JSONResponse({"error": "Frame not found"}, status_code=404)
    frame_masks = load_frame_correction_masks(job["output_dir"], frame_index, shape)
    save_global_static_mask(job["output_dir"], frame_masks["static_mask"], shape)
    return {"frame_index": frame_index, "applied_globally": True}


@app.get("/api/jobs/{job_id}/corrections/{frame_index}/overlay")
async def get_correction_overlay(job_id: str, frame_index: int):
    from PIL import Image
    import io

    job = _get_job(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)

    debug_path = Path(job["output_dir"]) / "correction_debug" / f"frame_{frame_index:06d}" / "correction_overlay.png"
    if debug_path.exists():
        return FileResponse(debug_path, media_type="image/png")

    frame, auto_mask, shape = _load_frame_and_auto_mask(job, frame_index)
    if frame is None:
        return JSONResponse({"error": "Frame not found"}, status_code=404)
    correction_masks = load_correction_masks(job["output_dir"], frame_index, shape)
    overlay = render_correction_overlay(frame, auto_mask, correction_masks)
    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


@app.get("/api/jobs/{job_id}/corrections/{frame_index}/difference_map")
async def get_correction_difference_map(job_id: str, frame_index: int):
    from annotations import (
        apply_guided_postprocess,
        load_frame_annotation,
        load_static_background,
    )
    from corrections import compose_full_guided_mask, enforce_full_guided_mask
    from PIL import Image
    import io

    job = _get_job(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)

    debug_path = Path(job["output_dir"]) / "correction_debug" / f"frame_{frame_index:06d}" / "difference_map.png"
    if debug_path.exists():
        return FileResponse(debug_path, media_type="image/png")

    frame, auto_mask, shape = _load_frame_and_auto_mask(job, frame_index)
    if frame is None:
        return JSONResponse({"error": "Frame not found"}, status_code=404)

    ann_dir = get_annotations_dir(job["output_dir"])
    ann = load_frame_annotation(ann_dir, frame_index, shape[1], shape[0])
    static_ann = load_static_background(ann_dir, shape[1], shape[0])
    correction_masks = load_correction_masks(job["output_dir"], frame_index, shape)
    from annotations import compute_guided_annotation_masks

    ann_masks = compute_guided_annotation_masks(ann, static_ann, shape)
    final_mask = enforce_full_guided_mask(
        compose_full_guided_mask(auto_mask, ann_masks, correction_masks),
        ann_masks,
        correction_masks,
    )
    diff = render_correction_difference_map(auto_mask, final_mask, correction_masks)
    buf = io.BytesIO()
    Image.fromarray(diff).save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


@app.get("/api/jobs/{job_id}/corrections/{frame_index}/metadata")
async def get_correction_metadata(job_id: str, frame_index: int):
    from annotations import compute_guided_annotation_masks, load_frame_annotation, load_static_background
    from corrections import compose_full_guided_mask, enforce_full_guided_mask

    job = _get_job(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)

    meta = load_correction_debug_metadata(job["output_dir"], frame_index)
    if meta is not None:
        return meta

    frame, auto_mask, shape = _load_frame_and_auto_mask(job, frame_index)
    if frame is None:
        return JSONResponse({"error": "Frame not found"}, status_code=404)

    ann_dir = get_annotations_dir(job["output_dir"])
    ann = load_frame_annotation(ann_dir, frame_index, shape[1], shape[0])
    static_ann = load_static_background(ann_dir, shape[1], shape[0])
    correction_masks = load_correction_masks(job["output_dir"], frame_index, shape)
    ann_masks = compute_guided_annotation_masks(ann, static_ann, shape)
    final_mask = enforce_full_guided_mask(
        compose_full_guided_mask(auto_mask, ann_masks, correction_masks),
        ann_masks,
        correction_masks,
    )
    return compute_correction_debug_metadata(auto_mask, final_mask, correction_masks)


@app.post("/api/jobs/{job_id}/corrections/{frame_index}/reset")
async def reset_corrections_frame(job_id: str, frame_index: int):
    job = _get_job(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    frame, _, shape = _load_frame_and_auto_mask(job, frame_index)
    if frame is None:
        return JSONResponse({"error": "Frame not found"}, status_code=404)
    reset_frame_corrections(job["output_dir"], frame_index, shape)
    return {"frame_index": frame_index, "reset": True}


@app.post("/api/jobs/{job_id}/corrections/reset-all")
async def reset_corrections_all(job_id: str):
    job = _get_job(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    frames_info = await list_frames(job_id)
    total = frames_info["total_frames"]
    if total <= 0:
        return JSONResponse({"error": "No frames"}, status_code=404)
    frame, _, shape = _load_frame_and_auto_mask(job, 0)
    reset_all_corrections(job["output_dir"], total, shape)
    return {"reset_frames": total}


@app.get("/api/jobs/{job_id}/keyframes")
async def get_keyframes(job_id: str):
    job = _get_job(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    frames_info = await list_frames(job_id)
    ann_dir = get_annotations_dir(job["output_dir"])
    return {
        "keyframes": list_keyframes(
            ann_dir,
            frames_info["total_frames"],
            frames_info["image_width"],
            frames_info["image_height"],
        )
    }


@app.post("/api/jobs/{job_id}/keyframes/suggest")
async def suggest_keyframes_endpoint(job_id: str, body: KeyframeSuggestModel):
    job = _get_job(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    frames_info = await list_frames(job_id)
    ann_dir = get_annotations_dir(job["output_dir"])
    marked = suggest_keyframes(
        ann_dir,
        frames_info["total_frames"],
        frames_info["image_width"],
        frames_info["image_height"],
        strategy=body.strategy,
        every_n=body.every_n,
    )
    return {"marked_keyframes": marked}


@app.post("/api/jobs/{job_id}/segment/auto")
async def run_auto_segmentation(job_id: str, background_tasks: BackgroundTasks):
    job = _get_job(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    if job["status"] not in ("setup", "ready"):
        return JSONResponse({"error": "Job is not ready for segmentation"}, status_code=400)

    save_setup_metadata(job["output_dir"], {"setup_completed": True})
    jobs[job_id]["status"] = "processing"
    jobs[job_id]["stage"] = "segmenting"
    jobs[job_id]["progress_percent"] = 0

    def task(jid):
        try:
            _run_segmentation(jid, guided=False, review_on_complete=True)
        except Exception as e:
            jobs[jid]["status"] = "failed"
            jobs[jid]["stage"] = "error"
            jobs[jid]["error"] = str(e)

    background_tasks.add_task(task, job_id)
    return {"job_id": job_id, "status": "processing"}


@app.post("/api/jobs/{job_id}/segment/guided")
async def run_guided_segmentation(job_id: str, background_tasks: BackgroundTasks):
    job = _get_job(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    if job["status"] not in ("ready", "review", "annotating"):
        return JSONResponse({"error": "Job is not ready for guided segmentation"}, status_code=400)

    jobs[job_id]["status"] = "processing"
    jobs[job_id]["stage"] = "segmenting"

    def task(jid):
        try:
            _run_segmentation(jid, guided=True)
        except Exception as e:
            jobs[jid]["status"] = "failed"
            jobs[jid]["stage"] = "error"
            jobs[jid]["error"] = str(e)

    background_tasks.add_task(task, job_id)
    return {"job_id": job_id, "status": "processing"}


@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    payload = dict(jobs[job_id])
    payload["job_id"] = job_id
    return payload


@app.get("/api/jobs/{job_id}/frames")
async def list_frames(job_id: str):
    job = _get_job(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)

    output_dir = Path(job["output_dir"])
    frames_dir = output_dir / "frames"
    total = get_frame_count(frames_dir)
    image_width = 0
    image_height = 0
    if total > 0:
        from PIL import Image
        with Image.open(frames_dir / "frame_0000.jpg") as img:
            image_width, image_height = img.size

    return {
        "total_frames": total,
        "image_width": image_width,
        "image_height": image_height,
        "frames": [
            _build_frame_asset_entry(job_id, i, output_dir)
            for i in range(total)
        ],
    }


@app.get("/api/jobs/{job_id}/frame-assets")
async def list_frame_assets(job_id: str):
    job = _get_job(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)

    output_dir = Path(job["output_dir"])
    frames_dir = output_dir / "frames"
    total = get_frame_count(frames_dir)
    image_width = 0
    image_height = 0
    if total > 0:
        from PIL import Image
        with Image.open(frames_dir / "frame_0000.jpg") as img:
            image_width, image_height = img.size

    return {
        "job_id": job_id,
        "total_frames": total,
        "image_width": image_width,
        "image_height": image_height,
        "video_url": f"/api/jobs/{job_id}/media/video",
        "frames": [
            _build_frame_asset_entry(job_id, i, output_dir)
            for i in range(total)
        ],
    }


@app.get("/api/jobs/{job_id}/frames/{frame_index}")
async def get_frame_image(job_id: str, frame_index: int):
    if job_id not in jobs:
        return JSONResponse({"error": "Job not found"}, status_code=404)

    frame_path = _frame_image_path(job_id, frame_index)
    if not frame_path.exists():
        return JSONResponse({"error": "Frame not found"}, status_code=404)
    return FileResponse(frame_path, media_type="image/jpeg")


@app.get("/api/jobs/{job_id}/preview/{frame_index}")
async def get_frame_preview(job_id: str, frame_index: int):
    return await get_frame_image(job_id, frame_index)


@app.get("/api/jobs/{job_id}/previews/{frame_index}/auto")
async def get_auto_preview(job_id: str, frame_index: int):
    job = _get_job(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    path = get_job_previews_dir(job["output_dir"]) / f"frame_{frame_index:06d}_auto.png"
    if not path.exists():
        return JSONResponse({"error": "Auto preview not found"}, status_code=404)
    return FileResponse(path, media_type="image/png")


@app.get("/api/jobs/{job_id}/previews/{frame_index}/guided")
async def get_guided_preview(job_id: str, frame_index: int):
    job = _get_job(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    path = get_job_previews_dir(job["output_dir"]) / f"frame_{frame_index:06d}_guided.png"
    if not path.exists():
        return JSONResponse({"error": "Guided preview not found"}, status_code=404)
    return FileResponse(path, media_type="image/png")


@app.get("/api/jobs/{job_id}/difference_maps/{frame_index}")
async def get_difference_map(job_id: str, frame_index: int):
    job = _get_job(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    path = get_difference_maps_dir(job["output_dir"]) / f"frame_{frame_index:06d}.png"
    if not path.exists():
        return JSONResponse({"error": "Difference map not found"}, status_code=404)
    return FileResponse(path, media_type="image/png")


@app.post("/api/jobs/{job_id}/frames/{frame_index}/preview-guided")
async def preview_guided_frame(job_id: str, frame_index: int, annotation: Optional[FrameAnnotationModel] = None):
    job = _get_job(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)

    params = _job_params(job)
    try:
        result = preview_frame_segmentation(
            job["output_dir"],
            frame_index,
            pixel_size_um=params["pixel_size_um"],
            frame_interval_min=params["frame_interval_min"],
            min_object_size_px=params["min_object_size_px"],
            dilation_radius=params["dilation_radius"],
            hole_fill_area=params["hole_fill_area"],
            min_branch_length_px=params["min_branch_length_px"],
            deepcell_token=params.get("deepcell_token"),
            frame_ann=_annotation_to_dict(annotation) if annotation else None,
            job_id=job_id,
        )
        jobs[job_id]["status"] = "review"
        return result
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/jobs/{job_id}/frames/{frame_index}/preview/accept")
async def accept_preview(job_id: str, frame_index: int):
    job = _get_job(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)

    ann_dir = get_annotations_dir(job["output_dir"])
    frames_info = await list_frames(job_id)
    ann = load_frame_annotation(
        ann_dir, frame_index, frames_info["image_width"], frames_info["image_height"]
    )
    ann["accepted_preview"] = True
    ann["preview_status"] = "accepted"
    config = load_job_config(ann_dir)
    if config.get("annotation_mode") == "keyframes":
        ann["is_keyframe"] = True
    save_frame_annotation(ann_dir, ann)
    return {"frame_index": frame_index, "accepted_preview": True}


@app.post("/api/jobs/{job_id}/frames/{frame_index}/preview/reject")
async def reject_preview(job_id: str, frame_index: int):
    job = _get_job(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)

    ann_dir = get_annotations_dir(job["output_dir"])
    frames_info = await list_frames(job_id)
    ann = load_frame_annotation(
        ann_dir, frame_index, frames_info["image_width"], frames_info["image_height"]
    )
    ann["accepted_preview"] = False
    ann["preview_status"] = "rejected"
    save_frame_annotation(ann_dir, ann)
    return {"frame_index": frame_index, "accepted_preview": False}


@app.get("/api/jobs/{job_id}/annotations/static_background")
async def get_static_background(job_id: str):
    job = _get_job(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)

    ann_dir = get_annotations_dir(job["output_dir"])
    frames_info = await list_frames(job_id)
    return load_static_background(
        ann_dir,
        frames_info["image_width"],
        frames_info["image_height"],
    )


@app.put("/api/jobs/{job_id}/annotations/static_background")
async def put_static_background(job_id: str, annotation: StaticBackgroundModel):
    job = _get_job(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)

    ann_dir = get_annotations_dir(job["output_dir"])
    save_static_background(ann_dir, _annotation_to_dict(annotation))
    jobs[job_id]["status"] = "review"
    return {"saved": True}


@app.post("/api/jobs/{job_id}/annotations/static_background/apply-all")
async def apply_static_to_all(job_id: str):
    job = _get_job(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)

    ann_dir = get_annotations_dir(job["output_dir"])
    frames_info = await list_frames(job_id)
    count = apply_static_background_to_all_frames(
        ann_dir,
        frames_info["total_frames"],
        frames_info["image_width"],
        frames_info["image_height"],
    )
    return {"applied_frames": count}


@app.get("/api/jobs/{job_id}/annotations/{frame_index}")
async def get_frame_annotations(job_id: str, frame_index: int):
    job = _get_job(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)

    ann_dir = get_annotations_dir(job["output_dir"])
    frames_info = await list_frames(job_id)
    data = load_frame_annotation(
        ann_dir,
        frame_index,
        frames_info["image_width"],
        frames_info["image_height"],
    )
    return data


@app.put("/api/jobs/{job_id}/annotations/{frame_index}")
async def put_frame_annotations(job_id: str, frame_index: int, annotation: FrameAnnotationModel):
    job = _get_job(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)

    if annotation.frame_index != frame_index:
        return JSONResponse({"error": "Frame index mismatch"}, status_code=400)

    output_dir = Path(job["output_dir"])
    ann_dir = get_annotations_dir(output_dir)
    data = _annotation_to_dict(annotation)
    if annotation_has_guidance(data):
        data["is_keyframe"] = True
    save_frame_annotation(ann_dir, data)

    frame_path = _frame_image_path(job_id, frame_index)
    if frame_path.exists():
        from PIL import Image
        import numpy as np
        img = np.array(Image.open(frame_path).convert("RGB"))
        static_ann = load_static_background(ann_dir, data["image_width"], data["image_height"])
        save_annotation_preview(ann_dir, frame_index, img, data, static_ann)

    jobs[job_id]["status"] = "review"
    return {"saved": True, "frame_index": frame_index}


@app.get("/api/jobs/{job_id}/overlays/{frame_index}")
async def get_overlay_frame(job_id: str, frame_index: int):
    job = _get_job(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)

    overlay_path = _overlay_image_path(job_id, frame_index)
    if not overlay_path.exists():
        return JSONResponse({"error": "Overlay not found"}, status_code=404)
    return FileResponse(overlay_path, media_type="image/png")


@app.get("/api/jobs/{job_id}/masks/{frame_index}")
async def get_mask_frame(job_id: str, frame_index: int):
    job = _get_job(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)

    mask_path = _mask_image_path(job_id, frame_index)
    if not mask_path.exists():
        return JSONResponse({"error": "Mask not found"}, status_code=404)

    if mask_path.suffix.lower() == ".png":
        return FileResponse(mask_path, media_type="image/png")

    from PIL import Image
    import numpy as np
    import tifffile

    arr = np.asarray(tifffile.imread(mask_path))
    if arr.ndim > 2:
        arr = arr[..., 0]
    png_path = mask_path.with_suffix(".png")
    Image.fromarray(arr.astype(np.uint8)).save(png_path)
    return FileResponse(png_path, media_type="image/png")


@app.get("/api/jobs/{job_id}/skeletons/{frame_index}")
async def get_skeleton_frame(
    job_id: str,
    frame_index: int,
    debug_raw_junctions: bool = False,
):
    import io

    import numpy as np
    import tifffile
    from PIL import Image

    from skeleton_branch_nodes import (
        load_prev_normalized_nodes,
        render_skeleton_overlay,
        save_skeleton_debug,
    )
    from sam2_pipeline import skeleton_and_branchpoints

    job = _get_job(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)

    output_dir = Path(job["output_dir"])
    skel_dir = output_dir / "skeletons"
    png_path = skel_dir / f"skeleton_{frame_index:04d}.png"
    tif_path = skel_dir / f"skeleton_{frame_index:04d}.tif"
    debug_overlay_path = (
        output_dir / "skeleton_debug" / f"frame_{frame_index:06d}" / "skeleton_overlay.png"
    )
    meta_path = output_dir / "skeleton_debug" / f"frame_{frame_index:06d}" / "metadata.json"

    needs_regeneration = debug_raw_junctions or not png_path.exists() or not meta_path.exists()

    if needs_regeneration:
        mask_path = output_dir / "masks" / f"mask_{frame_index:04d}.tif"
        mask_png = output_dir / "masks" / f"mask_{frame_index:04d}.png"
        mask_arr = None
        if mask_path.exists():
            mask_arr = np.asarray(tifffile.imread(str(mask_path)))
        elif mask_png.exists():
            mask_arr = np.asarray(Image.open(mask_png))
        if mask_arr is not None:
            if mask_arr.ndim > 2:
                mask_arr = mask_arr[..., 0]
            mask_bool = mask_arr > 0
            prev_nodes = load_prev_normalized_nodes(output_dir, frame_index)
            skel, _, _, _, branch_meta = skeleton_and_branchpoints(
                mask_bool.astype(np.uint8),
                prev_normalized_nodes=prev_nodes,
            )
            overlay = render_skeleton_overlay(
                branch_meta["skel_bool"],
                branch_meta["normalized_nodes"],
                raw_branch=branch_meta["raw_branch"],
                debug_raw_junctions=debug_raw_junctions,
            )
            skel_dir.mkdir(parents=True, exist_ok=True)
            if not debug_raw_junctions:
                Image.fromarray(overlay).save(png_path)
                save_skeleton_debug(
                    output_dir,
                    frame_index,
                    branch_meta["skel_bool"],
                    branch_meta["raw_branch"],
                    branch_meta["cluster_labels"],
                    branch_meta["normalized_nodes"],
                    overlay,
                )
            buf = io.BytesIO()
            Image.fromarray(overlay).save(buf, format="PNG")
            buf.seek(0)
            from fastapi.responses import StreamingResponse

            return StreamingResponse(buf, media_type="image/png")

    if png_path.exists() and not debug_raw_junctions:
        return FileResponse(png_path, media_type="image/png")

    if debug_overlay_path.exists():
        return FileResponse(debug_overlay_path, media_type="image/png")

    if not tif_path.exists():
        return JSONResponse({"error": "Skeleton not found"}, status_code=404)

    skel = np.asarray(tifffile.imread(str(tif_path)))
    if skel.ndim > 2:
        skel = skel[..., 0]
    skel_bool = skel > 0
    img = np.zeros((*skel_bool.shape, 3), dtype=np.uint8)
    img[skel_bool] = [255, 255, 255]
    skel_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(png_path)
    return FileResponse(png_path, media_type="image/png")


def _bacterial_tracking_allowed(job) -> bool:
    params = _job_params(job)
    target = normalize_preset_name(params.get("target_object_type"))
    return target in (PRESET_BACTERIA, PRESET_MIXED)


@app.get("/api/jobs/{job_id}/bacterial-tracking/info")
async def get_bacterial_tracking_info(job_id: str):
    job = _get_job(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    if not _bacterial_tracking_allowed(job):
        return {"available": False, "reason": "not_bacterial_workflow"}

    output_dir = Path(job["output_dir"])
    metadata = load_bacterial_tracking_metadata(output_dir)
    if not metadata:
        return {"available": False, "reason": "not_generated"}

    prefix = f"/api/jobs/{job_id}/bacterial-tracking"
    tracking_dir = get_bacterial_tracking_dir(output_dir)
    downloads = {}
    for key, filename in {
        "detections": "bacterial_detections.csv",
        "tracks": "bacterial_tracks.csv",
        "track_summary": "bacterial_track_summary.csv",
        "frame_metrics": "bacterial_frame_metrics.csv",
    }.items():
        if (tracking_dir / filename).exists():
            downloads[key] = f"{prefix}/csv/{key}"

    heatmaps = {}
    for key, filename in {
        "occupancy": "bacterial_occupancy_heatmap.png",
        "trajectory_density": "bacterial_trajectory_density_heatmap.png",
        "speed": "bacterial_speed_heatmap.png",
        "direction": "bacterial_direction_heatmap.png",
    }.items():
        if (tracking_dir / filename).exists():
            heatmaps[key] = f"{prefix}/heatmaps/{key}"

    video_url = None
    if (tracking_dir / "bacteria_tracking_overlay.mp4").exists():
        video_url = f"{prefix}/video"
    elif (tracking_dir / "bacterial_trajectory_overlay.mp4").exists():
        video_url = f"{prefix}/video"

    label_overlay_video_url = None
    if (tracking_dir / "bacteria_tracking_overlay.mp4").exists():
        label_overlay_video_url = f"{prefix}/label-video"

    return {
        "available": True,
        "metadata": metadata,
        "downloads": downloads,
        "heatmaps": heatmaps,
        "video_url": video_url,
        "label_overlay_video_url": label_overlay_video_url,
        "trajectory_frame_url_template": f"{prefix}/trajectory/{{frame_index}}",
        "label_overlay_frame_url_template": f"{prefix}/overlay/{{frame_index}}",
    }


@app.get("/api/jobs/{job_id}/bacterial-tracking/overlay/{frame_index}")
async def get_bacterial_tracking_label_overlay(job_id: str, frame_index: int):
    job = _get_job(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    path = resolve_tracking_label_overlay_path(
        get_bacterial_tracking_dir(Path(job["output_dir"])),
        frame_index,
    )
    if not path.exists():
        return JSONResponse({"error": "Tracking overlay not found"}, status_code=404)
    return FileResponse(path, media_type="image/png")


@app.get("/api/jobs/{job_id}/bacterial-tracking/trajectory/{frame_index}")
async def get_bacterial_trajectory_overlay(job_id: str, frame_index: int):
    job = _get_job(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    path = (
        get_bacterial_tracking_dir(Path(job["output_dir"]))
        / "bacterial_trajectory_overlay_frames"
        / f"overlay_{frame_index:04d}.png"
    )
    if not path.exists():
        return JSONResponse({"error": "Trajectory overlay not found"}, status_code=404)
    return FileResponse(path, media_type="image/png")


@app.get("/api/jobs/{job_id}/bacterial-tracking/heatmaps/{heatmap_name}")
async def get_bacterial_heatmap(job_id: str, heatmap_name: str):
    job = _get_job(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    names = {
        "occupancy": "bacterial_occupancy_heatmap.png",
        "trajectory_density": "bacterial_trajectory_density_heatmap.png",
        "speed": "bacterial_speed_heatmap.png",
        "direction": "bacterial_direction_heatmap.png",
    }
    if heatmap_name not in names:
        return JSONResponse({"error": "Unknown heatmap"}, status_code=404)
    path = get_bacterial_tracking_dir(Path(job["output_dir"])) / names[heatmap_name]
    if not path.exists():
        return JSONResponse({"error": "Heatmap not found"}, status_code=404)
    return FileResponse(path, media_type="image/png")


@app.get("/api/jobs/{job_id}/bacterial-tracking/csv/{csv_name}")
async def get_bacterial_tracking_csv(job_id: str, csv_name: str):
    job = _get_job(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    names = {
        "detections": "bacterial_detections.csv",
        "tracks": "bacterial_tracks.csv",
        "track_summary": "bacterial_track_summary.csv",
        "frame_metrics": "bacterial_frame_metrics.csv",
    }
    if csv_name not in names:
        return JSONResponse({"error": "Unknown CSV"}, status_code=404)
    path = get_bacterial_tracking_dir(Path(job["output_dir"])) / names[csv_name]
    if not path.exists():
        return JSONResponse({"error": "CSV not found"}, status_code=404)
    return FileResponse(path, media_type="text/csv", filename=names[csv_name])


@app.get("/api/jobs/{job_id}/bacterial-tracking/video")
async def get_bacterial_tracking_video(job_id: str):
    job = _get_job(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    tracking_dir = get_bacterial_tracking_dir(Path(job["output_dir"]))
    path = tracking_dir / "bacteria_tracking_overlay.mp4"
    if not path.exists():
        path = tracking_dir / "bacterial_trajectory_overlay.mp4"
    if not path.exists():
        return JSONResponse({"error": "Video not found"}, status_code=404)
    return FileResponse(path, media_type="video/mp4")


@app.get("/api/jobs/{job_id}/bacterial-tracking/label-video")
async def get_bacterial_tracking_label_video(job_id: str):
    job = _get_job(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    path = get_bacterial_tracking_dir(Path(job["output_dir"])) / "bacteria_tracking_overlay.mp4"
    if not path.exists():
        return JSONResponse({"error": "Label overlay video not found"}, status_code=404)
    return FileResponse(path, media_type="video/mp4")


@app.get("/api/jobs/{job_id}/bacterial-tracking-settings")
async def get_bacterial_tracking_settings(job_id: str):
    job = _get_job(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    params = _job_params(job)
    return merge_bacterial_tracking_config(params)


@app.put("/api/jobs/{job_id}/bacterial-tracking-settings")
async def put_bacterial_tracking_settings(job_id: str, body: dict = Body(...)):
    job = _get_job(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    params = validate_segmentation_params({**jobs[job_id].get("params", {}), **body})
    jobs[job_id]["params"] = params
    _sync_job_params_to_config(job_id, params)
    return merge_bacterial_tracking_config(params)


def _resolve_job_metrics_csv(job) -> Optional[Path]:
    results = job.get("results") or {}
    candidates = []
    for key in ("metrics_csv", "fungi_metrics_csv", "bacteria_metrics_csv"):
        value = results.get(key)
        if value:
            candidates.append(Path(value))

    output_dir = Path(job["output_dir"])
    candidates.extend(
        [
            output_dir / "results" / "hyphal_metrics.csv",
            output_dir / "fungi_metrics.csv",
            output_dir / "bacteria_metrics.csv",
        ]
    )
    for path in candidates:
        if path.exists():
            return path
    return None


def _build_results_payload(df: pd.DataFrame) -> dict:
    data = df.copy()
    if "time_min" not in data.columns and "frame" in data.columns:
        data["time_min"] = data["frame"]

    if "hyphal_length_um" in data.columns:
        growth_col = "length_growth_um_per_min"
        stats = {
            "metrics_type": "hyphal",
            "frames_processed": len(data),
            "max_growth_rate_um_min": float(data[growth_col].max())
            if growth_col in data.columns and not data[growth_col].isna().all()
            else 0.0,
            "avg_growth_rate_um_min": float(data[growth_col].mean())
            if growth_col in data.columns and not data[growth_col].isna().all()
            else 0.0,
            "total_branches_end": int(data.iloc[-1]["branch_points"])
            if not data.empty and "branch_points" in data.columns
            else 0,
            "max_tips": int(data["tip_count"].max())
            if not data.empty and "tip_count" in data.columns
            else 0,
        }
        chart_cols = [
            col
            for col in ("time_min", "hyphal_length_um", "branch_points", "tip_count")
            if col in data.columns
        ]
        chart_data = data[chart_cols].fillna(0).to_dict(orient="records")
        return {"stats": stats, "chart_data": chart_data}

    if "object_count" in data.columns:
        count_growth = "object_count_growth_per_min"
        stats = {
            "metrics_type": "bacterial",
            "frames_processed": len(data),
            "max_object_count": int(data["object_count"].max()) if not data.empty else 0,
            "final_object_count": int(data.iloc[-1]["object_count"]) if not data.empty else 0,
            "max_total_area_um2": float(data["total_area_um2"].max())
            if "total_area_um2" in data.columns and not data.empty
            else 0.0,
            "max_object_count_growth_per_min": float(data[count_growth].max())
            if count_growth in data.columns and not data[count_growth].isna().all()
            else 0.0,
            "avg_object_count_growth_per_min": float(data[count_growth].mean())
            if count_growth in data.columns and not data[count_growth].isna().all()
            else 0.0,
        }
        chart_cols = [
            col
            for col in ("time_min", "object_count", "total_area_um2", "mean_area_um2")
            if col in data.columns
        ]
        chart_data = data[chart_cols].fillna(0).to_dict(orient="records")
        return {"stats": stats, "chart_data": chart_data}

    stats = {"metrics_type": "generic", "frames_processed": len(data)}
    chart_data = data.fillna(0).to_dict(orient="records")
    return {"stats": stats, "chart_data": chart_data}


@app.get("/api/jobs/{job_id}/media/video")
async def get_job_video(job_id: str):
    job = _get_job(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    if "results" not in job:
        return JSONResponse({"error": "Video not ready"}, status_code=404)

    video_path = Path(job["results"]["video"])
    if not video_path.exists():
        return JSONResponse({"error": "Video not found"}, status_code=404)
    return FileResponse(video_path, media_type="video/mp4")


@app.get("/api/jobs/{job_id}/media/metrics_csv")
async def get_job_metrics_csv(job_id: str):
    job = _get_job(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    if "results" not in job:
        return JSONResponse({"error": "Metrics not ready"}, status_code=404)

    csv_path = _resolve_job_metrics_csv(job)
    if csv_path is None:
        return JSONResponse({"error": "CSV not found"}, status_code=404)
    return FileResponse(csv_path, media_type="text/csv", filename=csv_path.name)


@app.post("/api/jobs/{job_id}/annotations/{frame_index}/reset")
async def reset_frame_annotations(job_id: str, frame_index: int):
    job = _get_job(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)

    ann_dir = get_annotations_dir(job["output_dir"])
    frames_info = await list_frames(job_id)
    ann = reset_frame_annotation(
        ann_dir,
        frame_index,
        frames_info["image_width"],
        frames_info["image_height"],
    )
    return {"frame_index": frame_index, "reset": True, "annotation": ann}


@app.post("/api/jobs/{job_id}/annotations/reset-all")
async def reset_all_frame_annotations(job_id: str):
    job = _get_job(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)

    ann_dir = get_annotations_dir(job["output_dir"])
    frames_info = await list_frames(job_id)
    count = reset_all_annotations(
        ann_dir,
        frames_info["total_frames"],
        frames_info["image_width"],
        frames_info["image_height"],
    )
    return {"reset_frames": count}


@app.get("/api/jobs/{job_id}/annotations/{frame_index}/preview")
async def get_annotation_preview(job_id: str, frame_index: int):
    job = _get_job(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)

    preview_path = (
        get_annotations_dir(job["output_dir"]) / "previews" / f"frame_{frame_index:06d}.png"
    )
    if preview_path.exists():
        return FileResponse(preview_path, media_type="image/png")

    frame_path = _frame_image_path(job_id, frame_index)
    if not frame_path.exists():
        return JSONResponse({"error": "Frame not found"}, status_code=404)

    from PIL import Image
    import numpy as np

    ann_dir = get_annotations_dir(job["output_dir"])
    img = np.array(Image.open(frame_path).convert("RGB"))
    frame_ann = load_frame_annotation(ann_dir, frame_index, img.shape[1], img.shape[0])
    static_ann = load_static_background(ann_dir, img.shape[1], img.shape[0])
    preview = render_annotation_overlay(img, frame_ann, static_ann)
    save_annotation_preview(ann_dir, frame_index, img, frame_ann, static_ann)
    return FileResponse(
        get_annotations_dir(job["output_dir"]) / "previews" / f"frame_{frame_index:06d}.png",
        media_type="image/png",
    )


@app.get("/api/results/{job_id}")
async def get_results(job_id: str):
    if job_id not in jobs:
        return JSONResponse({"error": "Job not found"}, status_code=404)

    job = jobs[job_id]
    if job["status"] not in ("completed", "review"):
        return JSONResponse({"error": "Job not completed yet"}, status_code=400)

    metrics_path = _resolve_job_metrics_csv(job)
    if metrics_path is None:
        return JSONResponse({"error": "Metrics not available for this job"}, status_code=404)

    try:
        df = pd.read_csv(metrics_path)
    except Exception as exc:
        return JSONResponse({"error": f"Failed to read metrics: {exc}"}, status_code=500)

    payload = _build_results_payload(df)
    payload["metrics_csv_url"] = f"/api/jobs/{job_id}/media/metrics_csv"
    return payload


@app.get("/api/media/video")
async def get_video():
    results = _get_latest_media_paths()
    if results is None:
        video_path = OUT_DIR / "segmentation_overlay.mp4"
    else:
        video_path = Path(results["video"])
    if not video_path.exists():
        return JSONResponse({"error": "Video not found"}, status_code=404)
    return FileResponse(video_path, media_type="video/mp4")


@app.get("/api/media/metrics_csv")
async def get_metrics_csv():
    results = _get_latest_media_paths()
    if results is None:
        csv_path = OUT_DIR / "hyphal_metrics.csv"
    else:
        csv_path = Path(results["metrics_csv"])
    if not csv_path.exists():
        return JSONResponse({"error": "CSV not found"}, status_code=404)
    return FileResponse(csv_path, media_type="text/csv", filename="hyphal_metrics.csv")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
