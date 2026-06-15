from sam2_pipeline import process_file, OUT_DIR, BACKEND_DIR
from tuner import TuningSession, BOUNDS
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import uuid
import numpy as np
import pandas as pd
from pathlib import Path
from pydantic import BaseModel
from PIL import Image

app = FastAPI(title="Fungi SAM2 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store job status
jobs = {}
# Store tuning sessions
tuning_sessions = {}

@app.post("/api/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    pixel_size_um: float = Form(1.0),
    frame_interval_min: float = Form(1.0),
    min_object_size_px: int = Form(40),
    dilation_radius: int = Form(8),
    deepcell_token: str = Form(None)
):
    job_id = str(uuid.uuid4())
    upload_dir = BACKEND_DIR / "uploads"
    upload_dir.mkdir(exist_ok=True)
    
    file_path = upload_dir / f"{job_id}_{file.filename}"
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    jobs[job_id] = {"status": "processing", "file": str(file_path)}
    
    def run_pipeline(jid, path, pixel_size, frame_interval, min_size, radius, token):
        try:
            results = process_file(
                Path(path),
                pixel_size_um=pixel_size,
                frame_interval_min=frame_interval,
                min_object_size_px=min_size,
                dilation_radius=radius,
                deepcell_token=token
            )
            jobs[jid]["status"] = "completed"
            jobs[jid]["results"] = results
        except Exception as e:
            jobs[jid]["status"] = "failed"
            jobs[jid]["error"] = str(e)
            
    background_tasks.add_task(
        run_pipeline, 
        job_id, 
        file_path, 
        pixel_size_um, 
        frame_interval_min, 
        min_object_size_px, 
        dilation_radius, 
        deepcell_token
    )
    
    return {"job_id": job_id, "status": "processing"}

@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    return jobs[job_id]

@app.get("/api/results/{job_id}")
async def get_results(job_id: str):
    if job_id not in jobs:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    
    job = jobs[job_id]
    if job["status"] != "completed":
        return JSONResponse({"error": "Job not completed yet"}, status_code=400)
        
    metrics_path = job["results"]["metrics_csv"]
    df = pd.read_csv(metrics_path)
    
    # Extract simple statistics
    stats = {
        "frames_processed": len(df),
        "max_growth_rate_um_min": df["length_growth_um_per_min"].max() if not df["length_growth_um_per_min"].isna().all() else 0,
        "avg_growth_rate_um_min": df["length_growth_um_per_min"].mean() if not df["length_growth_um_per_min"].isna().all() else 0,
        "total_branches_end": int(df.iloc[-1]["branch_points"]) if not df.empty else 0,
        "max_tips": int(df["tip_count"].max()) if not df.empty else 0
    }
    
    # We return the raw data for charts as well
    chart_data = df[["time_min", "hyphal_length_um", "branch_points", "tip_count"]].fillna(0).to_dict(orient="records")
    
    return {"stats": stats, "chart_data": chart_data}

@app.get("/api/media/video")
async def get_video():
    video_path = OUT_DIR / "segmentation_overlay.mp4"
    if not video_path.exists():
        return JSONResponse({"error": "Video not found"}, status_code=404)
    return FileResponse(video_path, media_type="video/mp4")

@app.get("/api/media/metrics_csv")
async def get_metrics_csv():
    csv_path = OUT_DIR / "hyphal_metrics.csv"
    if not csv_path.exists():
        return JSONResponse({"error": "CSV not found"}, status_code=404)
    return FileResponse(csv_path, media_type="text/csv", filename="hyphal_metrics.csv")


# --- Preference-based Tuning Endpoints ---

def generate_session_overlays(session: TuningSession):
    session_dir = BACKEND_DIR / "uploads" / "tuner_overlays" / session.session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    from sam2_pipeline import clean_mask, skeleton_and_branchpoints, make_overlay
    for idx, cand in enumerate(session.candidates):
        frame_overlays = []
        for fi in range(len(session.frames)):
            frame = session.frames[fi]
            raw_mask = session.raw_masks[fi]
            cleaned = clean_mask(
                raw_mask,
                radius_px=cand["dilation_radius"],
                nearby_margin_px=cand["nearby_margin_px"],
                min_size=cand["min_object_size_px"]
            )
            skel, branches, _, _ = skeleton_and_branchpoints(
                cleaned,
                min_skan_branch_length_px=cand["min_skan_branch_length_px"]
            )
            overlay = make_overlay(frame, cleaned, skel, branches)
            frame_overlays.append(overlay)

            # Save individual frame overlay
            per_frame_path = session_dir / f"cand_{idx}_frame_{fi}.png"
            Image.fromarray(overlay).save(per_frame_path)

        # Stack all frame overlays vertically into one composite image
        composite = np.vstack(frame_overlays)
        out_path = session_dir / f"cand_{idx}.png"
        Image.fromarray(composite).save(out_path)

@app.post("/api/tune/start")
async def tune_start(
    file: UploadFile = File(...),
    deepcell_token: str = Form(None)
):
    session_id = str(uuid.uuid4())
    upload_dir = BACKEND_DIR / "uploads"
    upload_dir.mkdir(exist_ok=True)
    
    file_path = upload_dir / f"tune_{session_id}_{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        from sam2_pipeline import load_video_or_tiff, CellSAMWrapper, zero_shot_fluorescence_mask
        frames = load_video_or_tiff(file_path)

        # Select up to 5 evenly-spaced frames
        indices = np.linspace(0, len(frames) - 1, min(len(frames), 5), dtype=int)
        selected_frames = [frames[i] for i in indices]

        cellsam = CellSAMWrapper(deepcell_token)
        selected_raw_masks = []
        for frame in selected_frames:
            if cellsam.available:
                try:
                    import torch
                    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
                    raw_mask, _, _ = cellsam.segment_func(frame, device=device)
                except Exception as e:
                    print(f"[Tuner] CellSAM failed, falling back: {e}")
                    raw_mask = zero_shot_fluorescence_mask(frame)
            else:
                raw_mask = zero_shot_fluorescence_mask(frame)
            selected_raw_masks.append(raw_mask)

        session = TuningSession(session_id, selected_frames, selected_raw_masks)
        session.generate_initial_candidates()

        generate_session_overlays(session)

        tuning_sessions[session_id] = session

        return {
            "session_id": session_id,
            "round": session.round,
            "candidates": session.candidates,
            "max_rounds": session.max_rounds,
            "num_frames": len(selected_frames)
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/tune/feedback")
async def tune_feedback(
    session_id: str = Form(...),
    winner_idx: int = Form(...)
):
    if session_id not in tuning_sessions:
        return JSONResponse({"error": "Tuning session not found"}, status_code=404)
        
    session = tuning_sessions[session_id]
    
    try:
        session.record_feedback(winner_idx)
        best_rec, best_score = session.get_best_recommendation()
        
        if session.round >= session.max_rounds:
            return {
                "session_id": session_id,
                "completed": True,
                "best_params": best_rec,
                "best_score": best_score,
                "round": session.round
            }
            
        session.propose_next_candidates()
        generate_session_overlays(session)
        
        return {
            "session_id": session_id,
            "completed": False,
            "round": session.round,
            "candidates": session.candidates,
            "max_rounds": session.max_rounds,
            "best_params": best_rec,
            "best_score": best_score
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/api/tune/media/{session_id}/{candidate_idx}.png")
async def get_tune_media(session_id: str, candidate_idx: int):
    path = BACKEND_DIR / "uploads" / "tuner_overlays" / session_id / f"cand_{candidate_idx}.png"
    if not path.exists():
        return JSONResponse({"error": "Media not found"}, status_code=404)
    return FileResponse(path, media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

