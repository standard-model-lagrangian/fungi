from sam2_pipeline import process_file, OUT_DIR, BACKEND_DIR
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import uuid
import pandas as pd
from pathlib import Path
from pydantic import BaseModel

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
