#!/usr/bin/env python3
"""Run Fungi in Google Colab: FastAPI backend + built frontend on one port."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT / "backend"
STATIC_DIR = ROOT / "frontend" / "dist"

os.environ.setdefault("FUNGI_COLAB", "1")
sys.path.insert(0, str(BACKEND_DIR))
os.chdir(BACKEND_DIR)

from fastapi import HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from main import app  # noqa: E402


def mount_frontend() -> None:
    if not STATIC_DIR.exists():
        raise FileNotFoundError(
            f"Frontend build not found at {STATIC_DIR}. "
            "Run: cd frontend && npm install && npm run build"
        )

    assets_dir = STATIC_DIR / "assets"
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/favicon.svg", include_in_schema=False)
    async def favicon():
        path = STATIC_DIR / "favicon.svg"
        if path.is_file():
            return FileResponse(path)
        raise HTTPException(status_code=404)

    @app.get("/", include_in_schema=False)
    async def spa_root():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa_fallback(full_path: str):
        if full_path.startswith(("api/", "docs", "openapi", "redoc", "assets/")):
            raise HTTPException(status_code=404)
        candidate = STATIC_DIR / full_path
        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(STATIC_DIR / "index.html")


def main() -> None:
    mount_frontend()
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    print(f"Starting Fungi on 0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
