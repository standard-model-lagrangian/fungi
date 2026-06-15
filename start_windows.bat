@echo off
setlocal
cd /d "%~dp0"

echo ======================================
echo  Starting Fungi SAM2 Webapp (Windows)
echo ======================================

rem Check if .venv exists, if not, create it
if not exist ".venv" (
    echo Creating Python virtual environment (.venv)...
    python -m venv .venv
)

rem Upgrade pip in venv
echo Upgrading pip in venv...
.venv\Scripts\python.exe -m pip install --upgrade pip

rem Install fastapi, uvicorn, python-multipart in venv
echo Ensuring basic backend dependencies (fastapi, uvicorn, python-multipart) are installed in venv...
.venv\Scripts\python.exe -m pip install fastapi uvicorn python-multipart

echo Starting FastAPI backend...
cd backend
start /b ..\.venv\Scripts\python.exe main.py
cd ..

echo Starting Vite frontend...
cd frontend
start /b npm run dev

echo.
echo App is running! Close this window to stop (or terminate python/node processes in Task Manager).
echo Frontend: http://localhost:5173
echo Backend:  http://localhost:8000

timeout /t 3 /nobreak >nul
start http://localhost:5173

