#!/bin/bash
cd "$(dirname "$0")"

echo "======================================"
echo " Starting Fungi SAM2 Webapp (Mac)"
echo "======================================"

# Check if .venv exists, if not, create it
if [ ! -d ".venv" ]; then
    echo "Creating Python virtual environment (.venv)..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install fastapi, uvicorn, python-multipart in venv
echo "Ensuring basic backend dependencies (fastapi, uvicorn, python-multipart) are installed in venv..."
pip install fastapi uvicorn python-multipart

# Start backend in background
echo "Starting FastAPI backend..."
cd backend
../.venv/bin/python main.py &
BACKEND_PID=$!
cd ..

# Start frontend
echo "Starting Vite frontend..."
cd frontend
npm run dev &
FRONTEND_PID=$!

echo ""
echo "App is running! To stop, press Ctrl+C"
echo "Frontend: http://localhost:5173"
echo "Backend:  http://localhost:8000"

# Open browser
sleep 2
open http://localhost:5173

# Wait for Ctrl+C
trap 'kill $BACKEND_PID $FRONTEND_PID; exit' INT
wait

