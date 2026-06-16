# Fungi — Google Colab

Run the full Fungi web app (frontend + backend) in a single Colab notebook with GPU support.

## Quick start

1. Open [`Fungi_Colab.ipynb`](Fungi_Colab.ipynb) in Google Colab  
   ([Open in Colab](https://colab.research.google.com/github/standard-model-lagrangian/fungi/blob/main/colab/Fungi_Colab.ipynb))
2. **Runtime → Change runtime type → GPU** (recommended for CellSAM / SAM2)
3. Run all cells top to bottom
4. When the last cell finishes, click the link or use the popup window to open the app

## What the notebook does

1. Clones this repository from GitHub
2. Installs Node.js and builds the React frontend
3. Installs Python dependencies (PyTorch is already on Colab)
4. Downloads SAM2 weights on first run (~900 MB)
5. Starts a unified server on port **8000** (API + UI)
6. Opens the app inside Colab via `serve_kernel_port_as_window`

## Optional settings

| Variable | Description |
|----------|-------------|
| `REPO_URL` | Git clone URL (default: this repo) |
| `REPO_BRANCH` | Branch to check out (default: `main`) |
| `DEEPCELL_TOKEN` | [DeepCell token](https://users.deepcell.org) for CellSAM weights |

## Files

| File | Purpose |
|------|---------|
| `Fungi_Colab.ipynb` | Main Colab notebook |
| `serve_colab.py` | Serves API + built frontend on one port |
| `requirements-colab.txt` | Python packages (excludes torch; Colab provides it) |

## Notes

- **First run** can take 10–20 minutes while SAM2 / CellSAM dependencies install and weights download.
- Colab sessions time out; re-run the notebook to start a new session.
- Uploaded videos and results live under `backend/outputs/` and are lost when the runtime disconnects unless you download them.
- For local development, use `start_windows.bat` or `start_mac.command` in the repo root instead.
