# Fungi AI Pipeline Frontend

This directory contains the user interface for the Fungi AI Pipeline, built using **React**, **TypeScript**, and **Vite**.

## 1. UI Aesthetic & Style Guide

The UI is designed to look like a retro-modern, handcrafted **scientific instrument readout** rather than a generic SaaS product. Key design tokens are defined in [index.css](src/index.css):

- **Typography**:
  - Headings (`h1`, `h2`, `h3`): `Silkscreen` (monospaced retro-pixel display font).
  - Body, Buttons, & Labels: `Space Mono` (utilitarian monospace).
- **Color Palette**:
  - Background: Muted black-navy (`#0a0e14`).
  - Panels: Solid dark grey-navy (`#12171f`) with a `1px` subtle border (`#1e2a3a`).
  - Accent color: Warm instrument amber (`#e6b450` / hover: `#ffcc5c`).
  - Success indicators: Subdued green (`#7fd962`).
- **Texture**:
  - A fixed scanline vertical gradient overlay (`body::after` repeating-linear-gradient) to simulate a physical CRT monitor.
  - Rectilinear flat panels with angular `4px` maximum border-radius (no glassmorphism, no round circles, no glowing SaaS gradients).

## 2. Key Modules & Layout

The main interface is defined in [App.tsx](src/App.tsx) and features two workspaces:

### Tab A: Analyze Video (Standard Mode)
- **Upload Zone**: A dashed drag-and-drop box for time-lapse videos or multi-page TIFF files.
- **Settings Drawer**: Configurable constants (Pixel size in \(\mu m\)/px, frame capture interval, minimum object size threshold, and DeepCell access token).
- **Execution Lifecycle**:
  - Submits file to `/api/upload`.
  - Polls `/api/status/{job_id}` for completion.
  - Renders a responsive dashboard:
    - **KPI Cards**: Max growth rate, average growth rate, final branch points count, and maximum tip counts.
    - **Overlay Video**: An HTML5 video player playing the processed mask/skeleton overlay on top of the original time-lapse (`/api/media/video`).
    - **Recharts Analytics**: A dual-axis time-series chart showing total hyphal length (\(\mu m\)) and branch points over time.

### Tab B: Auto-Tune Parameters (Calibration Mode)
For segmenting complex or noisy slides where standard parameters fail.
- **Multi-Frame Sampling**: Automatically extracts up to 5 evenly spaced frames from the uploaded video.
- **Preference Loop**:
  - Sends file to `/api/tune/start` to begin a tuning session.
  - Presents a **2x2 grid** showing 4 different parameter candidate overlays.
  - Overlays are vertically stacked composites of all 5 frames, allowing the researcher to evaluate parameter robustness across the entire time-lapse.
  - The researcher selects the preferred overlay option.
  - Submits choice to `/api/tune/feedback`, which runs a **Gaussian Process** model on the backend to propose the next 4 candidates.
  - Completes after 6 rounds, showing the optimized settings and allowing the researcher to "Apply Parameters" to the main segmentation form.

## 3. Development Commands

From the `frontend` directory:

- **Launch Development Server**:
  ```bash
  npm run dev
  ```
  Runs the local dev server at `http://localhost:5173`. Proxies backend requests to `http://localhost:8000/api`.

- **Type-Check & Build**:
  ```bash
  npm run build
  ```
  Compiles TypeScript and bundles static assets into the `dist/` directory.

- **Lint Code**:
  ```bash
  npm run lint
  ```
  Runs ESLint rules.
