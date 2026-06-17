# Fungi AI Pipeline - Frontend

This is the Vite + React + TypeScript frontend for the Fungi AI Pipeline dashboard.

It features a modern, premium glassmorphic UI for tracking automated microscopy analysis jobs, visualizing growth kinetics with Recharts, and playing back overlay videos.

## Core Components

- `App.tsx`: The primary dashboard containing the upload forms, settings drawer, job status polling, and metrics charting.
- `PreSegmentationSetup.tsx`: An interactive canvas-based UI step allowing users to define specific Regions of Interest (ROIs) before the main segmentation pipeline runs.
- `AnnotationView.tsx`: Displays the generated frame-by-frame outputs.

## Styling

Styling is handled using vanilla CSS (`index.css` and `App.css`) to enforce strict glassmorphic design tokens (blur, semi-transparent backgrounds, vibrant accent colors) without relying on utility frameworks like Tailwind.

## Running Locally

In the context of the larger project, you should run the application using the `start_mac.command` or `start_windows.bat` scripts located in the root directory.

If you are developing strictly on the frontend:

```bash
npm install
npm run dev
```

(Ensure the FastAPI backend is running concurrently on port 8000 for full functionality).
