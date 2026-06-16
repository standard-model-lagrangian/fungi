import { useCallback, useEffect, useRef, useState } from 'react'
import { Play, SkipForward, Eraser, Square, Pentagon, Paintbrush } from 'lucide-react'
import {
  SEGMENTATION_PRESETS,
  SMALL_OBJECT_WARNING,
  shouldShowSmallObjectWarning,
  presetById,
  supportsBacterialTracking,
  type TargetObjectType,
  type SkeletonizationMode,
} from './segmentationPresets'

const API_URL = 'http://localhost:8000/api'
const BACKEND_ORIGIN = 'http://localhost:8000'

type SetupLayer = 'ignore' | 'roi'
type SetupTool = 'brush' | 'polygon' | 'bbox' | 'erase'

interface PreviewFrame {
  frame_index: number
  frame_url: string
}

interface SetupInfo {
  total_frames: number
  image_width: number
  image_height: number
  preview_frames: Record<string, PreviewFrame>
  roi_defined: boolean
  ignore_defined: boolean
}

interface TemporalSettings {
  use_temporal_continuity: boolean
  temporal_memory_frames: number
  temporal_persistence_weight: number
  max_allowed_area_drop_fraction: number
  recover_missing_middle_frames: boolean
  allow_tip_growth: boolean
  repair_disconnected_tubes: boolean
  max_bridge_gap_px: number
  max_bridge_angle_degrees: number
  min_bridge_intensity_percentile: number
  branch_node_merge_radius_px: number
  branch_node_temporal_smoothing: boolean
  branch_node_max_tracking_distance_px: number
}

interface PreSegmentationSetupProps {
  jobId: string
  onRunSegmentation: () => void
  onSkipAndRun: () => void
  extracting: boolean
}

function ensureCanvas(ref: React.MutableRefObject<HTMLCanvasElement | null>, w: number, h: number) {
  if (!ref.current) {
    ref.current = document.createElement('canvas')
  }
  const canvas = ref.current
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w
    canvas.height = h
    const ctx = canvas.getContext('2d')
    if (ctx) {
      ctx.clearRect(0, 0, w, h)
    }
  }
  return canvas
}

async function loadMaskToCanvas(canvas: HTMLCanvasElement, url: string, fillWhite = false) {
  const ctx = canvas.getContext('2d')
  if (!ctx) return
  ctx.clearRect(0, 0, canvas.width, canvas.height)
  if (fillWhite) {
    ctx.fillStyle = '#fff'
    ctx.fillRect(0, 0, canvas.width, canvas.height)
  }
  try {
    const res = await fetch(url)
    if (!res.ok) return
    const blob = await res.blob()
    const img = await createImageBitmap(blob)
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height)
    img.close()
  } catch {
    if (fillWhite) {
      ctx.fillStyle = '#fff'
      ctx.fillRect(0, 0, canvas.width, canvas.height)
    }
  }
}

async function canvasToBlob(canvas: HTMLCanvasElement): Promise<Blob> {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) resolve(blob)
      else reject(new Error('Failed to export canvas'))
    }, 'image/png')
  })
}

function getDisplayMetrics(
  container: DOMRect,
  imageWidth: number,
  imageHeight: number,
) {
  const scale = Math.min(container.width / imageWidth, container.height / imageHeight)
  const drawWidth = imageWidth * scale
  const drawHeight = imageHeight * scale
  const offsetX = (container.width - drawWidth) / 2
  const offsetY = (container.height - drawHeight) / 2
  return { drawWidth, drawHeight, offsetX, offsetY, scale }
}

function imageCoordsFromClient(
  clientX: number,
  clientY: number,
  container: HTMLElement,
  imageWidth: number,
  imageHeight: number,
) {
  const rect = container.getBoundingClientRect()
  const layout = getDisplayMetrics(rect, imageWidth, imageHeight)
  const x = (clientX - rect.left - layout.offsetX) / layout.scale
  const y = (clientY - rect.top - layout.offsetY) / layout.scale
  if (x < 0 || y < 0 || x >= imageWidth || y >= imageHeight) return null
  return { x, y }
}

function paintBrush(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  radius: number,
  erase: boolean,
) {
  ctx.save()
  ctx.globalCompositeOperation = erase ? 'destination-out' : 'source-over'
  if (!erase) {
    ctx.fillStyle = '#fff'
  }
  ctx.beginPath()
  ctx.arc(x, y, radius, 0, Math.PI * 2)
  ctx.fill()
  ctx.restore()
}

function drawStroke(
  ctx: CanvasRenderingContext2D,
  x0: number,
  y0: number,
  x1: number,
  y1: number,
  radius: number,
  erase: boolean,
) {
  const dist = Math.hypot(x1 - x0, y1 - y0)
  const steps = Math.max(1, Math.ceil(dist / (radius * 0.5)))
  for (let i = 0; i <= steps; i++) {
    const t = i / steps
    paintBrush(ctx, x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, radius, erase)
  }
}

function fillPolygon(ctx: CanvasRenderingContext2D, points: { x: number; y: number }[], erase: boolean) {
  if (points.length < 3) return
  ctx.save()
  ctx.globalCompositeOperation = erase ? 'destination-out' : 'source-over'
  if (!erase) ctx.fillStyle = '#fff'
  ctx.beginPath()
  ctx.moveTo(points[0].x, points[0].y)
  for (let i = 1; i < points.length; i++) ctx.lineTo(points[i].x, points[i].y)
  ctx.closePath()
  ctx.fill()
  ctx.restore()
}

function fillBBox(
  ctx: CanvasRenderingContext2D,
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  erase: boolean,
) {
  const left = Math.min(x1, x2)
  const top = Math.min(y1, y2)
  const w = Math.abs(x2 - x1)
  const h = Math.abs(y2 - y1)
  if (w < 2 || h < 2) return
  ctx.save()
  ctx.globalCompositeOperation = erase ? 'destination-out' : 'source-over'
  if (!erase) ctx.fillStyle = '#fff'
  ctx.fillRect(left, top, w, h)
  ctx.restore()
}

function clearMaskCanvas(ctx: CanvasRenderingContext2D, width: number, height: number) {
  ctx.clearRect(0, 0, width, height)
}

function tintMask(
  ctx: CanvasRenderingContext2D,
  mask: HTMLCanvasElement,
  color: string,
  layout: ReturnType<typeof getDisplayMetrics>,
  imageWidth: number,
  imageHeight: number,
) {
  const tmp = document.createElement('canvas')
  tmp.width = imageWidth
  tmp.height = imageHeight
  const tctx = tmp.getContext('2d')
  if (!tctx) return
  tctx.drawImage(mask, 0, 0)
  tctx.globalCompositeOperation = 'source-in'
  tctx.fillStyle = color
  tctx.fillRect(0, 0, imageWidth, imageHeight)
  ctx.drawImage(tmp, layout.offsetX, layout.offsetY, layout.drawWidth, layout.drawHeight)
}

function drawRoiOutline(
  ctx: CanvasRenderingContext2D,
  mask: HTMLCanvasElement,
  layout: ReturnType<typeof getDisplayMetrics>,
  imageWidth: number,
  imageHeight: number,
) {
  const tmp = document.createElement('canvas')
  tmp.width = imageWidth
  tmp.height = imageHeight
  const tctx = tmp.getContext('2d')
  if (!tctx) return
  tctx.drawImage(mask, 0, 0)
  const data = tctx.getImageData(0, 0, imageWidth, imageHeight)
  const edge = document.createElement('canvas')
  edge.width = imageWidth
  edge.height = imageHeight
  const ectx = edge.getContext('2d')
  if (!ectx) return
  const out = ectx.createImageData(imageWidth, imageHeight)
  const w = imageWidth
  const h = imageHeight
  const px = data.data
  const isOn = (x: number, y: number) => px[(y * w + x) * 4] > 127
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      if (!isOn(x, y)) continue
      const border = !isOn(x - 1, y) || !isOn(x + 1, y) || !isOn(x, y - 1) || !isOn(x, y + 1)
      if (border) {
        const i = (y * w + x) * 4
        out.data[i] = 250
        out.data[i + 1] = 204
        out.data[i + 2] = 21
        out.data[i + 3] = 255
      }
    }
  }
  ectx.putImageData(out, 0, 0)
  ctx.drawImage(edge, layout.offsetX, layout.offsetY, layout.drawWidth, layout.drawHeight)
}

export default function PreSegmentationSetup({
  jobId,
  onRunSegmentation,
  onSkipAndRun,
  extracting,
}: PreSegmentationSetupProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const viewerRef = useRef<HTMLDivElement>(null)
  const imageRef = useRef<HTMLImageElement | null>(null)
  const ignoreRef = useRef<HTMLCanvasElement | null>(null)
  const roiRef = useRef<HTMLCanvasElement | null>(null)

  const [setupInfo, setSetupInfo] = useState<SetupInfo | null>(null)
  const [previewKey, setPreviewKey] = useState<'first' | 'middle' | 'last'>('first')
  const [layer, setLayer] = useState<SetupLayer>('ignore')
  const [tool, setTool] = useState<SetupTool>('brush')
  const [brushSize, setBrushSize] = useState(18)
  const [saveStatus, setSaveStatus] = useState<'idle' | 'saving' | 'saved'>('idle')
  const [masksReady, setMasksReady] = useState(false)

  const [pixelSize, setPixelSize] = useState('1.0')
  const [holeFillArea, setHoleFillArea] = useState('200')
  const [frameInterval, setFrameInterval] = useState('1.0')
  const [minObjectSize, setMinObjectSize] = useState('40')
  const [dilationRadius, setDilationRadius] = useState('8')
  const [minBranchLength, setMinBranchLength] = useState('10')
  const [deepcellToken, setDeepcellToken] = useState('')
  const [targetObjectType, setTargetObjectType] = useState<TargetObjectType>('fungal_hyphae')
  const [skeletonizationMode, setSkeletonizationMode] = useState<SkeletonizationMode>('hyphae_only')
  const [skeletonMinObjectArea, setSkeletonMinObjectArea] = useState('40')
  const [hyphaeMinLength, setHyphaeMinLength] = useState('12')
  const [hyphaeMinAspectRatio, setHyphaeMinAspectRatio] = useState('2.5')
  const [bacteriaMaxLength, setBacteriaMaxLength] = useState('15')
  const [bacteriaMaxArea, setBacteriaMaxArea] = useState('200')
  const [maxComponentCount, setMaxComponentCount] = useState('5000')
  const [enableBacterialTracking, setEnableBacterialTracking] = useState(true)
  const [maxBacteriaDisplacement, setMaxBacteriaDisplacement] = useState('20')
  const [maxTrackGapFrames, setMaxTrackGapFrames] = useState('2')
  const [minTrackLengthFrames, setMinTrackLengthFrames] = useState('2')
  const [trajectoryTailFrames, setTrajectoryTailFrames] = useState('20')
  const [generateTrajectoryVideo, setGenerateTrajectoryVideo] = useState(true)
  const [generateHeatmaps, setGenerateHeatmaps] = useState(true)
  const [maxObjectsPerFrameTracking, setMaxObjectsPerFrameTracking] = useState('5000')

  const [temporalSettings, setTemporalSettings] = useState<TemporalSettings>({
    use_temporal_continuity: true,
    temporal_memory_frames: 3,
    temporal_persistence_weight: 0.5,
    max_allowed_area_drop_fraction: 0.15,
    recover_missing_middle_frames: true,
    allow_tip_growth: true,
    repair_disconnected_tubes: false,
    max_bridge_gap_px: 15,
    max_bridge_angle_degrees: 45,
    min_bridge_intensity_percentile: 60,
    branch_node_merge_radius_px: 6,
    branch_node_temporal_smoothing: true,
    branch_node_max_tracking_distance_px: 10,
  })

  const drawingRef = useRef(false)
  const lastPointRef = useRef<{ x: number; y: number } | null>(null)
  const polygonRef = useRef<{ x: number; y: number }[]>([])
  const bboxStartRef = useRef<{ x: number; y: number } | null>(null)
  const bboxPreviewRef = useRef<{ x0: number; y0: number; x1: number; y1: number } | null>(null)
  const layerRef = useRef<SetupLayer>(layer)
  const saveTimerRef = useRef<number | null>(null)
  const paramsHydratedRef = useRef(false)

  useEffect(() => {
    layerRef.current = layer
  }, [layer])

  const imageWidth = setupInfo?.image_width ?? 0
  const imageHeight = setupInfo?.image_height ?? 0
  const previewFrame = setupInfo?.preview_frames?.[previewKey]

  const activePreset = presetById(targetObjectType)

  const scheduleSave = useCallback(() => {
    if (saveTimerRef.current) window.clearTimeout(saveTimerRef.current)
    saveTimerRef.current = window.setTimeout(async () => {
      const ignore = ignoreRef.current
      const roi = roiRef.current
      if (!ignore || !roi) return
      setSaveStatus('saving')
      try {
        const ignoreForm = new FormData()
        ignoreForm.append('ignore_mask', await canvasToBlob(ignore), 'ignore.png')
        await fetch(`${API_URL}/jobs/${jobId}/setup/ignore-mask`, { method: 'PUT', body: ignoreForm })

        const roiForm = new FormData()
        roiForm.append('roi_mask', await canvasToBlob(roi), 'roi.png')
        await fetch(`${API_URL}/jobs/${jobId}/setup/roi-mask`, { method: 'PUT', body: roiForm })

        setSaveStatus('saved')
        window.setTimeout(() => setSaveStatus('idle'), 1500)
      } catch {
        setSaveStatus('idle')
      }
    }, 400)
  }, [jobId])

  const drawCanvas = useCallback(() => {
    const canvas = canvasRef.current
    const viewer = viewerRef.current
    const img = imageRef.current
    const ignore = ignoreRef.current
    const roi = roiRef.current
    if (!canvas || !viewer || !img || !ignore || !roi || imageWidth <= 0) return

    const rect = viewer.getBoundingClientRect()
    const dpr = window.devicePixelRatio || 1
    const layout = getDisplayMetrics(rect, imageWidth, imageHeight)
    canvas.width = Math.max(1, Math.floor(rect.width * dpr))
    canvas.height = Math.max(1, Math.floor(rect.height * dpr))
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, rect.width, rect.height)
    ctx.fillStyle = '#000'
    ctx.fillRect(0, 0, rect.width, rect.height)
    ctx.drawImage(img, layout.offsetX, layout.offsetY, layout.drawWidth, layout.drawHeight)
    tintMask(ctx, ignore, 'rgba(60, 120, 255, 0.55)', layout, imageWidth, imageHeight)
    tintMask(ctx, roi, 'rgba(250, 204, 21, 0.18)', layout, imageWidth, imageHeight)
    drawRoiOutline(ctx, roi, layout, imageWidth, imageHeight)

    // Dim area outside a constrained processing ROI
    const roiCtx = roi.getContext('2d')
    if (roiCtx) {
      const roiData = roiCtx.getImageData(0, 0, imageWidth, imageHeight).data
      const dim = document.createElement('canvas')
      dim.width = imageWidth
      dim.height = imageHeight
      const dctx = dim.getContext('2d')
      if (dctx) {
        const dimImage = dctx.createImageData(imageWidth, imageHeight)
        let hasExcluded = false
        for (let i = 0; i < roiData.length; i += 4) {
          if (roiData[i] < 128) {
            dimImage.data[i] = 0
            dimImage.data[i + 1] = 0
            dimImage.data[i + 2] = 0
            dimImage.data[i + 3] = 140
            hasExcluded = true
          }
        }
        if (hasExcluded) {
          dctx.putImageData(dimImage, 0, 0)
          ctx.drawImage(dim, layout.offsetX, layout.offsetY, layout.drawWidth, layout.drawHeight)
        }
      }
    }

    if (tool === 'polygon' && polygonRef.current.length > 0) {
      ctx.save()
      ctx.strokeStyle = layer === 'ignore' ? 'rgba(60, 120, 255, 0.9)' : 'rgba(250, 204, 21, 0.9)'
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.moveTo(
        layout.offsetX + polygonRef.current[0].x * layout.scale,
        layout.offsetY + polygonRef.current[0].y * layout.scale,
      )
      for (let i = 1; i < polygonRef.current.length; i++) {
        ctx.lineTo(
          layout.offsetX + polygonRef.current[i].x * layout.scale,
          layout.offsetY + polygonRef.current[i].y * layout.scale,
        )
      }
      ctx.stroke()
      ctx.restore()
    }

    if (tool === 'bbox' && bboxPreviewRef.current) {
      const { x0, y0, x1, y1 } = bboxPreviewRef.current
      const left = layout.offsetX + Math.min(x0, x1) * layout.scale
      const top = layout.offsetY + Math.min(y0, y1) * layout.scale
      const w = Math.abs(x1 - x0) * layout.scale
      const h = Math.abs(y1 - y0) * layout.scale
      ctx.save()
      ctx.strokeStyle = layer === 'ignore' ? 'rgba(60, 120, 255, 0.95)' : 'rgba(250, 204, 21, 0.95)'
      ctx.fillStyle = layer === 'ignore' ? 'rgba(60, 120, 255, 0.2)' : 'rgba(250, 204, 21, 0.15)'
      ctx.lineWidth = 2
      ctx.setLineDash([6, 4])
      ctx.fillRect(left, top, w, h)
      ctx.strokeRect(left, top, w, h)
      ctx.restore()
    }
  }, [imageWidth, imageHeight, layer, tool])

  const hydrateParamsFromServer = useCallback((params: Record<string, unknown>) => {
    setPixelSize(String(params.pixel_size_um ?? '1.0'))
    setHoleFillArea(String(params.hole_fill_area ?? '200'))
    setFrameInterval(String(params.frame_interval_min ?? '1.0'))
    setMinObjectSize(String(params.min_object_size_px ?? '40'))
    setDilationRadius(String(params.dilation_radius ?? '8'))
    setMinBranchLength(String(params.min_branch_length_px ?? '10'))
    if (params.deepcell_token) setDeepcellToken(String(params.deepcell_token))
    if (params.target_object_type || params.segmentation_preset) {
      setTargetObjectType(
        (params.target_object_type ?? params.segmentation_preset) as TargetObjectType,
      )
    }
    if (params.skeletonization_mode) {
      setSkeletonizationMode(params.skeletonization_mode as SkeletonizationMode)
    }
    if (params.skeleton_min_object_area_px != null) {
      setSkeletonMinObjectArea(String(params.skeleton_min_object_area_px))
    }
    if (params.hyphae_min_length_px != null) {
      setHyphaeMinLength(String(params.hyphae_min_length_px))
    }
    if (params.hyphae_min_aspect_ratio != null) {
      setHyphaeMinAspectRatio(String(params.hyphae_min_aspect_ratio))
    }
    if (params.bacteria_max_length_px != null) {
      setBacteriaMaxLength(String(params.bacteria_max_length_px))
    }
    if (params.bacteria_max_area_px2 != null) {
      setBacteriaMaxArea(String(params.bacteria_max_area_px2))
    }
    if (params.max_component_count_threshold != null) {
      setMaxComponentCount(String(params.max_component_count_threshold))
    }
    if (params.enable_bacterial_tracking != null) {
      setEnableBacterialTracking(Boolean(params.enable_bacterial_tracking))
    }
    if (params.max_bacteria_displacement_px != null) {
      setMaxBacteriaDisplacement(String(params.max_bacteria_displacement_px))
    }
    if (params.max_track_gap_frames != null) {
      setMaxTrackGapFrames(String(params.max_track_gap_frames))
    }
    if (params.min_track_length_frames != null) {
      setMinTrackLengthFrames(String(params.min_track_length_frames))
    }
    if (params.trajectory_tail_frames != null) {
      setTrajectoryTailFrames(String(params.trajectory_tail_frames))
    }
    if (params.generate_trajectory_overlay_video != null) {
      setGenerateTrajectoryVideo(Boolean(params.generate_trajectory_overlay_video))
    }
    if (params.generate_heatmaps != null) {
      setGenerateHeatmaps(Boolean(params.generate_heatmaps))
    }
    if (params.max_objects_per_frame_for_tracking != null) {
      setMaxObjectsPerFrameTracking(String(params.max_objects_per_frame_for_tracking))
    }
  }, [])

  const loadSetup = useCallback(async () => {
    const res = await fetch(`${API_URL}/jobs/${jobId}/setup/info`)
    if (!res.ok) return
    const data = await res.json()
    setSetupInfo(data)
    if (!paramsHydratedRef.current && data.params) {
      paramsHydratedRef.current = true
      hydrateParamsFromServer(data.params)
    }
  }, [jobId, hydrateParamsFromServer])

  const applyPreset = (presetId: TargetObjectType) => {
    paramsHydratedRef.current = true
    const preset = SEGMENTATION_PRESETS.find((p) => p.id === presetId) ?? SEGMENTATION_PRESETS[0]
    setTargetObjectType(preset.id)
    setMinObjectSize(String(preset.values.min_object_size_px))
    setHoleFillArea(String(preset.values.hole_fill_area))
    setDilationRadius(String(preset.values.dilation_radius))
    setMinBranchLength(String(preset.values.min_branch_length_px))
    setSkeletonizationMode(preset.values.skeletonization_mode)
    setSkeletonMinObjectArea(String(preset.values.skeleton_min_object_area_px))
    setHyphaeMinLength(String(preset.values.classification.hyphae_min_length_px))
    setHyphaeMinAspectRatio(String(preset.values.classification.hyphae_min_aspect_ratio))
    setBacteriaMaxLength(String(preset.values.classification.bacteria_max_length_px))
    setBacteriaMaxArea(String(preset.values.classification.bacteria_max_area_px2))
    setMaxComponentCount(String(preset.values.classification.max_component_count_threshold))
    setEnableBacterialTracking(preset.values.tracking.enable_bacterial_tracking)
    setMaxBacteriaDisplacement(String(preset.values.tracking.max_bacteria_displacement_px))
    setMaxTrackGapFrames(String(preset.values.tracking.max_track_gap_frames))
    setMinTrackLengthFrames(String(preset.values.tracking.min_track_length_frames))
    setTrajectoryTailFrames(String(preset.values.tracking.trajectory_tail_frames))
    setGenerateTrajectoryVideo(preset.values.tracking.generate_trajectory_overlay_video)
    setGenerateHeatmaps(preset.values.tracking.generate_heatmaps)
    setMaxObjectsPerFrameTracking(String(preset.values.tracking.max_objects_per_frame_for_tracking))
    setTemporalSettings((prev) => ({
      ...prev,
      use_temporal_continuity: preset.values.use_temporal_continuity,
      repair_disconnected_tubes: preset.values.repair_disconnected_tubes,
    }))
  }

  const loadMasks = useCallback(async (w: number, h: number) => {
    const ignore = ensureCanvas(ignoreRef, w, h)
    const roi = ensureCanvas(roiRef, w, h)
    await Promise.all([
      loadMaskToCanvas(ignore, `${BACKEND_ORIGIN}/api/jobs/${jobId}/setup/ignore-mask?t=${Date.now()}`),
      loadMaskToCanvas(roi, `${BACKEND_ORIGIN}/api/jobs/${jobId}/setup/roi-mask?t=${Date.now()}`, true),
    ])
    setMasksReady(true)
    drawCanvas()
  }, [jobId, drawCanvas])

  useEffect(() => {
    void loadSetup()
    const interval = window.setInterval(() => void loadSetup(), 1500)
    return () => window.clearInterval(interval)
  }, [loadSetup])

  useEffect(() => {
    void fetch(`${API_URL}/jobs/${jobId}/temporal-settings`).then(async (r) => {
      if (r.ok) setTemporalSettings(await r.json())
    })
  }, [jobId])

  useEffect(() => {
    if (!previewFrame || imageWidth <= 0 || imageHeight <= 0) return
    const img = new Image()
    img.crossOrigin = 'anonymous'
    img.onload = () => {
      imageRef.current = img
      void loadMasks(imageWidth, imageHeight)
    }
    img.src = `${BACKEND_ORIGIN}${previewFrame.frame_url}`
  }, [previewFrame, imageWidth, imageHeight, loadMasks])

  useEffect(() => {
    drawCanvas()
  }, [drawCanvas, masksReady, previewKey])

  const activeMaskCanvas = () => (layerRef.current === 'ignore' ? ignoreRef.current : roiRef.current)

  const prepareRoiReplace = (ctx: CanvasRenderingContext2D, target: HTMLCanvasElement, erase: boolean) => {
    if (layerRef.current === 'roi' && !erase) {
      clearMaskCanvas(ctx, target.width, target.height)
    }
  }

  const finishShape = () => {
    const target = activeMaskCanvas()
    if (!target) return
    const ctx = target.getContext('2d')
    if (!ctx) return
    const erase = tool === 'erase'

    if (tool === 'polygon' && polygonRef.current.length >= 3) {
      prepareRoiReplace(ctx, target, erase)
      fillPolygon(ctx, polygonRef.current, erase)
      polygonRef.current = []
      scheduleSave()
      drawCanvas()
      return
    }

    if (tool === 'bbox' && bboxStartRef.current && lastPointRef.current) {
      const start = bboxStartRef.current
      const end = lastPointRef.current
      const w = Math.abs(end.x - start.x)
      const h = Math.abs(end.y - start.y)
      if (w >= 2 && h >= 2) {
        prepareRoiReplace(ctx, target, erase)
        fillBBox(ctx, start.x, start.y, end.x, end.y, erase)
        scheduleSave()
      }
      bboxStartRef.current = null
      bboxPreviewRef.current = null
      drawCanvas()
    }
  }

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!masksReady || extracting) return
    const coords = imageCoordsFromClient(e.clientX, e.clientY, viewerRef.current!, imageWidth, imageHeight)
    if (!coords) return
    const target = activeMaskCanvas()
    if (!target) return
    const ctx = target.getContext('2d')
    if (!ctx) return
    const erase = tool === 'erase'

    if (tool === 'brush' || tool === 'erase') {
      drawingRef.current = true
      paintBrush(ctx, coords.x, coords.y, brushSize / 2, erase)
      lastPointRef.current = coords
      scheduleSave()
      drawCanvas()
      return
    }

    if (tool === 'polygon') {
      if (polygonRef.current.length >= 3) {
        const first = polygonRef.current[0]
        if (Math.hypot(coords.x - first.x, coords.y - first.y) < 10) {
          finishShape()
          return
        }
      }
      polygonRef.current = [...polygonRef.current, coords]
      drawCanvas()
      return
    }

    if (tool === 'bbox') {
      drawingRef.current = true
      bboxStartRef.current = coords
      lastPointRef.current = coords
      bboxPreviewRef.current = { x0: coords.x, y0: coords.y, x1: coords.x, y1: coords.y }
      drawCanvas()
      return
    }
  }

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!masksReady) return
    const coords = imageCoordsFromClient(e.clientX, e.clientY, viewerRef.current!, imageWidth, imageHeight)
    if (!coords) return

    if ((tool === 'brush' || tool === 'erase') && drawingRef.current) {
      const target = activeMaskCanvas()
      const ctx = target?.getContext('2d')
      if (!ctx || !lastPointRef.current) return
      drawStroke(ctx, lastPointRef.current.x, lastPointRef.current.y, coords.x, coords.y, brushSize / 2, tool === 'erase')
      lastPointRef.current = coords
      scheduleSave()
      drawCanvas()
      return
    }

    if (tool === 'bbox' && drawingRef.current && bboxStartRef.current) {
      lastPointRef.current = coords
      bboxPreviewRef.current = {
        x0: bboxStartRef.current.x,
        y0: bboxStartRef.current.y,
        x1: coords.x,
        y1: coords.y,
      }
      drawCanvas()
    }
  }

  const handleMouseUp = () => {
    if (tool === 'bbox' && drawingRef.current) {
      finishShape()
    }
    drawingRef.current = false
    lastPointRef.current = null
  }

  const persistSettingsAndRun = async (skipRoiOnly = false) => {
    await fetch(`${API_URL}/jobs/${jobId}/params`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        pixel_size_um: parseFloat(pixelSize),
        hole_fill_area: Math.max(0, parseInt(holeFillArea, 10) || 0),
        frame_interval_min: parseFloat(frameInterval),
        min_object_size_px: Math.max(1, parseInt(minObjectSize, 10) || 1),
        dilation_radius: Math.max(0, parseInt(dilationRadius, 10) || 0),
        min_branch_length_px: Math.max(1, parseInt(minBranchLength, 10) || 1),
        deepcell_token: deepcellToken || null,
        target_object_type: targetObjectType,
        segmentation_preset: targetObjectType,
        skeletonization_mode: skeletonizationMode,
        skeleton_min_object_area_px: Math.max(1, parseInt(skeletonMinObjectArea, 10) || 1),
        hyphae_min_length_px: Math.max(1, parseInt(hyphaeMinLength, 10) || 1),
        hyphae_min_aspect_ratio: Math.max(1, parseFloat(hyphaeMinAspectRatio) || 1),
        bacteria_max_length_px: Math.max(1, parseInt(bacteriaMaxLength, 10) || 1),
        bacteria_max_area_px2: Math.max(1, parseInt(bacteriaMaxArea, 10) || 1),
        max_component_count_threshold: Math.max(100, parseInt(maxComponentCount, 10) || 100),
        enable_bacterial_tracking: enableBacterialTracking,
        max_bacteria_displacement_px: Math.max(1, parseInt(maxBacteriaDisplacement, 10) || 1),
        max_track_gap_frames: Math.max(0, parseInt(maxTrackGapFrames, 10) || 0),
        min_track_length_frames: Math.max(1, parseInt(minTrackLengthFrames, 10) || 1),
        trajectory_tail_frames: Math.max(1, parseInt(trajectoryTailFrames, 10) || 1),
        generate_trajectory_overlay_video: generateTrajectoryVideo,
        generate_heatmaps: generateHeatmaps,
        max_objects_per_frame_for_tracking: Math.max(
          100,
          parseInt(maxObjectsPerFrameTracking, 10) || 100,
        ),
      }),
    })
    await fetch(`${API_URL}/jobs/${jobId}/temporal-settings`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(temporalSettings),
    })
    const ignore = ignoreRef.current
    const roi = roiRef.current
    if (ignore && roi) {
      if (skipRoiOnly) {
        const ctxI = ignore.getContext('2d')
        const ctxR = roi.getContext('2d')
        if (ctxI) { ctxI.clearRect(0, 0, ignore.width, ignore.height) }
        if (ctxR) {
          ctxR.clearRect(0, 0, roi.width, roi.height)
          ctxR.fillStyle = '#fff'
          ctxR.fillRect(0, 0, roi.width, roi.height)
        }
      }
      const ignoreForm = new FormData()
      ignoreForm.append('ignore_mask', await canvasToBlob(ignore), 'ignore.png')
      await fetch(`${API_URL}/jobs/${jobId}/setup/ignore-mask`, { method: 'PUT', body: ignoreForm })
      const roiForm = new FormData()
      roiForm.append('roi_mask', await canvasToBlob(roi), 'roi.png')
      await fetch(`${API_URL}/jobs/${jobId}/setup/roi-mask`, { method: 'PUT', body: roiForm })
    }
    await fetch(`${API_URL}/jobs/${jobId}/setup/metadata`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ setup_completed: true, roi_skipped: skipRoiOnly }),
    })
    if (skipRoiOnly) onSkipAndRun()
    else onRunSegmentation()
  }

  const inputStyle = {
    width: '100%',
    padding: 8,
    borderRadius: 6,
    border: '1px solid var(--panel-border)',
    background: 'rgba(0,0,0,0.2)',
    color: 'var(--text-primary)',
  } as const

  return (
    <div className="setup-workspace">
      <div className="setup-main glass-panel">
        <div className="setup-viewer-header">
          <h2>Pre-Segmentation ROI Setup</h2>
          {extracting && <span className="setup-status">Extracting frames…</span>}
          {saveStatus === 'saving' && <span className="setup-status">Saving…</span>}
          {saveStatus === 'saved' && <span className="setup-status">Saved</span>}
        </div>

        <div className="setup-preview-tabs">
          {(['first', 'middle', 'last'] as const).map((key) => (
            <button
              key={key}
              type="button"
              className={`setup-tab ${previewKey === key ? 'active' : ''}`}
              onClick={() => setPreviewKey(key)}
              disabled={!setupInfo?.preview_frames?.[key]}
            >
              {key === 'first' ? 'First Frame' : key === 'middle' ? 'Middle Frame' : 'Last Frame'}
            </button>
          ))}
        </div>

        <div className="setup-viewer" ref={viewerRef}>
          <canvas
            ref={canvasRef}
            className="setup-canvas"
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
          />
          {!masksReady && !extracting && <div className="setup-viewer-placeholder">Loading preview…</div>}
        </div>

        <p className="setup-legend">Ignore: blue · Processing ROI: yellow outline</p>

        <div className="setup-tool-row">
          <button type="button" className={`setup-tool-btn ${layer === 'ignore' ? 'active ignore' : ''}`} onClick={() => setLayer('ignore')}>Ignore Region</button>
          <button type="button" className={`setup-tool-btn ${layer === 'roi' ? 'active roi' : ''}`} onClick={() => setLayer('roi')}>Processing ROI</button>
        </div>

        <div className="setup-tool-row">
          {([
            ['brush', Paintbrush, 'Brush'],
            ['polygon', Pentagon, 'Polygon'],
            ['bbox', Square, 'BBox'],
            ['erase', Eraser, 'Erase'],
          ] as const).map(([id, Icon, label]) => (
            <button
              key={id}
              type="button"
              className={`setup-tool-btn ${tool === id ? 'active' : ''}`}
              onClick={() => {
                setTool(id)
                polygonRef.current = []
                bboxStartRef.current = null
                bboxPreviewRef.current = null
                drawingRef.current = false
              }}
            >
              <Icon size={14} /> {label}
            </button>
          ))}
          <label className="setup-brush-size">
            Brush
            <input type="range" min={4} max={80} value={brushSize} onChange={(e) => setBrushSize(parseInt(e.target.value, 10))} />
            {brushSize}px
          </label>
        </div>
      </div>

      <aside className="setup-sidebar">
        <section className="setup-section">
          <h3>Target Object Type</h3>
          <div className="setup-radio-group">
            {SEGMENTATION_PRESETS.map((preset) => (
              <label key={preset.id} className="setup-radio-row">
                <input
                  type="radio"
                  name="targetObjectType"
                  value={preset.id}
                  checked={targetObjectType === preset.id}
                  onChange={() => applyPreset(preset.id)}
                />
                <span>{preset.label}</span>
              </label>
            ))}
          </div>
        </section>

        <section className="setup-section">
          <h3>Object Classification</h3>
          <div className="setup-grid">
            <label>Hyphae Min Length (px)<input type="number" step="1" min={1} value={hyphaeMinLength} onChange={(e) => setHyphaeMinLength(e.target.value)} style={inputStyle} /></label>
            <label>Hyphae Min Aspect Ratio<input type="number" step="0.1" min={1} value={hyphaeMinAspectRatio} onChange={(e) => setHyphaeMinAspectRatio(e.target.value)} style={inputStyle} /></label>
            <label>Bacteria Max Length (px)<input type="number" step="1" min={1} value={bacteriaMaxLength} onChange={(e) => setBacteriaMaxLength(e.target.value)} style={inputStyle} /></label>
            <label>Bacteria Max Area (px²)<input type="number" step="1" min={1} value={bacteriaMaxArea} onChange={(e) => setBacteriaMaxArea(e.target.value)} style={inputStyle} /></label>
            <label>Max Component Count<input type="number" step="100" min={100} value={maxComponentCount} onChange={(e) => setMaxComponentCount(e.target.value)} style={inputStyle} /></label>
          </div>
        </section>

        <section className="setup-section">
          <h3>Segmentation Parameters</h3>
          {shouldShowSmallObjectWarning(parseInt(minObjectSize, 10) || 0) && (
            <p className="setup-warning">{SMALL_OBJECT_WARNING}</p>
          )}
          <div className="setup-grid">
            <label>Pixel Size (µm/px)<input type="number" step="0.01" min={0.001} value={pixelSize} onChange={(e) => setPixelSize(e.target.value)} style={inputStyle} /></label>
            <label>Hole Fill Area<input type="number" step="1" min={0} value={holeFillArea} onChange={(e) => setHoleFillArea(e.target.value)} style={inputStyle} /></label>
            <label>Frame Interval (min)<input type="number" step="0.1" min={0.001} value={frameInterval} onChange={(e) => setFrameInterval(e.target.value)} style={inputStyle} /></label>
            <label>Min Object Size (px)<input type="number" step="1" min={1} value={minObjectSize} onChange={(e) => setMinObjectSize(e.target.value)} style={inputStyle} /></label>
            <label>Dilation Radius (px)<input type="number" step="1" min={0} value={dilationRadius} onChange={(e) => setDilationRadius(e.target.value)} style={inputStyle} /></label>
            <label>Min Branch Length (px)<input type="number" step="1" min={1} value={minBranchLength} onChange={(e) => setMinBranchLength(e.target.value)} style={inputStyle} /></label>
          </div>
          {activePreset.workflow.enable_skeletonization && (
            <>
              <label style={{ display: 'block', marginTop: 12 }}>Skeletonization Mode
                <select
                  value={skeletonizationMode}
                  onChange={(e) => setSkeletonizationMode(e.target.value as SkeletonizationMode)}
                  style={inputStyle}
                >
                  <option value="hyphae_only">Hyphae Only</option>
                  <option value="all_objects">All Objects</option>
                </select>
              </label>
              {skeletonizationMode === 'hyphae_only' && (
                <label style={{ display: 'block', marginTop: 12 }}>Skeleton Min Object Area (px)
                  <input
                    type="number"
                    step="1"
                    min={1}
                    value={skeletonMinObjectArea}
                    onChange={(e) => setSkeletonMinObjectArea(e.target.value)}
                    style={inputStyle}
                  />
                </label>
              )}
            </>
          )}
          <label style={{ display: 'block', marginTop: 12 }}>DeepCell Token
            <input type="password" value={deepcellToken} onChange={(e) => setDeepcellToken(e.target.value)} style={inputStyle} />
          </label>
        </section>

        {supportsBacterialTracking(targetObjectType) && (
          <section className="setup-section">
            <h3>Bacterial Tracking</h3>
            <label className="annotation-toggle-row">
              <input
                type="checkbox"
                checked={enableBacterialTracking}
                onChange={(e) => setEnableBacterialTracking(e.target.checked)}
              />
              <span>Enable Bacterial Tracking</span>
            </label>
            <div className="setup-grid">
              <label>Max Displacement (px)
                <input type="number" min={1} value={maxBacteriaDisplacement} onChange={(e) => setMaxBacteriaDisplacement(e.target.value)} style={inputStyle} />
              </label>
              <label>Max Track Gap (frames)
                <input type="number" min={0} value={maxTrackGapFrames} onChange={(e) => setMaxTrackGapFrames(e.target.value)} style={inputStyle} />
              </label>
              <label>Min Track Length (frames)
                <input type="number" min={1} value={minTrackLengthFrames} onChange={(e) => setMinTrackLengthFrames(e.target.value)} style={inputStyle} />
              </label>
              <label>Trajectory Tail (frames)
                <input type="number" min={1} value={trajectoryTailFrames} onChange={(e) => setTrajectoryTailFrames(e.target.value)} style={inputStyle} />
              </label>
              <label>Max Objects / Frame
                <input type="number" min={100} value={maxObjectsPerFrameTracking} onChange={(e) => setMaxObjectsPerFrameTracking(e.target.value)} style={inputStyle} />
              </label>
            </div>
            <label className="annotation-toggle-row">
              <input type="checkbox" checked={generateTrajectoryVideo} onChange={(e) => setGenerateTrajectoryVideo(e.target.checked)} />
              <span>Generate Trajectory Overlay Video</span>
            </label>
            <label className="annotation-toggle-row">
              <input type="checkbox" checked={generateHeatmaps} onChange={(e) => setGenerateHeatmaps(e.target.checked)} />
              <span>Generate Heatmaps</span>
            </label>
            <p className="setup-hint">Pixel size and frame interval above are used for speed and displacement units.</p>
          </section>
        )}

        <section className="setup-section">
          <h3>Temporal Continuity</h3>
          <label className="annotation-toggle-row">
            <input type="checkbox" checked={temporalSettings.use_temporal_continuity} onChange={(e) => setTemporalSettings({ ...temporalSettings, use_temporal_continuity: e.target.checked })} />
            <span>Use Temporal Continuity</span>
          </label>
          <div className="setup-grid">
            <label>Memory Frames<input type="number" value={temporalSettings.temporal_memory_frames} onChange={(e) => setTemporalSettings({ ...temporalSettings, temporal_memory_frames: parseInt(e.target.value, 10) })} style={inputStyle} /></label>
            <label>Persistence Weight<input type="number" step="0.05" value={temporalSettings.temporal_persistence_weight} onChange={(e) => setTemporalSettings({ ...temporalSettings, temporal_persistence_weight: parseFloat(e.target.value) })} style={inputStyle} /></label>
            <label>Max Area Drop<input type="number" step="0.01" value={temporalSettings.max_allowed_area_drop_fraction} onChange={(e) => setTemporalSettings({ ...temporalSettings, max_allowed_area_drop_fraction: parseFloat(e.target.value) })} style={inputStyle} /></label>
            {activePreset.workflow.enable_tube_repair && (
              <>
                <label>Max Bridge Gap (px)<input type="number" value={temporalSettings.max_bridge_gap_px} onChange={(e) => setTemporalSettings({ ...temporalSettings, max_bridge_gap_px: parseInt(e.target.value, 10) })} style={inputStyle} /></label>
                <label>Max Bridge Angle (°)<input type="number" value={temporalSettings.max_bridge_angle_degrees} onChange={(e) => setTemporalSettings({ ...temporalSettings, max_bridge_angle_degrees: parseFloat(e.target.value) })} style={inputStyle} /></label>
              </>
            )}
            {activePreset.workflow.enable_branch_detection && (
              <>
                <label>Branch Merge Radius (px)<input type="number" value={temporalSettings.branch_node_merge_radius_px} onChange={(e) => setTemporalSettings({ ...temporalSettings, branch_node_merge_radius_px: parseInt(e.target.value, 10) })} style={inputStyle} /></label>
                <label>Branch Track Distance (px)<input type="number" value={temporalSettings.branch_node_max_tracking_distance_px} onChange={(e) => setTemporalSettings({ ...temporalSettings, branch_node_max_tracking_distance_px: parseInt(e.target.value, 10) })} style={inputStyle} /></label>
              </>
            )}
          </div>
          <label className="annotation-toggle-row">
            <input type="checkbox" checked={temporalSettings.recover_missing_middle_frames} onChange={(e) => setTemporalSettings({ ...temporalSettings, recover_missing_middle_frames: e.target.checked })} />
            <span>Recover Missing Middle Frames</span>
          </label>
          {activePreset.workflow.enable_tube_repair && (
            <label className="annotation-toggle-row">
              <input type="checkbox" checked={temporalSettings.repair_disconnected_tubes} onChange={(e) => setTemporalSettings({ ...temporalSettings, repair_disconnected_tubes: e.target.checked })} />
              <span>Repair Disconnected Tubes</span>
            </label>
          )}
          {activePreset.workflow.enable_branch_detection && (
            <label className="annotation-toggle-row">
              <input type="checkbox" checked={temporalSettings.branch_node_temporal_smoothing} onChange={(e) => setTemporalSettings({ ...temporalSettings, branch_node_temporal_smoothing: e.target.checked })} />
              <span>Branch Node Temporal Smoothing</span>
            </label>
          )}
        </section>

        <section className="setup-actions">
          <button type="button" className="btn" disabled={extracting} onClick={() => void persistSettingsAndRun(false)}>
            <Play size={18} /> Run Automated Segmentation
          </button>
          <button type="button" className="btn btn-secondary" disabled={extracting} onClick={() => void persistSettingsAndRun(true)}>
            <SkipForward size={18} /> Skip ROI Setup &amp; Run
          </button>
        </section>
      </aside>
    </div>
  )
}
