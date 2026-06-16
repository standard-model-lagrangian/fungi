import { useCallback, useEffect, useRef, useState } from 'react'
import {
  ChevronLeft,
  ChevronRight,
  Eraser,
  Play,
  Pause,
  Layers,
  Bookmark,
  Image as ImageIcon,
  BarChart3,
  RotateCcw,
  Undo2,
  Redo2,
  Plus,
  Minus,
  ScanEye,
} from 'lucide-react'

import { API_URL, BACKEND_ORIGIN } from './config'

export type ViewerMode = 'overlay' | 'skeleton' | 'original' | 'bacterial_tracking'
export type CorrectionTool = 'add' | 'remove' | 'static' | 'erase_correction'

export interface FrameAsset {
  frame_index: number
  original_frame_url: string | null
  auto_overlay_url: string | null
  auto_mask_url: string | null
  skeleton_frame_url: string | null
  bacterial_tracking_overlay_url?: string | null
}

interface BacterialTrackingInfo {
  available: boolean
  label_overlay_video_url?: string | null
  label_overlay_frame_url_template?: string | null
}

interface SegmentationPreviewResult {
  frame_index: number
  auto_overlay_url?: string
  guided_overlay_url?: string
  difference_map_url?: string
  auto_metrics?: Record<string, number | string>
  guided_metrics?: Record<string, number | string>
}

export interface FrameAnnotation {
  frame_index: number
  image_width: number
  image_height: number
  is_keyframe?: boolean
  object_points: number[][]
  background_points: number[][]
  static_background_points: number[][]
  object_brush_strokes: { points: number[][]; size: number }[]
  background_brush_strokes: { points: number[][]; size: number }[]
  static_background_brush_strokes: { points: number[][]; size: number }[]
  bounding_boxes: { x1: number; y1: number; x2: number; y2: number }[]
}

interface PropagationMetadata {
  frame_index: number
  directly_annotated: boolean
  source_keyframe: number | null
  propagation_distance: number | null
  mode: string
  bridged_gaps?: number
}

interface PropagationSettings {
  propagation_window: number
  use_temporal_propagation: boolean
  repair_disconnected_tubes: boolean
  max_bridge_gap_px: number
  max_bridge_angle_degrees: number
}

interface TemporalMetadata {
  frame_index: number
  pixels_recovered: number
  pixels_removed_flicker: number
  bridges_added: number
  middle_frame_recovery_applied: boolean
}

interface TemporalSettings {
  use_temporal_continuity: boolean
  temporal_memory_frames: number
  temporal_persistence_weight: number
  max_allowed_area_drop_fraction: number
  recover_missing_middle_frames: boolean
  min_temporal_component_persistence: number
  allow_tip_growth: boolean
  repair_disconnected_tubes: boolean
  max_bridge_gap_px: number
  max_bridge_angle_degrees: number
  min_bridge_intensity_percentile: number
}

interface CorrectionSnapshot {
  add: ImageData
  remove: ImageData
  static: ImageData
}

interface KeyframeInfo {
  frame_index: number
  is_keyframe: boolean
  has_guidance: boolean
}

interface DisplayMetrics {
  rect: DOMRect
  drawWidth: number
  drawHeight: number
  offsetX: number
  offsetY: number
}

const VIEWER_MODES: { id: ViewerMode; label: string }[] = [
  { id: 'overlay', label: 'Auto Overlay' },
  { id: 'skeleton', label: 'Skeletonised Frame' },
  { id: 'original', label: 'Original Frame' },
]

const CORRECTION_COLORS = {
  auto: 'rgba(40, 220, 120, 0.35)',
  add: 'rgba(0, 220, 255, 0.55)',
  remove: 'rgba(255, 0, 200, 0.55)',
  static: 'rgba(60, 120, 255, 0.55)',
} as const

const CORRECTION_TOOL_LABELS: Record<CorrectionTool, string> = {
  add: 'Add Segmentation',
  remove: 'Remove Segmentation',
  static: 'Static Background',
  erase_correction: 'Erase Correction',
}

function getDisplayMetrics(canvas: HTMLCanvasElement, imageWidth: number, imageHeight: number): DisplayMetrics {
  const rect = canvas.getBoundingClientRect()
  const containerAspect = rect.width / Math.max(rect.height, 1)
  const imageAspect = imageWidth / Math.max(imageHeight, 1)
  let drawWidth = rect.width
  let drawHeight = rect.height
  let offsetX = 0
  let offsetY = 0
  if (imageAspect > containerAspect) {
    drawWidth = rect.width
    drawHeight = rect.width / imageAspect
    offsetY = (rect.height - drawHeight) / 2
  } else {
    drawHeight = rect.height
    drawWidth = rect.height * imageAspect
    offsetX = (rect.width - drawWidth) / 2
  }
  return { rect, drawWidth, drawHeight, offsetX, offsetY }
}

function getImageCoordsFromClient(
  clientX: number,
  clientY: number,
  canvas: HTMLCanvasElement,
  imageWidth: number,
  imageHeight: number,
) {
  const layout = getDisplayMetrics(canvas, imageWidth, imageHeight)
  const localX = clientX - layout.rect.left - layout.offsetX
  const localY = clientY - layout.rect.top - layout.offsetY
  if (localX < 0 || localY < 0 || localX > layout.drawWidth || localY > layout.drawHeight) return null
  return {
    x: (localX / layout.drawWidth) * imageWidth,
    y: (localY / layout.drawHeight) * imageHeight,
  }
}

function ensureOffscreenCanvas(ref: React.MutableRefObject<HTMLCanvasElement | null>, width: number, height: number) {
  if (!ref.current) ref.current = document.createElement('canvas')
  if (ref.current.width !== width || ref.current.height !== height) {
    ref.current.width = width
    ref.current.height = height
    ref.current.getContext('2d')?.clearRect(0, 0, width, height)
  }
  return ref.current
}

function clearCorrectionCanvas(canvas: HTMLCanvasElement) {
  const ctx = canvas.getContext('2d')
  if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height)
}

function loadPngToCanvas(canvas: HTMLCanvasElement, url: string): Promise<boolean> {
  return new Promise((resolve) => {
    const img = new Image()
    img.crossOrigin = 'anonymous'
    img.onload = () => {
      const ctx = canvas.getContext('2d')
      if (ctx) {
        ctx.clearRect(0, 0, canvas.width, canvas.height)
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height)
      }
      resolve(true)
    }
    img.onerror = () => {
      clearCorrectionCanvas(canvas)
      resolve(false)
    }
    img.src = url
  })
}

function canvasToBlob(canvas: HTMLCanvasElement): Promise<Blob> {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => (blob ? resolve(blob) : reject(new Error('Failed to export mask'))), 'image/png')
  })
}

function snapshotCorrectionMasks(add: HTMLCanvasElement, remove: HTMLCanvasElement, staticC: HTMLCanvasElement): CorrectionSnapshot {
  const w = add.width
  const h = add.height
  return {
    add: add.getContext('2d')!.getImageData(0, 0, w, h),
    remove: remove.getContext('2d')!.getImageData(0, 0, w, h),
    static: staticC.getContext('2d')!.getImageData(0, 0, w, h),
  }
}

function restoreCorrectionSnapshot(snapshot: CorrectionSnapshot, add: HTMLCanvasElement, remove: HTMLCanvasElement, staticC: HTMLCanvasElement) {
  add.getContext('2d')!.putImageData(snapshot.add, 0, 0)
  remove.getContext('2d')!.putImageData(snapshot.remove, 0, 0)
  staticC.getContext('2d')!.putImageData(snapshot.static, 0, 0)
}

function paintCorrectionStroke(
  canvases: HTMLCanvasElement[],
  x: number,
  y: number,
  brushSize: number,
  erase: boolean,
  lastPoint: { x: number; y: number } | null,
) {
  for (const canvas of canvases) {
    const ctx = canvas.getContext('2d')
    if (!ctx) continue
    ctx.globalCompositeOperation = erase ? 'destination-out' : 'source-over'
    ctx.fillStyle = 'white'
    ctx.strokeStyle = 'white'
    ctx.lineWidth = brushSize
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'
    if (lastPoint) {
      ctx.beginPath()
      ctx.moveTo(lastPoint.x, lastPoint.y)
      ctx.lineTo(x, y)
      ctx.stroke()
    }
    ctx.beginPath()
    ctx.arc(x, y, brushSize / 2, 0, Math.PI * 2)
    ctx.fill()
  }
}

function tintMaskOnContext(
  ctx: CanvasRenderingContext2D,
  maskCanvas: HTMLCanvasElement,
  color: string,
  layout: DisplayMetrics,
  imageWidth: number,
  imageHeight: number,
) {
  const tmp = document.createElement('canvas')
  tmp.width = imageWidth
  tmp.height = imageHeight
  tmp.getContext('2d')!.drawImage(maskCanvas, 0, 0)
  const data = tmp.getContext('2d')!.getImageData(0, 0, imageWidth, imageHeight)
  const overlay = document.createElement('canvas')
  overlay.width = imageWidth
  overlay.height = imageHeight
  const octx = overlay.getContext('2d')!
  const rgba = color.match(/[\d.]+/g)?.map(Number) ?? [255, 255, 255, 0.5]
  const [r, g, b, a] = rgba
  const alpha = a <= 1 ? a : a / 255
  const img = octx.createImageData(imageWidth, imageHeight)
  for (let i = 0; i < data.data.length; i += 4) {
    if (data.data[i] > 127) {
      img.data[i] = r
      img.data[i + 1] = g
      img.data[i + 2] = b
      img.data[i + 3] = Math.round(alpha * 255)
    }
  }
  octx.putImageData(img, 0, 0)
  ctx.drawImage(overlay, layout.offsetX, layout.offsetY, layout.drawWidth, layout.drawHeight)
}

interface AnnotationViewProps {
  jobId: string
  videoVersion?: number
  temporalWarning?: string | null
  onRunGuided: () => void
  onViewResults?: () => void
  onBack: () => void
}

export default function AnnotationView({
  jobId,
  videoVersion = 0,
  temporalWarning = null,
  onRunGuided,
  onViewResults,
  onBack,
}: AnnotationViewProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const viewerRef = useRef<HTMLDivElement>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const imageRef = useRef<HTMLImageElement | null>(null)
  const annotationRef = useRef<FrameAnnotation | null>(null)
  const drawingRef = useRef(false)
  const syncingRef = useRef(false)
  const drawingKindRef = useRef<'correction' | null>(null)
  const rafDrawRef = useRef<number | null>(null)
  const correctionAddRef = useRef<HTMLCanvasElement | null>(null)
  const correctionRemoveRef = useRef<HTMLCanvasElement | null>(null)
  const correctionStaticRef = useRef<HTMLCanvasElement | null>(null)
  const globalStaticRef = useRef<HTMLCanvasElement | null>(null)
  const autoMaskCanvasRef = useRef<HTMLCanvasElement | null>(null)
  const correctionLastPointRef = useRef<{ x: number; y: number } | null>(null)
  const correctionDirtyRef = useRef(false)
  const activeCorrectionFrameRef = useRef<number | null>(null)
  const correctionLoadIdRef = useRef(0)
  const correctionHistoryByFrameRef = useRef<Map<number, { undo: CorrectionSnapshot[]; redo: CorrectionSnapshot[] }>>(new Map())
  const frameIndexRef = useRef(0)
  const initialAutoPlayRef = useRef(false)

  const [totalFrames, setTotalFrames] = useState(0)
  const [frameIndex, setFrameIndex] = useState(0)
  const [jumpInput, setJumpInput] = useState('1')
  const [annotation, setAnnotation] = useState<FrameAnnotation | null>(null)
  const [keyframes, setKeyframes] = useState<KeyframeInfo[]>([])
  const [brushSize, setBrushSize] = useState(16)
  const [viewerMode, setViewerMode] = useState<ViewerMode>('overlay')
  const [showRawJunctionDebug, setShowRawJunctionDebug] = useState(false)
  const [playing, setPlaying] = useState(false)
  const [frameAssets, setFrameAssets] = useState<FrameAsset[]>([])
  const [isDrawing, setIsDrawing] = useState(false)
  const [saveStatus, setSaveStatus] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle')
  const [statusMsg, setStatusMsg] = useState('')
  const [propagationMeta, setPropagationMeta] = useState<PropagationMetadata | null>(null)
  const [correctionTool, setCorrectionTool] = useState<CorrectionTool>('add')
  const [correctionUndoStack, setCorrectionUndoStack] = useState<CorrectionSnapshot[]>([])
  const [correctionRedoStack, setCorrectionRedoStack] = useState<CorrectionSnapshot[]>([])
  const [correctionsReady, setCorrectionsReady] = useState(false)
  const [propagationSettings, setPropagationSettings] = useState<PropagationSettings>({
    propagation_window: 10,
    use_temporal_propagation: true,
    repair_disconnected_tubes: true,
    max_bridge_gap_px: 15,
    max_bridge_angle_degrees: 45,
  })
  const [temporalMeta, setTemporalMeta] = useState<TemporalMetadata | null>(null)
  const [temporalSettings, setTemporalSettings] = useState<TemporalSettings>({
    use_temporal_continuity: true,
    temporal_memory_frames: 3,
    temporal_persistence_weight: 0.5,
    max_allowed_area_drop_fraction: 0.15,
    recover_missing_middle_frames: true,
    min_temporal_component_persistence: 2,
    allow_tip_growth: true,
    repair_disconnected_tubes: false,
    max_bridge_gap_px: 15,
    max_bridge_angle_degrees: 45,
    min_bridge_intensity_percentile: 60,
  })
  const [segmentationPreview, setSegmentationPreview] = useState<SegmentationPreviewResult | null>(null)
  const [previewLoading, setPreviewLoading] = useState(false)
  const [bacterialTrackingInfo, setBacterialTrackingInfo] = useState<BacterialTrackingInfo | null>(null)

  const cacheSuffix = videoVersion > 0 ? `?v=${videoVersion}` : ''
  const videoUrl = `${BACKEND_ORIGIN}/api/jobs/${jobId}/media/video${cacheSuffix}`
  const bacterialTrackingVideoUrl = bacterialTrackingInfo?.label_overlay_video_url
    ? `${BACKEND_ORIGIN}${bacterialTrackingInfo.label_overlay_video_url}${cacheSuffix}`
    : null

  const assetUrl = useCallback((path: string | null | undefined) => {
    if (!path) return null
    return `${BACKEND_ORIGIN}${path}${cacheSuffix}`
  }, [cacheSuffix])

  const fallbackAsset: FrameAsset = {
    frame_index: frameIndex,
    original_frame_url: `/api/jobs/${jobId}/frames/${frameIndex}`,
    auto_overlay_url: `/api/jobs/${jobId}/overlays/${frameIndex}`,
    auto_mask_url: `/api/jobs/${jobId}/masks/${frameIndex}`,
    skeleton_frame_url: `/api/jobs/${jobId}/skeletons/${frameIndex}`,
  }
  const currentAsset = frameAssets[frameIndex] ?? fallbackAsset
  const frameUrl = assetUrl(currentAsset.original_frame_url) ?? ''

  const previewImageUrl = useCallback((path: string | undefined) => {
    if (!path) return null
    const base = path.startsWith('http') ? path : `${BACKEND_ORIGIN}${path}`
    return `${base}${base.includes('?') ? '&' : '?'}t=${Date.now()}`
  }, [])

  const skeletonUrl = (() => {
    const base = assetUrl(currentAsset.skeleton_frame_url)
    if (!base) return null
    if (!showRawJunctionDebug) return base
    return `${base}${base.includes('?') ? '&' : '?'}debug_raw_junctions=true`
  })()
  const showSegmentationPreview =
    segmentationPreview != null && segmentationPreview.frame_index === frameIndex
  const previewOverlayUrl = showSegmentationPreview
    ? previewImageUrl(segmentationPreview?.guided_overlay_url)
    : null
  const bacterialTrackingFrameUrl = (() => {
    const assetUrlPath = currentAsset.bacterial_tracking_overlay_url
    if (assetUrlPath) return assetUrl(assetUrlPath)
    if (bacterialTrackingInfo?.available) {
      return `${BACKEND_ORIGIN}/api/jobs/${jobId}/bacterial-tracking/overlay/${frameIndex}${cacheSuffix}`
    }
    return null
  })()
  const bacterialTrackingAvailable = Boolean(
    bacterialTrackingInfo?.available && bacterialTrackingFrameUrl,
  )
  const showOverlayVideo = viewerMode === 'overlay' && playing && !showSegmentationPreview
  const showBacterialTrackingVideo =
    viewerMode === 'bacterial_tracking' && playing && Boolean(bacterialTrackingVideoUrl)
  const showViewerVideo = showOverlayVideo || showBacterialTrackingVideo
  const activeVideoUrl = showBacterialTrackingVideo && bacterialTrackingVideoUrl
    ? bacterialTrackingVideoUrl
    : videoUrl

  useEffect(() => {
    annotationRef.current = annotation
  }, [annotation])

  useEffect(() => {
    frameIndexRef.current = frameIndex
  }, [frameIndex])

  const registerUserInteraction = useCallback(() => {
    videoRef.current?.pause()
    setPlaying(false)
  }, [])

  const loadFrameAssets = useCallback(async () => {
    const res = await fetch(`${API_URL}/jobs/${jobId}/frame-assets`)
    if (!res.ok) return
    const data = await res.json()
    setTotalFrames(data.total_frames ?? 0)
    setFrameAssets(data.frames ?? [])
  }, [jobId])

  const loadAnnotation = useCallback(async (index: number) => {
    const res = await fetch(`${API_URL}/jobs/${jobId}/annotations/${index}`)
    const data = await res.json()
    setAnnotation(data)
  }, [jobId])

  const loadKeyframes = useCallback(async () => {
    const res = await fetch(`${API_URL}/jobs/${jobId}/keyframes`)
    const data = await res.json()
    setKeyframes(data.keyframes ?? [])
  }, [jobId])

  const loadPropagationMeta = useCallback(async (index: number) => {
    const res = await fetch(`${API_URL}/jobs/${jobId}/propagation/${index}`)
    if (res.ok) setPropagationMeta(await res.json())
  }, [jobId])

  const loadTemporalMeta = useCallback(async (index: number) => {
    const res = await fetch(`${API_URL}/jobs/${jobId}/temporal/${index}`)
    if (res.ok) setTemporalMeta(await res.json())
  }, [jobId])

  const persistCorrections = useCallback(async (targetFrameIndex?: number) => {
    const saveIndex = targetFrameIndex ?? activeCorrectionFrameRef.current ?? frameIndexRef.current
    const addCanvas = correctionAddRef.current
    const removeCanvas = correctionRemoveRef.current
    const staticCanvas = correctionStaticRef.current
    if (!addCanvas || !removeCanvas || !staticCanvas) return
    if (activeCorrectionFrameRef.current !== null && activeCorrectionFrameRef.current !== saveIndex) return

    setSaveStatus('saving')
    try {
      const form = new FormData()
      form.append('add_mask', await canvasToBlob(addCanvas), 'add.png')
      form.append('remove_mask', await canvasToBlob(removeCanvas), 'remove.png')
      form.append('static_mask', await canvasToBlob(staticCanvas), 'static.png')
      await fetch(`${API_URL}/jobs/${jobId}/corrections/${saveIndex}`, { method: 'PUT', body: form })
      correctionDirtyRef.current = false
      setSaveStatus('saved')
      window.setTimeout(() => setSaveStatus('idle'), 2000)
    } catch {
      setSaveStatus('error')
    }
  }, [jobId])

  const loadCorrectionMasks = useCallback(async (index: number, width: number, height: number) => {
    const loadId = ++correctionLoadIdRef.current
    setCorrectionsReady(false)

    const addCanvas = ensureOffscreenCanvas(correctionAddRef, width, height)
    const removeCanvas = ensureOffscreenCanvas(correctionRemoveRef, width, height)
    const staticCanvas = ensureOffscreenCanvas(correctionStaticRef, width, height)
    const globalCanvas = ensureOffscreenCanvas(globalStaticRef, width, height)
    const autoCanvas = ensureOffscreenCanvas(autoMaskCanvasRef, width, height)

    clearCorrectionCanvas(addCanvas)
    clearCorrectionCanvas(removeCanvas)
    clearCorrectionCanvas(staticCanvas)
    clearCorrectionCanvas(globalCanvas)
    clearCorrectionCanvas(autoCanvas)

    const cacheBust = `${cacheSuffix || ''}${cacheSuffix ? '&' : '?'}t=${Date.now()}`
    const asset = frameAssets[index]
    await Promise.all([
      loadPngToCanvas(addCanvas, `${BACKEND_ORIGIN}/api/jobs/${jobId}/corrections/${index}/add${cacheBust}`),
      loadPngToCanvas(removeCanvas, `${BACKEND_ORIGIN}/api/jobs/${jobId}/corrections/${index}/remove${cacheBust}`),
      loadPngToCanvas(staticCanvas, `${BACKEND_ORIGIN}/api/jobs/${jobId}/corrections/${index}/static${cacheBust}`),
      loadPngToCanvas(globalCanvas, `${BACKEND_ORIGIN}/api/jobs/${jobId}/corrections/global-static${cacheBust}`),
      loadPngToCanvas(
        autoCanvas,
        asset?.auto_mask_url
          ? `${BACKEND_ORIGIN}${asset.auto_mask_url}${cacheSuffix}`
          : `${BACKEND_ORIGIN}/api/jobs/${jobId}/masks/${index}${cacheSuffix}`,
      ),
    ])

    if (loadId !== correctionLoadIdRef.current) return

    activeCorrectionFrameRef.current = index
    correctionDirtyRef.current = false
    const history = correctionHistoryByFrameRef.current.get(index)
    setCorrectionUndoStack(history?.undo ?? [])
    setCorrectionRedoStack(history?.redo ?? [])
    setCorrectionsReady(true)
  }, [jobId, frameAssets, cacheSuffix])

  const drawWorkspaceCanvas = useCallback(() => {
    const canvas = canvasRef.current
    const img = imageRef.current
    const addCanvas = correctionAddRef.current
    const removeCanvas = correctionRemoveRef.current
    const staticCanvas = correctionStaticRef.current
    const globalCanvas = globalStaticRef.current
    const autoCanvas = autoMaskCanvasRef.current
    if (!canvas || !img || !annotation || !addCanvas || !removeCanvas || !staticCanvas) return
    if (viewerMode !== 'overlay' && viewerMode !== 'original') return
    if (activeCorrectionFrameRef.current !== frameIndexRef.current) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return
    const dpr = window.devicePixelRatio || 1
    const layout = getDisplayMetrics(canvas, annotation.image_width, annotation.image_height)
    canvas.width = Math.max(1, Math.floor(layout.rect.width * dpr))
    canvas.height = Math.max(1, Math.floor(layout.rect.height * dpr))
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, layout.rect.width, layout.rect.height)
    ctx.fillStyle = '#000'
    ctx.fillRect(0, 0, layout.rect.width, layout.rect.height)
    ctx.drawImage(img, layout.offsetX, layout.offsetY, layout.drawWidth, layout.drawHeight)
    if (viewerMode === 'overlay' && autoCanvas) {
      tintMaskOnContext(ctx, autoCanvas, CORRECTION_COLORS.auto, layout, annotation.image_width, annotation.image_height)
    }
    tintMaskOnContext(ctx, addCanvas, CORRECTION_COLORS.add, layout, annotation.image_width, annotation.image_height)
    tintMaskOnContext(ctx, removeCanvas, CORRECTION_COLORS.remove, layout, annotation.image_width, annotation.image_height)
    tintMaskOnContext(ctx, staticCanvas, CORRECTION_COLORS.static, layout, annotation.image_width, annotation.image_height)
    if (globalCanvas) {
      tintMaskOnContext(ctx, globalCanvas, 'rgba(60, 120, 255, 0.35)', layout, annotation.image_width, annotation.image_height)
    }
  }, [annotation, viewerMode])

  const scheduleRedraw = useCallback(() => {
    if (rafDrawRef.current !== null) return
    rafDrawRef.current = window.requestAnimationFrame(() => {
      rafDrawRef.current = null
      drawWorkspaceCanvas()
    })
  }, [drawWorkspaceCanvas])

  const seekVideoToFrame = useCallback((idx: number) => {
    const video = videoRef.current
    if (!video?.duration || totalFrames <= 1) return
    syncingRef.current = true
    video.currentTime = (idx / (totalFrames - 1)) * video.duration
    window.setTimeout(() => { syncingRef.current = false }, 120)
  }, [totalFrames])

  const switchToFrame = useCallback(async (idx: number) => {
    if (idx === frameIndexRef.current) return
    registerUserInteraction()

    const currentIndex = frameIndexRef.current
    if (correctionDirtyRef.current) {
      await persistCorrections(currentIndex)
    }
    correctionHistoryByFrameRef.current.set(currentIndex, {
      undo: correctionUndoStack,
      redo: correctionRedoStack,
    })

    setFrameIndex(idx)
    if (playing && (viewerMode === 'overlay' || viewerMode === 'bacterial_tracking')) seekVideoToFrame(idx)
  }, [registerUserInteraction, persistCorrections, correctionUndoStack, correctionRedoStack, playing, viewerMode, seekVideoToFrame])

  const goToFrame = useCallback((idx: number) => {
    void switchToFrame(idx)
  }, [switchToFrame])

  const togglePlay = () => {
    if (viewerMode !== 'overlay' && viewerMode !== 'bacterial_tracking') {
      setViewerMode('overlay')
    }
    const video = videoRef.current
    if (!video) {
      setPlaying(true)
      return
    }
    if (video.paused) {
      setPlaying(true)
      seekVideoToFrame(frameIndex)
      void video.play()
    } else {
      registerUserInteraction()
    }
  }

  const getCorrectionTargetCanvases = useCallback((): HTMLCanvasElement[] => {
    const add = correctionAddRef.current
    const remove = correctionRemoveRef.current
    const staticC = correctionStaticRef.current
    if (!add || !remove || !staticC) return []
    if (correctionTool === 'erase_correction') return [add, remove, staticC]
    if (correctionTool === 'add') return [add]
    if (correctionTool === 'remove') return [remove]
    return [staticC]
  }, [correctionTool])

  const finishDrawing = useCallback(() => {
    if (!drawingRef.current) return
    drawingRef.current = false
    drawingKindRef.current = null
    correctionLastPointRef.current = null
    setIsDrawing(false)
    correctionDirtyRef.current = true
    void persistCorrections(activeCorrectionFrameRef.current ?? frameIndexRef.current)
  }, [persistCorrections])

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!annotation || viewerMode !== 'overlay') return
    if (activeCorrectionFrameRef.current !== frameIndexRef.current) return
    registerUserInteraction()
    const coords = getImageCoordsFromClient(e.clientX, e.clientY, canvasRef.current!, annotation.image_width, annotation.image_height)
    if (!coords) return
    const add = correctionAddRef.current
    const remove = correctionRemoveRef.current
    const staticC = correctionStaticRef.current
    if (!add || !remove || !staticC) return
    const snapshot = snapshotCorrectionMasks(add, remove, staticC)
    setCorrectionUndoStack((prev) => {
      const next = [...prev.slice(-19), snapshot]
      correctionHistoryByFrameRef.current.set(frameIndexRef.current, { undo: next, redo: [] })
      return next
    })
    setCorrectionRedoStack([])
    drawingRef.current = true
    drawingKindRef.current = 'correction'
    setIsDrawing(true)
    const targets = getCorrectionTargetCanvases()
    paintCorrectionStroke(targets, coords.x, coords.y, brushSize, correctionTool === 'erase_correction', null)
    correctionLastPointRef.current = { x: coords.x, y: coords.y }
    scheduleRedraw()
  }

  const handleCorrectionUndo = () => {
    registerUserInteraction()
    const add = correctionAddRef.current
    const remove = correctionRemoveRef.current
    const staticC = correctionStaticRef.current
    if (!add || !remove || !staticC || correctionUndoStack.length === 0) return
    const prev = correctionUndoStack[correctionUndoStack.length - 1]
    const current = snapshotCorrectionMasks(add, remove, staticC)
    const nextUndo = correctionUndoStack.slice(0, -1)
    const nextRedo = [...correctionRedoStack, current]
    setCorrectionUndoStack(nextUndo)
    setCorrectionRedoStack(nextRedo)
    restoreCorrectionSnapshot(prev, add, remove, staticC)
    correctionHistoryByFrameRef.current.set(frameIndexRef.current, { undo: nextUndo, redo: nextRedo })
    scheduleRedraw()
    correctionDirtyRef.current = true
    void persistCorrections(frameIndexRef.current)
  }

  const handleCorrectionRedo = () => {
    registerUserInteraction()
    const add = correctionAddRef.current
    const remove = correctionRemoveRef.current
    const staticC = correctionStaticRef.current
    if (!add || !remove || !staticC || correctionRedoStack.length === 0) return
    const next = correctionRedoStack[correctionRedoStack.length - 1]
    const current = snapshotCorrectionMasks(add, remove, staticC)
    const nextRedo = correctionRedoStack.slice(0, -1)
    const nextUndo = [...correctionUndoStack, current]
    setCorrectionRedoStack(nextRedo)
    setCorrectionUndoStack(nextUndo)
    restoreCorrectionSnapshot(next, add, remove, staticC)
    correctionHistoryByFrameRef.current.set(frameIndexRef.current, { undo: nextUndo, redo: nextRedo })
    scheduleRedraw()
    correctionDirtyRef.current = true
    void persistCorrections(frameIndexRef.current)
  }

  const handleResetCorrectionsFrame = async () => {
    registerUserInteraction()
    const res = await fetch(`${API_URL}/jobs/${jobId}/corrections/${frameIndex}/reset`, { method: 'POST' })
    if (res.ok && annotation) {
      await loadCorrectionMasks(frameIndex, annotation.image_width, annotation.image_height)
      scheduleRedraw()
      setStatusMsg('Frame corrections reset')
    }
  }

  const handleApplyGlobalStatic = async () => {
    registerUserInteraction()
    const form = new FormData()
    form.append('frame_index', String(frameIndexRef.current))
    const res = await fetch(`${API_URL}/jobs/${jobId}/corrections/apply-global-static`, { method: 'POST', body: form })
    if (res.ok && annotation) {
      await loadCorrectionMasks(frameIndexRef.current, annotation.image_width, annotation.image_height)
      scheduleRedraw()
      setStatusMsg('Static background applied globally')
    }
  }

  const handleRunGuided = async () => {
    registerUserInteraction()
    if (correctionDirtyRef.current) await persistCorrections()
    onRunGuided()
  }

  const handlePreviewSegmentation = async () => {
    registerUserInteraction()
    if (!annotation) return
    if (correctionDirtyRef.current) await persistCorrections()
    setPreviewLoading(true)
    setStatusMsg('')
    try {
      const res = await fetch(`${API_URL}/jobs/${jobId}/frames/${frameIndex}/preview-guided`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(annotation),
      })
      const data = await res.json()
      if (!res.ok) {
        setStatusMsg(typeof data.error === 'string' ? data.error : 'Preview failed')
        return
      }
      setSegmentationPreview(data as SegmentationPreviewResult)
      setPlaying(false)
      setStatusMsg('Preview ready — change frame to return to overlay')
    } catch {
      setStatusMsg('Preview failed')
    } finally {
      setPreviewLoading(false)
    }
  }

  const toggleKeyframe = async () => {
    registerUserInteraction()
    if (!annotation) return
    const updated = { ...annotation, is_keyframe: !annotation.is_keyframe }
    setAnnotation(updated)
    await fetch(`${API_URL}/jobs/${jobId}/annotations/${annotation.frame_index}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(updated),
    })
    await loadKeyframes()
  }

  useEffect(() => {
    loadFrameAssets()
    loadKeyframes()
    void fetch(`${API_URL}/jobs/${jobId}/propagation-settings`).then(async (r) => {
      if (r.ok) setPropagationSettings(await r.json())
    })
    void fetch(`${API_URL}/jobs/${jobId}/temporal-settings`).then(async (r) => {
      if (r.ok) setTemporalSettings(await r.json())
    })
    void fetch(`${API_URL}/jobs/${jobId}/bacterial-tracking/info`).then(async (r) => {
      if (r.ok) setBacterialTrackingInfo(await r.json())
    })
  }, [jobId, loadFrameAssets, loadKeyframes, videoVersion])

  useEffect(() => {
    setSegmentationPreview(null)
  }, [frameIndex])

  useEffect(() => {
    loadPropagationMeta(frameIndex)
    loadTemporalMeta(frameIndex)
  }, [frameIndex, loadPropagationMeta, loadTemporalMeta, videoVersion])

  useEffect(() => {
    if (totalFrames > 0) loadAnnotation(frameIndex)
  }, [frameIndex, totalFrames, loadAnnotation])

  useEffect(() => {
    setJumpInput(String(frameIndex + 1))
  }, [frameIndex])

  useEffect(() => {
    if (totalFrames > 0 && !initialAutoPlayRef.current) {
      initialAutoPlayRef.current = true
      setPlaying(true)
    }
  }, [totalFrames])

  useEffect(() => {
    if (!annotation) return
    void loadCorrectionMasks(frameIndex, annotation.image_width, annotation.image_height).then(() => {
      scheduleRedraw()
    })
  }, [frameIndex, annotation?.image_width, annotation?.image_height, loadCorrectionMasks, videoVersion, scheduleRedraw])

  useEffect(() => {
    const img = new Image()
    img.crossOrigin = 'anonymous'
    img.onload = () => {
      imageRef.current = img
      drawWorkspaceCanvas()
    }
    img.src = frameUrl
  }, [frameUrl, drawWorkspaceCanvas])

  useEffect(() => {
    drawWorkspaceCanvas()
  }, [annotation, drawWorkspaceCanvas, viewerMode, correctionsReady])

  useEffect(() => {
    const viewer = viewerRef.current
    if (!viewer) return
    const observer = new ResizeObserver(() => drawWorkspaceCanvas())
    observer.observe(viewer)
    return () => observer.disconnect()
  }, [drawWorkspaceCanvas])

  useEffect(() => {
    const video = videoRef.current
    if (
      !playing
      || (viewerMode !== 'overlay' && viewerMode !== 'bacterial_tracking')
      || !video?.paused
    ) return
    seekVideoToFrame(frameIndex)
    void video.play()
  }, [playing, viewerMode, activeVideoUrl, frameIndex, seekVideoToFrame])

  useEffect(() => {
    if (viewerMode !== 'overlay' && viewerMode !== 'bacterial_tracking') registerUserInteraction()
  }, [viewerMode, registerUserInteraction])

  useEffect(() => {
    if (viewerMode === 'bacterial_tracking' && !bacterialTrackingAvailable) {
      setViewerMode('overlay')
    }
  }, [viewerMode, bacterialTrackingAvailable])

  useEffect(() => {
    if (!isDrawing) return
    if (activeCorrectionFrameRef.current !== frameIndexRef.current) return
    const onMove = (e: MouseEvent) => {
      if (!drawingRef.current || !annotationRef.current || !canvasRef.current) return
      const coords = getImageCoordsFromClient(e.clientX, e.clientY, canvasRef.current, annotationRef.current.image_width, annotationRef.current.image_height)
      if (!coords) return
      paintCorrectionStroke(
        getCorrectionTargetCanvases(),
        coords.x,
        coords.y,
        brushSize,
        correctionTool === 'erase_correction',
        correctionLastPointRef.current,
      )
      correctionLastPointRef.current = { x: coords.x, y: coords.y }
      scheduleRedraw()
    }
    const onUp = () => finishDrawing()
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
    return () => {
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
    }
  }, [isDrawing, brushSize, correctionTool, finishDrawing, getCorrectionTargetCanvases, scheduleRedraw])

  const handleVideoTimeUpdate = () => {
    if (syncingRef.current || !playing) return
    if (viewerMode !== 'overlay' && viewerMode !== 'bacterial_tracking') return
    const video = videoRef.current
    if (!video?.duration || totalFrames <= 0) return
    const idx = Math.min(totalFrames - 1, Math.max(0, Math.round((video.currentTime / video.duration) * (totalFrames - 1))))
    if (idx !== frameIndexRef.current) {
      const currentIndex = frameIndexRef.current
      if (correctionDirtyRef.current) {
        void persistCorrections(currentIndex)
      }
      correctionHistoryByFrameRef.current.set(currentIndex, {
        undo: correctionUndoStack,
        redo: correctionRedoStack,
      })
      setFrameIndex(idx)
    }
  }

  const jumpToFrame = () => {
    const n = parseInt(jumpInput, 10)
    if (!Number.isNaN(n) && n >= 1 && n <= totalFrames) goToFrame(n - 1)
  }

  const propagationStatusLabel = () => {
    if (!propagationMeta) return 'Propagation: unknown'
    if (propagationMeta.directly_annotated || propagationMeta.mode === 'keyframe') return 'Directly annotated keyframe'
    if (propagationMeta.mode === 'propagated' && propagationMeta.source_keyframe !== null) {
      return `Propagated from keyframe ${propagationMeta.source_keyframe + 1}`
    }
    return 'Auto segmentation only'
  }

  return (
    <div className="annotation-page">
      {totalFrames === 0 && <div className="annotation-loading-banner">Loading frame assets…</div>}
      {temporalWarning && <div className="annotation-warning-banner">{temporalWarning}</div>}

      <header className="annotation-toolbar-sticky">
        <div className="annotation-toolbar-inner">
          <div className="annotation-toolbar-group annotation-toolbar-nav">
            <button type="button" className="btn btn-secondary btn-compact" onClick={onBack}>Back</button>
            <span className="annotation-mode-pill">Review & Correct</span>
            <button type="button" className="btn btn-secondary btn-compact" onClick={togglePlay} title={playing ? 'Pause' : 'Play'}>
              {playing && (viewerMode === 'overlay' || viewerMode === 'bacterial_tracking') ? <Pause size={16} /> : <Play size={16} />}
            </button>
            <button type="button" className="btn btn-secondary btn-compact" disabled={frameIndex <= 0} onClick={() => goToFrame(frameIndex - 1)}><ChevronLeft size={16} /></button>
            <input
              type="range"
              min={0}
              max={Math.max(0, totalFrames - 1)}
              value={frameIndex}
              onChange={(e) => goToFrame(Number(e.target.value))}
              className="annotation-slider"
            />
            <button type="button" className="btn btn-secondary btn-compact" disabled={frameIndex >= totalFrames - 1} onClick={() => goToFrame(frameIndex + 1)}><ChevronRight size={16} /></button>
            <label className="annotation-jump">
              <span>Frame</span>
              <input type="number" min={1} max={Math.max(1, totalFrames)} value={jumpInput} onChange={(e) => setJumpInput(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && jumpToFrame()} />
              <button type="button" className="btn btn-secondary btn-compact" onClick={jumpToFrame}>Go</button>
            </label>
            <span className="annotation-frame-label">{frameIndex + 1} / {totalFrames}{annotation?.is_keyframe && <span className="keyframe-badge">KF</span>}</span>
            <span className="propagation-status-pill">{propagationStatusLabel()}</span>
          </div>

          <div className="annotation-toolbar-group">
            <span className="annotation-section-label">Correction</span>
            {(Object.keys(CORRECTION_TOOL_LABELS) as CorrectionTool[]).map((ct) => (
              <button
                key={ct}
                type="button"
                className={`btn btn-secondary btn-compact correction-tool-btn correction-${ct} ${correctionTool === ct ? 'active' : ''}`}
                onClick={() => { registerUserInteraction(); setCorrectionTool(ct) }}
              >
                {ct === 'add' && <Plus size={14} />}
                {ct === 'remove' && <Minus size={14} />}
                {ct === 'static' && <Layers size={14} />}
                {ct === 'erase_correction' && <Eraser size={14} />}
                {CORRECTION_TOOL_LABELS[ct]}
              </button>
            ))}
          </div>

          <div className="annotation-toolbar-group">
            <span className="annotation-section-label">Brush</span>
            <label className="annotation-brush-size">
              <input type="range" min={4} max={48} value={brushSize} onChange={(e) => setBrushSize(Number(e.target.value))} />
              <span>{brushSize}px</span>
            </label>
            <button type="button" className="btn btn-secondary btn-compact" onClick={handleCorrectionUndo} disabled={correctionUndoStack.length === 0}><Undo2 size={16} /> Undo</button>
            <button type="button" className="btn btn-secondary btn-compact" onClick={handleCorrectionRedo} disabled={correctionRedoStack.length === 0}><Redo2 size={16} /> Redo</button>
            <button type="button" className="btn btn-secondary btn-compact" onClick={handleResetCorrectionsFrame}><RotateCcw size={16} /> Reset Corrections</button>
            {correctionTool === 'static' && (
              <button type="button" className="btn btn-secondary btn-compact" onClick={() => void handleApplyGlobalStatic()}>
                <Layers size={14} /> Apply Static Globally
              </button>
            )}
          </div>

          <div className="annotation-toolbar-group annotation-toolbar-actions">
            <button type="button" className="btn btn-compact" onClick={handleRunGuided}><Play size={16} /> Run Guided Segmentation</button>
            <button
              type="button"
              className="btn btn-secondary btn-compact"
              disabled={previewLoading || !annotation}
              onClick={() => void handlePreviewSegmentation()}
            >
              <ScanEye size={16} /> {previewLoading ? 'Previewing…' : 'Preview Segmentation'}
            </button>
            {onViewResults && <button type="button" className="btn btn-secondary btn-compact" onClick={onViewResults}><BarChart3 size={16} /> Metrics</button>}
            {saveStatus === 'saved' && <span className="save-indicator">Saved</span>}
            {saveStatus === 'saving' && <span className="save-indicator saving">Saving…</span>}
            {statusMsg && <span className="annotation-status">{statusMsg}</span>}
          </div>
        </div>
      </header>

      <div className="annotation-main">
        <div className="annotation-viewer" ref={viewerRef} onMouseDown={() => registerUserInteraction()}>
          <div className="annotation-viewer-content">
            {showViewerVideo ? (
              <video
                ref={videoRef}
                key={activeVideoUrl}
                className="annotation-viewer-media annotation-overlay-video is-visible"
                src={activeVideoUrl}
                muted
                loop
                playsInline
                onTimeUpdate={handleVideoTimeUpdate}
                onPlay={() => setPlaying(true)}
                onPause={() => setPlaying(false)}
                onClick={registerUserInteraction}
              />
            ) : showSegmentationPreview && previewOverlayUrl ? (
              <div className="annotation-viewer-preview-wrap">
                <img
                  key={`preview-${frameIndex}-${previewOverlayUrl}`}
                  src={previewOverlayUrl}
                  alt="Segmentation preview with corrections"
                  className="annotation-viewer-media annotation-viewer-image"
                />
                <div className="annotation-viewer-preview-badge">Segmentation preview</div>
              </div>
            ) : viewerMode === 'bacterial_tracking' ? (
              bacterialTrackingFrameUrl ? (
                <img
                  key={`bacterial-tracking-${frameIndex}-${videoVersion}`}
                  src={bacterialTrackingFrameUrl}
                  alt="Bacterial tracking overlay with labels"
                  className="annotation-viewer-media annotation-viewer-image"
                />
              ) : (
                <div className="annotation-viewer-placeholder"><ImageIcon size={32} /><p>Bacterial tracking overlay not available.</p></div>
              )
            ) : viewerMode === 'skeleton' ? (
              skeletonUrl ? (
                <img
                  key={`skeleton-${frameIndex}-${videoVersion}-${showRawJunctionDebug ? 'raw' : 'norm'}`}
                  src={skeletonUrl}
                  alt="Skeleton with branch nodes"
                  className="annotation-viewer-media annotation-viewer-image"
                />
              ) : (
                <div className="annotation-viewer-placeholder"><ImageIcon size={32} /><p>Skeleton not available for this frame.</p></div>
              )
            ) : (
              <canvas ref={canvasRef} className="annotation-canvas" onMouseDown={handleMouseDown} onMouseUp={() => finishDrawing()} />
            )}
          </div>
        </div>

        <aside className="annotation-side-panel glass-panel">
          <h3>View</h3>
          <div className="viewer-mode-list">
            {VIEWER_MODES.map((mode) => (
              <button
                key={mode.id}
                type="button"
                className={`viewer-mode-btn ${viewerMode === mode.id ? 'active' : ''}`}
                onClick={() => { registerUserInteraction(); setViewerMode(mode.id) }}
              >
                {mode.label}
              </button>
            ))}
            {bacterialTrackingAvailable && (
              <button
                type="button"
                className={`viewer-mode-btn ${viewerMode === 'bacterial_tracking' ? 'active' : ''}`}
                onClick={() => { registerUserInteraction(); setViewerMode('bacterial_tracking') }}
              >
                Bacterial Tracking Overlay
              </button>
            )}
          </div>
          {viewerMode === 'bacterial_tracking' && (
            <p className="annotation-settings-hint">
              Bounding boxes and stable labels per tracked bacterium. Use Play to scrub the overlay video.
            </p>
          )}
          {viewerMode === 'skeleton' && (
            <label className="annotation-toggle-row">
              <input
                type="checkbox"
                checked={showRawJunctionDebug}
                onChange={(e) => {
                  registerUserInteraction()
                  setShowRawJunctionDebug(e.target.checked)
                }}
              />
              <span>Show raw junction debug nodes</span>
            </label>
          )}
          <p className="annotation-settings-hint correction-color-legend">
            Auto: green · Add: cyan · Remove: magenta · Static: blue
          </p>

          <div className="annotation-side-section">
            <h3>Keyframes</h3>
            <div className="annotation-side-actions">
              <button type="button" className="btn btn-secondary btn-compact" onClick={toggleKeyframe}><Bookmark size={14} /> {annotation?.is_keyframe ? 'Unmark' : 'Mark'} Keyframe</button>
            </div>
            <div className="keyframe-timeline">
              {keyframes.map((kf) => (
                <button key={kf.frame_index} type="button" className={`keyframe-chip ${kf.frame_index === frameIndex ? 'active' : ''} ${kf.has_guidance ? 'has-guidance' : ''}`} onClick={() => goToFrame(kf.frame_index)}>{kf.frame_index + 1}</button>
              ))}
            </div>
          </div>

          <div className="annotation-side-section">
            <h3>Temporal Continuity</h3>
            <label className="annotation-toggle-row">
              <input
                type="checkbox"
                checked={temporalSettings.use_temporal_continuity}
                onChange={(e) => {
                  const next = { ...temporalSettings, use_temporal_continuity: e.target.checked }
                  setTemporalSettings(next)
                  void fetch(`${API_URL}/jobs/${jobId}/temporal-settings`, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(next) })
                }}
              />
              <span>Use temporal continuity</span>
            </label>
            {temporalMeta && temporalMeta.pixels_recovered + temporalMeta.pixels_removed_flicker > 0 && (
              <div className="propagation-meta-card">
                <p>Recovered: {temporalMeta.pixels_recovered} · Flicker removed: {temporalMeta.pixels_removed_flicker}</p>
              </div>
            )}
          </div>

          <div className="annotation-side-section">
            <h3>Temporal Propagation</h3>
            <label className="annotation-toggle-row">
              <input
                type="checkbox"
                checked={propagationSettings.use_temporal_propagation}
                onChange={(e) => {
                  const next = { ...propagationSettings, use_temporal_propagation: e.target.checked }
                  setPropagationSettings(next)
                  void fetch(`${API_URL}/jobs/${jobId}/propagation-settings`, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(next) })
                }}
              />
              <span>Use temporal propagation</span>
            </label>
            {propagationMeta && <div className="propagation-meta-card"><p>{propagationStatusLabel()}</p></div>}
          </div>
        </aside>
      </div>
    </div>
  )
}
