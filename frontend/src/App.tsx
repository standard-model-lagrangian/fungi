import { useState, useCallback, useEffect } from 'react'
import { Upload, Download, Activity, Play, CheckCircle2, AlertCircle } from 'lucide-react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import AnnotationView from './AnnotationView'
import PreSegmentationSetup from './PreSegmentationSetup'

const API_URL = 'http://localhost:8000/api'
const BACKEND_ORIGIN = 'http://localhost:8000'
const JOB_STORAGE_KEY = 'fungi_active_job_id'

type ProcessingStage = 'extracting_frames' | 'segmenting' | 'temporal_overlays' | 'temporal' | 'tracking' | 'finished' | 'error' | 'ready' | 'review' | 'setup'
type AppView = 'upload' | 'setup' | 'annotating' | 'processing' | 'results' | 'failed'

interface JobStatus {
  status: 'extracting_frames' | 'ready' | 'review' | 'processing' | 'completed' | 'failed' | 'setup'
  job_id: string
  error?: string
  stage?: ProcessingStage
  current_frame_index?: number
  total_frames?: number
  current_frame_preview_url?: string | null
  progress_percent?: number
  temporal_warning?: string | null
}

const STAGE_LABELS: Record<ProcessingStage, string> = {
  extracting_frames: 'Extracting frames...',
  segmenting: 'Segmenting frames with CellSAM...',
  temporal_overlays: 'Saving auto overlays...',
  temporal: 'Applying temporal continuity...',
  tracking: 'Tracking growth and generating metrics...',
  finished: 'Processing complete',
  error: 'Processing failed',
  ready: 'Frames ready',
  setup: 'Configure ROI and segmentation',
  review: 'Review automated segmentation',
}

interface Results {
  stats: {
    frames_processed: number
    max_growth_rate_um_min: number
    avg_growth_rate_um_min: number
    total_branches_end: number
    max_tips: number
  }
  chart_data: any[]
}

function App() {
  const [file, setFile] = useState<File | null>(null)
  const [isDragging, setIsDragging] = useState(false)
  const [job, setJob] = useState<JobStatus | null>(null)
  const [results, setResults] = useState<Results | null>(null)
  const [view, setView] = useState<AppView>('upload')
  const [videoVersion, setVideoVersion] = useState(0)
  const [returnToAnnotating, setReturnToAnnotating] = useState(false)

  // Hyperparameters states
  const [pixelSize, setPixelSize] = useState('1.0')
  const [holeFillArea, setHoleFillArea] = useState('200')
  const [frameInterval, setFrameInterval] = useState('1.0')
  const [minObjectSize, setMinObjectSize] = useState('40')
  const [dilationRadius, setDilationRadius] = useState('8')
  const [minBranchLength, setMinBranchLength] = useState('8')
  const [deepcellToken, setDeepcellToken] = useState('')
  const [showSettings, setShowSettings] = useState(false)

  const inputStyle = {
    width: '100%',
    padding: 8,
    borderRadius: 6,
    border: '1px solid var(--panel-border)',
    background: 'rgba(0,0,0,0.2)',
    color: 'var(--text-primary)',
  } as const

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFile(e.dataTransfer.files[0])
    }
  }, [])

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0])
    }
  }

  const uploadFile = async () => {
    if (!file) return
    
    const formData = new FormData()
    formData.append('file', file)
    formData.append('pixel_size_um', pixelSize)
    formData.append('hole_fill_area', holeFillArea)
    formData.append('frame_interval_min', frameInterval)
    formData.append('min_object_size_px', minObjectSize)
    formData.append('dilation_radius', dilationRadius)
    formData.append('min_branch_length_px', minBranchLength)
    if (deepcellToken) {
      formData.append('deepcell_token', deepcellToken)
    }
    
    try {
      const res = await fetch(`${API_URL}/upload`, {
        method: 'POST',
        body: formData
      })
      const data = await res.json()
      sessionStorage.setItem(JOB_STORAGE_KEY, data.job_id)
      setJob({
        status: 'setup',
        job_id: data.job_id,
        stage: 'extracting_frames',
        current_frame_index: 0,
        total_frames: 0,
        current_frame_preview_url: null,
        progress_percent: 0,
      })
      setView('setup')
    } catch (error) {
      console.error(error)
      alert("Failed to connect to backend. Is it running?")
    }
  }

  const startAutoSegmentation = async () => {
    if (!job) return
    try {
      await fetch(`${API_URL}/jobs/${job.job_id}/segment/auto`, { method: 'POST' })
      setJob({ ...job, status: 'processing', stage: 'segmenting', progress_percent: 0 })
      setView('processing')
    } catch (error) {
      console.error(error)
      alert('Failed to start segmentation')
    }
  }

  const startGuidedSegmentation = async () => {
    if (!job) return
    try {
      await fetch(`${API_URL}/jobs/${job.job_id}/segment/guided`, { method: 'POST' })
      setReturnToAnnotating(true)
      setJob({ ...job, status: 'processing', stage: 'segmenting', progress_percent: 0 })
      setView('processing')
    } catch (error) {
      console.error(error)
      alert('Failed to start guided segmentation')
    }
  }

  const openResults = async () => {
    if (!job) return
    await fetchResults(job.job_id)
    setView('results')
  }

  useEffect(() => {
    const savedJobId = sessionStorage.getItem(JOB_STORAGE_KEY)
    if (!savedJobId || job) return

    fetch(`${API_URL}/status/${savedJobId}`)
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (!data || data.error) return
        setJob({
          status: data.status,
          job_id: data.job_id ?? savedJobId,
          stage: data.stage,
          current_frame_index: data.current_frame_index,
          total_frames: data.total_frames,
          current_frame_preview_url: data.current_frame_preview_url,
          progress_percent: data.progress_percent,
          error: data.error,
        })
        if (data.status === 'review') {
          setView('annotating')
        } else if (data.status === 'setup') {
          setView('setup')
        } else if (data.status === 'processing' || data.status === 'extracting_frames') {
          setView('processing')
        } else if (data.status === 'completed') {
          fetchResults(data.job_id)
          setView('results')
        }
      })
      .catch(() => {})
  }, [job])

  useEffect(() => {
    let interval: number

    if ((view === 'processing' || view === 'setup') && job?.job_id && (job.status === 'processing' || job.status === 'extracting_frames' || job.status === 'setup')) {
      interval = window.setInterval(async () => {
        try {
          const res = await fetch(`${API_URL}/status/${job.job_id}`)
          const data = await res.json()
          if (data.error) return

          if (data.status === 'review') {
            setJob({
              status: 'review',
              job_id: job.job_id,
              stage: 'review',
              total_frames: data.total_frames ?? 0,
              current_frame_preview_url: data.current_frame_preview_url ?? null,
              progress_percent: 100,
              temporal_warning: data.temporal_warning ?? null,
            })
            if (returnToAnnotating) {
              setVideoVersion((v) => v + 1)
              setReturnToAnnotating(false)
            }
            setView('annotating')
            clearInterval(interval)
          } else if (data.status === 'completed') {
            setJob({ status: 'completed', job_id: job.job_id, stage: 'finished', progress_percent: 100 })
            fetchResults(job.job_id)
            setView('results')
            clearInterval(interval)
          } else if (data.status === 'failed') {
            setJob({
              status: 'failed',
              job_id: job.job_id,
              error: data.error,
              stage: 'error',
              progress_percent: data.progress_percent ?? 0,
            })
            setView('failed')
            clearInterval(interval)
          } else {
            setJob({
              status: data.status,
              job_id: job.job_id,
              stage: data.stage ?? 'segmenting',
              current_frame_index: data.current_frame_index ?? 0,
              total_frames: data.total_frames ?? 0,
              current_frame_preview_url: data.current_frame_preview_url ?? null,
              progress_percent: data.progress_percent ?? 0,
              temporal_warning: data.temporal_warning ?? null,
            })
          }
        } catch (error) {
          console.error(error)
        }
      }, 1000)
    }

    return () => clearInterval(interval)
  }, [job, view, returnToAnnotating])

  const fetchResults = async (jobId: string) => {
    try {
      const res = await fetch(`${API_URL}/results/${jobId}`)
      const data = await res.json()
      setResults(data)
    } catch (error) {
      console.error("Failed to fetch results", error)
    }
  }

  const reset = () => {
    setFile(null)
    setJob(null)
    setResults(null)
    setView('upload')
    setReturnToAnnotating(false)
    sessionStorage.removeItem(JOB_STORAGE_KEY)
  }

  return (
    <div className={view === 'annotating' || view === 'setup' ? 'app-shell app-shell-annotation' : 'app-shell'}>
      {view !== 'annotating' && view !== 'setup' && (
      <header className="header">
        <h1>Fungi AI Pipeline</h1>
        <p style={{ color: 'var(--text-secondary)' }}>Automated CellSAM segmentation and growth tracking</p>
      </header>
      )}

      {view === 'upload' && !results && (
        <div className="glass-panel" style={{ maxWidth: 600, margin: '0 auto' }}>
          <div 
            className={`upload-zone ${isDragging ? 'drag-active' : ''}`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => document.getElementById('fileInput')?.click()}
          >
            <input 
              type="file" 
              id="fileInput" 
              style={{ display: 'none' }} 
              accept="video/*,.tif,.tiff"
              onChange={handleFileChange}
            />
            <Upload className="upload-icon" />
            <h3>{file ? file.name : "Drag & drop a video or TIFF stack here"}</h3>
            <p style={{ color: 'var(--text-secondary)', marginTop: 8 }}>
              or click to browse from your computer
            </p>
          </div>

          <div style={{ marginTop: 20 }}>
            <button 
              type="button" 
              onClick={() => setShowSettings(!showSettings)}
              style={{ 
                background: 'rgba(255, 255, 255, 0.05)', 
                color: 'var(--text-primary)', 
                border: '1px solid var(--panel-border)',
                width: '100%',
                padding: '10px',
                borderRadius: '8px',
                cursor: 'pointer',
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center',
                gap: '8px',
                fontSize: '0.95rem'
              }}
            >
              <span>{showSettings ? 'Hide Parameters' : 'Adjust Segmentation & Growth Parameters'}</span>
            </button>
            
            {showSettings && (
              <div style={{ 
                marginTop: 16, 
                padding: 16, 
                borderRadius: 8, 
                border: '1px solid var(--panel-border)', 
                background: 'rgba(255, 255, 255, 0.02)',
                display: 'flex',
                flexDirection: 'column',
                gap: 16,
                textAlign: 'left'
              }}>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
                  <div>
                    <label style={{ display: 'block', fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: 6 }}>
                      Pixel Size (µm/px)
                    </label>
                    <input 
                      type="number" 
                      step="0.01" 
                      min="0.01"
                      value={pixelSize} 
                      onChange={(e) => setPixelSize(e.target.value)} 
                      style={inputStyle}
                    />
                  </div>
                  <div>
                    <label style={{ display: 'block', fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: 6 }}>
                      Hole Fill Area
                    </label>
                    <input 
                      type="number" 
                      step="1" 
                      min="0"
                      value={holeFillArea} 
                      onChange={(e) => setHoleFillArea(e.target.value)} 
                      style={inputStyle}
                    />
                  </div>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
                  <div>
                    <label style={{ display: 'block', fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: 6 }}>
                      Frame Interval (min)
                    </label>
                    <input 
                      type="number" 
                      step="0.1" 
                      min="0.1"
                      value={frameInterval} 
                      onChange={(e) => setFrameInterval(e.target.value)} 
                      style={inputStyle}
                    />
                  </div>
                  <div>
                    <label style={{ display: 'block', fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: 6 }}>
                      Min Object Size (px)
                    </label>
                    <input 
                      type="number" 
                      step="1"
                      min="1"
                      value={minObjectSize} 
                      onChange={(e) => setMinObjectSize(e.target.value)} 
                      style={inputStyle}
                    />
                  </div>
                </div>
                
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
                  <div>
                    <label style={{ display: 'block', fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: 6 }}>
                      Dilation Radius (px)
                    </label>
                    <input 
                      type="number" 
                      step="1"
                      min="1"
                      value={dilationRadius} 
                      onChange={(e) => setDilationRadius(e.target.value)} 
                      style={inputStyle}
                    />
                  </div>
                  <div>
                    <label style={{ display: 'block', fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: 6 }}>
                      Min Branch Length (px)
                    </label>
                    <input 
                      type="number" 
                      step="1"
                      min="1"
                      value={minBranchLength} 
                      onChange={(e) => setMinBranchLength(e.target.value)} 
                      style={inputStyle}
                    />
                  </div>
                </div>

                <div>
                  <label style={{ display: 'block', fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: 6 }}>
                    DeepCell Access Token (for CellSAM weights)
                  </label>
                  <input 
                    type="password" 
                    placeholder="Enter your DeepCell access token..." 
                    value={deepcellToken} 
                    onChange={(e) => setDeepcellToken(e.target.value)} 
                    style={inputStyle}
                  />
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: 4, display: 'block' }}>
                    Required for CellSAM foundation model. Register at <a href="https://users.deepcell.org" target="_blank" rel="noreferrer" style={{ color: 'var(--accent)' }}>users.deepcell.org</a>.
                  </span>
                </div>
              </div>
            )}
          </div>
          
          <div style={{ textAlign: 'center', marginTop: 24 }}>
            <button 
              className="btn" 
              disabled={!file} 
              onClick={uploadFile}
              style={{ padding: '12px 32px', fontSize: '1.1rem' }}
            >
              <Play size={20} /> Upload Video
            </button>
          </div>
          <p style={{ textAlign: 'center', marginTop: 12, color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
            Configure ROI, ignore regions, and segmentation settings on the next screen.
          </p>
        </div>
      )}

      {view === 'setup' && job && (
        <PreSegmentationSetup
          jobId={job.job_id}
          extracting={job.stage === 'extracting_frames' || (job.total_frames ?? 0) === 0}
          onRunSegmentation={startAutoSegmentation}
          onSkipAndRun={startAutoSegmentation}
        />
      )}

      {view === 'processing' && job && (
        <div className="glass-panel progress-panel">
          {job.current_frame_preview_url ? (
            <div className="frame-preview-container">
              <img
                src={`${BACKEND_ORIGIN}${job.current_frame_preview_url}`}
                alt={`Processing frame ${(job.current_frame_index ?? 0) + 1}`}
                className="frame-preview-image"
              />
            </div>
          ) : (
            <div className="pulse-circle"></div>
          )}

          <h2>{STAGE_LABELS[job.stage ?? 'extracting_frames']}</h2>

          {(job.total_frames ?? 0) > 0 && (
            <p className="progress-frame-text">
              Processing frame {(job.current_frame_index ?? 0) + 1} / {job.total_frames}
            </p>
          )}

          <div className="progress-bar-track">
            <div
              className="progress-bar-fill"
              style={{ width: `${job.progress_percent ?? 0}%` }}
            />
          </div>
          <p className="progress-percent-text">{job.progress_percent ?? 0}%</p>

          <p className="progress-subtext">
            Segmentation is running. You will be taken to the review viewer when it finishes.
          </p>
        </div>
      )}

      {view === 'annotating' && job && (
        <AnnotationView
          jobId={job.job_id}
          videoVersion={videoVersion}
          temporalWarning={job.temporal_warning ?? null}
          onRunGuided={startGuidedSegmentation}
          onViewResults={openResults}
          onBack={reset}
        />
      )}

      {view === 'failed' && job && (
        <div className="glass-panel" style={{ maxWidth: 600, margin: '0 auto', textAlign: 'center' }}>
          <AlertCircle size={48} color="var(--accent)" style={{ margin: '0 auto 16px' }} />
          <h2>Processing Failed</h2>
          <p style={{ color: 'var(--text-secondary)', marginTop: 12 }}>
            {job.error || "An unknown error occurred during segmentation."}
          </p>
          <button className="btn" onClick={reset} style={{ marginTop: 24 }}>Try Again</button>
        </div>
      )}

      {view === 'results' && results && job && (
        <div className="dashboard-grid">
          <div className="main-content">
            <div className="glass-panel" style={{ marginBottom: 24, padding: 0, overflow: 'hidden' }}>
              <div className="video-container">
                <video 
                  src={`${API_URL}/jobs/${job.job_id}/media/video`} 
                  controls 
                  autoPlay 
                  loop
                  muted
                />
              </div>
              <div style={{ padding: '16px 24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <h3 style={{ margin: 0 }}>Segmentation Overlay</h3>
                <a href={`${API_URL}/jobs/${job.job_id}/media/video`} download className="btn" style={{ background: 'rgba(255,255,255,0.1)' }}>
                  <Download size={16} /> Save MP4
                </a>
              </div>
            </div>

            <div className="glass-panel">
              <h3 style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <Activity size={20} color="var(--accent)" /> Growth Dynamics
              </h3>
              <div className="chart-container">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={results.chart_data}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                    <XAxis 
                      dataKey="time_min" 
                      stroke="var(--text-secondary)" 
                      label={{ value: 'Time (min)', position: 'insideBottom', offset: -5, fill: 'var(--text-secondary)' }} 
                    />
                    <YAxis yAxisId="left" stroke="var(--text-secondary)" label={{ value: 'Length (μm)', angle: -90, position: 'insideLeft', fill: 'var(--text-secondary)' }} />
                    <YAxis yAxisId="right" orientation="right" stroke="var(--success)" label={{ value: 'Branch Points', angle: 90, position: 'insideRight', fill: 'var(--success)' }} />
                    <Tooltip 
                      contentStyle={{ background: 'var(--bg-color)', border: '1px solid var(--panel-border)', borderRadius: 8 }}
                      itemStyle={{ color: 'var(--text-primary)' }}
                    />
                    <Line yAxisId="left" type="monotone" dataKey="hyphal_length_um" stroke="var(--accent)" strokeWidth={3} dot={false} name="Length" />
                    <Line yAxisId="right" type="monotone" dataKey="branch_points" stroke="var(--success)" strokeWidth={3} dot={false} name="Branches" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          <div className="sidebar">
            <div className="glass-panel">
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 20 }}>
                <h3>Statistics</h3>
                <CheckCircle2 size={20} color="var(--success)" />
              </div>
              
              <div className="stat-card">
                <div className="stat-label">Max Growth Rate</div>
                <div className="stat-value">{results.stats.max_growth_rate_um_min.toFixed(2)}</div>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>μm / min</div>
              </div>

              <div className="stat-card">
                <div className="stat-label">Avg Growth Rate</div>
                <div className="stat-value">{results.stats.avg_growth_rate_um_min.toFixed(2)}</div>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>μm / min</div>
              </div>

              <div className="stat-card">
                <div className="stat-label">Total Branches</div>
                <div className="stat-value">{results.stats.total_branches_end}</div>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>at final frame</div>
              </div>

              <div className="stat-card">
                <div className="stat-label">Max Tips Detected</div>
                <div className="stat-value">{results.stats.max_tips}</div>
              </div>

              <div style={{ marginTop: 24 }}>
                <a href={`${API_URL}/jobs/${job.job_id}/media/metrics_csv`} download className="btn" style={{ width: '100%', justifyContent: 'center' }}>
                  <Download size={18} /> Export Full CSV Data
                </a>
                {job.status === 'review' && (
                  <button
                    onClick={() => setView('annotating')}
                    className="btn"
                    style={{ width: '100%', justifyContent: 'center', marginTop: 12 }}
                  >
                    Back to Review
                  </button>
                )}
                <button onClick={reset} className="btn" style={{ width: '100%', justifyContent: 'center', marginTop: 12, background: 'rgba(255,255,255,0.05)', color: 'var(--text-primary)' }}>
                  Process New Video
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default App
