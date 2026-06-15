import { useState, useCallback, useEffect } from 'react'
import { Upload, Download, Activity, Play, CheckCircle2, AlertCircle } from 'lucide-react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

const API_URL = 'http://localhost:8000/api'

interface JobStatus {
  status: 'processing' | 'completed' | 'failed'
  job_id: string
  error?: string
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

  // Hyperparameters states
  const [pixelSize, setPixelSize] = useState('1.0')
  const [frameInterval, setFrameInterval] = useState('1.0')
  const [minObjectSize, setMinObjectSize] = useState('40')
  const [dilationRadius, setDilationRadius] = useState('8')
  const [deepcellToken, setDeepcellToken] = useState('')
  const [showSettings, setShowSettings] = useState(false)

  // Preference-based tuning states
  const [mode, setMode] = useState<'standard' | 'tuning'>('standard')
  const [tuningSessionId, setTuningSessionId] = useState<string | null>(null)
  const [tuningRound, setTuningRound] = useState(1)
  const [tuningMaxRounds, setTuningMaxRounds] = useState(6)
  const [tuningCandidates, setTuningCandidates] = useState<any[]>([])
  const [tuningCompleted, setTuningCompleted] = useState(false)
  const [tuningBestParams, setTuningBestParams] = useState<any>(null)
  const [tuningBestScore, setTuningBestScore] = useState(0)
  const [isTuningLoading, setIsTuningLoading] = useState(false)
  const [numTuningFrames, setNumTuningFrames] = useState(1)

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
    formData.append('frame_interval_min', frameInterval)
    formData.append('min_object_size_px', minObjectSize)
    formData.append('dilation_radius', dilationRadius)
    if (deepcellToken) {
      formData.append('deepcell_token', deepcellToken)
    }
    
    try {
      const res = await fetch(`${API_URL}/upload`, {
        method: 'POST',
        body: formData
      })
      const data = await res.json()
      setJob({ status: 'processing', job_id: data.job_id })
    } catch (error) {
      console.error(error)
      alert("Failed to connect to backend. Is it running?")
    }
  }

  const startTuning = async () => {
    if (!file) return
    setIsTuningLoading(true)
    
    const formData = new FormData()
    formData.append('file', file)
    if (deepcellToken) {
      formData.append('deepcell_token', deepcellToken)
    }
    
    try {
      const res = await fetch(`${API_URL}/tune/start`, {
        method: 'POST',
        body: formData
      })
      if (!res.ok) {
        throw new Error(await res.text())
      }
      const data = await res.json()
      setTuningSessionId(data.session_id)
      setTuningRound(data.round)
      setTuningMaxRounds(data.max_rounds)
      setTuningCandidates(data.candidates)
      setTuningCompleted(false)
      setNumTuningFrames(data.num_frames || 1)
    } catch (error) {
      console.error(error)
      alert("Failed to start parameter tuning session.")
    } finally {
      setIsTuningLoading(false)
    }
  }

  const submitFeedback = async (winnerIdx: number) => {
    if (!tuningSessionId) return
    setIsTuningLoading(true)
    
    const formData = new FormData()
    formData.append('session_id', tuningSessionId)
    formData.append('winner_idx', winnerIdx.toString())
    
    try {
      const res = await fetch(`${API_URL}/tune/feedback`, {
        method: 'POST',
        body: formData
      })
      if (!res.ok) {
        throw new Error(await res.text())
      }
      const data = await res.json()
      setTuningBestParams(data.best_params)
      setTuningBestScore(data.best_score)
      
      if (data.completed) {
        setTuningCompleted(true)
      } else {
        setTuningRound(data.round)
        setTuningCandidates(data.candidates)
      }
    } catch (error) {
      console.error(error)
      alert("Failed to submit feedback.")
    } finally {
      setIsTuningLoading(false)
    }
  }

  const applyTuning = () => {
    if (tuningBestParams) {
      setDilationRadius(tuningBestParams.dilation_radius.toString())
      setMinObjectSize(tuningBestParams.min_object_size_px.toString())
      alert(`Applied parameters:\n- Dilation Radius: ${tuningBestParams.dilation_radius}px\n- Min Object Size: ${tuningBestParams.min_object_size_px}px\n- Nearby Margin: ${tuningBestParams.nearby_margin_px}px\n- Min Skan Branch Length: ${tuningBestParams.min_skan_branch_length_px}px`)
    }
    cancelTuning()
  }

  const cancelTuning = () => {
    setTuningSessionId(null)
    setTuningRound(1)
    setTuningCandidates([])
    setTuningCompleted(false)
    setTuningBestParams(null)
    setTuningBestScore(0)
    setNumTuningFrames(1)
    setMode('standard')
  }

  useEffect(() => {
    let interval: number
    
    if (job?.status === 'processing') {
      interval = window.setInterval(async () => {
        try {
          const res = await fetch(`${API_URL}/status/${job.job_id}`)
          const data = await res.json()
          
          if (data.status === 'completed') {
            setJob({ status: 'completed', job_id: job.job_id })
            fetchResults(job.job_id)
            clearInterval(interval)
          } else if (data.status === 'failed') {
            setJob({ status: 'failed', job_id: job.job_id, error: data.error })
            clearInterval(interval)
          }
        } catch (error) {
          console.error(error)
        }
      }, 2000)
    }
    
    return () => clearInterval(interval)
  }, [job])

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
  }

  return (
    <div style={{ maxWidth: 1200, margin: '0 auto', padding: '40px 20px' }}>
      <header className="header">
        <h1>Fungi AI Pipeline</h1>
        <p style={{ color: 'var(--text-secondary)' }}>Automated CellSAM segmentation and growth tracking</p>
      </header>

      {!job && !results && (
        <div className="instrument-panel" style={{ maxWidth: mode === 'tuning' && tuningSessionId ? 800 : 600, margin: '0 auto' }}>
          <div style={{ display: 'flex', borderBottom: '1px solid var(--panel-border)', marginBottom: 20, paddingBottom: 10, gap: 16 }}>
            <button
              onClick={() => { if (!tuningSessionId) setMode('standard') }}
              disabled={!!tuningSessionId}
              style={{
                background: 'none',
                border: 'none',
                color: mode === 'standard' ? 'var(--accent)' : 'var(--text-secondary)',
                fontWeight: mode === 'standard' ? 600 : 400,
                borderBottom: mode === 'standard' ? '2px solid var(--accent)' : 'none',
                paddingBottom: 4,
                cursor: tuningSessionId ? 'not-allowed' : 'pointer',
                fontSize: '1rem'
              }}
            >
              Analyze Video
            </button>
            <button
              onClick={() => { if (!tuningSessionId) setMode('tuning') }}
              disabled={!!tuningSessionId}
              style={{
                background: 'none',
                border: 'none',
                color: mode === 'tuning' ? 'var(--accent)' : 'var(--text-secondary)',
                fontWeight: mode === 'tuning' ? 600 : 400,
                borderBottom: mode === 'tuning' ? '2px solid var(--accent)' : 'none',
                paddingBottom: 4,
                cursor: tuningSessionId ? 'not-allowed' : 'pointer',
                fontSize: '1rem'
              }}
            >
              🎯 Auto-Tune Parameters
            </button>
          </div>

          {mode === 'standard' ? (
            <>
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
                          Pixel Size (μm/px)
                        </label>
                        <input 
                          type="number" 
                          step="0.01" 
                          value={pixelSize} 
                          onChange={(e) => setPixelSize(e.target.value)} 
                          style={{ 
                            width: '100%', 
                            padding: 8, 
                            borderRadius: 6, 
                            border: '1px solid var(--panel-border)', 
                            background: 'rgba(0,0,0,0.2)', 
                            color: 'var(--text-primary)' 
                          }} 
                        />
                      </div>
                      <div>
                        <label style={{ display: 'block', fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: 6 }}>
                          Frame Interval (min)
                        </label>
                        <input 
                          type="number" 
                          step="0.1" 
                          value={frameInterval} 
                          onChange={(e) => setFrameInterval(e.target.value)} 
                          style={{ 
                            width: '100%', 
                            padding: 8, 
                            borderRadius: 6, 
                            border: '1px solid var(--panel-border)', 
                            background: 'rgba(0,0,0,0.2)', 
                            color: 'var(--text-primary)' 
                          }} 
                        />
                      </div>
                    </div>
                    
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
                      <div>
                        <label style={{ display: 'block', fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: 6 }}>
                          Min Object Size (px)
                        </label>
                        <input 
                          type="number" 
                          value={minObjectSize} 
                          onChange={(e) => setMinObjectSize(e.target.value)} 
                          style={{ 
                            width: '100%', 
                            padding: 8, 
                            borderRadius: 6, 
                            border: '1px solid var(--panel-border)', 
                            background: 'rgba(0,0,0,0.2)', 
                            color: 'var(--text-primary)' 
                          }} 
                        />
                      </div>
                      <div>
                        <label style={{ display: 'block', fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: 6 }}>
                          Dilation/Cleanup Radius (px)
                        </label>
                        <input 
                          type="number" 
                          value={dilationRadius} 
                          onChange={(e) => setDilationRadius(e.target.value)} 
                          style={{ 
                            width: '100%', 
                            padding: 8, 
                            borderRadius: 6, 
                            border: '1px solid var(--panel-border)', 
                            background: 'rgba(0,0,0,0.2)', 
                            color: 'var(--text-primary)' 
                          }} 
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
                        style={{ 
                          width: '100%', 
                          padding: 8, 
                          borderRadius: 6, 
                          border: '1px solid var(--panel-border)', 
                          background: 'rgba(0,0,0,0.2)', 
                          color: 'var(--text-primary)' 
                        }} 
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
                  <Play size={20} /> Start Processing
                </button>
              </div>
            </>
          ) : (
            <div>
              {!tuningSessionId ? (
                <div>
                  <h3 style={{ marginBottom: 12 }}>1. Upload Trial Data</h3>
                  <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginBottom: 20 }}>
                    Upload a TIFF stack or video. We'll sample up to 5 frames (sparse, clumped, etc.) to evaluate each parameter config across diverse conditions.
                  </p>
                  
                  <div 
                    className={`upload-zone ${isDragging ? 'drag-active' : ''}`}
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                    onDrop={handleDrop}
                    onClick={() => document.getElementById('fileInputTuning')?.click()}
                  >
                    <input 
                      type="file" 
                      id="fileInputTuning" 
                      style={{ display: 'none' }} 
                      accept="video/*,.tif,.tiff,image/*"
                      onChange={handleFileChange}
                    />
                    <Upload className="upload-icon" />
                    <h3>{file ? file.name : "Drag & drop a video or TIFF stack"}</h3>
                    <p style={{ color: 'var(--text-secondary)', marginTop: 8 }}>
                      single frames also accepted
                    </p>
                  </div>

                  <div style={{ marginTop: 20, display: 'flex', flexDirection: 'column', gap: 12 }}>
                    <div>
                      <label style={{ display: 'block', fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: 6 }}>
                        DeepCell Access Token (optional, if using CellSAM)
                      </label>
                      <input 
                        type="password" 
                        placeholder="Enter token..." 
                        value={deepcellToken} 
                        onChange={(e) => setDeepcellToken(e.target.value)} 
                        style={{ 
                          width: '100%', 
                          padding: 8, 
                          borderRadius: 6, 
                          border: '1px solid var(--panel-border)', 
                          background: 'rgba(0,0,0,0.2)', 
                          color: 'var(--text-primary)' 
                        }} 
                      />
                    </div>
                    
                    <button 
                      className="btn" 
                      disabled={!file || isTuningLoading}
                      onClick={startTuning}
                      style={{ width: '100%', justifyContent: 'center', marginTop: 12 }}
                    >
                      {isTuningLoading ? 'Starting...' : 'Start Auto-Tuning Loop'}
                    </button>
                  </div>
                </div>
              ) : (
                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
                    <h3>Round {tuningRound} of {tuningMaxRounds}</h3>
                    <span style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
                      {numTuningFrames > 1 ? `${numTuningFrames} frames · ` : ''}pick best
                    </span>
                  </div>

                  {tuningBestScore > 0 && (
                    <div style={{ 
                      padding: 10, 
                      borderRadius: 8, 
                      background: 'rgba(16, 185, 129, 0.1)', 
                      border: '1px solid rgba(16, 185, 129, 0.2)', 
                      marginBottom: 16,
                      fontSize: '0.875rem',
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center'
                    }}>
                      <span>📈 Best estimated model score: <strong>{(tuningBestScore * 100).toFixed(0)}%</strong></span>
                      {tuningRound > 2 && (
                        <button 
                          onClick={applyTuning} 
                          style={{
                            background: 'var(--success)',
                            color: 'white',
                            border: 'none',
                            padding: '4px 10px',
                            borderRadius: 4,
                            cursor: 'pointer',
                            fontSize: '0.8rem'
                          }}
                        >
                          Accept Recommendation Now
                        </button>
                      )}
                    </div>
                  )}

                  {isTuningLoading ? (
                    <div style={{ textAlign: 'center', padding: '60px 0' }}>
                      <div className="pulse-circle"></div>
                      <p>Calculating next candidates using GP surrogate model...</p>
                    </div>
                  ) : tuningCompleted ? (
                    <div style={{ textAlign: 'center', padding: '24px 0' }}>
                      <CheckCircle2 size={48} color="var(--success)" style={{ margin: '0 auto 16px' }} />
                      <h2>Auto-Tuning Complete!</h2>
                      <p style={{ color: 'var(--text-secondary)', marginTop: 8, marginBottom: 24 }}>
                        We found the optimal configuration for your imaging session.
                      </p>
                      
                      <div style={{ 
                        background: 'rgba(255,255,255,0.03)', 
                        padding: 20, 
                        borderRadius: 12, 
                        border: '1px solid var(--panel-border)',
                        textAlign: 'left',
                        maxWidth: 400,
                        margin: '0 auto 24px',
                        display: 'flex',
                        flexDirection: 'column',
                        gap: 8
                      }}>
                        <div><strong>Dilation Radius:</strong> {tuningBestParams?.dilation_radius}px</div>
                        <div><strong>Min Object Size:</strong> {tuningBestParams?.min_object_size_px}px</div>
                        <div><strong>Nearby Margin:</strong> {tuningBestParams?.nearby_margin_px}px</div>
                        <div><strong>Min Skan Branch Length:</strong> {tuningBestParams?.min_skan_branch_length_px}px</div>
                      </div>

                      <div style={{ display: 'flex', gap: 12, justifyContent: 'center' }}>
                        <button className="btn" onClick={applyTuning}>Apply & Exit</button>
                        <button className="btn" style={{ background: 'rgba(255,255,255,0.05)', color: 'var(--text-primary)' }} onClick={cancelTuning}>Cancel</button>
                      </div>
                    </div>
                  ) : (
                    <div>
                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
                        {tuningCandidates.map((cand, idx) => (
                          <div 
                            key={idx} 
                            onClick={() => submitFeedback(idx)}
                            style={{
                              border: '1px solid var(--panel-border)',
                              borderRadius: 12,
                              padding: 8,
                              cursor: 'pointer',
                              background: 'rgba(255,255,255,0.02)',
                              transition: 'all 0.2s ease',
                              textAlign: 'center'
                            }}
                            onMouseEnter={(e) => {
                              e.currentTarget.style.borderColor = 'var(--accent)'
                              e.currentTarget.style.background = 'rgba(59, 130, 246, 0.05)'
                            }}
                            onMouseLeave={(e) => {
                              e.currentTarget.style.borderColor = 'var(--panel-border)'
                              e.currentTarget.style.background = 'rgba(255,255,255,0.02)'
                            }}
                          >
                            <div style={{ fontWeight: 600, marginBottom: 8, color: 'var(--accent)', fontSize: '0.9rem' }}>
                              Option {String.fromCharCode(65 + idx)}
                            </div>
                            <div style={{ aspectRatio: numTuningFrames > 1 ? '3/4' : '4/3', overflow: 'hidden', borderRadius: 2, background: '#000', maxHeight: 320 }}>
                              <img 
                                src={`${API_URL}/tune/media/${tuningSessionId}/${idx}.png?t=${Date.now()}`}
                                alt={`Option ${String.fromCharCode(65 + idx)}`}
                                style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                              />
                            </div>
                            <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)', marginTop: 8, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4 }}>
                              <div>Dilation: {cand.dilation_radius}px</div>
                              <div>Min Size: {cand.min_object_size_px}px</div>
                              <div>Margin: {cand.nearby_margin_px}px</div>
                              <div>Prune: {cand.min_skan_branch_length_px}px</div>
                            </div>
                          </div>
                        ))}
                      </div>

                      <div style={{ textAlign: 'center', marginTop: 24 }}>
                        <button className="btn" style={{ background: 'rgba(255,255,255,0.05)', color: 'var(--text-primary)' }} onClick={cancelTuning}>
                          Cancel Auto-Tuning
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {job?.status === 'processing' && (
        <div className="instrument-panel" style={{ maxWidth: 600, margin: '0 auto', textAlign: 'center', padding: '60px 20px' }}>
          <div className="pulse-circle"></div>
          <h2>Segmenting Frames with CellSAM...</h2>
          <p style={{ color: 'var(--text-secondary)', marginTop: 12 }}>
            This may take a few minutes depending on the video length and your GPU.
          </p>
        </div>
      )}

      {job?.status === 'failed' && (
        <div className="instrument-panel" style={{ maxWidth: 600, margin: '0 auto', textAlign: 'center' }}>
          <AlertCircle size={48} color="var(--accent)" style={{ margin: '0 auto 16px' }} />
          <h2>Processing Failed</h2>
          <p style={{ color: 'var(--text-secondary)', marginTop: 12 }}>
            {job.error || "An unknown error occurred during segmentation."}
          </p>
          <button className="btn" onClick={reset} style={{ marginTop: 24 }}>Try Again</button>
        </div>
      )}

      {results && job?.status === 'completed' && (
        <div className="dashboard-grid">
          <div className="main-content">
            <div className="instrument-panel" style={{ marginBottom: 24, padding: 0, overflow: 'hidden' }}>
              <div className="video-container">
                <video 
                  src={`${API_URL}/media/video`} 
                  controls 
                  autoPlay 
                  loop
                  muted
                />
              </div>
              <div style={{ padding: '16px 24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <h3 style={{ margin: 0 }}>Segmentation Overlay</h3>
                <a href={`${API_URL}/media/video`} download className="btn" style={{ background: 'rgba(255,255,255,0.1)' }}>
                  <Download size={16} /> Save MP4
                </a>
              </div>
            </div>

            <div className="instrument-panel">
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
            <div className="instrument-panel">
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
                <a href={`${API_URL}/media/metrics_csv`} download className="btn" style={{ width: '100%', justifyContent: 'center' }}>
                  <Download size={18} /> Export Full CSV Data
                </a>
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
