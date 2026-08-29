import { useEffect, useMemo, useRef, useState } from 'react'
import ForestScene from './components/ForestScene.jsx'
import TelemetryPanel from './components/TelemetryPanel.jsx'
import results from './data/forest-results.json'

const densityOrder = ['sparse', 'medium', 'dense']

function PlayIcon({ playing }) {
  return playing ? (
    <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 5h4v14H7zM13 5h4v14h-4z" /></svg>
  ) : (
    <svg viewBox="0 0 24 24" aria-hidden="true"><path d="m8 5 11 7-11 7z" /></svg>
  )
}

function CameraIcon() {
  return <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8.5 5 10 3h4l1.5 2H19a2 2 0 0 1 2 2v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2h3.5ZM12 8a4.5 4.5 0 1 0 0 9 4.5 4.5 0 0 0 0-9Zm0 2a2.5 2.5 0 1 1 0 5 2.5 2.5 0 0 1 0-5Z" /></svg>
}

export default function App() {
  const [density, setDensity] = useState('sparse')
  const densityScenarios = useMemo(
    () => results.scenarios.filter((scenario) => scenario.density === density),
    [density],
  )
  const [scenarioId, setScenarioId] = useState(densityScenarios[0]?.id)
  const [playing, setPlaying] = useState(false)
  const [progress, setProgress] = useState(0)
  const [showReference, setShowReference] = useState(true)
  const [showSafety, setShowSafety] = useState(true)
  const [cameraMode, setCameraMode] = useState('overview')
  const lastFrame = useRef(null)

  useEffect(() => {
    setScenarioId(densityScenarios[0]?.id)
    setProgress(0)
    setPlaying(false)
  }, [density, densityScenarios])

  const scenario = densityScenarios.find((item) => item.id === scenarioId) ?? densityScenarios[0]
  const summary = results.summaries.find((item) => item.density === density)
  const canPlay = Boolean(scenario?.realtime_path)

  useEffect(() => {
    if (!playing || !canPlay) {
      lastFrame.current = null
      return undefined
    }
    let frame
    const tick = (timestamp) => {
      if (lastFrame.current == null) lastFrame.current = timestamp
      const elapsed = timestamp - lastFrame.current
      lastFrame.current = timestamp
      setProgress((current) => {
        const next = current + elapsed / 9000
        if (next >= 1) {
          setPlaying(false)
          return 1
        }
        return next
      })
      frame = requestAnimationFrame(tick)
    }
    frame = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(frame)
  }, [canPlay, playing])

  const togglePlayback = () => {
    if (!canPlay) return
    if (progress >= 1) setProgress(0)
    setPlaying((value) => !value)
  }

  if (!scenario || !summary) return null

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand__mark" aria-hidden="true"><i /><i /><i /></div>
          <div>
            <h1>ForestFlight Lab</h1>
            <p>MVIE–FIRI 无人机穿林避障验证</p>
          </div>
        </div>
        <div className="budget-status">
          <i />
          <span>实时预算</span>
          <strong>{results.metadata.time_budget_ms.toFixed(0)} ms</strong>
        </div>
      </header>

      <main className="workspace">
        <section className="viewport-panel" aria-label="三维飞行回放">
          <div className="scene-toolbar">
            <div className="density-tabs" aria-label="树林密度">
              {densityOrder.map((item) => (
                <button
                  type="button"
                  key={item}
                  className={density === item ? 'is-active' : ''}
                  onClick={() => setDensity(item)}
                >
                  {results.summaries.find((entry) => entry.density === item)?.label}
                </button>
              ))}
            </div>
            <label className="scenario-select">
              <span>场景</span>
              <select value={scenario.id} onChange={(event) => {
                setScenarioId(event.target.value)
                setProgress(0)
                setPlaying(false)
              }}>
                {densityScenarios.map((item) => (
                  <option value={item.id} key={item.id}>Seed {String(item.seed).padStart(2, '0')}</option>
                ))}
              </select>
            </label>
          </div>

          <ForestScene
            scenario={scenario}
            progress={progress}
            showReference={showReference}
            showSafety={showSafety}
            cameraMode={cameraMode}
          />

          <div className="scene-legend" aria-label="图例">
            <span><i className="legend-path" />修复路径</span>
            <span><i className="legend-reference" />参考路径</span>
            <span><i className="legend-danger" />新感知枝障</span>
          </div>

          <div className="view-controls">
            <button
              type="button"
              aria-label="切换相机模式"
              title="切换相机模式"
              onClick={() => setCameraMode((mode) => mode === 'overview' ? 'follow' : 'overview')}
              className={cameraMode === 'follow' ? 'is-active' : ''}
            ><CameraIcon /></button>
            <label><input type="checkbox" checked={showReference} onChange={(event) => setShowReference(event.target.checked)} />参考路径</label>
            <label><input type="checkbox" checked={showSafety} onChange={(event) => setShowSafety(event.target.checked)} />安全域</label>
          </div>

          <div className="playback">
            <button
              type="button"
              className="playback__button"
              onClick={togglePlayback}
              disabled={!canPlay}
              aria-label={canPlay ? (playing ? '暂停回放' : '开始回放') : '无安全路径，无人机保持悬停'}
            ><PlayIcon playing={playing} /></button>
            <label className="timeline">
              <span>{canPlay ? '飞行进度' : '未输出路径 · 保持悬停'}</span>
              <input
                type="range"
                min="0"
                max="100"
                value={Math.round(progress * 100)}
                disabled={!canPlay}
                onChange={(event) => {
                  setProgress(Number(event.target.value) / 100)
                  setPlaying(false)
                }}
              />
            </label>
            <output>{Math.round(progress * 100)}%</output>
          </div>
        </section>

        <TelemetryPanel scenario={scenario} summary={summary} budget={results.metadata.time_budget_ms} />
      </main>
    </div>
  )
}
