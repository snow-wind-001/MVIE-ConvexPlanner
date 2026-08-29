function format(value, digits = 2) {
  return value == null ? '—' : Number(value).toFixed(digits)
}

const directionLabels = { left: '左', right: '右', up: '上', down: '下' }

function Metric({ label, value, unit, tone }) {
  return (
    <div className={`metric ${tone ? `metric--${tone}` : ''}`}>
      <span>{label}</span>
      <strong>{value}<small>{unit}</small></strong>
    </div>
  )
}

export default function TelemetryPanel({ scenario, summary, budget }) {
  const metrics = scenario.metrics
  const success = metrics.reactive_success && metrics.reactive_safe
  return (
    <aside className="telemetry" aria-label="算法测试遥测">
      <section className="verdict">
        <div className={`verdict__mark ${success ? 'is-safe' : 'is-hold'}`} aria-hidden="true">
          {success ? '✓' : 'Ⅱ'}
        </div>
        <div>
          <span>本场景判定</span>
          <h2>{success ? '安全绕行' : '安全悬停'}</h2>
          <p>{success ? '三维精确复核通过' : '预算内未找到安全解，不输出路径'}</p>
        </div>
      </section>

      <section className="metric-grid" aria-label="本场景指标">
        <Metric
          label="实时规划"
          value={format(metrics.reactive_planning_ms)}
          unit="ms"
          tone={metrics.reactive_planning_ms <= budget ? 'good' : 'bad'}
        />
        <Metric
          label="最小净空"
          value={format(metrics.reactive_min_clearance_m)}
          unit="m"
          tone={metrics.reactive_min_clearance_m >= 0.3 ? 'good' : 'bad'}
        />
        <Metric label="主绕行方向" value={directionLabels[metrics.avoidance_direction] ?? '—'} unit="" />
        <Metric label="路径增量" value={format(metrics.length_overhead_percent, 1)} unit="%" />
      </section>

      <section className="evidence-section">
        <div className="section-heading">
          <h3>{summary.label}批量证据</h3>
          <span>{summary.scenarios} seeds</span>
        </div>
        <div className="bar-row">
          <div>
            <span>安全成功</span>
            <strong>{summary.reactive_safe_count}/{summary.scenarios}</strong>
          </div>
          <div className="bar"><i style={{ width: `${summary.reactive_safe_count / summary.scenarios * 100}%` }} /></div>
        </div>
        <div className="bar-row">
          <div>
            <span>20 ms 达标</span>
            <strong>{summary.deadline_met_count}/{summary.scenarios}</strong>
          </div>
          <div className="bar"><i className="is-blue" style={{ width: `${summary.deadline_met_count / summary.scenarios * 100}%` }} /></div>
        </div>
      </section>

      <section className="timing-list">
        <div><span>中位数</span><strong>{format(summary.timing.median_ms)} ms</strong></div>
        <div><span>P95</span><strong>{format(summary.timing.p95_ms)} ms</strong></div>
        <div><span>最大</span><strong>{format(summary.timing.max_ms)} ms</strong></div>
        <div><span>低频 full</span><strong>{format(metrics.full_planning_ms, 0)} ms</strong></div>
      </section>

      <section className="protocol-note">
        <h3>测试协议</h3>
        <p>树干、斜向树枝与多球树冠全部参与三维碰撞；在航路上加入新感知横穿树枝，再执行一次 realtime 全向局部修复。</p>
      </section>
    </aside>
  )
}
