
import { useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  Download,
  Eye,
  FileText,
  Moon,
  Network,
  RefreshCw,
  Server,
  ShieldCheck,
  Sun,
  Upload,
} from "lucide-react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  PieChart,
  Pie,
  Cell,
} from "recharts";
import "./App.css";

const APP_NAME = "Argus";
const API_URL = "http://127.0.0.1:8000";
const PIE_COLORS = ["var(--chart-neutral)", "var(--warn)"];
const BAR_BENIGN = "var(--chart-benign)";
const BAR_ATTACK = "var(--danger)";
const TOOLTIP_STYLE = {
  background: "var(--surface)",
  border: "1px solid var(--line-strong)",
  borderRadius: 8,
  color: "var(--text)",
  fontSize: 12,
  boxShadow: "0 6px 20px rgba(0,0,0,0.15)",
};
const AXIS_TICK = { fontSize: 12, fill: "var(--dim)" };
const EMPTY_CLASS_COUNTS = Object.freeze({});
const EMPTY_PREVIEW_ROWS = Object.freeze([]);

function initialTheme() {
  try {
    const saved = localStorage.getItem("argus-theme");
    if (saved === "light" || saved === "dark") return saved;
  } catch {
    /* storage unavailable */
  }
  return "dark";
}

const FIELD_PALETTE = {
  idle: ["57,255,136", "205,255,225"],
  ok: ["57,255,136", "205,255,225"],
  warn: ["245,185,68", "255,228,165"],
  danger: ["255,77,106", "255,196,202"],
};

/* Animated network field: nodes drift, packets travel to the protected core.
   Colour and packet rate follow the analysis result. */
function NetworkField({ tone, busy }) {
  const canvasRef = useRef(null);
  const liveRef = useRef({ tone, busy });

  useEffect(() => {
    liveRef.current = { tone, busy };
  }, [tone, busy]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return undefined;
    const ctx = canvas.getContext("2d");
    if (!ctx) return undefined;
    const reduce = window.matchMedia?.(
      "(prefers-reduced-motion: reduce)"
    ).matches;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const nodes = [];
    let w = 0;
    let h = 0;
    let raf = 0;
    const rand = (a, b) => a + Math.random() * (b - a);

    function resize() {
      const rect = canvas.getBoundingClientRect();
      w = rect.width;
      h = rect.height;
      canvas.width = Math.max(1, Math.round(w * dpr));
      canvas.height = Math.max(1, Math.round(h * dpr));
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      nodes.length = 0;
      const count = Math.round(Math.min(64, Math.max(22, (w * h) / 9000)));
      for (let i = 0; i < count; i += 1) {
        nodes.push({
          x: rand(0, w),
          y: rand(0, h),
          vx: rand(-0.14, 0.14),
          vy: rand(-0.14, 0.14),
          r: rand(1.2, 2.8),
        });
      }
    }

    function draw(time) {
      const { tone: t, busy: isBusy } = liveRef.current;
      const [c1] = FIELD_PALETTE[t] || FIELD_PALETTE.idle;
      const cx = w * 0.5;
      const cy = h * 0.47;
      ctx.clearRect(0, 0, w, h);

      for (const n of nodes) {
        if (!reduce) {
          n.x += n.vx;
          n.y += n.vy;
          if (n.x < 0 || n.x > w) n.vx *= -1;
          if (n.y < 0 || n.y > h) n.vy *= -1;
        }
      }
      for (let i = 0; i < nodes.length; i += 1) {
        for (let j = i + 1; j < nodes.length; j += 1) {
          const dx = nodes[i].x - nodes[j].x;
          const dy = nodes[i].y - nodes[j].y;
          const d = Math.hypot(dx, dy);
          if (d < 130) {
            ctx.strokeStyle = `rgba(80,150,255,${(1 - d / 130) * 0.24})`;
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(nodes[i].x, nodes[i].y);
            ctx.lineTo(nodes[j].x, nodes[j].y);
            ctx.stroke();
          }
        }
      }
      for (const n of nodes) {
        ctx.fillStyle = "rgba(130,185,255,0.7)";
        ctx.beginPath();
        ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
        ctx.fill();
      }

      // pulsing rings around the protected core
      for (let k = 0; k < 3; k += 1) {
        const phase = ((time / (isBusy ? 900 : 2200) + k / 3) % 1);
        ctx.strokeStyle = `rgba(${c1},${(1 - phase) * 0.4})`;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(cx, cy, 30 + phase * 110, 0, Math.PI * 2);
        ctx.stroke();
      }
      const glow = ctx.createRadialGradient(cx, cy, 0, cx, cy, 60);
      glow.addColorStop(0, `rgba(${c1},0.55)`);
      glow.addColorStop(1, `rgba(${c1},0)`);
      ctx.fillStyle = glow;
      ctx.beginPath();
      ctx.arc(cx, cy, 60, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = `rgb(${c1})`;
      ctx.beginPath();
      ctx.arc(cx, cy, 9, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "rgba(255,255,255,0.9)";
      ctx.beginPath();
      ctx.arc(cx, cy, 3.5, 0, Math.PI * 2);
      ctx.fill();

    }

    function loop(time) {
      draw(time);
      raf = requestAnimationFrame(loop);
    }

    resize();
    if (reduce) {
      draw(0);
    } else {
      raf = requestAnimationFrame(loop);
    }
    const observer = new ResizeObserver(() => {
      resize();
      if (reduce) draw(0);
    });
    observer.observe(canvas);
    return () => {
      cancelAnimationFrame(raf);
      observer.disconnect();
    };
  }, []);

  return <canvas ref={canvasRef} className="hero-canvas" aria-hidden="true" />;
}

function CountUp({ value }) {
  const [shown, setShown] = useState(0);

  useEffect(() => {
    const reduce = window.matchMedia?.(
      "(prefers-reduced-motion: reduce)"
    ).matches;
    const duration = reduce ? 0 : 900;
    const start = performance.now();
    let raf = 0;
    const tick = (now) => {
      const t = duration ? Math.min(1, (now - start) / duration) : 1;
      setShown(Math.round(value * (1 - (1 - t) ** 3)));
      if (t < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [value]);

  return <>{shown.toLocaleString()}</>;
}

function StatCard({ icon: Icon, title, raw, share, subtitle, tone = "neutral" }) {
  return (
    <div className="stat-card" data-tone={tone}>
      <div className="stat-top">
        <span className="stat-icon">
          <Icon size={15} />
        </span>
        <span>{title}</span>
      </div>
      <div className="stat-value">
        {raw === null || raw === undefined ? "—" : <CountUp value={raw} />}
      </div>
      {share !== undefined && (
        <div className="stat-bar" aria-hidden="true">
          <i style={{ width: `${Math.min(100, Math.max(0, share * 100))}%` }} />
        </div>
      )}
      <div className="stat-subtitle">{subtitle}</div>
    </div>
  );
}

function formatValue(value) {
  if (value === null || value === undefined) return "—";
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (typeof value === "number") {
    return Number.isFinite(value)
      ? Number(value.toFixed(5)).toString()
      : "—";
  }
  return String(value);
}

function isTrue(value) {
  return (
    value === true ||
    value === 1 ||
    String(value).toLowerCase() === "true"
  );
}

export default function App() {
  const [file, setFile] = useState(null);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState("all");
  const [theme, setTheme] = useState(initialTheme);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try {
      localStorage.setItem("argus-theme", theme);
    } catch {
      /* storage unavailable */
    }
  }, [theme]);

  async function analyzeFile() {
    if (!file || loading) return;

    setLoading(true);
    setError("");
    setResults(null);
    setSearch("");
    setFilter("all");

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${API_URL}/analyze`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Analysis failed.");
      }

      setResults(data);
    } catch (err) {
      setError(
        err.message ||
          "Could not connect to the Python API. Check that it is running."
      );
    } finally {
      setLoading(false);
    }
  }

  function handleFile(event) {
    setFile(event.target.files?.[0] ?? null);
    setResults(null);
    setError("");
    setSearch("");
    setFilter("all");
  }

  const classCounts = results?.class_counts ?? EMPTY_CLASS_COUNTS;

  const sortedClasses = useMemo(
    () =>
      Object.entries(classCounts)
        .map(([name, count]) => ({
          name,
          count: Number(count) || 0,
        }))
        .sort((a, b) => b.count - a.count),
    [classCounts]
  );

  const benignCount =
    sortedClasses.find(
      (item) => item.name.trim().toLowerCase() === "benign"
    )?.count ?? 0;

  const predictedNonBenign = Math.max(
    0,
    sortedClasses.reduce((sum, item) => sum + item.count, 0) -
      benignCount
  );

  const attackClasses = sortedClasses.filter(
    (item) => item.name.trim().toLowerCase() !== "benign"
  );

  const anomalyCount = Number(results?.anomaly_count ?? 0);
  const generalizedAttackCount = Number(
    results?.generalized_attack_count ?? 0
  );
  const modelGeneralizedAttackCount = Number(
    results?.model_generalized_attack_count ?? 0
  );
  const behavioralFindings = results?.behavioral_findings;
  const targetFloodFlowCount = Number(behavioralFindings?.target_flood_flow_count ?? 0);
  const targetFloodEvent = behavioralFindings?.target_flood_events?.[0];
  const behavioralFlowCount = Math.max(
    Number(behavioralFindings?.high_rate_multiport_flow_count ?? 0),
    Number(behavioralFindings?.source_burst_flow_count ?? 0),
    targetFloodFlowCount,
  );
  const topBehavioralEvent = behavioralFindings?.events?.[0];
  const evaluation = results?.evaluation;
  const labeledClasses = Object.entries(
    evaluation?.classification_report ?? {}
  )
    .filter(([, metrics]) => Number(metrics?.support) > 0)
    .sort((left, right) => Number(right[1].support) - Number(left[1].support));
  const totalFlows = Number(results?.total_flows ?? 0);
  const normalByIsolationForest = Math.max(
    0,
    totalFlows - anomalyCount
  );

  const classChartData = sortedClasses.slice(0, 8).map((item) => ({
    name:
      item.name.length > 19
        ? `${item.name.slice(0, 17)}…`
        : item.name,
    fullName: item.name,
    flows: item.count,
  }));

  const anomalyChartData = results
    ? [
        {
          name: "Not flagged",
          value: normalByIsolationForest,
        },
        {
          name: "Anomalies",
          value: anomalyCount,
        },
      ]
    : [];

  // The API deliberately sends only a 500-row preview.
  const previewRows = results?.results ?? EMPTY_PREVIEW_ROWS;

  const filteredRows = useMemo(() => {
    return previewRows.filter((row) => {
      const predicted = String(
        row.predicted_class ?? row.hybrid_status ?? ""
      )
        .trim()
        .toLowerCase();

      const anomaly = isTrue(row.is_anomaly);

      const matchesFilter =
        filter === "all" ||
        (filter === "anomaly" && anomaly) ||
        (filter === "behavioral" && (
          isTrue(row.is_source_burst_alert) ||
          isTrue(row.is_high_rate_multiport_activity)
        )) ||
        (filter === "attack" &&
          ((predicted !== "benign" && predicted !== "") ||
            isTrue(row.is_model_generalized_attack)));

      const matchesSearch =
        !search ||
        Object.values(row).some((value) =>
          String(value ?? "")
            .toLowerCase()
            .includes(search.toLowerCase())
        );

      return matchesFilter && matchesSearch;
    });
  }, [previewRows, filter, search]);

  const resultColumns = results?.columns ?? [];

  function downloadFullCSV() {
    if (!results?.download_url) return;
    window.location.href = `${API_URL}${results.download_url}`;
  }

  const modelsFlagged = predictedNonBenign > 0 || generalizedAttackCount > 0;
  const behavioralSubject = topBehavioralEvent
    ? ` (top source ${topBehavioralEvent.source_ip})`
    : targetFloodEvent
      ? ` (aimed at ${targetFloodEvent.destination_ip})`
      : "";
  let verdictTone = "ok";
  let verdictTitle = "No threats detected";
  let verdictText = `Models and behavioral rules found nothing suspicious in ${totalFlows.toLocaleString()} flows.`;
  if (modelsFlagged) {
    verdictTone = "danger";
    verdictTitle = "Attack traffic predicted";
    verdictText =
      `Random Forest classed ${predictedNonBenign.toLocaleString()} flows as non-benign` +
      (attackClasses[0] ? ` (mostly ${attackClasses[0].name})` : "") +
      `; the learned binary model flagged ${generalizedAttackCount.toLocaleString()}.` +
      (behavioralFlowCount > 0
        ? ` ${behavioralFlowCount.toLocaleString()} flows also matched behavioral rules${behavioralSubject}.`
        : "");
  } else if (behavioralFlowCount > 0) {
    verdictTone = "warn";
    verdictTitle = "Possible scan or high-rate activity detected by behavioral rules";
    verdictText =
      `The trained models flagged no attacks, but ${behavioralFlowCount.toLocaleString()} flows${behavioralSubject} ` +
      "matched the port-diversity / source-rate / unanswered-flood rules. The models may not " +
      "recognise this traffic pattern; review the Behavioral Heuristic panel below.";
  }

  // Moves the specular highlight on whichever glass card the pointer is over.
  function trackLight(event) {
    const card = event.target.closest?.(".panel, .stat-card, .verdict");
    if (!card) return;
    const rect = card.getBoundingClientRect();
    card.style.setProperty("--mx", `${event.clientX - rect.left}px`);
    card.style.setProperty("--my", `${event.clientY - rect.top}px`);
  }

  return (
    <div className="app-shell">
      <aside className="hero">
        <NetworkField tone={results ? verdictTone : "idle"} busy={loading} />
        <div className="hero-top">
          <div className="brand">
            <div className="brand-logo">
              <Eye size={19} />
            </div>
            <h2>{APP_NAME}</h2>
          </div>
        </div>

        <div className="hero-title">
          <h2>Network</h2>
          <span className="hero-chip">
            <Network size={13} /> Hybrid intrusion detection
          </span>
        </div>

        {results && (
          <div className="hero-cards">
            <div className="hero-card">
              <small>Flows analyzed</small>
              <strong>{totalFlows.toLocaleString()}</strong>
            </div>
            <div className="hero-card" data-tone={verdictTone}>
              <small>Verdict</small>
              <strong>{verdictTone === "ok" ? "Clear" : verdictTone === "warn" ? "Rule alert" : "Attack"}</strong>
            </div>
          </div>
        )}

      </aside>

      <main
        className="main-content"
        id="dashboard"
        onPointerMove={trackLight}
      >
        <header className="topbar">
          <div>
            <p className="eyebrow">Network intrusion detection</p>
            <h1>Traffic Analysis</h1>
          </div>
          <div className="topbar-actions">
            <button
              className="icon-button"
              onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
              aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
              title="Toggle theme"
            >
              {theme === "dark" ? <Sun size={16} /> : <Moon size={16} />}
            </button>
          </div>
        </header>

        <section className="panel analyze-panel" id="analyze">
          <div className="panel-heading">
            <div>
              <h3>Analyze a capture</h3>
              <p>
                Upload a CIC flow CSV or a PCAP / PCAPNG capture. Random
                Forest, Isolation Forest and behavioral rules report
                separately.
              </p>
            </div>
          </div>

          <div className="analyze-row">
            <label className="drop-zone">
              <span className="drop-icon">
                {file ? <FileText size={20} /> : <Upload size={20} />}
              </span>
              <span className="drop-text">
                <strong>{file ? file.name : "Choose a CSV or PCAP file"}</strong>
                <small>
                  {file
                    ? `${(file.size / 1024 / 1024).toFixed(2)} MB`
                    : "CSV, PCAP or PCAPNG · up to 20 GiB"}
                </small>
              </span>
              <input
                type="file"
                accept=".csv,.pcap,.pcapng,.pcap10,text/csv,application/vnd.tcpdump.pcap"
                onChange={handleFile}
              />
            </label>

            <button
              className="analyze-button"
              disabled={!file || loading}
              onClick={analyzeFile}
            >
              {loading ? (
                <RefreshCw className="spin" size={16} />
              ) : (
                <Activity size={16} />
              )}
              {loading ? "Analyzing traffic..." : "Analyze Network Traffic"}
            </button>
          </div>

          {error && (
            <div className="error-message" role="alert">
              <strong>Analysis failed</strong>
              <p>{error}</p>
            </div>
          )}

          {results && (
            <div className="result-message">
              <div className="result-success">
                <CheckCircle2 size={17} />
                <strong>Analysis completed</strong>
              </div>
              <dl className="result-facts">
                <div>
                  <dt>File</dt>
                  <dd>{results.filename}</dd>
                </div>
                <div>
                  <dt>Total flows</dt>
                  <dd>{totalFlows.toLocaleString()}</dd>
                </div>
                <div>
                  <dt>Anomalies</dt>
                  <dd>{anomalyCount.toLocaleString()}</dd>
                </div>
                <div>
                  <dt>Table preview</dt>
                  <dd>{results.preview_rows ?? previewRows.length} rows</dd>
                </div>
              </dl>
              <button
                className="download-button"
                onClick={downloadFullCSV}
                disabled={!results.download_url}
              >
                <Download size={15} /> Download full results CSV
              </button>
            </div>
          )}

          <p className="upload-note">
            PCAP conversion requires CICFlowMeter on the API server. No fixed
            flow-count limit.
          </p>
        </section>

        {results && (
          <section
            className="verdict"
            data-tone={verdictTone}
            role={verdictTone === "ok" ? "status" : "alert"}
          >
            {verdictTone === "ok" ? (
              <CheckCircle2 size={20} />
            ) : (
              <AlertTriangle size={20} />
            )}
            <div>
              <strong>{verdictTitle}</strong>
              <p>{verdictText}</p>
            </div>
          </section>
        )}

        <section className="stats-grid">
          <StatCard
            icon={Activity}
            title="Analyzed Flows"
            raw={results ? totalFlows : null}
            subtitle={results ? results.filename : "Upload a capture to begin"}
          />
          <StatCard
            icon={AlertTriangle}
            title="Non-Benign Predictions"
            raw={results ? predictedNonBenign : null}
            share={results && totalFlows ? predictedNonBenign / totalFlows : 0}
            subtitle="Based on predicted classes"
            tone={results && predictedNonBenign > 0 ? "danger" : "neutral"}
          />
          <StatCard
            icon={ShieldCheck}
            title="Learned Binary Attack Flags"
            raw={results ? generalizedAttackCount : null}
            share={results && totalFlows ? generalizedAttackCount / totalFlows : 0}
            subtitle={
              results
                ? "Calibrated classifier only; behavior rules are separate"
                : "Learned model predictions only"
            }
            tone={results && generalizedAttackCount > 0 ? "danger" : "neutral"}
          />
          <StatCard
            icon={AlertTriangle}
            title="Behavioral Alerts"
            raw={results ? behavioralFlowCount : null}
            share={results && totalFlows ? behavioralFlowCount / totalFlows : 0}
            subtitle={
              results
                ? topBehavioralEvent
                  ? `Top source ${topBehavioralEvent.source_ip} (heuristic, not a model)`
                  : targetFloodEvent
                    ? `Flood aimed at ${targetFloodEvent.destination_ip} (heuristic, not a model)`
                    : "No source-rate, port-scan or flood rule matched"
                : "Rule-based signals"
            }
            tone={results && behavioralFlowCount > 0 ? "warn" : "neutral"}
          />
          <StatCard
            icon={CheckCircle2}
            title="Anomalies"
            raw={results ? anomalyCount : null}
            share={results && totalFlows ? anomalyCount / totalFlows : 0}
            subtitle={
              results && totalFlows
                ? `${((anomalyCount / totalFlows) * 100).toFixed(2)}% flagged`
                : "Awaiting analysis"
            }
          />
        </section>

        {evaluation && (
          <section className="panel evaluation-panel">
            <div className="panel-heading">
              <div>
                <h3>Ground-Truth Label Evaluation</h3>
                <p>
                  Actual class counts and strict label comparison. Unsupported
                  classes are counted as unmatched, not remapped.
                </p>
              </div>
              <span className="panel-chip">LABELED INPUT</span>
            </div>
            <div className="evaluation-metrics">
              <span>
                Exact-label agreement: {(Number(evaluation.accuracy) * 100).toFixed(2)}%
              </span>
              <span>
                Balanced accuracy: {(
                  Number(evaluation.balanced_accuracy) * 100
                ).toFixed(2)}%
              </span>
              <span>
                Macro F1: {(Number(evaluation.f1_macro) * 100).toFixed(2)}%
              </span>
              <span>
                Evaluated rows: {Number(
                  evaluation.evaluated_rows ?? 0
                ).toLocaleString()}
              </span>
            </div>
            <div className="evaluation-table-wrap">
              <table className="evaluation-table">
                <thead>
                  <tr>
                    <th>Actual class</th>
                    <th>Rows</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1</th>
                  </tr>
                </thead>
                <tbody>
                  {labeledClasses.map(([name, metrics]) => (
                    <tr key={name}>
                      <td>{name}</td>
                      <td>{Number(metrics.support).toLocaleString()}</td>
                      <td>{(Number(metrics.precision) * 100).toFixed(2)}%</td>
                      <td>{(Number(metrics.recall) * 100).toFixed(2)}%</td>
                      <td>{(Number(metrics["f1-score"]) * 100).toFixed(2)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="table-note">
              {evaluation.note} CSV labels are used only for this evaluation
              and are never passed to the detector as model inputs. Unsupported
              actual labels: {Object.entries(
                evaluation.unsupported_actual_label_counts ?? {}
              ).map(([name, count]) => `${name} (${Number(count).toLocaleString()})`).join(", ") || "none"}.
            </p>
          </section>
        )}

        {results && behavioralFindings && (
          <section
            className={`panel behavioral-summary${behavioralFlowCount === 0 ? " is-clear" : ""}`}
          >
            <div className="panel-heading">
              <div>
                <h3>Behavioral Heuristic (separate from model predictions)</h3>
                <p>Source-rate and port-diversity rules can flag benign high-volume traffic.</p>
              </div>
              <ShieldCheck size={19} />
            </div>
            <h2>{behavioralFindings.classification}</h2>
            <p>
              {Number(behavioralFindings.source_burst_event_count ?? 0).toLocaleString()} source-minute source-burst events cover {Number(behavioralFindings.source_burst_flow_count ?? 0).toLocaleString()} flows. {Number(behavioralFindings.high_rate_multiport_event_count ?? 0).toLocaleString()} also exceeded the port-diversity threshold, covering {Number(behavioralFindings.high_rate_multiport_flow_count ?? 0).toLocaleString()} flows.
              {` Random Forest non-benign predictions: ${predictedNonBenign.toLocaleString()}; learned binary-model flags: ${modelGeneralizedAttackCount.toLocaleString()}. Heuristic matches are not model predictions.`}
            </p>
            {behavioralFindings.events?.map((event) => (
              <p className="event-line" key={event.event_id}>
                {event.classification}: {event.source_ip}, {event.window_start} to {event.last_observed_at}; {Number(event.flow_count).toLocaleString()} associated flows, {Number(event.unique_destination_ports).toLocaleString()} destination ports across {Number(event.unique_destination_ips).toLocaleString()} destination IPs. Top destination: {event.top_destination_ips?.[0]?.destination_ip ?? "unavailable"}. Reason: {event.reasons?.join("; ")}.
              </p>
            ))}
            {behavioralFindings.target_flood_events?.map((event) => (
              <p className="event-line" key={`${event.destination_ip}|${event.window_start}`}>
                {event.classification}: target {event.destination_ip}, minute starting {event.window_start}; {Number(event.flows).toLocaleString()} flows, {(Number(event.unanswered_share) * 100).toFixed(1)}% unanswered{event.unique_sources != null ? `, ${Number(event.unique_sources).toLocaleString()} distinct sources` : ""}. Threshold: more than {Number(event.flow_threshold).toLocaleString()} flows per minute (uncalibrated development rule).
              </p>
            ))}
            <p className="table-note">{behavioralFindings.note}</p>
          </section>
        )}

        <section className="charts-grid" id="traffic">
          <div className="panel traffic-panel">
            <div className="panel-heading">
              <div>
                <h3>Predicted Class Distribution</h3>
                <p>
                  {results
                    ? "Model predictions; supplied CSV labels are shown separately"
                    : "Upload a CSV to populate this chart"}
                </p>
              </div>
              <span className="panel-chip">
                {results ? "MODEL RESULTS" : "NO DATA"}
              </span>
            </div>

            {results && classChartData.length > 0 ? (
              <>
                <div className="chart-area">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={classChartData}
                      margin={{
                        top: 8,
                        right: 10,
                        left: 0,
                        bottom: 35,
                      }}
                    >
                      <CartesianGrid
                        stroke="var(--line)"
                        strokeDasharray="4 4"
                        vertical={false}
                      />
                      <XAxis
                        dataKey="name"
                        stroke="var(--line-strong)"
                        tickLine={false}
                        axisLine={false}
                        interval={0}
                        angle={-25}
                        textAnchor="end"
                        height={60}
                        tick={AXIS_TICK}
                      />
                      <YAxis
                        stroke="var(--line-strong)"
                        tickLine={false}
                        axisLine={false}
                        tick={AXIS_TICK}
                      />
                      <Tooltip
                        contentStyle={TOOLTIP_STYLE}
                        formatter={(value) => [
                          Number(value).toLocaleString(),
                          "Predicted flows",
                        ]}
                        labelFormatter={(label, payload) =>
                          payload?.[0]?.payload?.fullName ?? label
                        }
                      />
                      <Bar
                        dataKey="flows"
                        name="Predicted flows"
                        radius={[4, 4, 0, 0]}
                      >
                        {classChartData.map((entry) => (
                          <Cell
                            key={entry.fullName}
                            fill={
                              entry.fullName.trim().toLowerCase() === "benign"
                                ? BAR_BENIGN
                                : BAR_ATTACK
                            }
                          />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <p className="chart-note">
                  Showing up to eight most frequent predicted classes.
                </p>
              </>
            ) : (
              <div className="empty-state">
                Class distribution will appear here after a successful
                analysis.
              </div>
            )}
          </div>

          <div className="panel distribution-panel">
            <div className="panel-heading">
              <div>
                <h3>Anomaly Distribution</h3>
                <p>
                  {results
                    ? "Actual Isolation Forest flags"
                    : "Upload a CSV to populate this chart"}
                </p>
              </div>
            </div>

            {results ? (
              <>
                <div className="donut-wrap">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={anomalyChartData}
                        dataKey="value"
                        nameKey="name"
                        innerRadius={64}
                        outerRadius={90}
                        paddingAngle={4}
                        stroke="none"
                      >
                        {anomalyChartData.map((entry, index) => (
                          <Cell
                            key={entry.name}
                            fill={PIE_COLORS[index % PIE_COLORS.length]}
                          />
                        ))}
                      </Pie>
                      <Tooltip
                        contentStyle={TOOLTIP_STYLE}
                        formatter={(value) => [
                          Number(value).toLocaleString(),
                          "Flows",
                        ]}
                      />
                    </PieChart>
                  </ResponsiveContainer>
                  <div className="donut-center">
                    <strong>
                      {totalFlows
                        ? `${((anomalyCount / totalFlows) * 100).toFixed(1)}%`
                        : "0%"}
                    </strong>
                    <span>Flagged anomalous</span>
                  </div>
                </div>

                <div className="legend">
                  {anomalyChartData.map((item, index) => (
                    <span key={item.name}>
                      <i
                        style={{
                          background:
                            PIE_COLORS[index % PIE_COLORS.length],
                        }}
                      />
                      {item.name}: {item.value.toLocaleString()}
                    </span>
                  ))}
                </div>
              </>
            ) : (
              <div className="empty-state">
                Anomaly and non-anomaly counts will appear here after
                analysis.
              </div>
            )}
          </div>
        </section>

        <section className="threats-section">
          <div className="panel threats-panel" id="threats">
            <div className="panel-heading">
              <div>
                <h3>Predicted Attack Categories</h3>
                <p>
                  {results
                    ? "Predictions returned by the detector"
                    : "Upload a dataset to show predictions"}
                </p>
              </div>
              <AlertTriangle size={19} />
            </div>

            {results ? (
              attackClasses.length > 0 ? (
                <div className="threat-list">
                  {attackClasses.map(({ name, count }) => (
                    <div className="threat-row" key={name}>
                      <div className="threat-symbol">
                        <AlertTriangle size={17} />
                      </div>
                      <div className="threat-name">
                        <strong>{name}</strong>
                        <small>Predicted flows</small>
                      </div>
                      <strong className="threat-count">
                        {count.toLocaleString()}
                      </strong>
                      <span className="severity high">Predicted</span>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="empty-state">
                  No non-benign predicted classes were returned.
                </div>
              )
            ) : (
              <div className="empty-state">
                No analysis yet. Upload a compatible flow CSV or a PCAP capture.
              </div>
            )}
          </div>
        </section>

        {results && (
          <section className="panel results-panel" id="results">
            <div className="panel-heading results-heading">
              <div>
                <h3>Detection Results</h3>
                <p>
                  {filteredRows.length.toLocaleString()} matching preview
                  rows from {totalFlows.toLocaleString()} analyzed flows
                </p>
              </div>
              <Server size={19} />
            </div>

            <div className="table-controls">
              <input
                type="search"
                value={search}
                onChange={(event) => setSearch(event.target.value)}
                placeholder="Search preview..."
                aria-label="Search preview results"
              />
              <select
                value={filter}
                onChange={(event) => setFilter(event.target.value)}
                aria-label="Filter detection results"
              >
                <option value="all">All preview flows</option>
                <option value="anomaly">Anomalies only</option>
                <option value="behavioral">Behavioral alerts</option>
                <option value="attack">Non-benign predictions</option>
              </select>
            </div>

            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    {resultColumns.map((column) => (
                      <th key={column}>
                        {column.replaceAll("_", " ")}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {filteredRows.map((row, index) => (
                    <tr key={index}>
                      {resultColumns.map((column) => (
                        <td key={column}>
                          {column === "is_anomaly" ? (
                            <span
                              className={`table-status ${
                                isTrue(row[column]) ? "danger" : "safe"
                              }`}
                            >
                              {isTrue(row[column])
                                ? "Anomaly"
                                : "Not flagged"}
                            </span>
                          ) : (
                            formatValue(row[column])
                          )}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <p className="table-note">
              This table contains up to 500 preview rows. Search and filters
              apply to the preview only. Download the full CSV for every
              analyzed flow.
            </p>

            {filteredRows.length === 0 && (
              <div className="empty-state">
                No rows match your current search or filter.
              </div>
            )}
          </section>
        )}
      </main>
    </div>
  );
}
