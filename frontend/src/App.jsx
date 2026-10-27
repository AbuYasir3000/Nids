
import { useMemo, useState } from "react";
import {
  ShieldCheck,
  Activity,
  Network,
  AlertTriangle,
  Upload,
  LayoutDashboard,
  FileWarning,
  CheckCircle2,
  Zap,
  RefreshCw,
  Server,
  Download,
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

const API_URL = "http://127.0.0.1:8000";
const PIE_COLORS = ["#4de0b2", "#ff7e8a"];
const BAR_COLOR = "#4de0b2";
const EMPTY_CLASS_COUNTS = Object.freeze({});
const EMPTY_PREVIEW_ROWS = Object.freeze([]);

function StatCard({ icon: Icon, title, value, subtitle, color }) {
  return (
    <div className="stat-card">
      <div className="stat-top">
        <span>{title}</span>
        <span className="stat-icon" style={{ color }}>
          <Icon size={20} />
        </span>
      </div>
      <div className="stat-value">{value}</div>
      <div className="stat-subtitle">{subtitle}</div>
    </div>
  );
}

function formatValue(value) {
  if (value === null || value === undefined) return "â€”";
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (typeof value === "number") {
    return Number.isFinite(value)
      ? Number(value.toFixed(5)).toString()
      : "â€”";
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
  const [apiStatus, setApiStatus] = useState("Checking");
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState("all");

  async function checkAPI() {
    setApiStatus("Checking");

    try {
      const response = await fetch(`${API_URL}/health`);
      if (!response.ok) throw new Error("API unavailable");

      const data = await response.json();
      setApiStatus(
        !data.detector_ready
          ? "Models unavailable"
          : data.generalized_model_ready
            ? "Online"
            : "Generalization model missing"
      );
    } catch {
      setApiStatus("Offline");
    }
  }

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
        ? `${item.name.slice(0, 17)}â€¦`
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

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-logo">
            <ShieldCheck size={25} />
          </div>
          <div>
            <h2>
              Sentinel<span>AI</span>
            </h2>
            <p>NETWORK DEFENSE</p>
          </div>
        </div>

        <div className="nav-label">WORKSPACE</div>
        <a className="nav-item active" href="#dashboard">
          <LayoutDashboard size={18} /> Dashboard
        </a>
        <a className="nav-item" href="#traffic">
          <Network size={18} /> Prediction Charts
        </a>
        <a className="nav-item" href="#threats">
          <FileWarning size={18} /> Threat Detection
        </a>
        <a className="nav-item" href="#analyze">
          <Upload size={18} /> Analyze Dataset
        </a>

        <div className="sidebar-bottom">
          <div className="system-status">
            <span
              className={`status-dot ${
                apiStatus === "Online" ? "" : "status-warning"
              }`}
            />
            <div>
              <strong>
                {apiStatus === "Online" ? "Models Online" : apiStatus}
              </strong>
              <small>Python detection service</small>
            </div>
          </div>
          <p className="version">HYBRID NIDS Â· DEVELOPMENT</p>
        </div>
      </aside>

      <main className="main-content" id="dashboard">
        <header className="topbar">
          <div>
            <p className="eyebrow">SECURITY OPERATIONS CENTER</p>
            <h1>Network Overview</h1>
          </div>
          <button className="api-button" onClick={checkAPI}>
            <RefreshCw size={15} />
            <span>API: {apiStatus}</span>
          </button>
        </header>

        <section className="welcome-panel">
          <div>
            <div className="welcome-tag">
              <Zap size={14} /> AI-POWERED SECURITY
            </div>
            <h2>
              Network Intelligence, <span>Reimagined.</span>
            </h2>
            <p>
              Analyze network flows with your trained Random Forest and
              Isolation Forest models.
            </p>
          </div>
          <div className="welcome-art">
            <ShieldCheck size={92} strokeWidth={1} />
          </div>
        </section>

        <section className="stats-grid">
          <StatCard
            icon={Activity}
            title="Analyzed Flows"
            value={results ? totalFlows.toLocaleString() : "â€”"}
            subtitle={results ? results.filename : "Upload a CSV to begin"}
            color="#66e3bf"
          />
          <StatCard
            icon={AlertTriangle}
            title="Non-Benign Predictions"
            value={
              results ? predictedNonBenign.toLocaleString() : "â€”"
            }
            subtitle="Based on predicted classes"
            color="#ffb86b"
          />
          <StatCard
            icon={ShieldCheck}
            title="Learned Binary Attack Flags"
            value={
              results ? generalizedAttackCount.toLocaleString() : "—"
            }
            subtitle={
              results
                ? "Calibrated classifier only; behavior rules are separate"
                : "Learned model predictions only"
            }
            color="#a99bff"
          />
          <StatCard
            icon={CheckCircle2}
            title="Anomalies"
            value={results ? anomalyCount.toLocaleString() : "â€”"}
            subtitle={
              results && totalFlows
                ? `${((anomalyCount / totalFlows) * 100).toFixed(2)}% flagged`
                : "Awaiting analysis"
            }
            color="#65b9ff"
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
          <section className="panel behavioral-summary">
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
              <p key={event.event_id}>
                {event.classification}: {event.source_ip}, {event.window_start} to {event.last_observed_at}; {Number(event.flow_count).toLocaleString()} associated flows, {Number(event.unique_destination_ports).toLocaleString()} destination ports across {Number(event.unique_destination_ips).toLocaleString()} destination IPs. Top destination: {event.top_destination_ips?.[0]?.destination_ip ?? "unavailable"}. Reason: {event.reasons?.join("; ")}.
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
                        stroke="#233345"
                        strokeDasharray="4 4"
                        vertical={false}
                      />
                      <XAxis
                        dataKey="name"
                        stroke="#71839a"
                        tickLine={false}
                        axisLine={false}
                        interval={0}
                        angle={-25}
                        textAnchor="end"
                        height={60}
                        tick={{ fontSize: 10 }}
                      />
                      <YAxis
                        stroke="#71839a"
                        tickLine={false}
                        axisLine={false}
                        tick={{ fontSize: 10 }}
                      />
                      <Tooltip
                        contentStyle={{
                          background: "#101c2a",
                          border: "1px solid #2a3b50",
                          borderRadius: 8,
                        }}
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
                        fill={BAR_COLOR}
                        radius={[5, 5, 0, 0]}
                      />
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
                        innerRadius={62}
                        outerRadius={88}
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
                        contentStyle={{
                          background: "#101c2a",
                          border: "1px solid #2a3b50",
                          borderRadius: 8,
                        }}
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

        <section className="bottom-grid">
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

          <div className="panel upload-panel" id="analyze">
            <div className="panel-heading">
              <div>
                <h3>Analyze a Dataset</h3>
                <p>Run your trained Python models</p>
              </div>
              <Upload size={19} />
            </div>

            <label className="drop-zone">
              <Upload size={27} />
              <strong>{file ? file.name : "Choose a CSV or PCAP file"}</strong>
              <span>
                {file
                  ? `${(file.size / 1024 / 1024).toFixed(2)} MB`
                  : "CSV, PCAP, or PCAPNG traffic capture"}
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
                <RefreshCw className="spin" size={17} />
              ) : (
                <Activity size={17} />
              )}
              {loading ? "Analyzing traffic..." : "Analyze Network Traffic"}
            </button>

            {error && (
              <div className="error-message">
                <strong>Analysis failed</strong>
                <p>{error}</p>
              </div>
            )}

            {results && (
              <div className="result-message">
                <div className="result-success">
                  <CheckCircle2 size={18} />
                  <strong>Analysis completed</strong>
                </div>
                <p>File: {results.filename}</p>
                <p>Total flows: {totalFlows.toLocaleString()}</p>
                <p>Anomalies: {anomalyCount.toLocaleString()}</p>
                <p>
                  Table preview: {results.preview_rows ?? previewRows.length} rows
                </p>
                <button
                  className="download-button"
                  onClick={downloadFullCSV}
                  disabled={!results.download_url}
                >
                  <Download size={15} /> Download Full Results CSV
                </button>
              </div>
            )}

            <p className="upload-note">
              Maximum 20 GiB per upload with no fixed flow-count limit. PCAP
              conversion requires CICFlowMeter on the API server.
            </p>
          </div>
        </section>

        {results && (
          <section className="panel results-panel">
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

        <footer>
          SentinelAI Â· Hybrid Network Intrusion Detection System Â· Student
          project
        </footer>
      </main>
    </div>
  );
}
