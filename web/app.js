const DEFAULT_CSV = "/api/evaluation-results/default";

const performanceMetrics = [
  { key: "normalized_score_mean", label: "Normalized Score", better: "high", format: "score" },
  { key: "success_rate", label: "Success Rate", better: "high", format: "percent" },
  { key: "raw_reward_mean", label: "Raw Reward", better: "high", format: "score" },
  { key: "length_mean", label: "Episode Length", better: "low", format: "score" },
];

const humanMetrics = [
  { key: "dtw_action_dtw_mean_agg_mean", label: "Action DTW", better: "low", format: "score" },
  { key: "dtw_state_dtw_mean_agg_mean", label: "State DTW", better: "low", format: "score" },
  { key: "min_dtw_action_dtw_min_agg_mean", label: "Min Action DTW", better: "low", format: "score" },
  { key: "min_dtw_state_dtw_min_agg_mean", label: "Min State DTW", better: "low", format: "score" },
  { key: "wasserstein_action_w2_dist_agg_mean", label: "Action Wasserstein", better: "low", format: "score" },
  { key: "wasserstein_state_w2_dist_agg_mean", label: "State Wasserstein", better: "low", format: "score" },
  { key: "occupancy_state_action_mmd_agg_mean", label: "Occupancy MMD", better: "low", format: "small" },
  { key: "traj_qd_action_frechet_knn_quality_agg_mean", label: "Action Frechet Quality", better: "low", format: "score" },
  { key: "traj_qd_state_frechet_knn_quality_agg_mean", label: "State Frechet Quality", better: "low", format: "score" },
];

const palette = ["#0f766e", "#b42318", "#b7791f", "#6d28d9", "#2f855a", "#be185d", "#2563eb", "#854d0e"];
const metadataColumns = new Set(["agent_type", "env_id", "model_path", "base_seed", "eval_episodes", "horizon", "default_horizon"]);
const metricPreferenceKey = "maq-dashboard-selected-metrics-v1";
const rankingMetricPreferenceKey = "maq-dashboard-human-ranking-metrics-v1";
const humanBaselinePreferenceKey = "maq-dashboard-use-human-baseline-v1";
const sequenceFilterPreferenceKey = "maq-dashboard-sequence-filter-v1";
const codebookFilterPreferenceKey = "maq-dashboard-codebook-filter-v1";
const coreMetricKeys = [
  "normalized_score_mean",
  "success_rate",
  "raw_reward_mean",
  "length_mean",
  "dtw_action_dtw_mean_agg_mean",
  "dtw_state_dtw_mean_agg_mean",
  "wasserstein_action_w2_dist_agg_mean",
  "wasserstein_state_w2_dist_agg_mean",
  "occupancy_state_action_mmd_agg_mean",
];

let state = {
  rows: [],
  sourceName: "",
  runs: [],
  groups: [],
  filteredGroups: [],
  humanBaselinesBySeed: new Map(),
  sequenceValues: [],
  codebookValues: [],
  selectedSequences: [],
  selectedCodebooks: [],
  metricColumns: [],
  selectedMetricColumns: [],
  rankingMetricColumns: [],
  metricSearch: "",
  rankingMetricSearch: "",
  performanceMetric: "normalized_score_mean",
  humanMetric: "dtw_action_dtw_mean_agg_mean",
  sortMetric: "normalized_score_mean",
  sortDirection: "desc",
  rankingSortKey: "default",
  rankingSortDirection: "asc",
  methodFilter: "all",
  humanSweepAxis: "sequence",
  showRuns: false,
  useHumanBaseline: loadBooleanPreference(humanBaselinePreferenceKey),
};

const els = {
  status: document.getElementById("status"),
  csvFile: document.getElementById("csvFile"),
  reloadDefault: document.getElementById("reloadDefault"),
  performanceMetric: document.getElementById("performanceMetric"),
  humanMetric: document.getElementById("humanMetric"),
  sortMetric: document.getElementById("sortMetric"),
  methodFilter: document.getElementById("methodFilter"),
  humanSweepAxis: document.getElementById("humanSweepAxis"),
  showRuns: document.getElementById("showRuns"),
  useHumanBaseline: document.getElementById("useHumanBaseline"),
  sequenceFilterOptions: document.getElementById("sequenceFilterOptions"),
  codebookFilterOptions: document.getElementById("codebookFilterOptions"),
  metricPicker: document.getElementById("metricPicker"),
  toggleMetricPicker: document.getElementById("toggleMetricPicker"),
  metricSearch: document.getElementById("metricSearch"),
  metricChecklist: document.getElementById("metricChecklist"),
  selectCoreMetrics: document.getElementById("selectCoreMetrics"),
  selectAllMetrics: document.getElementById("selectAllMetrics"),
  clearMetrics: document.getElementById("clearMetrics"),
  rankingMetricPicker: document.getElementById("rankingMetricPicker"),
  toggleRankingMetricPicker: document.getElementById("toggleRankingMetricPicker"),
  rankingMetricSearch: document.getElementById("rankingMetricSearch"),
  rankingMetricChecklist: document.getElementById("rankingMetricChecklist"),
  selectHumanRankingMetrics: document.getElementById("selectHumanRankingMetrics"),
  selectAllRankingMetrics: document.getElementById("selectAllRankingMetrics"),
  clearRankingMetrics: document.getElementById("clearRankingMetrics"),
  rankingCaption: document.getElementById("rankingCaption"),
  humanRanking: document.getElementById("humanRanking"),
  metricGuide: document.getElementById("metricGuide"),
  metricGuideSummary: document.getElementById("metricGuideSummary"),
  performanceCaption: document.getElementById("performanceCaption"),
  tradeoffCaption: document.getElementById("tradeoffCaption"),
  humanChartCaption: document.getElementById("humanChartCaption"),
  chartTooltip: document.getElementById("chartTooltip"),
  summary: document.getElementById("summary"),
  performanceChart: document.getElementById("performanceChart"),
  tradeoffChart: document.getElementById("tradeoffChart"),
  humanChart: document.getElementById("humanChart"),
  resultsHead: document.getElementById("resultsHead"),
  resultsBody: document.getElementById("resultsBody"),
  downloadSummary: document.getElementById("downloadSummary"),
  tableCaption: document.getElementById("tableCaption"),
};

function init() {
  fillMetricSelect(els.performanceMetric, performanceMetrics);
  fillMetricSelect(els.humanMetric, humanMetrics);
  fillMetricSelect(els.sortMetric, [...performanceMetrics, ...humanMetrics]);

  els.performanceMetric.value = state.performanceMetric;
  els.humanMetric.value = state.humanMetric;
  els.sortMetric.value = state.sortMetric;
  els.useHumanBaseline.checked = state.useHumanBaseline;

  els.performanceMetric.addEventListener("change", () => {
    state.performanceMetric = els.performanceMetric.value;
    render();
  });
  els.humanMetric.addEventListener("change", () => {
    state.humanMetric = els.humanMetric.value;
    render();
  });
  els.sortMetric.addEventListener("change", () => {
    state.sortMetric = els.sortMetric.value;
    state.sortDirection = defaultSortDirection(getMetric(state.sortMetric));
    render();
  });
  els.methodFilter.addEventListener("change", () => {
    state.methodFilter = els.methodFilter.value;
    render();
  });
  els.humanSweepAxis.addEventListener("change", () => {
    state.humanSweepAxis = els.humanSweepAxis.value;
    renderHumanChart();
  });
  els.showRuns.addEventListener("change", () => {
    state.showRuns = els.showRuns.checked;
    renderTable();
  });
  els.useHumanBaseline.addEventListener("change", () => {
    setHumanBaselineMode(els.useHumanBaseline.checked);
  });
  els.toggleMetricPicker.addEventListener("click", () => {
    els.metricPicker.hidden = !els.metricPicker.hidden;
    els.toggleMetricPicker.textContent = els.metricPicker.hidden ? "Choose metrics" : "Hide metrics";
  });
  els.metricSearch.addEventListener("input", () => {
    state.metricSearch = els.metricSearch.value.trim().toLowerCase();
    renderMetricChecklist();
  });
  els.selectCoreMetrics.addEventListener("click", () => {
    setSelectedMetrics(coreMetricKeys.filter((key) => state.metricColumns.includes(key)));
  });
  els.selectAllMetrics.addEventListener("click", () => {
    setSelectedMetrics(state.metricColumns);
  });
  els.clearMetrics.addEventListener("click", () => {
    setSelectedMetrics([]);
  });
  els.toggleRankingMetricPicker.addEventListener("click", () => {
    els.rankingMetricPicker.hidden = !els.rankingMetricPicker.hidden;
    els.toggleRankingMetricPicker.textContent = els.rankingMetricPicker.hidden ? "Choose ranking metrics" : "Hide ranking metrics";
  });
  els.rankingMetricSearch.addEventListener("input", () => {
    state.rankingMetricSearch = els.rankingMetricSearch.value.trim().toLowerCase();
    renderRankingMetricChecklist();
  });
  els.selectHumanRankingMetrics.addEventListener("click", () => {
    setRankingMetrics(defaultRankingMetricKeys(state.metricColumns));
  });
  els.selectAllRankingMetrics.addEventListener("click", () => {
    setRankingMetrics(state.metricColumns);
  });
  els.clearRankingMetrics.addEventListener("click", () => {
    setRankingMetrics([]);
  });
  els.csvFile.addEventListener("change", loadUploadedFile);
  els.reloadDefault.addEventListener("click", loadDefaultCsv);
  els.downloadSummary.addEventListener("click", downloadSummaryCsv);

  loadDefaultCsv();
}

function loadBooleanPreference(key) {
  try {
    return localStorage.getItem(key) === "true";
  } catch {
    return false;
  }
}

function setHumanBaselineMode(enabled) {
  state.useHumanBaseline = enabled;
  localStorage.setItem(humanBaselinePreferenceKey, String(enabled));
  if (state.sortMetric !== "runs") {
    state.sortDirection = enabled ? "asc" : defaultSortDirection(getMetric(state.sortMetric));
  }
  state.groups = aggregateRuns(state.runs);
  updateSortMetricSelect();
  render();
}

function fillMetricSelect(select, metrics) {
  select.innerHTML = metrics
    .map((metric) => {
      const help = metricHelp(metric.key);
      return `<option value="${metric.key}">${escapeHtml(metric.label)} - ${escapeHtml(help.shortDirection)}</option>`;
    })
    .join("");
}

async function loadDefaultCsv() {
  setStatus("Loading evaluation CSV...");
  try {
    const response = await fetch(DEFAULT_CSV, { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const text = await response.text();
    loadCsvText(text, DEFAULT_CSV);
  } catch (error) {
    setStatus(
      `Could not fetch ${DEFAULT_CSV}. Start the Flask service or use Load CSV.`,
      true,
    );
  }
}

function loadUploadedFile(event) {
  const file = event.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => loadCsvText(String(reader.result), file.name);
  reader.onerror = () => setStatus(`Could not read ${file.name}`, true);
  reader.readAsText(file);
}

function loadCsvText(text, sourceName) {
  const rows = parseCsv(text);
  state.sourceName = sourceName;
  state.rows = rows;
  state.metricColumns = detectMetricColumns(rows);
  state.selectedMetricColumns = loadSelectedMetricPreference(state.metricColumns);
  state.rankingMetricColumns = loadRankingMetricPreference(state.metricColumns);
  if (!state.metricColumns.includes(state.sortMetric)) {
    state.sortMetric = firstAvailableMetric(["normalized_score_mean", "success_rate"]);
    state.sortDirection = state.sortMetric ? defaultSortDirection(getMetric(state.sortMetric)) : "desc";
  }
  updateHumanMetricSelect();
  if (!state.metricColumns.includes(state.humanMetric)) {
    state.humanMetric = firstAvailableMetric(["dtw_action_dtw_mean_agg_mean", "dtw_state_dtw_mean_agg_mean"]);
  }
  els.humanMetric.value = state.humanMetric;
  state.runs = rows.map(normalizeRun).filter(Boolean);
  state.humanBaselinesBySeed = buildHumanBaselinesBySeed(state.runs);
  state.sequenceValues = uniqueSortedNumbers(state.runs.map((run) => run.sequenceLength));
  state.codebookValues = uniqueSortedNumbers(state.runs.map((run) => run.codebookSize));
  state.selectedSequences = loadNumberFilterPreference(sequenceFilterPreferenceKey, state.sequenceValues);
  state.selectedCodebooks = loadNumberFilterPreference(codebookFilterPreferenceKey, state.codebookValues);
  state.groups = aggregateRuns(state.runs);
  state.methodFilter = "all";
  updateSortMetricSelect();
  updateMethodFilter();
  renderMacroFilters();
  renderMetricChecklist();
  renderRankingMetricChecklist();
  renderMetricGuide();
  setStatus(statusText(sourceName));
  render();
}

function statusText(sourceName) {
  const baselineCount = state.humanBaselinesBySeed.size;
  const baselineText = baselineCount
    ? `${baselineCount} human-human seed baseline${baselineCount === 1 ? "" : "s"}`
    : "no human-human seed baselines";
  const modeText = state.useHumanBaseline ? "Human-human distance mode on." : "Raw metric mode.";
  const shown = getFilteredGroups().length;
  return `${sourceName}: ${state.runs.length} runs, ${shown}/${state.groups.length} grouped configurations shown, ${state.metricColumns.length} numeric metrics, ${baselineText}. ${modeText}`;
}

function loadSelectedMetricPreference(availableMetrics) {
  let stored = [];
  try {
    stored = JSON.parse(localStorage.getItem(metricPreferenceKey) || "[]");
  } catch {
    stored = [];
  }
  const selected = stored.filter((key) => availableMetrics.includes(key));
  if (selected.length) return selected;
  return coreMetricKeys.filter((key) => availableMetrics.includes(key));
}

function loadRankingMetricPreference(availableMetrics) {
  let stored = [];
  try {
    stored = JSON.parse(localStorage.getItem(rankingMetricPreferenceKey) || "[]");
  } catch {
    stored = [];
  }
  const selected = stored.filter((key) => availableMetrics.includes(key));
  if (selected.length) return selected;
  return defaultRankingMetricKeys(availableMetrics);
}

function defaultRankingMetricKeys(availableMetrics) {
  return availableMetrics.filter((key) => isHumanSimilarityMetric(String(key).toLowerCase()));
}

function loadNumberFilterPreference(key, availableValues) {
  let stored = [];
  try {
    stored = JSON.parse(localStorage.getItem(key) || "[]");
  } catch {
    stored = [];
  }
  const availableSet = new Set(availableValues.map(String));
  const selected = stored.map(Number).filter((value) => availableSet.has(String(value)));
  return selected.length ? selected : [...availableValues];
}

function setNumberFilterPreference(key, values) {
  localStorage.setItem(key, JSON.stringify(values));
}

function uniqueSortedNumbers(values) {
  return [...new Set(values.filter(isFiniteNumber))].sort((a, b) => a - b);
}

function setSelectedMetrics(metrics) {
  const selectedSet = new Set(metrics);
  state.selectedMetricColumns = state.metricColumns.filter((key) => selectedSet.has(key));
  localStorage.setItem(metricPreferenceKey, JSON.stringify(state.selectedMetricColumns));
  renderMetricChecklist();
  render();
}

function firstAvailableMetric(preferredKeys) {
  return preferredKeys.find((key) => state.metricColumns.includes(key)) || state.metricColumns[0] || "";
}

function toggleMetricSelection(key, checked) {
  const next = new Set(state.selectedMetricColumns);
  if (checked) next.add(key);
  else next.delete(key);
  setSelectedMetrics([...next]);
}

function setRankingMetrics(metrics) {
  const selectedSet = new Set(metrics);
  state.rankingMetricColumns = state.metricColumns.filter((key) => selectedSet.has(key));
  localStorage.setItem(rankingMetricPreferenceKey, JSON.stringify(state.rankingMetricColumns));
  renderRankingMetricChecklist();
  renderHumanRanking();
}

function toggleRankingMetricSelection(key, checked) {
  const next = new Set(state.rankingMetricColumns);
  if (checked) next.add(key);
  else next.delete(key);
  setRankingMetrics([...next]);
}

function renderMetricChecklist() {
  if (!els.metricChecklist) return;
  const query = state.metricSearch;
  const selected = new Set(state.selectedMetricColumns);
  const metrics = state.metricColumns.filter((key) => {
    if (!query) return true;
    const label = prettifyMetricLabel(key).toLowerCase();
    return key.toLowerCase().includes(query) || label.includes(query);
  });

  els.metricChecklist.innerHTML = metrics
    .map((key) => {
      const metric = getMetric(key);
      const help = metricHelp(key);
      return `<label class="metric-option">
        <input type="checkbox" value="${escapeHtml(key)}" ${selected.has(key) ? "checked" : ""} />
        <span>${escapeHtml(metric.label)}<small>${escapeHtml(help.shortDirection)}. ${escapeHtml(help.description)} (${escapeHtml(key)})</small></span>
      </label>`;
    })
    .join("");

  els.metricChecklist.querySelectorAll("input[type='checkbox']").forEach((input) => {
    input.addEventListener("change", () => toggleMetricSelection(input.value, input.checked));
  });
}

function renderRankingMetricChecklist() {
  if (!els.rankingMetricChecklist) return;
  const query = state.rankingMetricSearch;
  const selected = new Set(state.rankingMetricColumns);
  const metrics = state.metricColumns.filter((key) => {
    if (!query) return true;
    const label = prettifyMetricLabel(key).toLowerCase();
    return key.toLowerCase().includes(query) || label.includes(query);
  });

  els.rankingMetricChecklist.innerHTML = metrics
    .map((key) => {
      const metric = getMetric(key);
      const help = metricHelp(key);
      return `<label class="metric-option">
        <input type="checkbox" value="${escapeHtml(key)}" ${selected.has(key) ? "checked" : ""} />
        <span>${escapeHtml(metric.label)}<small>${escapeHtml(help.shortDirection)}. ${escapeHtml(help.description)} (${escapeHtml(key)})</small></span>
      </label>`;
    })
    .join("");

  els.rankingMetricChecklist.querySelectorAll("input[type='checkbox']").forEach((input) => {
    input.addEventListener("change", () => toggleRankingMetricSelection(input.value, input.checked));
  });
}

function renderMacroFilters() {
  renderNumberFilter(
    els.sequenceFilterOptions,
    state.sequenceValues,
    state.selectedSequences,
    "sq",
    (next) => {
      state.selectedSequences = next;
      setNumberFilterPreference(sequenceFilterPreferenceKey, next);
      render();
      renderMacroFilters();
    },
  );
  renderNumberFilter(
    els.codebookFilterOptions,
    state.codebookValues,
    state.selectedCodebooks,
    "k",
    (next) => {
      state.selectedCodebooks = next;
      setNumberFilterPreference(codebookFilterPreferenceKey, next);
      render();
      renderMacroFilters();
    },
  );
}

function renderNumberFilter(container, values, selectedValues, prefix, onChange) {
  if (!container) return;
  const selected = new Set(selectedValues.map(String));
  if (!values.length) {
    container.innerHTML = `<span class="filter-empty">No ${escapeHtml(prefix)} values</span>`;
    return;
  }
  container.innerHTML = values
    .map((value) => `<label class="filter-option">
      <input type="checkbox" value="${value}" ${selected.has(String(value)) ? "checked" : ""} />
      Show ${escapeHtml(prefix)}=${escapeHtml(value)}
    </label>`)
    .join("");
  container.querySelectorAll("input[type='checkbox']").forEach((input) => {
    input.addEventListener("change", () => {
      const checked = [...container.querySelectorAll("input[type='checkbox']:checked")]
        .map((checkbox) => checkbox.value);
      const checkedSet = new Set(checked);
      onChange(values.filter((value) => checkedSet.has(String(value))));
    });
  });
}

function renderMetricGuide() {
  if (!els.metricGuide) return;
  const categories = [
    ["Task performance", (key) => metricCategory(key) === "performance"],
    ["Human similarity distances", (key) => metricCategory(key) === "human"],
    ["Episode length and counts", (key) => metricCategory(key) === "episode"],
    ["Other CSV metrics", (key) => metricCategory(key) === "other"],
  ];

  const sections = categories
    .map(([title, predicate]) => {
      const keys = state.metricColumns.filter(predicate);
      if (!keys.length) return "";
      const rows = keys
        .map((key) => {
          const metric = getMetric(key);
          const help = metricHelp(key);
          return `<article class="metric-guide-item">
            <div>
              <strong>${escapeHtml(metric.label)}</strong>
              <code>${escapeHtml(key)}</code>
            </div>
            <p>${escapeHtml(help.description)}</p>
            <span class="metric-direction ${escapeHtml(help.tone)}">${escapeHtml(help.direction)}</span>
          </article>`;
        })
        .join("");
      return `<section class="metric-guide-section">
        <h3>${escapeHtml(title)}</h3>
        <div class="metric-guide-list">${rows}</div>
      </section>`;
    })
    .join("");

  els.metricGuide.innerHTML = sections || `<div class="empty-state">No numeric metrics found.</div>`;
  if (els.metricGuideSummary) {
    els.metricGuideSummary.textContent = `${state.metricColumns.length} selectable mean/non-std metrics`;
  }
}

function detectMetricColumns(rows) {
  if (!rows.length) return [];
  const columns = Object.keys(rows[0]);
  const numericColumns = columns.filter((column) => {
    if (metadataColumns.has(column)) return false;
    if (isStdMetricColumn(column)) return false;
    return rows.some((row) => isFiniteNumber(numberOrNull(row[column])));
  });

  const preferred = [
    "normalized_score_mean",
    "success_rate",
    "raw_reward_mean",
    "length_mean",
    "dtw_action_dtw_mean_agg_mean",
    "dtw_state_dtw_mean_agg_mean",
    "wasserstein_action_w2_dist_agg_mean",
    "wasserstein_state_w2_dist_agg_mean",
    "occupancy_state_action_mmd_agg_mean",
  ].filter((column) => numericColumns.includes(column));

  const remaining = numericColumns
    .filter((column) => !preferred.includes(column))
    .sort((a, b) => prettifyMetricLabel(a).localeCompare(prettifyMetricLabel(b)));
  return [...preferred, ...remaining];
}

function isStdMetricColumn(column) {
  return /(^|_)std($|_)/i.test(column) || /_agg_std$/i.test(column);
}

function parseCsv(text) {
  const rows = [];
  let row = [];
  let value = "";
  let inQuotes = false;

  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    const next = text[i + 1];
    if (char === '"') {
      if (inQuotes && next === '"') {
        value += '"';
        i += 1;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (char === "," && !inQuotes) {
      row.push(value);
      value = "";
    } else if ((char === "\n" || char === "\r") && !inQuotes) {
      if (char === "\r" && next === "\n") i += 1;
      row.push(value);
      if (row.some((cell) => cell !== "")) rows.push(row);
      row = [];
      value = "";
    } else {
      value += char;
    }
  }

  if (value || row.length) {
    row.push(value);
    rows.push(row);
  }

  if (rows.length < 2) return [];
  const header = rows[0].map((h) => h.trim());
  return rows.slice(1).map((cells) => {
    const out = {};
    header.forEach((key, index) => {
      out[key] = cells[index] ?? "";
    });
    return out;
  });
}

function normalizeRun(row) {
  const agentType = row.agent_type || "UnknownAgent";
  const parsed = parseAgent(agentType, row.model_path || "");
  const run = {
    raw: row,
    agentType,
    methodRaw: parsed.methodRaw,
    method: displayMethod(parsed.methodRaw),
    seed: parsed.seed ?? numberOrNull(row.base_seed),
    sequenceLength: parsed.sequenceLength,
    codebookSize: parsed.codebookSize,
    env: row.env_id || "",
    modelPath: row.model_path || "",
    isHumanHuman: parsed.isHumanHuman || false,
    isMacro: parsed.sequenceLength != null || parsed.codebookSize != null,
    metrics: {},
  };

  state.metricColumns.forEach((key) => {
    run.metrics[key] = numberOrNull(row[key]);
  });

  run.configKey = run.isMacro
    ? `${run.method}|sq=${run.sequenceLength ?? "?"}|k=${run.codebookSize ?? "?"}`
    : `${run.method}|baseline`;
  run.groupLabel = run.isMacro
    ? `${run.method} sq=${run.sequenceLength ?? "?"} k=${run.codebookSize ?? "?"}`
    : run.method;
  run.runLabel = run.seed != null ? `${run.groupLabel} seed=${run.seed}` : run.groupLabel;
  return run;
}

function parseAgent(agentType, modelPath) {
  const source = `${agentType} ${modelPath}`;
  const humanHumanMatch = agentType.match(/^human_human_ratio([^_]+)_seed(\d+)/i);
  if (humanHumanMatch) {
    return {
      methodRaw: `human_human_ratio${humanHumanMatch[1]}`,
      seed: Number(humanHumanMatch[2]),
      sequenceLength: null,
      codebookSize: null,
      isHumanHuman: true,
    };
  }
  const methodMatch = agentType.match(/^(.+?Agent)/);
  const methodRaw = methodMatch ? methodMatch[1] : agentType.replace(/seed\d+.*/, "");
  const seedMatch = source.match(/seed(\d+)/i);
  const sqMatch = source.match(/(?:_sq|sq|SEQ)(\d+)/i);
  const kMatch = source.match(/(?:_k|[^a-z]k)(\d+)/i);

  return {
    methodRaw,
    seed: seedMatch ? Number(seedMatch[1]) : null,
    sequenceLength: sqMatch ? Number(sqMatch[1]) : null,
    codebookSize: kMatch ? Number(kMatch[1]) : null,
    isHumanHuman: false,
  };
}

function displayMethod(methodRaw) {
  const map = {
    RLPDMAQAgent: "MAQ+RLPD",
    DSACMAQAgent: "MAQ+DSAC",
    MAQIQLAgent: "MAQ+IQL",
    SB3Agent: "SAC",
    IQLAgent: "IQL",
    RLPDAgent: "RLPD",
    RndAgent: "Random",
  };
  const humanHumanMatch = methodRaw.match(/^human_human_ratio(.+)$/i);
  if (humanHumanMatch) return `Human-Human ratio=${humanHumanMatch[1]}`;
  return map[methodRaw] || methodRaw.replace(/Agent$/, "");
}

function aggregateRuns(runs) {
  const byKey = new Map();
  runs.forEach((run) => {
    if (!byKey.has(run.configKey)) {
      byKey.set(run.configKey, {
        key: run.configKey,
        label: run.groupLabel,
        method: run.method,
        methodRaw: run.methodRaw,
        sequenceLength: run.sequenceLength,
        codebookSize: run.codebookSize,
        isMacro: run.isMacro,
        isHumanHuman: run.isHumanHuman,
        runs: [],
        metrics: {},
      });
    }
    byKey.get(run.configKey).runs.push(run);
  });

  const groups = [...byKey.values()];
  groups.forEach((group, index) => {
    group.color = palette[index % palette.length];
    state.metricColumns.forEach((key) => {
      const values = group.runs.map((run) => getRunMetric(run, key)).filter(isFiniteNumber);
      group.metrics[key] = {
        mean: values.length ? mean(values) : null,
        std: values.length > 1 ? std(values) : 0,
        n: values.length,
      };
    });
  });
  return groups;
}

function buildHumanBaselinesBySeed(runs) {
  const baselines = new Map();
  runs.forEach((run) => {
    if (run.isHumanHuman && run.seed != null && !baselines.has(String(run.seed))) {
      baselines.set(String(run.seed), run);
    }
  });
  return baselines;
}

function updateMethodFilter() {
  const methods = [...new Set(state.groups.map((group) => group.method))].sort();
  els.methodFilter.innerHTML = [
    `<option value="all">All methods</option>`,
    ...methods.map((method) => `<option value="${escapeHtml(method)}">${escapeHtml(method)}</option>`),
  ].join("");
  els.methodFilter.value = state.methodFilter;
}

function updateSortMetricSelect() {
  const metrics = state.metricColumns.map((key) => getMetric(key));
  fillMetricSelect(els.sortMetric, metrics);
  els.sortMetric.value = state.sortMetric;
  els.sortMetric.disabled = !metrics.length;
}

function updateHumanMetricSelect() {
  const preferred = humanMetrics
    .map((metric) => metric.key)
    .filter((key) => state.metricColumns.includes(key));
  const orderedKeys = [...preferred, ...state.metricColumns.filter((key) => !preferred.includes(key))];
  fillMetricSelect(els.humanMetric, orderedKeys.map((key) => getMetric(key)));
  els.humanMetric.disabled = !orderedKeys.length;
}

function render() {
  state.filteredGroups = getSortedFilteredGroups();
  setStatus(statusText(state.sourceName || "Loaded CSV"));
  renderSummary();
  renderPerformanceChart();
  renderTradeoffChart();
  renderHumanChart();
  renderTable();
  renderHumanRanking();
}

function getFilteredGroups() {
  return state.groups.filter(
    (group) =>
      (state.methodFilter === "all" || group.method === state.methodFilter) &&
      passesMacroFilters(group),
  );
}

function getSortedFilteredGroups() {
  const groups = getFilteredGroups();
  sortGroups(groups);
  return groups;
}

function passesMacroFilters(group) {
  if (isFiniteNumber(group.sequenceLength) && !state.selectedSequences.includes(group.sequenceLength)) {
    return false;
  }
  if (isFiniteNumber(group.codebookSize) && !state.selectedCodebooks.includes(group.codebookSize)) {
    return false;
  }
  return true;
}

function renderSummary() {
  const scoreMetric = getMetric("normalized_score_mean");
  const successMetric = getMetric("success_rate");
  const humanMetric = getMetric(state.humanMetric);
  const groups = modelGroups(getSortedFilteredGroups());

  const bestScore = state.metricColumns.includes("normalized_score_mean") ? bestGroup(groups, "normalized_score_mean", "high") : null;
  const bestSuccess = state.metricColumns.includes("success_rate") ? bestGroup(groups, "success_rate", "high") : null;
  const bestHuman = state.humanMetric ? bestGroup(groups, state.humanMetric, "low") : null;
  const bestTradeoff = state.humanMetric && state.metricColumns.includes("normalized_score_mean") ? bestTradeoffGroup(groups, state.humanMetric) : null;

  els.summary.innerHTML = [
    summaryCard("Best score", bestScore, scoreMetric),
    summaryCard("Best success", bestSuccess, successMetric),
    summaryCard(`Best ${humanMetric.label}`, bestHuman, humanMetric),
    tradeoffCard("Best tradeoff", bestTradeoff, humanMetric),
  ].join("");
}

function summaryCard(label, group, metric) {
  if (!group) return `<article class="summary-card"><div class="label">${escapeHtml(label)}</div><div class="detail">No data</div></article>`;
  const value = metricMean(group, metric.key);
  return `<article class="summary-card">
    <div class="label">${escapeHtml(label)}</div>
    <div class="value">${formatValue(value, metric)}</div>
    <div class="detail">${escapeHtml(group.label)}<br>${group.runs.length} run${group.runs.length === 1 ? "" : "s"}${baselineDetail()}</div>
  </article>`;
}

function tradeoffCard(label, group, humanMetric) {
  if (!group) return `<article class="summary-card"><div class="label">${escapeHtml(label)}</div><div class="detail">No data</div></article>`;
  const score = metricMean(group, "normalized_score_mean");
  const human = metricMean(group, humanMetric.key);
  return `<article class="summary-card">
    <div class="label">${escapeHtml(label)}</div>
    <div class="value">${formatValue(score, getMetric("normalized_score_mean"))}</div>
    <div class="detail">${escapeHtml(group.label)}<br>${escapeHtml(displayMetricLabel(humanMetric))} ${formatValue(human, humanMetric)}${baselineDetail()}</div>
  </article>`;
}

function renderPerformanceChart() {
  const metric = getMetric(state.performanceMetric);
  if (!metric.key || !state.metricColumns.includes(metric.key)) {
    return emptyChart(els.performanceChart, "No performance metric available in this CSV.");
  }
  if (els.performanceCaption) {
    els.performanceCaption.textContent = `${metricHelp(metric.key).direction} ${baselineCaption(metric.key)}Grouped by method, sequence length, and codebook size.`;
  }
  const groups = getSortedFilteredGroups().filter((group) => isFiniteNumber(metricMean(group, metric.key)));
  if (!groups.length) return emptyChart(els.performanceChart, "No performance data for this filter.");

  const sorted = sortForSweep(groups);
  const margin = { top: 20, right: 18, bottom: 74, left: 58 };
  const width = 920;
  const height = 420;
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;
  const maxY = Math.max(...sorted.map((g) => metricMean(g, metric.key)), 1);
  const minY = Math.min(0, ...sorted.map((g) => metricMean(g, metric.key)));
  const y = scaleLinear(minY, maxY * 1.08, margin.top + innerH, margin.top);
  const band = innerW / sorted.length;
  const barW = Math.min(42, band * 0.6);

  let svg = svgOpen(width, height);
  svg += drawYAxis({ y, min: minY, max: maxY * 1.08, x: margin.left, width: innerW, height, label: displayMetricLabel(metric) });
  sorted.forEach((group, index) => {
    const cx = margin.left + band * index + band / 2;
    const value = metricMean(group, metric.key);
    const top = Math.min(y(value), y(0));
    const bottom = Math.max(y(value), y(0));
    const barH = Math.max(1, bottom - top);
    const series = defaultLegendLabel(group);
    svg += `<rect class="chart-mark tip-target" data-series="${escapeAttr(series)}" data-tip="${escapeAttr(groupTooltip(group, [[displayMetricLabel(metric), formatValue(value, metric), metric.key]]))}" x="${cx - barW / 2}" y="${top}" width="${barW}" height="${barH}" rx="4" fill="${group.color}"></rect>`;
    svg += xTickLabel(cx, height - 48, compactLabel(group));
  });
  svg += `<line class="axis" x1="${margin.left}" x2="${width - margin.right}" y1="${y(0)}" y2="${y(0)}"/>`;
  svg += `</svg>${legend(sorted)}`;
  els.performanceChart.innerHTML = svg;
  attachChartTooltips(els.performanceChart);
}

function renderTradeoffChart() {
  const humanMetric = getMetric(state.humanMetric);
  if (!humanMetric.key || !state.metricColumns.includes(humanMetric.key) || !state.metricColumns.includes("normalized_score_mean")) {
    return emptyChart(els.tradeoffChart, "No paired normalized score and selected metric data in this CSV.");
  }
  const humanHelp = metricHelp(humanMetric.key);
  if (els.tradeoffCaption) {
    els.tradeoffCaption.textContent = state.useHumanBaseline && isBaselineAdjustedMetric(humanMetric.key)
      ? `Absolute distance from same-seed human-human baseline. Lower values are closer to the baseline.`
      : `Y-axis normalized score: higher is better task performance. X-axis ${humanMetric.label}: ${humanHelp.direction}`;
  }
  const groups = getSortedFilteredGroups().filter(
    (group) =>
      isFiniteNumber(metricMean(group, state.humanMetric)) &&
      isFiniteNumber(metricMean(group, "normalized_score_mean")),
  );
  if (!groups.length) return emptyChart(els.tradeoffChart, "No paired score and human-likeness data.");

  const width = 560;
  const height = 360;
  const margin = { top: 18, right: 20, bottom: 54, left: 60 };
  const xs = groups.map((g) => metricMean(g, state.humanMetric));
  const ys = groups.map((g) => metricMean(g, "normalized_score_mean"));
  const xDomain = paddedDomain(xs, 0.04);
  const yDomain = paddedDomain(ys, 0.08, true);
  const x = scaleLinear(xDomain[0], xDomain[1], margin.left, width - margin.right);
  const y = scaleLinear(yDomain[0], yDomain[1], height - margin.bottom, margin.top);
  let svg = svgOpen(width, height);
  svg += drawYAxis({ y, min: yDomain[0], max: yDomain[1], x: margin.left, width: width - margin.left - margin.right, height, label: displayMetricLabel(getMetric("normalized_score_mean")) });
  svg += drawXAxis({ x, min: xDomain[0], max: xDomain[1], y: height - margin.bottom, width, label: `${displayMetricLabel(humanMetric)} (${humanHelp.shortDirection})` });
  groups.forEach((group) => {
    const score = metricMean(group, "normalized_score_mean");
    const human = metricMean(group, state.humanMetric);
    const success = metricMean(group, "success_rate");
    const radius = isFiniteNumber(success) ? 6 + success * 10 : 7;
    const series = defaultLegendLabel(group);
    svg += `<circle class="chart-mark tip-target" data-series="${escapeAttr(series)}" data-tip="${escapeAttr(groupTooltip(group, [
        ["Score", formatNumber(score), "normalized_score_mean"],
        [displayMetricLabel(humanMetric), formatValue(human, humanMetric), humanMetric.key],
        ["Success", formatPercent(success), "success_rate"],
      ]))}" cx="${x(human)}" cy="${y(score)}" r="${radius}" fill="${group.color}" opacity="0.82" stroke="#24221f" stroke-width="1"></circle>`;
  });
  svg += `</svg>${legend(groups)}`;
  els.tradeoffChart.innerHTML = svg;
  attachChartTooltips(els.tradeoffChart);
}

function renderHumanChart() {
  const metric = getMetric(state.humanMetric);
  if (!metric.key || !state.metricColumns.includes(metric.key)) {
    return emptyChart(els.humanChart, "No selected human metric available in this CSV.");
  }
  const axis = state.humanSweepAxis;
  const axisKey = axis === "codebook" ? "codebookSize" : "sequenceLength";
  const axisLabel = axis === "codebook" ? "Codebook size" : "Sequence length";
  if (els.humanChartCaption) {
    els.humanChartCaption.textContent = `${displayMetricLabel(metric)} by ${axisLabel.toLowerCase()}: ${state.useHumanBaseline && isBaselineAdjustedMetric(metric.key) ? "absolute distance from same-seed human-human baseline; lower is closer." : metricHelp(metric.key).direction}`;
  }
  const groups = sortForSweep(getSortedFilteredGroups()).filter(
    (group) => group.isMacro && isFiniteNumber(group[axisKey]) && isFiniteNumber(metricMean(group, metric.key)),
  );
  if (!groups.length) return emptyChart(els.humanChart, `No macro sweep data for this metric and ${axisLabel.toLowerCase()}.`);

  const width = 560;
  const height = 360;
  const margin = { top: 18, right: 20, bottom: 54, left: 62 };
  const xs = groups.map((g) => g[axisKey]);
  const ys = groups.map((g) => metricMean(g, metric.key));
  const x = scaleLinear(Math.min(...xs), Math.max(...xs), margin.left, width - margin.right);
  const yDomain = paddedDomain(ys, 0.04);
  const y = scaleLinear(yDomain[0], yDomain[1], height - margin.bottom, margin.top);

  let svg = svgOpen(width, height);
  svg += drawYAxis({ y, min: yDomain[0], max: yDomain[1], x: margin.left, width: width - margin.left - margin.right, height, label: displayMetricLabel(metric) });
  svg += drawXAxis({ x, min: Math.min(...xs), max: Math.max(...xs), y: height - margin.bottom, width, label: axisLabel });
  const seriesLabel = humanSweepLegendLabel(axis);
  const byMethod = groupBy(groups, seriesLabel);
  byMethod.forEach((methodGroups) => {
    methodGroups.sort((a, b) => a[axisKey] - b[axisKey]);
    const points = methodGroups.map((g) => `${x(g[axisKey])},${y(metricMean(g, metric.key))}`).join(" ");
    const series = seriesLabel(methodGroups[0]);
    svg += `<polyline class="chart-line" data-series="${escapeAttr(series)}" points="${points}" fill="none" stroke="${methodGroups[0].color}" stroke-width="2.5"/>`;
    methodGroups.forEach((group) => {
      const value = metricMean(group, metric.key);
      svg += `<circle class="chart-mark tip-target" data-series="${escapeAttr(series)}" data-tip="${escapeAttr(groupTooltip(group, [[displayMetricLabel(metric), formatValue(value, metric), metric.key]]))}" cx="${x(group[axisKey])}" cy="${y(value)}" r="5.5" fill="${group.color}" stroke="#24221f" stroke-width="1"></circle>`;
    });
  });
  svg += `</svg>${legend(groups, seriesLabel)}`;
  els.humanChart.innerHTML = svg;
  attachChartTooltips(els.humanChart);
}

function renderTable() {
  const tableGroups = getSortedFilteredGroups();
  const visibleMetricKeys = getVisibleTableMetricKeys();
  const headerCells = [
    `<th>Experiment</th>`,
    `<th class="sortable ${state.sortMetric === "runs" ? "sorted" : ""}" data-sort="runs">Runs ${sortGlyph("runs")}</th>`,
    ...visibleMetricKeys.map((key) => {
      const metric = getMetric(key);
      const help = metricHelp(key);
      return `<th class="sortable ${state.sortMetric === key ? "sorted" : ""}" data-sort="${escapeHtml(key)}" title="${escapeHtml(help.description)}&#10;${escapeHtml(help.direction)}&#10;${escapeHtml(baselineTooltipLine())}&#10;${escapeHtml(key)}">${escapeHtml(displayMetricLabel(metric))} ${sortGlyph(key)}</th>`;
    }),
    `<th>Model</th>`,
  ];
  els.resultsHead.innerHTML = `<tr>${headerCells.join("")}</tr>`;
  els.tableCaption.textContent = state.showRuns
    ? "Showing grouped configurations and their seed-level runs."
    : `Aggregated rows hide repeated seeds by default. ${state.useHumanBaseline ? "Human-similarity metric cells show mean absolute distance from same-seed human-human rows; performance metrics stay raw. " : ""}Choose metrics to decide which columns appear.`;

  const rows = [];
  tableGroups.forEach((group) => {
    rows.push(groupRow(group, visibleMetricKeys));
    if (state.showRuns) {
      const runs = [...group.runs].sort((a, b) => (a.seed ?? 0) - (b.seed ?? 0));
      runs.forEach((run) => rows.push(runRow(run, visibleMetricKeys)));
    }
  });
  els.resultsBody.innerHTML = rows.join("");
  els.resultsHead.querySelectorAll("th.sortable").forEach((th) => {
    th.addEventListener("click", () => {
      setSortMetric(th.dataset.sort);
    });
  });
}

function renderHumanRanking() {
  if (!els.humanRanking) return;
  const rankingGroups = getSortedFilteredGroups();
  const metrics = state.rankingMetricColumns.filter((key) => state.metricColumns.includes(key));
  const ranking = computeHumanRanking(modelGroups(rankingGroups), metrics);
  if (els.rankingCaption) {
    els.rankingCaption.textContent = metrics.length && state.useHumanBaseline
      ? `Using ${metrics.length} selected metric${metrics.length === 1 ? "" : "s"}. Each metric gives rank #1 to the model closest to the same-seed human-human baseline.`
      : metrics.length
      ? `Using ${metrics.length} selected metric${metrics.length === 1 ? "" : "s"}. Each metric gives rank #1 to the most human-like value for that metric.`
      : "Choose at least one ranking metric to compute the most human-like model.";
  }

  if (!metrics.length) {
    els.humanRanking.innerHTML = `<div class="empty-state">No ranking metrics selected.</div>`;
    return;
  }
  if (!ranking.length) {
    els.humanRanking.innerHTML = `<div class="empty-state">No finite values for the selected ranking metrics.</div>`;
    return;
  }

  const sortedRanking = sortRankingEntries(ranking);
  const winner = ranking[0];
  const winnerModel = bestRunForGroup(winner.group)?.modelPath || "";
  const metricLabels = metrics.map((key) => displayMetricLabel(getMetric(key))).join(", ");
  const rows = sortedRanking
    .map((entry, index) => {
      const bestModel = bestRunForGroup(entry.group)?.modelPath || "";
      return `<tr>
        <td>${index + 1}</td>
        <td><span class="method-pill">${escapeHtml(entry.group.method)}</span> ${escapeHtml(configSuffix(entry.group))}</td>
        <td>${entry.firstPlaces}</td>
        <td>${formatNumber(entry.averageRank)}</td>
        <td>${entry.validMetricCount}</td>
        <td class="dim">${escapeHtml(entry.winningLabels.join(", ") || "-")}</td>
        <td class="dim" title="${escapeHtml(bestModel)}">${escapeHtml(shortPath(bestModel))}</td>
      </tr>`;
    })
    .join("");

  els.humanRanking.innerHTML = `<div class="ranking-winner">
      <div>
        <span class="label">Most human-like by selected metrics</span>
        <strong>${escapeHtml(winner.group.label)}</strong>
        <p>${winner.firstPlaces} first-place metric${winner.firstPlaces === 1 ? "" : "s"}; average rank ${formatNumber(winner.averageRank)} across ${winner.validMetricCount} metric${winner.validMetricCount === 1 ? "" : "s"}${baselineDetail()}.</p>
        <p class="dim">Selected metrics: ${escapeHtml(metricLabels)}</p>
        ${winnerModel ? `<p class="dim">Representative model: ${escapeHtml(shortPath(winnerModel))}</p>` : ""}
      </div>
    </div>
    <div class="table-wrap ranking-table-wrap">
      <table>
        <thead>
          <tr>
            <th class="sortable ${rankingSortedClass("rank")}" data-ranking-sort="rank">Rank ${rankingSortGlyph("rank")}</th>
            <th class="sortable ${rankingSortedClass("experiment")}" data-ranking-sort="experiment">Experiment ${rankingSortGlyph("experiment")}</th>
            <th class="sortable ${rankingSortedClass("firstPlaces")}" data-ranking-sort="firstPlaces">#1 Metrics ${rankingSortGlyph("firstPlaces")}</th>
            <th class="sortable ${rankingSortedClass("averageRank")}" data-ranking-sort="averageRank">Avg Rank ${rankingSortGlyph("averageRank")}</th>
            <th class="sortable ${rankingSortedClass("validMetricCount")}" data-ranking-sort="validMetricCount">Valid Metrics ${rankingSortGlyph("validMetricCount")}</th>
            <th class="sortable ${rankingSortedClass("winningLabels")}" data-ranking-sort="winningLabels">#1 On ${rankingSortGlyph("winningLabels")}</th>
            <th class="sortable ${rankingSortedClass("model")}" data-ranking-sort="model">Model ${rankingSortGlyph("model")}</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    </div>`;
  els.humanRanking.querySelectorAll("th[data-ranking-sort]").forEach((th) => {
    th.addEventListener("click", () => setRankingSort(th.dataset.rankingSort));
  });
}

function setRankingSort(key) {
  if (!key) return;
  if (state.rankingSortKey === key) {
    state.rankingSortDirection = state.rankingSortDirection === "asc" ? "desc" : "asc";
  } else {
    state.rankingSortKey = key;
    state.rankingSortDirection = defaultRankingSortDirection(key);
  }
  renderHumanRanking();
}

function defaultRankingSortDirection(key) {
  if (key === "firstPlaces" || key === "validMetricCount") return "desc";
  return "asc";
}

function rankingSortedClass(key) {
  return state.rankingSortKey === key ? "sorted" : "";
}

function rankingSortGlyph(key) {
  if (state.rankingSortKey !== key) return "";
  return state.rankingSortDirection === "asc" ? "▲" : "▼";
}

function sortRankingEntries(entries) {
  return [...entries].sort((a, b) => {
    const direction = state.rankingSortDirection === "asc" ? 1 : -1;
    const av = rankingSortValue(a, state.rankingSortKey);
    const bv = rankingSortValue(b, state.rankingSortKey);
    if (typeof av === "number" && typeof bv === "number") {
      if (!isFiniteNumber(av) && !isFiniteNumber(bv)) return defaultRankingCompare(a, b);
      if (!isFiniteNumber(av)) return 1;
      if (!isFiniteNumber(bv)) return -1;
      if (av !== bv) return (av - bv) * direction;
      return defaultRankingCompare(a, b);
    }
    const cmp = String(av ?? "").localeCompare(String(bv ?? ""));
    return cmp ? cmp * direction : defaultRankingCompare(a, b);
  });
}

function rankingSortValue(entry, key) {
  if (key === "rank" || key === "default") return entry.defaultRank;
  if (key === "experiment") return entry.group.label;
  if (key === "firstPlaces") return entry.firstPlaces;
  if (key === "averageRank") return entry.averageRank;
  if (key === "validMetricCount") return entry.validMetricCount;
  if (key === "winningLabels") return entry.winningLabels.join(", ");
  if (key === "model") return shortPath(bestRunForGroup(entry.group)?.modelPath || "");
  return entry.defaultRank;
}

function defaultRankingCompare(a, b) {
  return a.defaultRank - b.defaultRank;
}

function computeHumanRanking(groups, metricKeys) {
  const records = new Map();
  groups.forEach((group) => {
    records.set(group.key, {
      group,
      firstPlaces: 0,
      rankSum: 0,
      validMetricCount: 0,
      winningLabels: [],
      averageRank: Infinity,
    });
  });

  metricKeys.forEach((key) => {
    const metric = getMetric(key);
    const direction = defaultSortDirection(metric);
    const sorted = groups
      .filter((group) => isFiniteNumber(metricMean(group, key)))
      .sort((a, b) => {
        const av = rankingMetricSortValue(metricMean(a, key), key);
        const bv = rankingMetricSortValue(metricMean(b, key), key);
        return direction === "asc" ? av - bv : bv - av;
      });
    let previousValue = null;
    let previousRank = 0;
    sorted.forEach((group, index) => {
      const value = metricMean(group, key);
      const sortValue = rankingMetricSortValue(value, key);
      const rank = previousValue != null && valuesEqual(sortValue, previousValue) ? previousRank : index + 1;
      const record = records.get(group.key);
      record.rankSum += rank;
      record.validMetricCount += 1;
      if (rank === 1) {
        record.firstPlaces += 1;
        record.winningLabels.push(displayMetricLabel(metric));
      }
      previousValue = sortValue;
      previousRank = rank;
    });
  });

  return [...records.values()]
    .filter((record) => record.validMetricCount > 0)
    .map((record) => ({
      ...record,
      averageRank: record.rankSum / record.validMetricCount,
    }))
    .sort((a, b) => {
      if (a.firstPlaces !== b.firstPlaces) return b.firstPlaces - a.firstPlaces;
      if (a.averageRank !== b.averageRank) return a.averageRank - b.averageRank;
      if (a.validMetricCount !== b.validMetricCount) return b.validMetricCount - a.validMetricCount;
      return a.group.label.localeCompare(b.group.label);
    })
    .map((record, index) => ({
      ...record,
      defaultRank: index + 1,
    }));
}

function getVisibleTableMetricKeys() {
  const selected = state.selectedMetricColumns;
  return selected;
}

function groupRow(group, visibleMetricKeys) {
  const bestModel = bestRunForGroup(group)?.modelPath || "";
  const metricCells = visibleMetricKeys
    .map((key) => {
      const metric = getMetric(key);
      const stat = group.metrics[key] || { mean: null, std: 0 };
      return `<td>${formatValue(stat.mean ?? null, metric)}</td>`;
    })
    .join("");
  return `<tr>
    <td><span class="method-pill">${escapeHtml(group.method)}</span> ${escapeHtml(configSuffix(group))}</td>
    <td>${group.runs.length}</td>
    ${metricCells}
    <td class="dim" title="${escapeHtml(bestModel)}">${escapeHtml(shortPath(bestModel))}</td>
  </tr>`;
}

function runRow(run, visibleMetricKeys) {
  const metricCells = visibleMetricKeys
    .map((key) => `<td>${formatValue(getRunMetric(run, key), getMetric(key))}</td>`)
    .join("");
  return `<tr class="run-row">
    <td class="dim">seed=${run.seed ?? "-"} ${escapeHtml(run.groupLabel)}</td>
    <td>1</td>
    ${metricCells}
    <td class="dim" title="${escapeHtml(run.modelPath)}">${escapeHtml(shortPath(run.modelPath))}</td>
  </tr>`;
}

function downloadSummaryCsv() {
  const metrics = state.selectedMetricColumns.length ? state.selectedMetricColumns : state.metricColumns;
  const header = ["method", "sequence_length", "codebook_size", "runs", ...metrics.map((m) => `${m}_mean`)];
  const lines = [header.join(",")];
  state.groups.forEach((group) => {
    const row = [
      csvEscape(group.method),
      group.sequenceLength ?? "",
      group.codebookSize ?? "",
      group.runs.length,
      ...metrics.map((m) => valueForCsv(metricMean(group, m))),
    ];
    lines.push(row.join(","));
  });
  const blob = new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "maq_experiment_summary.csv";
  link.click();
  URL.revokeObjectURL(url);
}

function setSortMetric(key) {
  if (!key) return;
  if (state.sortMetric === key) {
    state.sortDirection = state.sortDirection === "asc" ? "desc" : "asc";
  } else {
    state.sortMetric = key;
    state.sortDirection = key === "runs" ? "desc" : state.useHumanBaseline ? "asc" : defaultSortDirection(getMetric(key));
  }
  if (state.metricColumns.includes(state.sortMetric)) {
    els.sortMetric.value = state.sortMetric;
  }
  render();
}

function sortGroups(groups) {
  const metric = getMetric(state.sortMetric);
  const direction = state.sortDirection === "asc" ? 1 : -1;
  groups.sort((a, b) => {
    const av = state.sortMetric === "runs" ? a.runs.length : metricMean(a, state.sortMetric);
    const bv = state.sortMetric === "runs" ? b.runs.length : metricMean(b, state.sortMetric);
    if (!isFiniteNumber(av) && !isFiniteNumber(bv)) return a.label.localeCompare(b.label);
    if (!isFiniteNumber(av)) return 1;
    if (!isFiniteNumber(bv)) return -1;
    return (av - bv) * direction;
  });
}

function sortForSweep(groups) {
  return [...groups].sort((a, b) => {
    if (a.isMacro !== b.isMacro) return a.isMacro ? -1 : 1;
    if ((a.sequenceLength ?? 999) !== (b.sequenceLength ?? 999)) return (a.sequenceLength ?? 999) - (b.sequenceLength ?? 999);
    return a.label.localeCompare(b.label);
  });
}

function bestGroup(groups, metricKey, better) {
  const valid = groups.filter((group) => isFiniteNumber(metricMean(group, metricKey)));
  if (!valid.length) return null;
  if (state.useHumanBaseline && isBaselineAdjustedMetric(metricKey)) {
    return valid.reduce((best, group) => {
      const value = metricMean(group, metricKey);
      const bestValue = metricMean(best, metricKey);
      return value < bestValue ? group : best;
    }, valid[0]);
  }
  return valid.reduce((best, group) => {
    const value = metricMean(group, metricKey);
    const bestValue = metricMean(best, metricKey);
    return better === "low" ? (value < bestValue ? group : best) : value > bestValue ? group : best;
  }, valid[0]);
}

function bestTradeoffGroup(groups, humanMetricKey) {
  const valid = groups.filter(
    (group) =>
      isFiniteNumber(metricMean(group, "normalized_score_mean")) &&
      isFiniteNumber(metricMean(group, humanMetricKey)),
  );
  if (!valid.length) return null;
  const scores = valid.map((g) => metricMean(g, "normalized_score_mean"));
  const humans = valid.map((g) => metricMean(g, humanMetricKey));
  const sMin = Math.min(...scores);
  const sMax = Math.max(...scores);
  const hMin = Math.min(...humans);
  const hMax = Math.max(...humans);
  return valid.reduce((best, group) => {
    const maxHumanDistance = Math.max(...humans);
    const scoreValue = normalize01(metricMean(group, "normalized_score_mean"), sMin, sMax);
    const humanValue = state.useHumanBaseline && isBaselineAdjustedMetric(humanMetricKey)
      ? 1 - normalize01(metricMean(group, humanMetricKey), 0, maxHumanDistance)
      : 1 - normalize01(metricMean(group, humanMetricKey), hMin, hMax);
    const scoreNorm = Number.isFinite(scoreValue) ? scoreValue : 0.5;
    const humanNorm = Number.isFinite(humanValue) ? humanValue : 0.5;
    const tradeoff = scoreNorm * 0.55 + humanNorm * 0.45;
    const bestTradeoff = best._tradeoff ?? -Infinity;
    group._tradeoff = tradeoff;
    return tradeoff > bestTradeoff ? group : best;
  }, valid[0]);
}

function bestRunForGroup(group) {
  if (!state.metricColumns.includes("normalized_score_mean")) {
    return group.runs[0];
  }
  return [...group.runs]
    .filter((run) => isFiniteNumber(run.metrics.normalized_score_mean))
    .sort((a, b) => b.metrics.normalized_score_mean - a.metrics.normalized_score_mean)[0];
}

function groupTooltip(group, entries = []) {
  const lines = [group.label];
  entries.forEach(([label, value, key]) => {
    lines.push(`${label}: ${value}`);
    if (key) lines.push(metricHelp(key).direction);
  });
  lines.push(`Runs: ${group.runs.length}`);

  const bestRun = bestRunForGroup(group) || group.runs[0];
  if (bestRun?.modelPath) {
    lines.push(`Best model: ${bestRun.modelPath}`);
  }

  if (group.runs.length > 1) {
    lines.push("Models:");
    group.runs.forEach((run) => {
      const seed = run.seed != null ? `seed ${run.seed}` : "run";
      lines.push(`- ${seed}: ${run.modelPath || "no model path"}`);
    });
  }

  return lines.join("\n");
}

function attachChartTooltips(container) {
  container.querySelectorAll(".tip-target").forEach((target) => {
    target.addEventListener("mouseenter", (event) => {
      showChartTooltip(target.dataset.tip || "", event);
    });
    target.addEventListener("mousemove", moveChartTooltip);
    target.addEventListener("mouseleave", hideChartTooltip);
  });
  attachLegendHighlights(container);
}

function attachLegendHighlights(container) {
  container.querySelectorAll(".legend-item[data-series]").forEach((item) => {
    item.addEventListener("mouseenter", () => {
      setChartSeriesHighlight(container, item.dataset.series);
    });
    item.addEventListener("focus", () => {
      setChartSeriesHighlight(container, item.dataset.series);
    });
    item.addEventListener("mouseleave", () => {
      clearChartSeriesHighlight(container);
    });
    item.addEventListener("blur", () => {
      clearChartSeriesHighlight(container);
    });
  });
}

function setChartSeriesHighlight(container, series) {
  container.querySelectorAll(".chart-mark[data-series], .chart-line[data-series]").forEach((mark) => {
    const isMatch = mark.dataset.series === series;
    mark.classList.toggle("is-highlighted", isMatch);
    mark.classList.toggle("is-dimmed", !isMatch);
  });
  container.querySelectorAll(".legend-item[data-series]").forEach((item) => {
    const isMatch = item.dataset.series === series;
    item.classList.toggle("is-highlighted", isMatch);
    item.classList.toggle("is-dimmed", !isMatch);
  });
}

function clearChartSeriesHighlight(container) {
  container.querySelectorAll(".chart-mark, .chart-line, .legend-item").forEach((element) => {
    element.classList.remove("is-highlighted", "is-dimmed");
  });
}

function showChartTooltip(text, event) {
  if (!text) return;
  els.chartTooltip.textContent = text;
  els.chartTooltip.hidden = false;
  moveChartTooltip(event);
}

function moveChartTooltip(event) {
  if (els.chartTooltip.hidden) return;
  const pad = 14;
  const rect = els.chartTooltip.getBoundingClientRect();
  let left = event.clientX + pad;
  let top = event.clientY + pad;

  if (left + rect.width > window.innerWidth - 8) {
    left = event.clientX - rect.width - pad;
  }
  if (top + rect.height > window.innerHeight - 8) {
    top = event.clientY - rect.height - pad;
  }

  els.chartTooltip.style.left = `${Math.max(8, left)}px`;
  els.chartTooltip.style.top = `${Math.max(8, top)}px`;
}

function hideChartTooltip() {
  els.chartTooltip.hidden = true;
}

function getRunMetric(run, key) {
  const rawValue = run.metrics[key];
  if (!state.useHumanBaseline || !isBaselineAdjustedMetric(key) || !isFiniteNumber(rawValue)) {
    return rawValue;
  }
  if (run.isHumanHuman) return 0;
  const baseline = state.humanBaselinesBySeed.get(String(run.seed));
  const baselineValue = baseline?.metrics?.[key];
  if (!isFiniteNumber(baselineValue)) return null;
  return Math.abs(rawValue - baselineValue);
}

function metricMean(group, key) {
  return group?.metrics?.[key]?.mean ?? null;
}

function modelGroups(groups) {
  return groups.filter((group) => !group.isHumanHuman);
}

function displayMetricLabel(metric) {
  return state.useHumanBaseline && isBaselineAdjustedMetric(metric.key)
    ? `${metric.label} Distance To Human-Human`
    : metric.label;
}

function baselineCaption(key) {
  return state.useHumanBaseline && isBaselineAdjustedMetric(key)
    ? "Values are absolute distance to same-seed human-human baseline. "
    : "";
}

function baselineDetail() {
  return state.useHumanBaseline ? "<br>distance to human-human baseline" : "";
}

function baselineTooltipLine() {
  return state.useHumanBaseline
    ? "Human-human baseline mode adjusts only human-similarity metrics: value = abs(agent metric - same-seed human-human metric). Performance metrics stay raw."
    : "";
}

function rankingMetricSortValue(value) {
  return value;
}

function isBaselineAdjustedMetric(key) {
  return isHumanSimilarityMetric(String(key).toLowerCase());
}

function getMetric(key) {
  return [...performanceMetrics, ...humanMetrics].find((metric) => metric.key === key) || {
    key,
    label: prettifyMetricLabel(key),
    better: inferBetterDirection(key),
    format: inferFormat(key),
  };
}

function metricHelp(key) {
  const lower = String(key).toLowerCase();
  const subject = metricSubject(lower);
  const population = metricPopulation(lower);
  const statistic = metricStatistic(lower);
  const humanSimilarity = isHumanSimilarityMetric(lower);

  const earlyTermination = lower.includes("early_termination_rate");

  const performance =
    lower.includes("normalized_score") ||
    lower.includes("raw_reward") ||
    lower === "success_rate";

  const count = lower.includes("_count") || lower.includes("total_");
  const exceedDefault = lower.includes("exceed_default");
  const length = lower.includes("length");
  const better = inferBetterDirection(key);
  const distance = metricTrajectoryDistance(lower);

  let description;

  if (lower.includes("normalized_score")) {
    description =
      "Mean D4RL normalized score from env.get_normalized_score(total raw reward) * 100 over evaluation episodes.";
  } else if (lower.includes("raw_reward")) {
    description =
      "Mean total raw environment reward over evaluation episodes.";
  } else if (lower === "success_rate") {
    description =
      "Fraction of evaluation episodes where the final environment info reported success or goal_achieved.";
  } else if (earlyTermination) {
    description =
      "Fraction of episodes marked as early-terminated by the evaluator; in the current evaluator this is usually 0 because the early-success break is disabled.";
  } else if (lower.includes("exceed_default_pct")) {
    description =
      `${population} percentage of episodes longer than the environment default horizon.`;
  } else if (lower.includes("exceed_default_count")) {
    description =
      `${population} count of episodes longer than the environment default horizon.`;
  } else if (lower.includes("total_success_episodes")) {
    description =
      "Number of successful evaluation episodes.";
  } else if (lower.includes("total_non_success_episodes")) {
    description =
      "Number of failed evaluation episodes.";
  } else if (lower.includes("js_diversity")) {
    description =
      `${population} Jensen-Shannon divergence between the agent-to-human ${distance} distance distribution and the human-to-human reference diversity distribution; ${statistic}.`;
  } else if (lower.includes("knn_quality")) {
    description =
      `${population} mean distance to the 5 nearest human ${subject} trajectories using ${distance}; ${statistic}.`;
  } else if (lower.includes("min_dtw")) {
    description =
      `${population} closest Dynamic Time Warping distance from the agent ${subject} trajectory to any human reference trajectory; ${statistic}.`;
  } else if (lower.includes("dtw")) {
    description =
      `${population} average Dynamic Time Warping distance from the agent ${subject} trajectory to all human reference trajectories; ${statistic}.`;
  } else if (lower.includes("wasserstein") || lower.includes("w2")) {
    description =
      `${population} Wasserstein distance between scaled agent and human ${subject} distributions using optimal transport; ${statistic}.`;
  } else if (lower.includes("mmd")) {
    description =
      `${population} squared RBF-kernel MMD between scaled agent and human state-action occupancy samples; ${statistic}.`;
  } else if (lower.includes("frechet")) {
    description =
      `${population} discrete Frechet trajectory distance after scaling ${subject} features; ${statistic}.`;
  } else if (lower.includes("mse")) {
    description =
      `${population} length-penalized trajectory MSE after scaling ${subject} features; ${statistic}.`;
  } else if (length) {
    description =
      `${population} episode length in primitive environment steps; ${statistic}.`;
  } else {
    description =
      `${population} evaluation metric from the CSV; ${statistic}.`;
  }

  let direction;
  let shortDirection;
  let tone = "context";

  if (humanSimilarity) {
    direction =
      better === "low"
        ? "Lower is more human-like."
        : "Higher is more human-like.";

    shortDirection =
      better === "low"
        ? "Lower = more human-like"
        : "Higher = more human-like";

    tone = better === "low" ? "low" : "high";
  } else if (earlyTermination) {
    direction =
      "Not meaningful as a better/worse score in the current evaluator; usually 0 because early-success termination is disabled.";

    shortDirection = "Usually 0; contextual only";
    tone = "context";
  } else if (performance) {
    if (lower.includes("raw_reward")) {
      direction =
        "Higher is better task performance within the same environment; avoid comparing raw reward across different environments.";

      shortDirection = "Higher = better within same env";
    } else {
      direction = "Higher is better task performance.";
      shortDirection = "Higher = better task performance";
    }

    tone = "high";
  } else if (exceedDefault) {
    direction =
      "Lower means fewer episodes exceeded the default horizon; not a human-likeness score alone.";

    shortDirection = "Lower = fewer over-horizon episodes";
    tone = "low";
  } else if (count) {
    direction = "Count metric; interpret with the episode total.";
    shortDirection = "Contextual count";
    tone = "context";
  } else if (length) {
    direction =
      "Lower means shorter episodes; not necessarily more human-like alone.";

    shortDirection = "Lower = shorter episodes";
    tone = "low";
  } else {
    direction = better === "low" ? "Lower is better." : "Higher is better.";

    shortDirection =
      better === "low" ? "Lower = better" : "Higher = better";

    tone = better === "low" ? "low" : "high";
  }

  return {
    description,
    direction,
    shortDirection,
    tone,
  };
}

function metricCategory(key) {
  const lower = String(key).toLowerCase();
  if (isHumanSimilarityMetric(lower)) return "human";
  if (
    lower.includes("normalized_score") ||
    lower.includes("raw_reward") ||
    lower === "success_rate" ||
    lower.includes("early_termination_rate")
  ) {
    return "performance";
  }
  if (
    lower.includes("length") ||
    lower.includes("_count") ||
    lower.includes("total_") ||
    lower.includes("exceed_default")
  ) {
    return "episode";
  }
  return "other";
}

function metricSubject(lower) {
  if (lower.includes("state_action")) return "state-action";
  if (lower.includes("action")) return "action";
  if (lower.includes("state")) return "state";
  return "trajectory";
}

function metricPopulation(lower) {
  if (lower.startsWith("success_")) return "Successful-episode";
  if (lower.startsWith("non_success_")) return "Failed-episode";
  return "All-episode";
}

function metricStatistic(lower) {
  if (lower.endsWith("_agg_mean")) return "mean over evaluated episodes";
  if (lower.endsWith("_mean")) return "mean";
  if (lower.endsWith("_min")) return "minimum";
  if (lower.endsWith("_max")) return "maximum";
  if (lower.endsWith("_pct")) return "percentage";
  if (lower.endsWith("_count")) return "count";
  return "reported value";
}

function metricTrajectoryDistance(lower) {
  if (lower.includes("frechet")) return "discrete Frechet";
  if (lower.includes("mse")) return "length-penalized MSE";
  if (lower.includes("dtw")) return "DTW";
  if (lower.includes("wasserstein") || lower.includes("w2")) return "Wasserstein";
  return "trajectory";
}

function isHumanSimilarityMetric(lower) {
  return (
    lower.includes("dtw") ||
    lower.includes("wasserstein") ||
    lower.includes("w2") ||
    lower.includes("mmd") ||
    lower.includes("frechet") ||
    lower.includes("mse") ||
    lower.includes("js_diversity")
  );
}

function prettifyMetricLabel(key) {
  return String(key)
    .replace(/^success_/, "Success ")
    .replace(/^non_success_/, "Fail ")
    .replace(/_agg_/g, " ")
    .replace(/_/g, " ")
    .replace(/\bdtw\b/g, "DTW")
    .replace(/\bw2\b/g, "W2")
    .replace(/\bmmd\b/g, "MMD")
    .replace(/\bqd\b/g, "QD")
    .replace(/\bstd\b/g, "Std")
    .replace(/\bpct\b/g, "%")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function inferBetterDirection(key) {
  const lower = String(key).toLowerCase();
  if (
    lower.includes("dtw") ||
    lower.includes("wasserstein") ||
    lower.includes("w2") ||
    lower.includes("mmd") ||
    lower.includes("frechet") ||
    lower.includes("mse") ||
    lower.includes("js_diversity") ||
    lower.includes("exceed_default") ||
    lower.includes("length") ||
    lower.includes("loss")
  ) {
    return "low";
  }
  return "high";
}

function inferFormat(key) {
  const lower = String(key).toLowerCase();
  if (lower.includes("pct")) return "pct";
  if (lower.includes("rate")) return "percent";
  if (lower.includes("mmd") || lower.includes("js_diversity")) return "small";
  return "score";
}

function defaultSortDirection(metric) {
  return metric.better === "low" ? "asc" : "desc";
}

function sortGlyph(key) {
  if (state.sortMetric !== key) return "";
  return state.sortDirection === "asc" ? "▲" : "▼";
}

function emptyChart(container, message) {
  container.innerHTML = `<div class="empty-state">${escapeHtml(message)}</div>`;
}

function svgOpen(width, height) {
  return `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Experiment chart">`;
}

function drawYAxis({ y, min, max, x, width, height, label }) {
  const ticks = makeTicks(min, max, 5);
  let out = "";
  ticks.forEach((tick) => {
    out += `<line class="grid-line" x1="${x}" x2="${x + width}" y1="${y(tick)}" y2="${y(tick)}"/>`;
    out += `<text class="tick-label" x="${x - 8}" y="${y(tick) + 4}" text-anchor="end">${formatNumber(tick)}</text>`;
  });
  out += `<line class="axis" x1="${x}" x2="${x}" y1="18" y2="${height - 54}"/>`;
  out += `<text class="axis-label" x="${x}" y="12">${escapeHtml(label)}</text>`;
  return out;
}

function drawXAxis({ x, min, max, y, width, label }) {
  const ticks = makeTicks(min, max, 5);
  let out = `<line class="axis" x1="60" x2="${width - 20}" y1="${y}" y2="${y}"/>`;
  ticks.forEach((tick) => {
    out += `<line class="axis" x1="${x(tick)}" x2="${x(tick)}" y1="${y}" y2="${y + 5}"/>`;
    out += `<text class="tick-label" x="${x(tick)}" y="${y + 20}" text-anchor="middle">${formatNumber(tick)}</text>`;
  });
  out += `<text class="axis-label" x="${width / 2}" y="${y + 42}" text-anchor="middle">${escapeHtml(label)}</text>`;
  return out;
}

function xTickLabel(x, y, text) {
  const words = text.split(" ");
  const first = escapeHtml(words.slice(0, 2).join(" "));
  const second = escapeHtml(words.slice(2).join(" "));
  return `<text class="tick-label" x="${x}" y="${y}" text-anchor="middle">
    <tspan x="${x}">${first}</tspan>
    <tspan x="${x}" dy="14">${second}</tspan>
  </text>`;
}

function defaultLegendLabel(group) {
  return group.label;
}

function humanSweepLegendLabel(axis) {
  return (group) => axis === "codebook"
    ? `${group.method} sq=${group.sequenceLength ?? "?"}`
    : `${group.method} k=${group.codebookSize ?? "?"}`;
}

function legend(groups, labelFn = defaultLegendLabel) {
  const seen = new Map();
  groups.forEach((group) => {
    const label = labelFn(group);
    if (!seen.has(label)) seen.set(label, group.color);
  });
  return `<div class="legend">${[...seen.entries()]
    .slice(0, 12)
    .map(([label, color]) => `<span class="legend-item" tabindex="0" data-series="${escapeAttr(label)}"><i class="legend-dot" style="background:${color}"></i>${escapeHtml(label)}</span>`)
    .join("")}</div>`;
}

function scaleLinear(domainMin, domainMax, rangeMin, rangeMax) {
  const d = domainMax - domainMin || 1;
  return (value) => rangeMin + ((value - domainMin) / d) * (rangeMax - rangeMin);
}

function paddedDomain(values, paddingRatio = 0.05, includeZero = false) {
  const finite = values.filter(isFiniteNumber);
  if (!finite.length) return [0, 1];
  let min = Math.min(...finite);
  let max = Math.max(...finite);
  if (includeZero) {
    min = Math.min(min, 0);
    max = Math.max(max, 0);
  }
  const span = max - min || Math.max(Math.abs(max), 1);
  const pad = span * paddingRatio;
  return [min - pad, max + pad];
}

function makeTicks(min, max, count) {
  const span = max - min || 1;
  const step = niceStep(span / Math.max(1, count - 1));
  const first = Math.ceil(min / step) * step;
  const ticks = [];
  for (let value = first; value <= max + step * 0.5; value += step) ticks.push(value);
  return ticks.slice(0, 7);
}

function niceStep(raw) {
  const power = 10 ** Math.floor(Math.log10(raw || 1));
  const normalized = raw / power;
  if (normalized <= 1) return power;
  if (normalized <= 2) return 2 * power;
  if (normalized <= 5) return 5 * power;
  return 10 * power;
}

function configSuffix(group) {
  return group.isMacro ? `sq=${group.sequenceLength ?? "?"} k=${group.codebookSize ?? "?"}` : "baseline";
}

function compactLabel(group) {
  if (!group.isMacro) return group.method;
  return `sq ${group.sequenceLength} k ${group.codebookSize}`;
}

function shortPath(path) {
  if (!path) return "";
  const parts = path.split("/");
  return parts.slice(-2).join("/");
}

function formatValue(value, metric) {
  if (!isFiniteNumber(value)) return "NA";
  if (metric.format === "percent") return formatPercent(value);
  if (metric.format === "pct") return `${formatNumber(value)}%`;
  if (metric.format === "small") return value.toFixed(4);
  return formatNumber(value);
}

function formatPercent(value) {
  if (!isFiniteNumber(value)) return "NA";
  return `${(value * 100).toFixed(value >= 0.1 ? 0 : 1)}%`;
}

function formatNumber(value) {
  if (!isFiniteNumber(value)) return "NA";
  const abs = Math.abs(value);
  if (abs >= 100) return value.toFixed(0);
  if (abs >= 10) return value.toFixed(1);
  if (abs >= 1) return value.toFixed(2);
  return value.toFixed(3);
}

function numberOrNull(value) {
  if (value == null || value === "" || String(value).toLowerCase() === "nan") return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function isFiniteNumber(value) {
  return typeof value === "number" && Number.isFinite(value);
}

function mean(values) {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function std(values) {
  const avg = mean(values);
  return Math.sqrt(mean(values.map((value) => (value - avg) ** 2)));
}

function normalize01(value, min, max) {
  if (max === min) return 0.5;
  return (value - min) / (max - min);
}

function valuesEqual(a, b) {
  return Math.abs(a - b) <= 1e-12;
}

function groupBy(items, keyFn) {
  const map = new Map();
  items.forEach((item) => {
    const key = keyFn(item);
    if (!map.has(key)) map.set(key, []);
    map.get(key).push(item);
  });
  return map;
}

function valueForCsv(value) {
  return isFiniteNumber(value) ? String(value) : "";
}

function csvEscape(value) {
  const text = String(value ?? "");
  return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function escapeAttr(value) {
  return escapeHtml(value).replaceAll("\n", "&#10;");
}

function setStatus(message, isError = false) {
  els.status.textContent = message;
  els.status.classList.toggle("error", isError);
}

init();
