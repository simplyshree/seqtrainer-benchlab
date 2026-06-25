let currentDatasetId = null;
let currentRunId = null;
let currentDataset = null;
let currentPlan = null;
let currentRunConfig = null;
let currentMode = localStorage.getItem("seqtrainerBenchLabMode") || "beginner";

const toast = document.querySelector("#toast");
const previewLabels = {
  "dataset-preview": ["Show Preview", "Hide Preview"],
  "feature-preview": ["Show Preview", "Hide Preview"],
  "predictions-table": ["Show Predictions", "Hide Predictions"],
};

function showToast(message) {
  toast.textContent = message;
  toast.classList.remove("hidden");
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => toast.classList.add("hidden"), 4200);
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const data = await response.json();
      detail = data.detail || detail;
    } catch {
      // Keep status text.
    }
    throw new Error(detail);
  }
  return response.json();
}

function setText(selector, value) {
  const node = document.querySelector(selector);
  if (node) node.textContent = value;
}

function goToPanel(panelName) {
  document.querySelectorAll(".nav-button").forEach((item) => item.classList.toggle("active", item.dataset.panel === panelName));
  document.querySelectorAll(".panel").forEach((panel) => panel.classList.remove("visible"));
  document.querySelector(`#panel-${panelName}`).classList.add("visible");
}

function applyWorkflowMode(mode, navigate = false) {
  currentMode = mode;
  localStorage.setItem("seqtrainerBenchLabMode", mode);
  document.body.dataset.mode = mode;
  document.querySelectorAll(".mode-card").forEach((card) => card.classList.remove("selected"));
  document.querySelector(`#mode-card-${mode}`)?.classList.add("selected");
  if (navigate) {
    showToast(mode === "advanced" ? "Advanced mode enabled." : "Beginner mode enabled.");
    goToPanel("dataset");
  }
}

async function withLoading(button, fn) {
  if (!button) return fn();
  try {
    button.classList.add("loading");
    button.disabled = true;
    return await fn();
  } finally {
    button.classList.remove("loading");
    button.disabled = false;
  }
}

function renderMeta(target, payload) {
  target.innerHTML = "";
  Object.entries(payload).forEach(([key, value]) => {
    const dt = document.createElement("dt");
    const dd = document.createElement("dd");
    dt.textContent = key.replaceAll("_", " ");
    dd.textContent = Array.isArray(value) ? value.join(", ") : String(value);
    target.append(dt, dd);
  });
}

function renderTable(target, rows) {
  target.classList.remove("empty-state");
  target.classList.remove("preview-collapsed");
  if (!rows || rows.length === 0) {
    target.classList.add("empty-state");
    target.textContent = "No rows.";
    return;
  }
  const columns = Object.keys(rows[0]);
  const table = document.createElement("table");
  const thead = document.createElement("thead");
  const tbody = document.createElement("tbody");
  const headerRow = document.createElement("tr");
  columns.forEach((column) => {
    const th = document.createElement("th");
    th.textContent = column;
    headerRow.append(th);
  });
  thead.append(headerRow);
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    columns.forEach((column) => {
      const td = document.createElement("td");
      const value = row[column];
      td.textContent = typeof value === "number" ? Number(value.toFixed(6)).toString() : value ?? "";
      tr.append(td);
    });
    tbody.append(tr);
  });
  table.append(thead, tbody);
  target.innerHTML = "";
  target.append(table);
}

function previewButton(targetId) {
  return document.querySelector(`.preview-toggle[data-preview-target="${targetId}"]`);
}

function setPreviewExpanded(targetId, expanded) {
  const target = document.querySelector(`#${targetId}`);
  const button = previewButton(targetId);
  if (!target || !button) return;

  const hasRows = !target.classList.contains("empty-state") && target.querySelector("table");
  button.disabled = !hasRows;
  if (!hasRows) {
    button.textContent = previewLabels[targetId]?.[0] || "Show";
    target.classList.remove("preview-collapsed");
    return;
  }

  target.classList.toggle("preview-collapsed", !expanded);
  button.textContent = previewLabels[targetId]?.[expanded ? 1 : 0] || (expanded ? "Hide" : "Show");
}

function renderPreviewTable(target, rows) {
  renderTable(target, rows);
  setPreviewExpanded(target.id, false);
}

function preprocessingConfig() {
  return {
    use_gc: document.querySelector("#use-gc").checked,
    use_kmers: document.querySelector("#use-kmers").checked,
    normalize_kmers: document.querySelector("#normalize-kmers").checked,
    use_one_hot: document.querySelector("#use-one-hot").checked,
    kmer_size: Number(document.querySelector("#kmer-size").value),
    sequence_length: Number(document.querySelector("#sequence-length").value),
  };
}

function updateDatasetState(dataset) {
  currentDataset = dataset;
  setText("#state-dataset", dataset.original_name);
  setText("#dataset-rows", dataset.row_count);
  setText("#dataset-cols", dataset.columns.length);
  setText("#dataset-format", dataset.source_format.replace(".", "").toUpperCase());
  setText("#preview-count", `${dataset.row_count} rows, ${dataset.columns.length} columns`);

  if (dataset.suggested_sequence_col) {
    document.querySelector("#sequence-col").value = dataset.suggested_sequence_col;
  }
  if (dataset.suggested_target_col) {
    document.querySelector("#target-col").value = dataset.suggested_target_col;
  }

  const summary = dataset.target_summary || {};
  const uploadAnalysis = document.querySelector("#upload-analysis");
  const guidance = document.querySelector("#benchmark-guidance");
  const localRunLimit = dataset.local_small_run_limit;
  const classText = summary.available
    ? `Target ${summary.target_col}: ${Object.entries(summary.class_counts || {})
        .map(([label, count]) => `${label}=${count}`)
        .join(", ")}. ${summary.recommendation}`
    : "No label/target column detected. Choose the target column before benchmarking.";
  const largeText = dataset.large_dataset_warning
    ? ` Local quick-run mode will use the first ${localRunLimit} rows unless you export the plan for Colab/HPC.`
    : "";
  const analysisMarkup = `<span>Dataset Analysis</span><strong>${summary.available ? "Labels detected" : "Needs target review"}</strong><p>${classText}${largeText}</p>`;
  uploadAnalysis.innerHTML = analysisMarkup;
  guidance.textContent = `${classText}${largeText} Default protocol: shared user/literature threshold, false positives treated as costly, 3 reruns, fixed materialized split.`;
  document.querySelector("#continue-preprocess").disabled = false;
  document.querySelector("#config-json-file").disabled = false;
  document.querySelector("#go-config-export").disabled = false;
  setText("#json-import-help", "Dataset is loaded. Import a saved BenchLab JSON now, or go directly to export settings.");
  updateBenchmarkPreview();

  if (summary.imbalance_detected) {
    showToast("Class imbalance detected. Review the class balancing option before running.");
  } else if (dataset.large_dataset_warning) {
    showToast(`Large dataset detected. Local quick runs use the first ${localRunLimit} rows by default.`);
  }
}

function numericValue(selector, fallback = null) {
  const value = document.querySelector(selector).value;
  if (value === "") return fallback;
  return Number(value);
}

function selectedValues(selector) {
  return [...document.querySelectorAll(selector)].filter((item) => item.checked).map((item) => item.value);
}

function benchmarkPayload() {
  return {
    dataset_id: currentDatasetId,
    sequence_col: document.querySelector("#sequence-col").value,
    target_col: document.querySelector("#target-col").value,
    models: selectedValues(".model-option"),
    comparison_models: selectedValues(".comparison-model-option"),
    preprocessing: preprocessingConfig(),
    test_size: Number(document.querySelector("#test-size").value),
    validation_size: Number(document.querySelector("#validation-size").value),
    random_seed: Number(document.querySelector("#random-seed").value),
    split_strategy: document.querySelector("#split-strategy").value,
    threshold_strategy: document.querySelector("#threshold-strategy").value,
    threshold_value: numericValue("#threshold-value"),
    threshold_scope: document.querySelector("#threshold-scope").value,
    biological_goal: document.querySelector("#biological-goal").value,
    balance_strategy: document.querySelector("#balance-strategy").value,
    max_rows: Number(document.querySelector("#max-rows").value),
    reruns: Number(document.querySelector("#reruns").value),
    cv_folds: Number(document.querySelector("#cv-folds").value),
    training_cycles: Number(document.querySelector("#training-cycles").value),
    early_stopping_patience: Number(document.querySelector("#early-stopping-patience").value),
  };
}

function renderPlanOutput(plan, prompt, runConfig = null) {
  currentPlan = { plan, codex_prompt: prompt, run_config: runConfig || currentRunConfig };
  const output = document.querySelector("#plan-output");
  output.value = `${JSON.stringify(plan, null, 2)}\n\n--- CODEX PROMPT ---\n${prompt}`;
  output.classList.remove("hidden");
  document.querySelector("#download-plan").disabled = false;
  document.querySelector("#copy-prompt").disabled = false;
}

function optionText(selector) {
  const node = document.querySelector(selector);
  return node.options[node.selectedIndex]?.textContent || node.value;
}

function updateBenchmarkPreview() {
  const payload = benchmarkPayload();
  const rows = currentDataset?.row_count;
  const capApplied = rows && rows > payload.max_rows;
  setText("#preview-data-scope", rows ? `${Math.min(rows, payload.max_rows)} of ${rows} rows` : "No dataset loaded");
  setText(
    "#preview-data-detail",
    rows
      ? `${capApplied ? "Local quick run is capped; export JSON for full Colab/HPC run." : "Local quick run can use the uploaded rows."} Balance: ${optionText("#balance-strategy")}.`
      : "Upload data to see local-run limits and class readiness."
  );
  setText("#preview-split-plan", optionText("#split-strategy"));
  setText("#preview-split-detail", `Test ${payload.test_size}, validation ${payload.validation_size}, seed ${payload.random_seed}, ${payload.cv_folds}-fold CV, ${payload.reruns} reruns, ${payload.training_cycles} cycles.`);
  setText("#preview-threshold-rule", optionText("#threshold-strategy"));
  setText("#preview-threshold-detail", `${optionText("#threshold-scope")}; goal: ${optionText("#biological-goal")}${payload.threshold_value !== null ? `; value ${payload.threshold_value}` : ""}.`);
  const comparison = payload.comparison_models.length ? payload.comparison_models.join(", ") : "No Colab/HPC models selected";
  const runnable = payload.models.length ? payload.models.join(", ") : "No local baselines selected";
  setText("#preview-model-targets", comparison);
  setText("#preview-model-detail", `Local small run: ${runnable}. Export JSON for reproducible notebooks.`);
}

function renderCapabilities(data) {
  if (data.source) {
    setText("#source-commit", `main ${data.source.verified_main_commit}, checked ${data.source.verified_on}`);
  }

  const capabilityList = document.querySelector("#capability-list");
  if (capabilityList) {
    capabilityList.innerHTML = "";
    (data.seqtrainer_functions || []).slice(0, 6).forEach((name) => {
      const item = document.createElement("div");
      item.textContent = name;
      capabilityList.append(item);
    });
  }
}

document.querySelectorAll(".nav-button").forEach((button) => {
  button.addEventListener("click", () => goToPanel(button.dataset.panel));
});

document.querySelectorAll("[data-next-panel]").forEach((button) => {
  button.addEventListener("click", () => goToPanel(button.dataset.nextPanel));
});

document.querySelectorAll("[data-mode-choice]").forEach((button) => {
  button.addEventListener("click", () => applyWorkflowMode(button.dataset.modeChoice, true));
});

const fileInputEl = document.querySelector("#dataset-file");
const fileInfoEl = document.querySelector("#file-info");

if (fileInputEl && fileInfoEl) {
  fileInputEl.addEventListener("change", () => {
    const file = fileInputEl.files && fileInputEl.files[0];
    fileInfoEl.textContent = file ? `${file.name} (${Math.max(1, Math.round(file.size / 1024))} KB)` : "No file chosen";
  });
}

async function checkHealth() {
  const dot = document.querySelector("#health-dot");
  const text = document.querySelector("#health-text");
  try {
    await api("/api/health");
    dot.classList.add("ok");
    text.textContent = "Service online";
  } catch {
    dot.classList.remove("ok");
    text.textContent = "Service unavailable";
  }
}

document.querySelector("#upload-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const fileInput = document.querySelector("#dataset-file");
  if (!fileInput.files.length) return showToast("No file selected.");
  const submitBtn = document.querySelector("#upload-form button[type='submit']");
  await withLoading(submitBtn, async () => {
    const body = new FormData();
    body.append("file", fileInput.files[0]);
    try {
      const data = await api("/api/datasets", { method: "POST", body });
      currentDatasetId = data.dataset.dataset_id;
      renderMeta(document.querySelector("#dataset-meta"), data.dataset);
      renderPreviewTable(document.querySelector("#dataset-preview"), data.preview);
      updateDatasetState(data.dataset);
      showToast("Dataset uploaded. Review the analysis before continuing.");
    } catch (error) {
      showToast(error.message);
    }
  });
});

document.querySelector("#continue-preprocess").addEventListener("click", () => {
  if (!currentDatasetId) {
    showToast("Upload a dataset first.");
    return;
  }
  goToPanel("preprocess");
});

document.querySelector("#continue-benchmark").addEventListener("click", () => {
  if (document.querySelector("#continue-benchmark").disabled) {
    showToast("Preview features first.");
    return;
  }
  goToPanel("benchmark");
});

document.querySelector("#continue-results").addEventListener("click", () => {
  if (!currentRunId) {
    showToast("Run a benchmark or export a run config first.");
    return;
  }
  goToPanel("results");
});

document.querySelector("#preprocess-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!currentDatasetId) {
    showToast("Upload a dataset first.");
    return;
  }
  const submitBtn = document.querySelector("#preprocess-form button[type='submit']");
  await withLoading(submitBtn, async () => {
    const body = new FormData();
    body.append("sequence_col", document.querySelector("#sequence-col").value);
    body.append("config", JSON.stringify(preprocessingConfig()));
    try {
      const data = await api(`/api/preprocess/${currentDatasetId}`, { method: "POST", body });
      const summary = `${data.row_count} rows, ${data.feature_count} features. Showing first ${data.columns.length} columns.`;
      document.querySelector("#feature-summary").textContent = summary;
      setText("#state-features", `${data.feature_count} features`);
      setText("#feature-count", `${data.feature_count} generated columns`);
      renderPreviewTable(document.querySelector("#feature-preview"), data.preview);
      document.querySelector("#continue-benchmark").disabled = false;
      showToast("Feature preview ready.");
      goToPanel("benchmark");
    } catch (error) {
      showToast(error.message);
    }
  });
});

document.querySelector("#benchmark-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!currentDatasetId) {
    showToast("Upload a dataset first.");
    return;
  }
  const submitBtn = document.querySelector("#benchmark-form button[type='submit']");
  await withLoading(submitBtn, async () => {
    const payload = benchmarkPayload();
    const models = payload.models;
    if (!models.length) {
      showToast("Choose at least one runnable model.");
      return;
    }
    if (currentDataset && currentDataset.row_count > payload.max_rows) {
      const proceed = window.confirm(`This local quick run will use only the first ${payload.max_rows} rows from ${currentDataset.row_count} uploaded rows. Export the JSON plan for Colab/HPC if you need the full dataset. Continue?`);
      if (!proceed) return;
    }
    try {
      const data = await api("/api/benchmarks", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      await showRunResult(data);
      showToast(data.dataset_removed ? "Benchmark complete. Uploaded dataset was deleted after run." : "Benchmark complete.");
      goToPanel("results");
    } catch (error) {
      showToast(error.message);
    }
  });
});

async function showRunResult(data) {
  currentRunId = data.run.run_id;
  currentRunConfig = data.run_config || null;
  currentPlan = data.benchmark_plan
    ? { plan: data.benchmark_plan, codex_prompt: data.codex_prompt || "", run_config: currentRunConfig }
    : currentPlan;
  renderMetrics(data.metrics || {});
  renderRunDetails(data);
  renderPreviewTable(document.querySelector("#predictions-table"), data.predictions || []);
  setExportLink(currentRunId);
  document.querySelector("#download-run-config").disabled = !currentRunConfig;
  setText("#state-run", currentRunId.slice(0, 8));
  document.querySelector("#continue-results").disabled = false;
  await loadRuns();
}

function renderMetrics(metrics) {
  const rows = Object.entries(metrics).map(([model, values]) => ({ model, ...values }));
  renderTable(document.querySelector("#metrics-table"), rows);
  renderMetricSummary(metrics);
}

function formatMetricValue(value) {
  if (typeof value !== "number") return value ?? "-";
  if (Math.abs(value) >= 100) return Math.round(value).toString();
  return Number(value.toFixed(4)).toString();
}

function renderMetricSummary(metrics) {
  const target = document.querySelector("#metrics-summary");
  target.innerHTML = "";
  const entries = Object.entries(metrics || {});
  if (!entries.length) {
    target.classList.add("empty-state");
    target.textContent = "No metrics yet.";
    return;
  }
  target.classList.remove("empty-state");

  entries.forEach(([model, values]) => {
    const classificationKeys = ["accuracy", "precision", "recall", "f1", "mcc"];
    const metricKeys = Object.keys(values).filter((key) => key !== "task_type");
    const priority = classificationKeys.filter((key) => key in values);
    const keys = [...priority, ...metricKeys.filter((key) => !priority.includes(key))];
    const card = document.createElement("article");
    card.className = "metric-card";
    const title = document.createElement("div");
    title.className = "metric-card-title";
    title.innerHTML = `<strong>${model.replaceAll("_", " ")}</strong><span>${values.task_type || "benchmark"}</span>`;
    const grid = document.createElement("div");
    grid.className = "metric-pill-grid";
    keys.forEach((key) => {
      const pill = document.createElement("div");
      pill.className = "metric-pill";
      pill.innerHTML = `<span>${key.replaceAll("_", " ")}</span><strong>${formatMetricValue(values[key])}</strong>`;
      grid.append(pill);
    });
    card.append(title, grid);
    target.append(card);
  });
}

function renderRunDetails(payload) {
  const run = payload.run || {};
  const dataset = payload.dataset || {};
  const training = payload.training_config || {};
  const preprocessing = payload.preprocessing_config || {};
  const environment = payload.environment || {};
  const targetKind = run.target_kind === "binary_numeric_label" ? "Binary 0/1 label" : "Numeric regression target";
  const cleanup = run.source_dataset_removed_after_run || payload.dataset_removed ? "Uploaded source file deleted after run" : "Uploaded source file retained";
  const preprocessingText =
    Object.entries(preprocessing)
      .filter(([, value]) => value !== false && value !== null && value !== undefined)
      .map(([key, value]) => `${key}=${value}`)
      .join(", ") || "none";

  renderMeta(document.querySelector("#run-details"), {
    run_id: run.run_id || currentRunId || "none",
    run_mode: run.run_mode || "local_benchmark",
    created_at: run.created_at ? new Date(run.created_at).toLocaleString() : "unknown",
    completed_at: run.completed_at ? new Date(run.completed_at).toLocaleString() : "unknown",
    elapsed_seconds: run.elapsed_seconds ?? "-",
    dataset: dataset.original_name || run.dataset_id || "unknown",
    sequence_column: training.sequence_col || "unknown",
    target_column: training.target_col || "unknown",
    target_kind: targetKind,
    models: training.models || [],
    split: `${training.train_rows ?? run.train_rows ?? "-"} train / ${training.test_rows ?? run.test_rows ?? "-"} test`,
    rows_used: training.rows_used ?? run.rows_used ?? "-",
    local_row_limit: training.local_row_limit ?? "-",
    row_cap_applied: training.row_cap_applied ? "yes" : "no",
    class_balance: training.class_balance_applied ? `${training.class_balance_strategy} applied` : training.class_balance_strategy || "none",
    threshold: training.classification_threshold ?? training.threshold_strategy ?? "not used",
    random_seed: training.random_seed ?? "-",
    reruns: training.reruns ?? "-",
    cv_folds: training.cv_folds ?? "-",
    training_cycles: training.training_cycles ?? "-",
    early_stopping_patience: training.early_stopping_patience ?? "-",
    python_version: environment.python_version || "-",
    preprocessing: preprocessingText,
    data_cleanup: cleanup,
  });
  renderExportSummary(payload, preprocessingText);
}

function selectedMetricNames(metrics) {
  const names = new Set();
  Object.values(metrics || {}).forEach((values) => {
    Object.keys(values || {}).forEach((key) => {
      if (key !== "task_type") names.add(key);
    });
  });
  return [...names].sort();
}

function renderExportSummary(payload, preprocessingText) {
  const run = payload.run || {};
  const dataset = payload.dataset || {};
  const training = payload.training_config || {};
  const metrics = payload.metrics || {};
  const environment = payload.environment || {};
  const runConfig = payload.run_config || {};
  const packages = environment.packages || {};
  const dependencies = runConfig.dependencies || {};
  const artifacts = [...new Set([
    ...(run.artifact_paths ? Object.values(run.artifact_paths) : []),
    ...(payload.replay_artifacts || []),
  ])];
  const displayedArtifacts = artifacts.length ? artifacts : [
    "run_config.json",
    "metrics.json",
    "predictions.csv",
    "run_manifest.json",
    "dataset_manifest.json",
    "preprocessing_config.json",
    "training_config.json",
    "benchmark_plan.json",
    "environment.json",
  ];

  renderMeta(document.querySelector("#export-summary"), {
    selected_models: training.models || [],
    selected_metrics: selectedMetricNames(metrics),
    preprocessing: preprocessingText || "none",
    split_and_seed: `${training.split_strategy || "-"}; seed ${training.random_seed ?? "-"}`,
    threshold_policy: `${training.threshold_strategy || "-"}; ${training.threshold_scope || "-"}`,
    rows_and_timing: `${training.rows_used ?? run.rows_used ?? "-"} rows; ${run.elapsed_seconds ?? "-"} seconds`,
    python_version: environment.python_version || dependencies.python_version || "-",
    docker_base_image: dependencies.docker_base_image || "-",
    dataset_sha256: dataset.sha256 || run.dataset_sha256 || "-",
    reproducibility_status: payload.reproducibility_status || "-",
    package_versions: Object.entries(packages).map(([name, version]) => `${name} ${version}`),
    export_artifacts: displayedArtifacts,
  });
}

function setExportLink(runId) {
  const link = document.querySelector("#export-link");
  link.href = `/api/runs/${runId}/export`;
  link.classList.remove("hidden");
}

async function loadRuns() {
  const data = await api("/api/runs");
  const list = document.querySelector("#run-list");
  list.innerHTML = "";
  if (!data.runs.length) {
    list.classList.add("empty-state");
    list.textContent = "No runs yet.";
    return;
  }
  list.classList.remove("empty-state");
  data.runs.slice().reverse().forEach((run) => {
    const button = document.createElement("button");
    button.className = "run-item";
    button.innerHTML = `<strong>${run.run_id.slice(0, 8)}</strong><span>${new Date(run.created_at).toLocaleString()}</span>`;
    button.addEventListener("click", () => loadRun(run.run_id));
    list.append(button);
  });
}

async function loadRun(runId) {
  const data = await api(`/api/runs/${runId}`);
  currentRunId = runId;
  renderMetrics(data.metrics);
  renderRunDetails(data);
  renderPreviewTable(document.querySelector("#predictions-table"), data.predictions);
  setExportLink(runId);
  currentRunConfig = data.run_config && Object.keys(data.run_config).length ? data.run_config : null;
  document.querySelector("#download-run-config").disabled = !currentRunConfig;
  setText("#state-run", runId.slice(0, 8));
}

document.querySelector("#refresh-runs").addEventListener("click", () => {
  loadRuns().catch((error) => showToast(error.message));
});

document.querySelector("#generate-plan").addEventListener("click", async () => {
  if (!currentDatasetId) {
    showToast("Upload a dataset first.");
    return;
  }
  try {
    const data = await api("/api/benchmark-plan", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(benchmarkPayload()),
    });
    renderPlanOutput(data.plan, data.codex_prompt);
    showToast("Benchmark JSON and Codex prompt generated.");
  } catch (error) {
    showToast(error.message);
  }
});

document.querySelector("#go-config-export").addEventListener("click", () => {
  if (!currentDatasetId) {
    showToast("Upload a dataset first.");
    return;
  }
  updateBenchmarkPreview();
  goToPanel("benchmark");
  showToast("Review settings, then export the run config without training.");
});

document.querySelector("#export-run-config").addEventListener("click", async () => {
  if (!currentDatasetId) {
    showToast("Upload a dataset first.");
    return;
  }
  const button = document.querySelector("#export-run-config");
  await withLoading(button, async () => {
    try {
      const data = await api("/api/run-configs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(benchmarkPayload()),
      });
      await showRunResult(data);
      renderPlanOutput(data.benchmark_plan, data.codex_prompt, data.run_config);
      showToast(data.dataset_removed ? "Run config export ready. Uploaded dataset was deleted." : "Run config export ready.");
      goToPanel("results");
    } catch (error) {
      showToast(error.message);
    }
  });
});

document.querySelector("#download-plan").addEventListener("click", () => {
  if (!currentPlan) return;
  const blob = new Blob([JSON.stringify(currentPlan.plan, null, 2)], { type: "application/json" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = "seqtrainer-benchmark-plan.json";
  link.click();
  URL.revokeObjectURL(link.href);
});

document.querySelector("#download-run-config").addEventListener("click", () => {
  const runConfig = currentRunConfig || currentPlan?.run_config;
  if (!runConfig) {
    showToast("No Run Config JSON is available for this run.");
    return;
  }
  const blob = new Blob([JSON.stringify(runConfig, null, 2)], { type: "application/json" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = "run_config.json";
  link.click();
  URL.revokeObjectURL(link.href);
});

document.querySelector("#copy-prompt").addEventListener("click", async () => {
  if (!currentPlan) return;
  await navigator.clipboard.writeText(currentPlan.codex_prompt);
  showToast("Codex prompt copied.");
});

function setIfPresent(selector, value) {
  const node = document.querySelector(selector);
  if (node && value !== undefined && value !== null) node.value = value;
}

function applyImportedConfig(plan) {
  const columns = plan.columns || {};
  const split = plan.split || {};
  const threshold = plan.threshold || {};
  const balance = plan.class_balance || {};
  const training = plan.training || {};
  setIfPresent("#sequence-col", columns.sequence_col);
  setIfPresent("#target-col", columns.target_col);
  setIfPresent("#split-strategy", split.strategy);
  setIfPresent("#test-size", split.test_size);
  setIfPresent("#validation-size", split.validation_size);
  setIfPresent("#random-seed", split.random_seed);
  setIfPresent("#cv-folds", split.cv_folds);
  setIfPresent("#reruns", split.reruns);
  setIfPresent("#threshold-strategy", threshold.strategy);
  if (typeof threshold.value === "number") setIfPresent("#threshold-value", threshold.value);
  setIfPresent("#threshold-scope", threshold.scope);
  setIfPresent("#biological-goal", threshold.biological_goal);
  setIfPresent("#balance-strategy", balance.strategy);
  setIfPresent("#training-cycles", training.training_cycles);
  setIfPresent("#early-stopping-patience", training.early_stopping_patience);
  updateBenchmarkPreview();
}

function applyImportedRunConfig(config) {
  const dataset = config.dataset || {};
  const split = config.split || {};
  const preprocessing = config.preprocessing || {};
  const threshold = config.threshold || config.training || {};
  const balance = config.balance || config.training || {};
  const training = config.training || {};
  const modelSelection = config.model_selection || {};

  setIfPresent("#sequence-col", dataset.sequence_column);
  setIfPresent("#target-col", dataset.target_column);
  setIfPresent("#split-strategy", split.split_strategy);
  setIfPresent("#test-size", split.test_size);
  setIfPresent("#validation-size", split.validation_size);
  setIfPresent("#random-seed", split.random_seed);
  setIfPresent("#cv-folds", split.cv_folds);
  setIfPresent("#reruns", split.reruns);
  setIfPresent("#threshold-strategy", threshold.threshold_strategy);
  if (typeof threshold.threshold_value === "number") setIfPresent("#threshold-value", threshold.threshold_value);
  setIfPresent("#threshold-scope", threshold.threshold_scope);
  setIfPresent("#biological-goal", threshold.biological_goal);
  setIfPresent("#balance-strategy", balance.balance_strategy);
  setIfPresent("#max-rows", balance.local_row_limit || training.local_row_limit);
  setIfPresent("#training-cycles", training.cycles || training.training_cycles);
  setIfPresent("#early-stopping-patience", training.early_stopping_patience);
  setIfPresent("#kmer-size", preprocessing.kmer_size);
  setIfPresent("#sequence-length", preprocessing.sequence_length);
  document.querySelector("#use-gc").checked = preprocessing.use_gc ?? true;
  document.querySelector("#use-kmers").checked = preprocessing.use_kmers ?? true;
  document.querySelector("#normalize-kmers").checked = preprocessing.normalize_kmers ?? true;
  document.querySelector("#use-one-hot").checked = preprocessing.use_one_hot ?? false;

  const localModels = modelSelection.local_models || (config.models || []).map((model) => model.model_name);
  document.querySelectorAll(".model-option").forEach((node) => {
    node.checked = localModels.includes(node.value);
  });
  const comparisonModels = modelSelection.comparison_models || [];
  document.querySelectorAll(".comparison-model-option").forEach((node) => {
    node.checked = comparisonModels.includes(node.value);
  });
  updateBenchmarkPreview();
}

document.querySelector("#config-json-file").addEventListener("change", async (event) => {
  const file = event.target.files && event.target.files[0];
  if (!file) return;
  if (!currentDatasetId) {
    showToast("Upload the matching dataset before importing JSON.");
    event.target.value = "";
    return;
  }
  try {
    const imported = JSON.parse(await file.text());
    if (imported.schema_version && imported.dataset && imported.split && imported.preprocessing) {
      applyImportedRunConfig(imported);
      showToast("Run Config JSON imported.");
    } else {
      applyImportedConfig(imported);
      showToast("Benchmark plan JSON imported.");
    }
    const output = document.querySelector("#plan-output");
    output.value = JSON.stringify(imported, null, 2);
    output.classList.remove("hidden");
    document.querySelector("#continue-preprocess").disabled = false;
    updateBenchmarkPreview();
    goToPanel("benchmark");
  } catch (error) {
    showToast(`Could not import JSON: ${error.message}`);
  }
});

document.querySelectorAll(".preview-toggle").forEach((button) => {
  setPreviewExpanded(button.dataset.previewTarget, true);
  button.addEventListener("click", () => {
    const target = document.querySelector(`#${button.dataset.previewTarget}`);
    if (!target || target.classList.contains("empty-state")) return;
    setPreviewExpanded(button.dataset.previewTarget, target.classList.contains("preview-collapsed"));
  });
});

document
  .querySelectorAll(
    "#benchmark-form input, #benchmark-form select, #benchmark-form textarea"
  )
  .forEach((node) => {
    node.addEventListener("input", updateBenchmarkPreview);
    node.addEventListener("change", updateBenchmarkPreview);
  });

document.querySelector("#email-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!currentRunId) {
    showToast("Run a benchmark or select a run first.");
    return;
  }
  const email = document.querySelector("#email-address").value.trim();
  if (!email) {
    showToast("Enter an email address.");
    return;
  }
  const submitBtn = document.querySelector("#email-form button[type='submit']");
  await withLoading(submitBtn, async () => {
    try {
      const data = await api(`/api/runs/${currentRunId}/email`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email }),
      });
      if (data.mode === "mailto" && data.mailto) {
        window.location.href = data.mailto;
      }
      showToast(data.message || "Email prepared.");
    } catch (error) {
      showToast(error.message);
    }
  });
});

checkHealth();
api("/api/capabilities").then(renderCapabilities).catch(() => {});
loadRuns().catch(() => {});
applyWorkflowMode(currentMode, false);
updateBenchmarkPreview();
