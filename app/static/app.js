let currentDatasetId = null;
let currentRunId = null;

const toast = document.querySelector("#toast");

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
      renderTable(document.querySelector("#dataset-preview"), data.preview);
      updateDatasetState(data.dataset);
      showToast("Dataset uploaded.");
      goToPanel("preprocess");
    } catch (error) {
      showToast(error.message);
    }
  });
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
      renderTable(document.querySelector("#feature-preview"), data.preview);
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
    const models = [...document.querySelectorAll(".model-option:checked")].map((item) => item.value);
    if (!models.length) {
      showToast("Choose at least one runnable model.");
      return;
    }
    const payload = {
      dataset_id: currentDatasetId,
      sequence_col: document.querySelector("#sequence-col").value,
      target_col: document.querySelector("#target-col").value,
      models,
      preprocessing: preprocessingConfig(),
      test_size: Number(document.querySelector("#test-size").value),
      random_seed: Number(document.querySelector("#random-seed").value),
    };
    try {
      const data = await api("/api/benchmarks", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      currentRunId = data.run.run_id;
      renderMetrics(data.metrics);
      renderRunDetails(data);
      renderTable(document.querySelector("#predictions-table"), data.predictions);
      setExportLink(currentRunId);
      setText("#state-run", currentRunId.slice(0, 8));
      await loadRuns();
      showToast(data.dataset_removed ? "Benchmark complete. Uploaded dataset was deleted after run." : "Benchmark complete.");
      goToPanel("results");
    } catch (error) {
      showToast(error.message);
    }
  });
});

function renderMetrics(metrics) {
  const rows = Object.entries(metrics).map(([model, values]) => ({ model, ...values }));
  renderTable(document.querySelector("#metrics-table"), rows);
}

function renderRunDetails(payload) {
  const run = payload.run || {};
  const dataset = payload.dataset || {};
  const training = payload.training_config || {};
  const preprocessing = payload.preprocessing_config || {};
  const targetKind = run.target_kind === "binary_numeric_label" ? "Binary 0/1 label" : "Numeric regression target";
  const cleanup = run.source_dataset_removed_after_run || payload.dataset_removed ? "Uploaded source file deleted after run" : "Uploaded source file retained";
  const preprocessingText =
    Object.entries(preprocessing)
      .filter(([, value]) => value !== false && value !== null && value !== undefined)
      .map(([key, value]) => `${key}=${value}`)
      .join(", ") || "none";

  renderMeta(document.querySelector("#run-details"), {
    run_id: run.run_id || currentRunId || "none",
    created_at: run.created_at ? new Date(run.created_at).toLocaleString() : "unknown",
    dataset: dataset.original_name || run.dataset_id || "unknown",
    sequence_column: training.sequence_col || "unknown",
    target_column: training.target_col || "unknown",
    target_kind: targetKind,
    models: training.models || [],
    split: `${training.train_rows ?? run.train_rows ?? "-"} train / ${training.test_rows ?? run.test_rows ?? "-"} test`,
    rows_used: training.rows_used ?? run.rows_used ?? "-",
    random_seed: training.random_seed ?? "-",
    preprocessing: preprocessingText,
    data_cleanup: cleanup,
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
  renderTable(document.querySelector("#predictions-table"), data.predictions);
  setExportLink(runId);
  setText("#state-run", runId.slice(0, 8));
}

document.querySelector("#refresh-runs").addEventListener("click", () => {
  loadRuns().catch((error) => showToast(error.message));
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
