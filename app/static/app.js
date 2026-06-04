let currentDatasetId = null;
let currentRunId = null;

const toast = document.querySelector("#toast");

function showToast(message) {
  toast.textContent = message;
  toast.classList.remove("hidden");
  setTimeout(() => toast.classList.add("hidden"), 4200);
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

function renderMeta(target, payload) {
  target.innerHTML = "";
  Object.entries(payload).forEach(([key, value]) => {
    const dt = document.createElement("dt");
    const dd = document.createElement("dd");
    dt.textContent = key;
    dd.textContent = Array.isArray(value) ? value.join(", ") : String(value);
    target.append(dt, dd);
  });
}

function renderTable(target, rows) {
  if (!rows || rows.length === 0) {
    target.innerHTML = "<p class=\"empty\">No rows.</p>";
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
      td.textContent = row[column] ?? "";
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

document.querySelectorAll(".nav-button").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".nav-button").forEach((item) => item.classList.remove("active"));
    document.querySelectorAll(".panel").forEach((panel) => panel.classList.remove("visible"));
    button.classList.add("active");
    document.querySelector(`#panel-${button.dataset.panel}`).classList.add("visible");
  });
});

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
  if (!fileInput.files.length) return;
  const body = new FormData();
  body.append("file", fileInput.files[0]);
  try {
    const data = await api("/api/datasets", { method: "POST", body });
    currentDatasetId = data.dataset.dataset_id;
    renderMeta(document.querySelector("#dataset-meta"), data.dataset);
    renderTable(document.querySelector("#dataset-preview"), data.preview);
    showToast("Dataset uploaded.");
  } catch (error) {
    showToast(error.message);
  }
});

document.querySelector("#preprocess-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!currentDatasetId) {
    showToast("Upload a dataset first.");
    return;
  }
  const body = new FormData();
  body.append("sequence_col", document.querySelector("#sequence-col").value);
  body.append("config", JSON.stringify(preprocessingConfig()));
  try {
    const data = await api(`/api/preprocess/${currentDatasetId}`, { method: "POST", body });
    document.querySelector("#feature-summary").textContent =
      `${data.row_count} rows, ${data.feature_count} features. Showing first ${data.columns.length} columns.`;
    renderTable(document.querySelector("#feature-preview"), data.preview);
    showToast("Feature preview ready.");
  } catch (error) {
    showToast(error.message);
  }
});

document.querySelector("#benchmark-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!currentDatasetId) {
    showToast("Upload a dataset first.");
    return;
  }
  const models = [...document.querySelectorAll(".model-option:checked")].map((item) => item.value);
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
    renderTable(document.querySelector("#predictions-table"), data.predictions);
    setExportLink(currentRunId);
    await loadRuns();
    showToast("Benchmark complete.");
  } catch (error) {
    showToast(error.message);
  }
});

function renderMetrics(metrics) {
  const rows = Object.entries(metrics).map(([model, values]) => ({ model, ...values }));
  renderTable(document.querySelector("#metrics-table"), rows);
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
  data.runs.slice().reverse().forEach((run) => {
    const button = document.createElement("button");
    button.className = "run-item";
    button.textContent = `${run.created_at} | ${run.run_id}`;
    button.addEventListener("click", () => loadRun(run.run_id));
    list.append(button);
  });
}

async function loadRun(runId) {
  const data = await api(`/api/runs/${runId}`);
  currentRunId = runId;
  renderMetrics(data.metrics);
  renderTable(document.querySelector("#predictions-table"), data.predictions);
  setExportLink(runId);
}

document.querySelector("#refresh-runs").addEventListener("click", () => {
  loadRuns().catch((error) => showToast(error.message));
});

checkHealth();
loadRuns().catch(() => {});
