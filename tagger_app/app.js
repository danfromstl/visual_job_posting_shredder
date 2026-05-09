const fileSelect = document.querySelector("#fileSelect");
const searchInput = document.querySelector("#searchInput");
const filterSelect = document.querySelector("#filterSelect");
const statusText = document.querySelector("#statusText");
const countText = document.querySelector("#countText");
const saveText = document.querySelector("#saveText");
const snippetList = document.querySelector("#snippetList");
const detailCategory = document.querySelector("#detailCategory");
const detailTitle = document.querySelector("#detailTitle");
const detailCompany = document.querySelector("#detailCompany");
const detailLine = document.querySelector("#detailLine");
const detailJob = document.querySelector("#detailJob");
const detailId = document.querySelector("#detailId");
const detailText = document.querySelector("#detailText");
const tagIndicator = document.querySelector("#tagIndicator");
const markButton = document.querySelector("#markButton");
const clearButton = document.querySelector("#clearButton");

const state = {
  files: [],
  sourceFile: "",
  rows: [],
  visibleRows: [],
  selectedRowIndex: -1,
  search: "",
  filter: "all",
  saving: false,
};

function text(value, fallback = "-") {
  if (value === null || value === undefined || value === "") {
    return fallback;
  }
  return String(value);
}

function setStatus(message) {
  statusText.textContent = message;
}

function setSaveState(message) {
  saveText.textContent = message;
}

function escapeLower(value) {
  return text(value, "").toLowerCase();
}

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.error || response.statusText);
  }
  return payload;
}

async function loadFiles() {
  setStatus("Loading sources");
  const payload = await fetchJson("/api/files");
  state.files = payload.files || [];
  fileSelect.replaceChildren();

  for (const file of state.files) {
    const option = document.createElement("option");
    option.value = file.name;
    option.textContent = `${file.name} (${file.line_count})`;
    fileSelect.append(option);
  }

  const savedFile = localStorage.getItem("tagger.sourceFile");
  const firstFile = state.files[0]?.name || "";
  state.sourceFile = state.files.some((file) => file.name === savedFile) ? savedFile : firstFile;
  fileSelect.value = state.sourceFile;
}

async function loadSnippets(sourceFile) {
  if (!sourceFile) {
    state.rows = [];
    render();
    setStatus("No JSONL files found");
    return;
  }

  setStatus("Loading snippets");
  setSaveState("Idle");
  const payload = await fetchJson(`/api/snippets?file=${encodeURIComponent(sourceFile)}`);
  state.sourceFile = sourceFile;
  state.rows = (payload.rows || []).map((row, index) => ({
    ...row,
    _rowIndex: index,
    _not_relevant: Boolean(row._not_relevant),
  }));
  state.selectedRowIndex = state.rows.length ? 0 : -1;
  localStorage.setItem("tagger.sourceFile", sourceFile);
  render();
  setStatus(`${sourceFile}`);
  if (payload.errors?.length) {
    setSaveState(`${payload.errors.length} skipped`);
  }
}

function rowMatchesSearch(row) {
  if (!state.search) {
    return true;
  }
  const haystack = [
    row.text,
    row.company,
    row.title,
    row.job_name,
    row.category,
    row.id,
  ]
    .map((item) => escapeLower(item))
    .join(" ");
  return haystack.includes(state.search);
}

function rowMatchesFilter(row) {
  if (state.filter === "notRelevant") {
    return row._not_relevant;
  }
  if (state.filter === "untagged") {
    return !row._not_relevant;
  }
  return true;
}

function getVisibleRows() {
  return state.rows.filter((row) => rowMatchesFilter(row) && rowMatchesSearch(row));
}

function render() {
  state.visibleRows = getVisibleRows();
  if (
    state.selectedRowIndex < 0 ||
    !state.visibleRows.some((row) => row._rowIndex === state.selectedRowIndex)
  ) {
    state.selectedRowIndex = state.visibleRows[0]?._rowIndex ?? -1;
  }

  renderList();
  renderDetail();
  renderCounts();
}

function renderCounts() {
  const tagged = state.rows.filter((row) => row._not_relevant).length;
  countText.textContent = `${state.visibleRows.length} shown / ${state.rows.length} rows`;
  setStatus(`${tagged} not relevant`);
}

function renderList() {
  snippetList.replaceChildren();
  if (!state.visibleRows.length) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.textContent = "No rows";
    snippetList.append(empty);
    return;
  }

  const fragment = document.createDocumentFragment();
  for (const row of state.visibleRows) {
    const item = document.createElement("div");
    item.className = "snippet-row";
    if (row._not_relevant) {
      item.classList.add("is-not-relevant");
    }
    if (row._rowIndex === state.selectedRowIndex) {
      item.classList.add("is-active");
    }
    item.dataset.rowIndex = String(row._rowIndex);

    const meta = document.createElement("div");
    meta.className = "row-meta";

    const rowNumber = document.createElement("span");
    rowNumber.className = "row-number";
    rowNumber.textContent = `#${text(row._line_number)}`;
    meta.append(rowNumber, document.createTextNode(text(row.category, "snippet")));

    const main = document.createElement("div");
    main.className = "snippet-main";

    const title = document.createElement("div");
    title.className = "snippet-title";
    title.textContent = [row.company, row.title].filter(Boolean).join(" - ") || text(row.job_name);

    const body = document.createElement("div");
    body.className = "snippet-text";
    body.textContent = text(row.text, "");

    main.append(title, body);
    item.append(meta, main);
    fragment.append(item);
  }

  snippetList.append(fragment);
  scrollActiveIntoView(false);
}

function renderDetail() {
  const row = selectedRow();
  if (!row) {
    detailCategory.textContent = "No row selected";
    detailTitle.textContent = "-";
    detailCompany.textContent = "-";
    detailLine.textContent = "-";
    detailJob.textContent = "-";
    detailId.textContent = "-";
    detailText.textContent = "";
    tagIndicator.textContent = "Untagged";
    tagIndicator.classList.remove("is-not-relevant");
    return;
  }

  detailCategory.textContent = text(row.category, "snippet");
  detailTitle.textContent = text(row.title || row.job_name);
  detailCompany.textContent = text(row.company);
  detailLine.textContent = text(row._line_number);
  detailJob.textContent = text(row.job_key || row.job_name);
  detailId.textContent = text(row.id);
  detailText.textContent = text(row.text, "");
  tagIndicator.textContent = row._not_relevant ? "Not relevant" : "Untagged";
  tagIndicator.classList.toggle("is-not-relevant", row._not_relevant);
}

function selectedRow() {
  return state.rows[state.selectedRowIndex] || null;
}

function visiblePosition() {
  return state.visibleRows.findIndex((row) => row._rowIndex === state.selectedRowIndex);
}

function selectVisiblePosition(position) {
  if (!state.visibleRows.length) {
    state.selectedRowIndex = -1;
    renderDetail();
    return;
  }

  const bounded = Math.max(0, Math.min(position, state.visibleRows.length - 1));
  state.selectedRowIndex = state.visibleRows[bounded]._rowIndex;
  updateActiveRow();
  renderDetail();
  scrollActiveIntoView(true);
}

function selectRowIndex(rowIndex) {
  state.selectedRowIndex = rowIndex;
  updateActiveRow();
  renderDetail();
  scrollActiveIntoView(true);
}

function updateActiveRow() {
  for (const node of snippetList.querySelectorAll(".snippet-row")) {
    node.classList.toggle("is-active", Number(node.dataset.rowIndex) === state.selectedRowIndex);
  }
}

function updateRowTagClass(row) {
  const node = snippetList.querySelector(`.snippet-row[data-row-index="${row._rowIndex}"]`);
  if (node) {
    node.classList.toggle("is-not-relevant", row._not_relevant);
  }
}

function scrollActiveIntoView(smooth) {
  const node = snippetList.querySelector(".snippet-row.is-active");
  if (!node) {
    return;
  }
  node.scrollIntoView({ block: "nearest", behavior: smooth ? "smooth" : "auto" });
}

async function setNotRelevant(row, value, advance) {
  if (!row || state.saving) {
    return;
  }
  if (row._not_relevant === value && advance) {
    moveSelection(1);
    return;
  }

  const previousVisiblePosition = visiblePosition();
  const previous = row._not_relevant;
  row._not_relevant = value;
  state.saving = true;
  updateRowTagClass(row);
  renderDetail();
  renderCounts();
  setSaveState("Saving");

  try {
    const payload = await fetchJson("/api/tags", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source_file: state.sourceFile,
        id: row.id,
        not_relevant: value,
        job_key: row.job_key,
        job_name: row.job_name,
        company: row.company,
        title: row.title,
        category: row.category,
        text: row.text,
      }),
    });
    setSaveState(`${payload.count} saved`);
    if (advance) {
      const rowStillVisible = rowMatchesFilter(row) && rowMatchesSearch(row);
      state.visibleRows = getVisibleRows();
      if (state.visibleRows.length) {
        const nextPosition = rowStillVisible ? previousVisiblePosition + 1 : previousVisiblePosition;
        const bounded = Math.max(0, Math.min(nextPosition, state.visibleRows.length - 1));
        state.selectedRowIndex = state.visibleRows[bounded]._rowIndex;
      }
      render();
    } else if (!rowMatchesFilter(row) || !rowMatchesSearch(row)) {
      render();
    }
  } catch (error) {
    row._not_relevant = previous;
    updateRowTagClass(row);
    renderDetail();
    renderCounts();
    setSaveState("Save failed");
    alert(error.message);
  } finally {
    state.saving = false;
  }
}

function moveSelection(delta) {
  const current = visiblePosition();
  if (current === -1) {
    selectVisiblePosition(0);
    return;
  }
  selectVisiblePosition(current + delta);
}

function currentTargetAllowsShortcut(event) {
  const target = event.target;
  if (!(target instanceof HTMLElement)) {
    return true;
  }
  return !["INPUT", "SELECT", "TEXTAREA", "BUTTON"].includes(target.tagName);
}

fileSelect.addEventListener("change", () => {
  loadSnippets(fileSelect.value).catch((error) => {
    setStatus("Load failed");
    alert(error.message);
  });
});

searchInput.addEventListener("input", () => {
  state.search = searchInput.value.trim().toLowerCase();
  render();
});

filterSelect.addEventListener("change", () => {
  state.filter = filterSelect.value;
  render();
});

snippetList.addEventListener("click", (event) => {
  const target = event.target instanceof Element ? event.target : null;
  const rowNode = target?.closest(".snippet-row");
  if (!rowNode) {
    return;
  }
  selectRowIndex(Number(rowNode.dataset.rowIndex));
  snippetList.focus();
});

markButton.addEventListener("click", () => {
  setNotRelevant(selectedRow(), true, false);
});

clearButton.addEventListener("click", () => {
  setNotRelevant(selectedRow(), false, false);
});

document.addEventListener("keydown", (event) => {
  if (!currentTargetAllowsShortcut(event)) {
    return;
  }

  const key = event.key.toLowerCase();
  if (key === "arrowdown" || key === "j") {
    event.preventDefault();
    moveSelection(1);
  } else if (key === "arrowup" || key === "k") {
    event.preventDefault();
    moveSelection(-1);
  } else if (key === "r") {
    event.preventDefault();
    setNotRelevant(selectedRow(), true, true);
  } else if (key === "u") {
    event.preventDefault();
    setNotRelevant(selectedRow(), false, true);
  } else if (key === " ") {
    event.preventDefault();
    const row = selectedRow();
    setNotRelevant(row, row ? !row._not_relevant : false, true);
  }
});

loadFiles()
  .then(() => loadSnippets(state.sourceFile))
  .catch((error) => {
    setStatus("Startup failed");
    setSaveState("Error");
    alert(error.message);
  });
