const fileSelect = document.querySelector("#fileSelect");
const searchInput = document.querySelector("#searchInput");
const filterSelect = document.querySelector("#filterSelect");
const statusText = document.querySelector("#statusText");
const countText = document.querySelector("#countText");
const saveText = document.querySelector("#saveText");
const scoreText = document.querySelector("#scoreText");
const snippetList = document.querySelector("#snippetList");
const prevPostButton = document.querySelector("#prevPostButton");
const nextPostButton = document.querySelector("#nextPostButton");
const postingTitle = document.querySelector("#postingTitle");
const postingMeta = document.querySelector("#postingMeta");
const detailCategory = document.querySelector("#detailCategory");
const detailTitle = document.querySelector("#detailTitle");
const detailCompany = document.querySelector("#detailCompany");
const detailLine = document.querySelector("#detailLine");
const detailJob = document.querySelector("#detailJob");
const detailId = document.querySelector("#detailId");
const detailText = document.querySelector("#detailText");
const tagIndicator = document.querySelector("#tagIndicator");
const suggestionIndicator = document.querySelector("#suggestionIndicator");
const markButton = document.querySelector("#markButton");
const clearButton = document.querySelector("#clearButton");
const scoringDialog = document.querySelector("#scoringDialog");
const scoringTitle = document.querySelector("#scoringTitle");
const scoringDetail = document.querySelector("#scoringDetail");
const scoringProgress = document.querySelector("#scoringProgress");

const state = {
  files: [],
  sourceFile: "",
  rows: [],
  postings: [],
  postingIndex: 0,
  visibleRows: [],
  selectedRowIndex: -1,
  search: "",
  filter: "all",
  reviewing: false,
  navigating: false,
  scoringDialogOpen: false,
  pendingTagSaves: new Set(),
  recommendationRequest: 0,
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

function setScoreState(message) {
  scoreText.textContent = message;
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
    state.postings = [];
    render();
    setStatus("No JSONL files found");
    return;
  }

  setStatus("Loading snippets");
  setSaveState("Idle");
  setScoreState("Scoring idle");
  const payload = await fetchJson(`/api/snippets?file=${encodeURIComponent(sourceFile)}`);
  state.sourceFile = sourceFile;
  state.rows = (payload.rows || []).map((row, index) => ({
    ...row,
    _rowIndex: index,
    _postingKey: text(row._posting_key || row.job_key || row.job_id || row.job_name, `posting-${index}`),
    _not_relevant: Boolean(row._not_relevant),
    _reviewed_relevant: Boolean(row._reviewed_relevant),
    _recommendation: null,
    _saveState: "idle",
    _saveVersion: 0,
  }));
  buildPostings();
  restorePostingPosition();
  selectFirstVisibleRowInPosting();
  localStorage.setItem("tagger.sourceFile", sourceFile);
  render();
  loadRecommendationsForCurrentPosting();
  if (payload.errors?.length) {
    setSaveState(`${payload.errors.length} skipped`);
  }
}

function buildPostings() {
  const groups = new Map();
  for (const row of state.rows) {
    if (!groups.has(row._postingKey)) {
      groups.set(row._postingKey, {
        key: row._postingKey,
        company: row.company,
        title: row.title,
        jobName: row.job_name,
        rowIndexes: [],
      });
    }
    groups.get(row._postingKey).rowIndexes.push(row._rowIndex);
  }
  state.postings = Array.from(groups.values()).map((posting, index) => ({
    ...posting,
    index,
    label: postingLabel(posting),
  }));
}

function postingLabel(posting) {
  const title = [posting.company, posting.title].filter(Boolean).join(" - ");
  return title || text(posting.jobName || posting.key, "Untitled posting");
}

function restorePostingPosition() {
  const savedKey = localStorage.getItem(postingStorageKey());
  const savedIndex = state.postings.findIndex((posting) => posting.key === savedKey);
  state.postingIndex = savedIndex >= 0 ? savedIndex : 0;
}

function postingStorageKey() {
  return `tagger.posting.${state.sourceFile}`;
}

function currentPosting() {
  return state.postings[state.postingIndex] || null;
}

function currentPostingRows() {
  const posting = currentPosting();
  if (!posting) {
    return [];
  }
  return posting.rowIndexes.map((index) => state.rows[index]).filter(Boolean);
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
  return currentPostingRows().filter((row) => rowMatchesFilter(row) && rowMatchesSearch(row));
}

function render() {
  state.visibleRows = getVisibleRows();
  if (
    state.selectedRowIndex < 0 ||
    !state.visibleRows.some((row) => row._rowIndex === state.selectedRowIndex)
  ) {
    state.selectedRowIndex = state.visibleRows[0]?._rowIndex ?? -1;
  }

  renderPostingNav();
  renderList();
  renderDetail();
  renderCounts();
}

function renderPostingNav() {
  const posting = currentPosting();
  prevPostButton.disabled = state.navigating || state.postingIndex <= 0;
  nextPostButton.disabled = state.navigating || state.postingIndex >= state.postings.length - 1;

  if (!posting) {
    postingTitle.textContent = "No posting selected";
    postingMeta.textContent = "0 of 0";
    return;
  }

  const postingRows = currentPostingRows();
  const taggedHere = postingRows.filter((row) => row._not_relevant).length;
  const reviewedRelevantHere = postingRows.filter((row) => row._reviewed_relevant).length;
  postingTitle.textContent = posting.label;
  postingMeta.textContent = `Posting ${state.postingIndex + 1} of ${state.postings.length} | ${postingRows.length} snippets | ${taggedHere} not relevant | ${reviewedRelevantHere} reviewed relevant`;
}

function renderCounts() {
  const postingRows = currentPostingRows();
  const taggedHere = postingRows.filter((row) => row._not_relevant).length;
  const taggedTotal = state.rows.filter((row) => row._not_relevant).length;
  countText.textContent = `${state.visibleRows.length} shown / ${postingRows.length} rows`;
  setStatus(`${taggedHere} here | ${taggedTotal} total not relevant`);
}

function renderList() {
  snippetList.replaceChildren();
  if (!state.visibleRows.length) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.textContent = "No rows in this posting match the current view";
    snippetList.append(empty);
    return;
  }

  const fragment = document.createDocumentFragment();
  for (const row of state.visibleRows) {
    const item = document.createElement("div");
    item.className = "snippet-row";
    if (row._saveState && row._saveState !== "idle") {
      item.classList.add("has-save-status");
    }
    if (row._not_relevant) {
      item.classList.add("is-not-relevant");
    }
    if (row._reviewed_relevant) {
      item.classList.add("is-reviewed-relevant");
    }
    if (row._recommendation?.suggested_label === "not_relevant") {
      item.classList.add("is-suggested-not-relevant");
    }
    if (row._recommendation?.suggested_label === "relevant") {
      item.classList.add("is-suggested-relevant");
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
    const badges = recommendationBadges(row);
    if (badges) {
      main.append(badges);
    }
    item.append(meta, main);
    const saveStatus = rowSaveStatus(row);
    if (saveStatus) {
      item.append(saveStatus);
    }
    fragment.append(item);
  }

  snippetList.append(fragment);
  scrollActiveIntoView(false);
}

function rowSaveStatus(row) {
  if (!row._saveState || row._saveState === "idle") {
    return null;
  }
  const status = document.createElement("div");
  status.className = "row-save-status";
  if (row._saveState === "error") {
    status.classList.add("is-error");
    status.textContent = "save failed";
    return status;
  }
  if (row._saveState === "saved") {
    status.classList.add("is-saved");
    status.textContent = "saved";
    return status;
  }

  const spinner = document.createElement("span");
  spinner.className = "mini-spinner";
  spinner.setAttribute("aria-hidden", "true");
  status.append(spinner, document.createTextNode("saving"));
  return status;
}

function recommendationBadges(row) {
  if (!row._recommendation) {
    return null;
  }
  const wrapper = document.createElement("div");
  wrapper.className = "row-badges";

  const badge = document.createElement("span");
  badge.className = "score-badge";
  const notRelevantScore = formatScore(row._recommendation.not_relevant_score);
  const reviewedRelevantScore = formatScore(row._recommendation.reviewed_relevant_score);
  if (row._recommendation.suggested_label === "not_relevant") {
    badge.classList.add("is-not-relevant");
    badge.textContent = `Suggest not relevant | NR ${notRelevantScore} / R ${reviewedRelevantScore}`;
  } else {
    badge.classList.add("is-relevant");
    badge.textContent = `Suggest relevant | NR ${notRelevantScore} / R ${reviewedRelevantScore}`;
  }
  wrapper.append(badge);

  if (row._recommendation.irrelevance_cluster) {
    const clusterBadge = document.createElement("span");
    clusterBadge.className = "score-badge is-cluster";
    clusterBadge.textContent = `${row._recommendation.irrelevance_cluster.cluster_id} ${formatScore(row._recommendation.irrelevance_cluster.score)}`;
    clusterBadge.title = row._recommendation.irrelevance_cluster.exemplar?.text || "";
    wrapper.append(clusterBadge);
  }
  return wrapper;
}

function formatScore(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "--";
  }
  return Number(value).toFixed(2);
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
    tagIndicator.classList.remove("is-reviewed-relevant");
    suggestionIndicator.textContent = "No suggestion";
    suggestionIndicator.className = "suggestion-indicator";
    return;
  }

  detailCategory.textContent = text(row.category, "snippet");
  detailTitle.textContent = text(row.title || row.job_name);
  detailCompany.textContent = text(row.company);
  detailLine.textContent = text(row._line_number);
  detailJob.textContent = text(row.job_key || row.job_name);
  detailId.textContent = text(row.id);
  detailText.textContent = text(row.text, "");
  tagIndicator.textContent = row._not_relevant
    ? "Not relevant"
    : row._reviewed_relevant
      ? "Reviewed relevant"
      : "Untagged";
  tagIndicator.classList.toggle("is-not-relevant", row._not_relevant);
  tagIndicator.classList.toggle("is-reviewed-relevant", row._reviewed_relevant && !row._not_relevant);
  renderSuggestion(row);
}

function renderSuggestion(row) {
  suggestionIndicator.className = "suggestion-indicator";
  if (!row._recommendation) {
    suggestionIndicator.textContent = "No suggestion";
    return;
  }

  const notRelevantScore = formatScore(row._recommendation.not_relevant_score);
  const reviewedRelevantScore = formatScore(row._recommendation.reviewed_relevant_score);
  const cluster = row._recommendation.irrelevance_cluster;
  const clusterText = cluster ? ` | ${cluster.cluster_id} ${formatScore(cluster.score)}` : "";
  if (row._recommendation.suggested_label === "not_relevant") {
    suggestionIndicator.classList.add("is-not-relevant");
    suggestionIndicator.textContent = `Suggest not relevant | NR ${notRelevantScore} / R ${reviewedRelevantScore}${clusterText}`;
  } else {
    suggestionIndicator.classList.add("is-relevant");
    suggestionIndicator.textContent = `Suggest relevant | NR ${notRelevantScore} / R ${reviewedRelevantScore}${clusterText}`;
  }
}

function selectedRow() {
  return state.rows[state.selectedRowIndex] || null;
}

function visiblePosition() {
  return state.visibleRows.findIndex((row) => row._rowIndex === state.selectedRowIndex);
}

function selectFirstVisibleRowInPosting() {
  state.visibleRows = getVisibleRows();
  state.selectedRowIndex = state.visibleRows[0]?._rowIndex ?? -1;
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
    node.classList.toggle("is-reviewed-relevant", row._reviewed_relevant);
  }
}

function scrollActiveIntoView(smooth) {
  const node = snippetList.querySelector(".snippet-row.is-active");
  if (!node) {
    return;
  }
  node.scrollIntoView({ block: "nearest", behavior: smooth ? "smooth" : "auto" });
}

function setNotRelevant(row, value, advance) {
  if (!row) {
    return;
  }
  if (row._not_relevant === value && advance) {
    moveSelection(1);
    return;
  }

  const previousVisiblePosition = visiblePosition();
  const previousState = {
    notRelevant: row._not_relevant,
    reviewedRelevant: row._reviewed_relevant,
  };
  const saveVersion = (row._saveVersion || 0) + 1;
  row._saveVersion = saveVersion;
  row._not_relevant = value;
  if (value) {
    row._reviewed_relevant = false;
  }
  row._saveState = "saving";

  if (advance) {
    const rowStillVisible = rowMatchesFilter(row) && rowMatchesSearch(row);
    state.visibleRows = getVisibleRows();
    if (state.visibleRows.length) {
      const nextPosition = rowStillVisible ? previousVisiblePosition + 1 : previousVisiblePosition;
      const bounded = Math.max(0, Math.min(nextPosition, state.visibleRows.length - 1));
      state.selectedRowIndex = state.visibleRows[bounded]._rowIndex;
    } else {
      state.selectedRowIndex = -1;
    }
  }
  if (!advance && (!rowMatchesFilter(row) || !rowMatchesSearch(row))) {
    state.visibleRows = getVisibleRows();
    state.selectedRowIndex = state.visibleRows[0]?._rowIndex ?? -1;
  }

  renderDetail();
  renderCounts();
  renderPostingNav();
  render();
  persistTag(row, value, saveVersion, previousState);
}

function persistTag(row, value, saveVersion, previousState) {
  const sourceFile = state.sourceFile;
  const saveRequest = () =>
    fetchJson("/api/tags", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source_file: sourceFile,
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
  const promise = (row._saveChain || Promise.resolve()).catch(() => {}).then(saveRequest);
  row._saveChain = promise;

  state.pendingTagSaves.add(promise);
  updatePendingSaveSummary();

  promise
    .then((payload) => {
      if (row._saveVersion !== saveVersion) {
        return;
      }
      row._saveState = "saved";
      setSaveState(`${payload.count} not relevant saved`);
      setScoreState("Updating scorer");
      render();
      window.setTimeout(() => {
        if (row._saveVersion === saveVersion && row._saveState === "saved") {
          row._saveState = "idle";
          render();
        }
      }, 650);
    })
    .catch((error) => {
      if (row._saveVersion === saveVersion) {
        row._not_relevant = previousState.notRelevant;
        row._reviewed_relevant = previousState.reviewedRelevant;
        row._saveState = "error";
        render();
      }
      setSaveState("Save failed");
      alert(error.message);
    })
    .finally(() => {
      state.pendingTagSaves.delete(promise);
      updatePendingSaveSummary();
    });
}

function updatePendingSaveSummary() {
  if (state.pendingTagSaves.size) {
    setSaveState(`${state.pendingTagSaves.size} saving`);
  } else if (saveText.textContent.endsWith("saving")) {
    setSaveState("Idle");
  }
}

async function waitForPendingTagSaves() {
  while (state.pendingTagSaves.size) {
    const pending = Array.from(state.pendingTagSaves);
    await Promise.allSettled(pending);
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

async function movePosting(delta) {
  if (!state.postings.length || state.navigating) {
    return;
  }
  const nextIndex = Math.max(0, Math.min(state.postingIndex + delta, state.postings.length - 1));
  if (nextIndex === state.postingIndex) {
    return;
  }

  state.navigating = true;
  renderPostingNav();
  showScoringDialog("Preparing next posting", "Finishing queued tag saves", `${state.pendingTagSaves.size} pending`);
  try {
    await waitForPendingTagSaves();
    updateScoringDialog("Preparing next posting", "Saving reviewed-relevant examples", "Current posting");
    const reviewed = await markCurrentPostingReviewed({ quiet: true });
    if (!reviewed) {
      updateScoringDialog("Review save failed", "Could not advance to the next posting", "Try again");
      window.setTimeout(() => {
        hideScoringDialog();
      }, 900);
      return;
    }

    state.postingIndex = nextIndex;
    const posting = currentPosting();
    if (posting) {
      localStorage.setItem(postingStorageKey(), posting.key);
    }
    snippetList.scrollTop = 0;
    selectFirstVisibleRowInPosting();
    setScoreState("Scoring posting");
    render();
    await loadRecommendationsForCurrentPosting({ showDialog: true, keepDialogOpen: true });
  } finally {
    state.navigating = false;
    renderPostingNav();
  }
}

async function markCurrentPostingReviewed(options = {}) {
  const posting = currentPosting();
  if (!posting) {
    return false;
  }
  if (state.reviewing) {
    return false;
  }

  state.reviewing = true;
  setSaveState("Marking reviewed");
  try {
    const payload = await fetchJson("/api/review-posting", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source_file: state.sourceFile,
        posting_key: posting.key,
      }),
    });
    for (const row of currentPostingRows()) {
      row._reviewed_relevant = !row._not_relevant;
    }
    setSaveState(`${payload.reviewed_relevant_count} relevant seeds`);
    return true;
  } catch (error) {
    setSaveState("Review save failed");
    if (!options.quiet) {
      alert(error.message);
    }
    return false;
  } finally {
    state.reviewing = false;
  }
}

async function loadRecommendationsForCurrentPosting(options = {}) {
  const posting = currentPosting();
  if (!posting) {
    setScoreState("No posting");
    return;
  }

  const requestId = ++state.recommendationRequest;
  setScoreState("Scoring posting");
  if (options.showDialog) {
    showScoringDialog("Scoring posting", posting.label, "Embedding snippets and comparing examples");
  }
  try {
    const payload = await fetchJson(
      `/api/recommendations?file=${encodeURIComponent(state.sourceFile)}&posting_key=${encodeURIComponent(posting.key)}`
    );
    if (requestId !== state.recommendationRequest) {
      return;
    }

    const byId = new Map((payload.recommendations || []).map((item) => [item.id, item]));
    for (const row of currentPostingRows()) {
      row._recommendation = byId.get(row.id) || null;
    }

    if (payload.available) {
      setScoreState(
        `Scored ${payload.rows_scored} | ${payload.not_relevant_seed_count} NR / ${payload.reviewed_relevant_seed_count} R | ${payload.irrelevance_cluster_count} clusters`
      );
      if (options.showDialog) {
        updateScoringDialog(
          "Scoring complete",
          `${payload.rows_scored} snippets scored`,
          `${payload.not_relevant_seed_count} NR / ${payload.reviewed_relevant_seed_count} R | ${payload.irrelevance_cluster_count} clusters`
        );
      }
    } else {
      setScoreState(payload.reason === "no_seed_tags" ? "No seed tags" : "Scoring unavailable");
      if (payload.message) {
        console.warn(payload.message);
      }
      if (options.showDialog) {
        updateScoringDialog("Scoring unavailable", posting.label, payload.message || "No score returned");
      }
    }
    render();
  } catch (error) {
    if (requestId !== state.recommendationRequest) {
      return;
    }
    setScoreState("Scoring failed");
    if (options.showDialog) {
      updateScoringDialog("Scoring failed", posting.label, error.message);
    }
    console.error(error);
  } finally {
    if (options.showDialog) {
      window.setTimeout(() => {
        hideScoringDialog();
      }, 550);
    }
  }
}

function showScoringDialog(title, detail, progress) {
  state.scoringDialogOpen = true;
  updateScoringDialog(title, detail, progress);
  scoringDialog.hidden = false;
}

function updateScoringDialog(title, detail, progress) {
  scoringTitle.textContent = title;
  scoringDetail.textContent = detail;
  scoringProgress.textContent = progress;
}

function hideScoringDialog() {
  scoringDialog.hidden = true;
  state.scoringDialogOpen = false;
}

function currentTargetAllowsShortcut(event) {
  if (state.scoringDialogOpen || state.navigating) {
    return false;
  }
  const target = event.target;
  if (!(target instanceof HTMLElement)) {
    return true;
  }
  return !["INPUT", "SELECT", "TEXTAREA", "BUTTON"].includes(target.tagName);
}

fileSelect.addEventListener("change", async () => {
  if (state.rows.length) {
    await waitForPendingTagSaves();
    const reviewed = await markCurrentPostingReviewed();
    if (!reviewed) {
      fileSelect.value = state.sourceFile;
      return;
    }
  }
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
  selectFirstVisibleRowInPosting();
  render();
});

prevPostButton.addEventListener("click", () => {
  movePosting(-1);
});

nextPostButton.addEventListener("click", () => {
  movePosting(1);
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
  } else if (key === "arrowright") {
    event.preventDefault();
    movePosting(1);
  } else if (key === "arrowleft") {
    event.preventDefault();
    movePosting(-1);
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
