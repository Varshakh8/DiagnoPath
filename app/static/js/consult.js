
"use strict";

// =========================================================
// ELEMENTS
// =========================================================

const caseId = window.CASE_ID;

let tab = "symptoms";
let snapshot = null;

const actionList = document.getElementById("action-list");
const filter = document.getElementById("action-filter");
const metrics = document.getElementById("metrics");
const title = document.getElementById("case-title");

const legacyFindings = document.getElementById("legacy-findings");
const legacyTimeline = document.getElementById("legacy-timeline");
const legacyResultBanner = document.getElementById("legacy-result-banner");

const findings = document.getElementById("findings");
const timeline = document.getElementById("timeline");
const resultBanner = document.getElementById("result-banner");
const progress = document.getElementById("ai-progress");

const aiSuggestBtn = document.getElementById("ai-suggest-btn");
const aiSuggestionBox = document.getElementById("ai-suggestion");

let manualCaseId =
  localStorage.getItem("diagnopath_manual_case_id") || null;

let manualBusy = false;
let manualCompleted = false;

// =========================================================
// ORIGINAL SIMULATED CONSULTATION — PRESERVED
// =========================================================

function chip(label, value) {
  const span = document.createElement("span");
  span.className = "chip";

  const strong = document.createElement("strong");
  strong.textContent = value ?? 0;

  span.append(document.createTextNode(label + ": "), strong);
  return span.outerHTML;
}

function render() {
  if (!snapshot || !snapshot.state) return;

  const state = snapshot.state;

  metrics.innerHTML = [
    chip("Steps", state.stats.steps),
    chip("Symptoms asked", state.stats.symptoms_asked),
    chip("Tests ordered", state.stats.tests_ordered),
    chip("Cost", state.stats.total_test_cost),
  ].join("");

  const revealed = [
    ...state.revealed_symptoms.map((s) => ({
      ...s,
      kind: "Symptom",
    })),
    ...state.revealed_tests.map((s) => ({
      ...s,
      kind: "Test",
    })),
  ];

  legacyFindings.innerHTML = revealed.length
    ? revealed.map((f) => {
        const tag = document.createElement("span");
        tag.className = `tag ${f.present ? "yes" : "no"}`;
        tag.textContent =
          `${f.kind}: ${f.label} · ${f.present ? "Present" : "Absent"}`;
        return tag.outerHTML;
      }).join("")
    : `<p class="muted">No simulated findings yet.</p>`;

  legacyTimeline.replaceChildren();

  (state.history || []).forEach((line) => {
    const li = document.createElement("li");
    li.textContent = line;
    legacyTimeline.appendChild(li);
  });

  if (state.done) {
    legacyResultBanner.classList.remove("hidden");
    legacyResultBanner.textContent = state.correct
      ? `Simulated diagnosis: ${state.predicted}. Matches the case record.`
      : `Simulated diagnosis: ${state.predicted}. Reference label: ${state.true_label}.`;
  } else {
    legacyResultBanner.classList.add("hidden");
    legacyResultBanner.textContent = "";
  }

  drawActions();
}

function drawActions() {
  if (!snapshot || !snapshot.state) return;

  const state = snapshot.state;

  if (state.done) {
    actionList.textContent = "This simulated consultation is complete.";
    return;
  }

  const query = filter.value.trim().toLowerCase();
  let items = [];

  if (tab === "symptoms") {
    items = state.pending_symptoms.map((s) => ({
      ...s,
      type: "ask",
    }));
  } else if (tab === "tests") {
    items = state.pending_tests.map((s) => ({
      ...s,
      type: "test",
    }));
  } else {
    items = state.diseases.map((d) => ({
      name: d,
      label: d,
      cost: 0,
      type: "diagnose",
    }));
  }

  items = items.filter((item) =>
    String(item.label || item.name)
      .toLowerCase()
      .includes(query)
  );

  actionList.replaceChildren();

  items.slice(0, 80).forEach((item) => {
    const button = document.createElement("button");
    button.type = "button";

    const strong = document.createElement("strong");
    strong.textContent = item.label || item.name;

    const detail = document.createElement("div");
    detail.className = "muted";
    detail.textContent =
      item.type === "diagnose"
        ? "Submit diagnosis"
        : `Cost ${item.cost}`;

    button.append(strong, detail);
    button.addEventListener("click", () => {
      act(item.type, item.name);
    });

    actionList.appendChild(button);
  });

  if (!items.length) {
    actionList.textContent = "No matching actions.";
  }
}

async function loadCase() {
  try {
    const response = await fetch(
      `/api/cases/${encodeURIComponent(caseId)}`
    );

    const data = await response.json();

    if (!response.ok) {
      console.error("Simulated case could not load:", data);
      return;
    }

    snapshot = data;
    render();
  } catch (error) {
    console.error("Simulated case request failed:", error);
  }
}

async function act(type, target) {
  try {
    const response = await fetch(
      `/api/cases/${encodeURIComponent(caseId)}/actions`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ type, target }),
      }
    );

    const data = await response.json();

    if (!response.ok) {
      alert(data.error || "Action failed");
      return;
    }

    snapshot = data;
    render();
  } catch (error) {
    alert("Action failed: " + error.message);
  }
}

document.querySelectorAll(".tabs button").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".tabs button").forEach((item) => {
      item.classList.remove("active");
    });

    button.classList.add("active");
    tab = button.dataset.tab;
    drawActions();
  });
});

filter.addEventListener("input", drawActions);

// =========================================================
// MANUAL RL CONSULTATION
// =========================================================

async function manualRequest(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json();

  if (!response.ok) {
    throw new Error(
      data.detail || data.error || data.message ||
      `Request failed (${response.status})`
    );
  }

  return data;
}

function displayProgress(data) {
  progress.replaceChildren();

  const entries = [
    ["Steps", data.step ?? 0],
    ["Symptoms answered", data.symptoms_answered ?? 0],
    ["Investigations recorded", data.tests_answered ?? 0],
  ];

  entries.forEach(([label, value]) => {
    const item = document.createElement("span");
    item.className = "chip";

    const strong = document.createElement("strong");
    strong.textContent = String(value);

    item.append(document.createTextNode(label + ": "), strong);
    progress.appendChild(item);
  });
}

async function refreshManualFindings() {
  if (!manualCaseId) return;

  const data = await manualRequest(
    `/api/manual-cases/${encodeURIComponent(manualCaseId)}`
  );

  findings.replaceChildren();

  const addFinding = (kind, label, value) => {
    if (value === null || value === undefined) return;

    const present = Number(value) === 1;
    const tag = document.createElement("span");
    tag.className = `tag ${present ? "yes" : "no"}`;
    tag.textContent =
      `${kind}: ${label.replaceAll("_", " ")} · ` +
      `${present ? "Present / Positive" : "Absent / Negative"}`;

    findings.appendChild(tag);
  };

  Object.entries(data.symptoms || {}).forEach(([label, value]) => {
    addFinding("Symptom", label, value);
  });

  Object.entries(data.tests || {}).forEach(([label, value]) => {
    addFinding("Test", label, value);
  });

  if (!findings.children.length) {
    const empty = document.createElement("p");
    empty.className = "muted";
    empty.textContent = "No findings recorded yet.";
    findings.appendChild(empty);
  }

  timeline.replaceChildren();

  const events = Array.isArray(data.timeline) ? data.timeline : [];

  if (events.length) {
    events.forEach((event) => {
      const li = document.createElement("li");
      li.textContent = event;
      timeline.appendChild(li);
    });
  } else {
    const li = document.createElement("li");
    li.textContent = "Consultation started; waiting for the first answer.";
    timeline.appendChild(li);
  }

  displayProgress(data);
}

function showManualError(error) {
  console.error("Manual RL consultation error:", error);

  aiSuggestionBox.replaceChildren();

  const message = document.createElement("p");
  message.className = "tag muted";
  message.textContent = "AI consultation error: " + error.message;

  const retry = document.createElement("button");
  retry.type = "button";
  retry.textContent = "Continue consultation";

  retry.addEventListener("click", suggest);

  aiSuggestionBox.append(message, retry);
}

function renderManualQuestion(data) {
  aiSuggestionBox.replaceChildren();

  const panel = document.createElement("div");
  panel.className = "panel";

  const heading = document.createElement("h3");
  heading.textContent = "🤖 RL Agent";

  const prediction = data.prediction || data.diagnosis;

  if (
    data.status === "diagnosis" ||
    data.status === "complete" ||
    data.status === "done" ||
    prediction
  ) {
    manualCompleted = true;

    const result = document.createElement("p");
    result.textContent = "Predicted disease: " + prediction;

    const disclaimer = document.createElement("p");
    disclaimer.className = "muted";
    disclaimer.textContent =
      "Experimental prediction only. Not a clinically validated diagnosis.";

    const restart = document.createElement("button");
    restart.type = "button";
    restart.textContent = "Start new AI consultation";

    restart.addEventListener("click", () => {
      startManualConsultation();
    });

    panel.append(heading, result, disclaimer, restart);
    aiSuggestionBox.appendChild(panel);

    aiSuggestBtn.textContent = "🤖 AI Consultation Complete";

    resultBanner.classList.remove("hidden");
    resultBanner.textContent = "Experimental prediction: " + prediction;

    return;
  }

  manualCompleted = false;

  const isSymptom = data.type === "ask";
  const label = data.label || data.target;

  const instruction = document.createElement("p");
  instruction.className = "muted";
  instruction.textContent = isSymptom
    ? "RL action: Ask symptom"
    : "RL action: Request investigation result";

  const question = document.createElement("p");
  question.textContent = isSymptom
    ? `Does the patient have ${label}?`
    : `What is the result of ${label}?`;

  const yes = document.createElement("button");
  yes.type = "button";
  yes.textContent = isSymptom ? "Yes" : "Positive";
  yes.addEventListener("click", () => submitManualAnswer(true));

  const no = document.createElement("button");
  no.type = "button";
  no.textContent = isSymptom ? "No" : "Negative";
  no.addEventListener("click", () => submitManualAnswer(false));

  panel.append(heading, instruction, question, yes, no);
  aiSuggestionBox.appendChild(panel);

  aiSuggestBtn.textContent = "🤖 AI Consultation Active";
  resultBanner.classList.add("hidden");
}

async function startManualConsultation() {
  if (manualBusy) return;

  manualBusy = true;
  aiSuggestBtn.disabled = true;

  try {
    aiSuggestionBox.textContent = "Starting new AI consultation...";

    const data = await manualRequest("/api/manual-cases", {
      method: "POST",
    });

    manualCaseId = data.case_id;
    if (!manualCaseId) {
      throw new Error("Server did not return a consultation ID.");
    }

    localStorage.setItem(
      "diagnopath_manual_case_id",
      manualCaseId
    );

    manualCompleted = false;
    resultBanner.classList.add("hidden");

    await refreshManualFindings();
    await getNextManualAction();
  } catch (error) {
    showManualError(error);
  } finally {
    manualBusy = false;
    aiSuggestBtn.disabled = false;
  }
}

async function getNextManualAction() {
  const data = await manualRequest(
    `/api/manual-cases/${encodeURIComponent(manualCaseId)}/next`
  );

  renderManualQuestion(data);
}

async function submitManualAnswer(answer) {
  if (manualBusy || !manualCaseId || manualCompleted) return;

  manualBusy = true;
  aiSuggestBtn.disabled = true;

  try {
    const response = await manualRequest(
      `/api/manual-cases/${encodeURIComponent(manualCaseId)}/answer`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ answer }),
      }
    );

    await refreshManualFindings();

    // The answer endpoint returns the updated step.
    displayProgress(response);

    await getNextManualAction();
  } catch (error) {
    showManualError(error);
  } finally {
    manualBusy = false;
    aiSuggestBtn.disabled = false;
  }
}

async function suggest() {
  if (manualBusy) return;

  if (!manualCaseId || manualCompleted) {
    await startManualConsultation();
    return;
  }

  manualBusy = true;
  aiSuggestBtn.disabled = true;

  try {
    await refreshManualFindings();
    await getNextManualAction();
  } catch (error) {
    showManualError(error);
  } finally {
    manualBusy = false;
    aiSuggestBtn.disabled = false;
  }
}

aiSuggestBtn.addEventListener("click", suggest);

// =========================================================
// INITIAL LOAD
// =========================================================

title.textContent = "AI-assisted consultation";
loadCase();
