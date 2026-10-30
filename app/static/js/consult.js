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

const xaiButton = document.getElementById("explain-xai-btn");
const xaiResult = document.getElementById("xai-result");

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

// =========================================================
// MEMBER 3 — EXPLAINABLE AI
// =========================================================


async function explainRecommendation() {
  console.log("NEW XAI FUNCTION IS RUNNING!");
  const activeCaseId =
    manualCaseId ||
    localStorage.getItem("diagnopath_manual_case_id");

  if (!activeCaseId) {
    alert("Please start an AI consultation first.");
    return;
  }

  // Open the new tab immediately after the button click.
  const explanationWindow = window.open("", "_blank");

  if (!explanationWindow) {
    alert("Please allow pop-ups for this site and try again.");
    return;
  }

  xaiButton.disabled = true;
  xaiButton.textContent = "Loading explanation...";

  explanationWindow.document.open();
  explanationWindow.document.write(`
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>DiagnoPath AI - XAI Explanation</title>
      <style>
        body {
          font-family: Arial, sans-serif;
          max-width: 950px;
          margin: 30px auto;
          padding: 20px;
          background: #f3f6fb;
          color: #243047;
          line-height: 1.6;
        }
        h1, h2 { color: #1756a9; }
        section {
          background: white;
          padding: 22px;
          margin: 20px 0;
          border-radius: 12px;
          box-shadow: 0 2px 8px #0001;
        }
        li { margin: 8px 0; overflow-wrap: anywhere; }
        .muted { color: #64748b; }
        .error { color: #b91c1c; }
        .score { font-weight: bold; }
        button {
          padding: 10px 16px;
          border: 0;
          border-radius: 7px;
          background: #1756a9;
          color: white;
          cursor: pointer;
        }
      </style>
    </head>
    <body>
      <h1>DiagnoPath AI — Explainable AI</h1>
      <p>Explanation of the selected Reinforcement Learning action.</p>
      <div id="content">Loading RL, SHAP and LIME explanations...</div>
    </body>
    </html>
  `);
  explanationWindow.document.close();

  try {
    const response = await fetch(
      `/api/xai/explain?case_id=${encodeURIComponent(activeCaseId)}`
    );

    const data = await response.json();

    if (!response.ok || data.status !== "ok") {
      throw new Error(
        data.detail || data.message || "Failed to load explanation."
      );
    }

    if (explanationWindow.closed) return;

    const doc = explanationWindow.document;
    const content = doc.getElementById("content");
    content.replaceChildren();

    function addText(tag, text, parent = content, className = "") {
      const element = doc.createElement(tag);
      element.textContent = String(text ?? "");
      if (className) element.className = className;
      parent.appendChild(element);
      return element;
    }

    function addSection(heading) {
      const section = doc.createElement("section");
      content.appendChild(section);
      addText("h2", heading, section);
      return section;
    }

    function renderFeatures(parent, features, method) {
      if (!Array.isArray(features) || features.length === 0) {
        addText("p", "No feature contributions available.", parent);
        return;
      }

      const list = doc.createElement("ul");

      features.forEach((feature) => {
        let description;

        if (method === "SHAP") {
          description =
            `${feature.feature} (${feature.finding}): ` +
            `${feature.contribution}`;
        } else if (method === "LIME") {
          description =
            `${feature.description}: ${feature.weight}`;
        } else {
          description =
            `${feature.feature} (${feature.finding}): ` +
            `${feature.contribution}`;
        }

        addText("li", description, list);
      });

      parent.appendChild(list);
    }

    // 1. RL feature contributions
    const rlSection = addSection("1. Custom RL Feature Contributions");
    
    // Generate a human-readable explanation of the recommendation.
    const paragraphSection = addSection(
      "Why was this recommendation made?"
    );

    const positiveFeatures = (
      data.top_positive_contributions || []
    ).slice(0, 3);

    const shapPositive = (
      data.shap?.features || []
    )
      .filter((feature) => Number(feature.contribution) > 0)
      .slice(0, 3);

    const limePositive = (
      data.lime?.features || []
    )
      .filter((feature) => Number(feature.weight) > 0)
      .slice(0, 3);

    const featureNames = positiveFeatures
      .map((feature) => feature.feature)
      .filter(Boolean);

    const actionLabel = data.action?.label || data.action?.target || "the selected action";

    let explanation;

    if (data.action?.type === "diagnose") {
      if (featureNames.length > 0) {
        explanation =
          `The RL agent recommended ${actionLabel} because its learned ` +
          `model assigned this diagnostic action a Q-score of ${data.q_score}. ` +
          `The custom feature-contribution analysis identified ` +
          `${featureNames.join(", ")} as positive contributors to the ` +
          `selected action's score. `;
      } else {
        explanation =
          `The RL agent recommended ${actionLabel} with a Q-score of ` +
          `${data.q_score}. The custom feature-contribution analysis did ` +
          `not identify any positive known-feature contributions in its ` +
          `reported results. The recommendation therefore needs to be ` +
          `interpreted using the model's learned parameters and the ` +
          `complete consultation state. `;
      }
    } else {
      explanation =
        `The RL agent selected the action "${actionLabel}" with a Q-score ` +
        `of ${data.q_score}. This score reflects the agent's learned ` +
        `assessment of that action for the current consultation state. `;

      if (featureNames.length > 0) {
        explanation +=
          `The strongest positive custom contributions reported were ` +
          `${featureNames.join(", ")}. `;
      }
    }

    if (shapPositive.length > 0) {
      explanation +=
        `SHAP also identified positive contributions from ` +
        `${shapPositive.map((feature) => feature.feature).join(", ")}. `;
    }

    if (limePositive.length > 0) {
      explanation +=
        `LIME's local approximation assigned positive weights to ` +
        `${limePositive.map((feature) => feature.description).join(", ")}. `;
    }

    explanation +=
      "These methods explain aspects of the model's selected-action score; " +
      "they do not establish that the disease is present or prove causation. " +
      "The recommendation is experimental and requires clinical assessment.";

    addText("p", explanation, paragraphSection);


    addText(
      "p",
      `Recommended action: ${data.action.type} — ${data.action.label}`,
      rlSection
    );

    addText("p", `Q-score: ${data.q_score}`, rlSection, "score");

    addText("h3", "Top positive contributions", rlSection);
    renderFeatures(
      rlSection,
      data.top_positive_contributions,
      "RL"
    );

    addText("h3", "Top negative contributions", rlSection);
    renderFeatures(
      rlSection,
      data.top_negative_contributions,
      "RL"
    );

    if (data.note) {
      addText("p", data.note, rlSection, "muted");
    }

    // 2. SHAP explanation
    const shapSection = addSection("2. SHAP Explanation");
    const shap = data.shap;

    if (!shap || shap.status !== "ok") {
      addText(
        "p",
        shap?.message || "SHAP explanation is unavailable.",
        shapSection,
        "error"
      );
    } else {
      addText(
        "p",
        `Explained quantity: ${shap.explained_quantity || "selected action Q-score"}`,
        shapSection
      );
      addText("p", `Base value: ${shap.base_value}`, shapSection);
      renderFeatures(shapSection, shap.features, "SHAP");

      if (shap.note) {
        addText("p", shap.note, shapSection, "muted");
      }
    }

    // 3. LIME explanation
    const limeSection = addSection("3. LIME Explanation");
    const lime = data.lime;

    if (!lime || lime.status !== "ok") {
      addText(
        "p",
        lime?.message || "LIME explanation is unavailable.",
        limeSection,
        "error"
      );
    } else {
      addText(
        "p",
        `Explained quantity: ${lime.explained_quantity || "selected action Q-score"}`,
        limeSection
      );

      addText(
        "p",
        `Local surrogate R²: ${lime.local_surrogate_r2}`,
        limeSection
      );

      renderFeatures(limeSection, lime.features, "LIME");

      if (lime.note) {
        addText("p", lime.note, limeSection, "muted");
      }
    }

    addText(
      "p",
      "These explanations describe model scores, not disease probabilities or medical causation.",
      content,
      "muted"
    );

    const closeButton = doc.createElement("button");
    closeButton.textContent = "Close explanation";
    closeButton.addEventListener("click", () => {
      explanationWindow.close();
    });
    content.appendChild(closeButton);

  } catch (error) {
    console.error("XAI explanation failed:", error);

    if (!explanationWindow.closed) {
      const content = explanationWindow.document.getElementById("content");
      if (content) {
        content.replaceChildren();
        const message = explanationWindow.document.createElement("p");
        message.className = "error";
        message.textContent = error.message || "Failed to load explanation.";
        content.appendChild(message);
      }
    }
  } finally {
    xaiButton.disabled = false;
    xaiButton.textContent = "Explain Recommendation";
  }
}


xaiButton.addEventListener("click", explainRecommendation);

// =========================================================
// ORIGINAL SIMULATED CONSULTATION — TAB HANDLING
// =========================================================

document.querySelectorAll(".tabs button").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tabs button").forEach((button) => {
      button.classList.remove("active");
    });

    btn.classList.add("active");
    tab = btn.dataset.tab;
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