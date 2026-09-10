const caseId = window.CASE_ID;
let tab = "symptoms";
let snapshot = null;

const actionList = document.getElementById("action-list");
const filter = document.getElementById("action-filter");
const findings = document.getElementById("findings");
const timeline = document.getElementById("timeline");
const metrics = document.getElementById("metrics");
const title = document.getElementById("case-title");
const resultBanner = document.getElementById("result-banner");

function chip(label, value) {
  return `<span class="chip">${label}: <strong>${value}</strong></span>`;
}

function render() {
  const state = snapshot.state;
  title.textContent = `Patient ${state.patient_id}`;
  metrics.innerHTML = [
    chip("Steps", state.stats.steps),
    chip("Symptoms asked", state.stats.symptoms_asked),
    chip("Tests ordered", state.stats.tests_ordered),
    chip("Cost", state.stats.total_test_cost),
  ].join("");

  const revealed = [
    ...state.revealed_symptoms.map((s) => ({ ...s, kind: "Symptom" })),
    ...state.revealed_tests.map((s) => ({ ...s, kind: "Test" })),
  ];
  findings.innerHTML = revealed.length
    ? revealed
        .map(
          (f) =>
            `<span class="tag ${f.present ? "yes" : "no"}">${f.kind}: ${f.label} · ${
              f.present ? "Present" : "Absent"
            }</span>`
        )
        .join("")
    : `<p class="muted">No findings yet. Ask a symptom to begin.</p>`;

  timeline.innerHTML = state.history.map((line) => `<li>${line}</li>`).join("");

  if (state.done) {
    resultBanner.classList.remove("hidden");
    resultBanner.className = `result ${state.correct ? "ok" : "bad"}`;
    resultBanner.innerHTML = state.correct
      ? `Diagnosis recorded: <em>${state.predicted}</em>. Matches the case record.`
      : `You diagnosed <em>${state.predicted}</em>. Recorded reference label: <em>${state.true_label}</em>.`;
  }

  drawActions();
}

function drawActions() {
  const state = snapshot.state;
  if (state.done) {
    actionList.innerHTML = `<p class="muted">This consultation is complete. Return to the dashboard for a new case.</p>`;
    return;
  }
  const q = filter.value.trim().toLowerCase();
  let items = [];
  if (tab === "symptoms") {
    items = state.pending_symptoms.map((s) => ({ ...s, type: "ask" }));
  } else if (tab === "tests") {
    items = state.pending_tests.map((s) => ({ ...s, type: "test" }));
  } else {
    items = state.diseases.map((d) => ({
      name: d,
      label: d,
      cost: 0,
      type: "diagnose",
    }));
  }
  items = items.filter((i) => i.label.toLowerCase().includes(q) || i.name.toLowerCase().includes(q));
  actionList.innerHTML = items
    .slice(0, 80)
    .map((i) => {
      const extra = i.type === "diagnose" ? "Submit diagnosis" : `Cost ${i.cost}`;
      return `<button data-type="${i.type}" data-target="${encodeURIComponent(i.name)}"><strong>${i.label}</strong><div class="muted">${extra}</div></button>`;
    })
    .join("");
  actionList.querySelectorAll("button").forEach((btn) => {
    btn.addEventListener("click", () => act(btn.dataset.type, decodeURIComponent(btn.dataset.target)));
  });
}

async function loadCase() {
  const res = await fetch(`/api/cases/${caseId}`);
  snapshot = await res.json();
  if (!res.ok) {
    title.textContent = snapshot.error || "Case not found";
    return;
  }
  render();
}

async function act(type, target) {
  const res = await fetch(`/api/cases/${caseId}/actions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ type, target }),
  });
  const data = await res.json();
  if (!res.ok) {
    alert(data.error || "Action failed");
    return;
  }
  snapshot = data;
  render();
}

document.querySelectorAll(".tabs button").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tabs button").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    tab = btn.dataset.tab;
    drawActions();
  });
});
filter.addEventListener("input", drawActions);
loadCase();
