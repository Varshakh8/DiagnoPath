(() => {
  "use strict";

  const container = document.getElementById("whatif-content");
  if (!container) return;

  const escapeHTML = value => String(value ?? "").replace(/[&<>"']/g, c => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  })[c]);
  const stateLabel = value => value === 1 ? "Present / Positive" : value === 0 ? "Absent / Negative" : "Unknown";

  container.innerHTML = `
    <p class="muted">Create a baseline patient state, then change one or more findings. Both states are evaluated by the existing trained RL agent. This simulation never edits saved consultation records.</p>
    <div style="display:flex;gap:10px;flex-wrap:wrap;align-items:end;margin:14px 0;">
      <label style="flex:2;min-width:220px;">Clinical feature<br><select id="wi-feature" style="width:100%;"></select></label>
      <label style="flex:1;min-width:150px;">Baseline value<br><select id="wi-base" style="width:100%;"><option value="">Unknown</option><option value="0">Absent / Negative</option><option value="1">Present / Positive</option></select></label>
      <label style="flex:1;min-width:150px;">What-if value<br><select id="wi-mod" style="width:100%;"><option value="">Unknown</option><option value="0">Absent / Negative</option><option value="1">Present / Positive</option></select></label>
      <button id="wi-add" type="button">Add / Update finding</button>
    </div>
    <div id="wi-findings"><p class="muted">No findings added yet. Add the patient's known symptoms or investigation results.</p></div>
    <div style="display:flex;gap:10px;flex-wrap:wrap;margin:14px 0;">
      <button id="wi-run" class="primary" type="button">Compare RL scenarios</button>
      <button id="wi-clear" class="ghost" type="button">Clear scenario</button>
    </div>
    <p id="wi-status" class="muted" aria-live="polite"></p>
    <div id="wi-results" aria-live="polite"></div>
  `;

  const featureSelect = document.getElementById("wi-feature");
  const baseSelect = document.getElementById("wi-base");
  const modSelect = document.getElementById("wi-mod");
  const rowsBox = document.getElementById("wi-findings");
  const results = document.getElementById("wi-results");
  const status = document.getElementById("wi-status");
  const findings = new Map();

  const valueFrom = select => select.value === "" ? null : Number(select.value);
  const valueOptions = selected => `
    <option value="" ${selected === null ? "selected" : ""}>Unknown</option>
    <option value="0" ${selected === 0 ? "selected" : ""}>Absent / Negative</option>
    <option value="1" ${selected === 1 ? "selected" : ""}>Present / Positive</option>`;

  function renderFindings() {
    if (!findings.size) {
      rowsBox.innerHTML = '<p class="muted">No findings added yet.</p>';
      return;
    }
    rowsBox.innerHTML = `<div style="overflow-x:auto;"><table><thead><tr><th>Finding</th><th>Baseline</th><th>What-if</th><th></th></tr></thead><tbody>${
      [...findings.entries()].map(([name, item]) => `
        <tr data-feature="${escapeHTML(name)}">
          <td>${escapeHTML(item.label)} <small class="muted">(${escapeHTML(item.kind)})</small></td>
          <td><select class="wi-row-base">${valueOptions(item.baseline)}</select></td>
          <td><select class="wi-row-mod">${valueOptions(item.modified)}</select></td>
          <td><button type="button" class="ghost wi-remove">Remove</button></td>
        </tr>`).join("")}</tbody></table></div>`;
    rowsBox.querySelectorAll("tr[data-feature]").forEach(row => {
      const name = row.dataset.feature;
      row.querySelector(".wi-row-base").addEventListener("change", event => {
        findings.get(name).baseline = event.target.value === "" ? null : Number(event.target.value);
      });
      row.querySelector(".wi-row-mod").addEventListener("change", event => {
        findings.get(name).modified = event.target.value === "" ? null : Number(event.target.value);
      });
      row.querySelector(".wi-remove").addEventListener("click", () => {
        findings.delete(name);
        renderFindings();
      });
    });
  }

  function scenarioObject(key) {
    const output = {};
    findings.forEach((item, name) => {
      if (item[key] !== null) output[name] = item[key];
    });
    return output;
  }

  function renderExplain(title, explanation) {
    if (!explanation) return `<p class="muted">${title}: explanation unavailable.</p>`;
    const methods = ["shap", "lime"].map(method => {
      const item = explanation[method];
      if (!item || item.status !== "ok") {
        return `<div class="xai-method"><h4>${method.toUpperCase()}</h4><p class="muted">${escapeHTML(item?.message || "Not available")}</p></div>`;
      }
      const features = (item.features || []).slice(0, 6).map(feature => ({
        label: feature.feature || feature.description || "Feature",
        value: Number(feature.contribution !== undefined ? feature.contribution : feature.weight) || 0
      }));
      const max = Math.max(0.0001, ...features.map(f => Math.abs(f.value)));
      const bars = features.map(feature => `
        <div class="xai-row">
          <span>${escapeHTML(feature.label)}</span>
          <div class="xai-track"><div class="xai-fill ${feature.value < 0 ? "negative" : ""}" style="width:${Math.max(1, Math.abs(feature.value)/max*100)}%"></div></div>
          <strong>${feature.value.toFixed(4)}</strong>
        </div>`).join("");
      return `<div class="xai-method"><h4>${method.toUpperCase()}</h4><p class="muted">${escapeHTML(item.note || "")}</p>${bars || '<p class="muted">No contributions returned.</p>'}</div>`;
    }).join("");
    return `<section style="margin-top:12px;"><h4>${escapeHTML(title)} — feature contribution visualisations</h4>${methods}</section>`;
  }

  function renderScenario(title, data) {
    const ranking = (data.top_diseases || []).map(item =>
      `<div class="score-row"><span>${escapeHTML(item.disease)}</span><div class="score-track"><div class="score-fill" style="width:${Math.max(0, Math.min(100, Number(item.relative_score) || 0))}%"></div></div><strong>${Number(item.relative_score || 0).toFixed(1)}%</strong></div>`
    ).join("");
    const action = data.next_rl_action || {};
    return `<article class="panel" style="margin:14px 0;">
      <h3>${escapeHTML(title)}</h3>
      <p><strong>Leading disease:</strong> ${escapeHTML(data.best_disease)}</p>
      <p><strong>RL next action:</strong> ${escapeHTML(action.type)} — ${escapeHTML(String(action.target || "").replaceAll("_", " "))}</p>
      <p class="muted">Known findings: ${escapeHTML(data.known_findings)}. Q-scores are internal RL model scores, not calibrated probabilities or clinical confidence.</p>
      <h4>Leading disease score comparison</h4><div>${ranking}</div>
      ${renderExplain(title, data.explanation)}
    </article>`;
  }

  async function loadFeatures() {
    const response = await fetch("/api/whatif/features", {credentials: "same-origin"});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
    const featureMap = new Map(data.features.map(feature => [feature.name, feature]));
    featureSelect.innerHTML = data.features.map(feature =>
      `<option value="${escapeHTML(feature.name)}">${escapeHTML(feature.label)} (${escapeHTML(feature.kind)})</option>`
    ).join("");

    // Start from the actual completed consultation so What-if changes are
    // a comparison against this patient's original recorded findings.
    const caseId = window.DIAGNO_CASE_ID;
    if (caseId) {
      const caseResponse = await fetch(`/api/manual-cases/${encodeURIComponent(caseId)}`, {credentials:"same-origin"});
      const caseData = await caseResponse.json();
      if (!caseResponse.ok) throw new Error(caseData.error || `HTTP ${caseResponse.status}`);
      Object.entries(caseData.symptoms || {}).forEach(([name, value]) => {
        if (value !== null && featureMap.has(name)) {
          const feature = featureMap.get(name);
          findings.set(name, {label:feature.label, kind:feature.kind, baseline:Number(value), modified:Number(value)});
        }
      });
      Object.entries(caseData.tests || {}).forEach(([name, value]) => {
        if (value !== null && featureMap.has(name)) {
          const feature = featureMap.get(name);
          findings.set(name, {label:feature.label, kind:feature.kind, baseline:Number(value), modified:Number(value)});
        }
      });
      renderFindings();
      status.textContent = findings.size
        ? "Original consultation findings loaded. Change the What-if column to simulate a different patient state."
        : "No findings were recorded. Add the patient's baseline findings, then modify them for comparison.";
    }
  }

  document.getElementById("wi-add").addEventListener("click", () => {
    const name = featureSelect.value;
    if (!name) return;
    const option = featureSelect.selectedOptions[0];
    const match = option.textContent.match(/^(.*) \((symptom|test)\)$/);
    findings.set(name, {
      label: match ? match[1] : name,
      kind: match ? match[2] : "feature",
      baseline: valueFrom(baseSelect),
      modified: valueFrom(modSelect)
    });
    renderFindings();
  });

  document.getElementById("wi-run").addEventListener("click", async () => {
    status.textContent = "Evaluating both states with the trained RL agent and generating explanations…";
    results.innerHTML = "";
    const baseline = scenarioObject("baseline");
    const modified = scenarioObject("modified");
    if (!Object.keys(baseline).length && !Object.keys(modified).length) {
      status.textContent = "Add at least one known finding before comparing scenarios.";
      return;
    }
    const button = document.getElementById("wi-run");
    button.disabled = true;
    try {
      const response = await fetch("/api/whatif/evaluate", {
        method: "POST", credentials: "same-origin",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({baseline, modified})
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
      results.innerHTML = renderScenario("Original patient state", data.baseline) + renderScenario("Modified what-if state", data.modified);
      const changed = data.baseline.best_disease !== data.modified.best_disease;
      status.textContent = changed
        ? "The leading RL disease prediction changed in this hypothetical scenario."
        : "The leading RL disease prediction remained the same in this hypothetical scenario.";
    } catch (error) {
      status.textContent = "What-if analysis failed: " + error.message;
    } finally {
      button.disabled = false;
    }
  });

  document.getElementById("wi-clear").addEventListener("click", () => {
    findings.clear();
    results.innerHTML = "";
    status.textContent = "Scenario cleared. No consultation records were changed.";
    renderFindings();
  });

  loadFeatures().catch(error => {
    status.textContent = "Could not load clinical features: " + error.message;
  });
})();
