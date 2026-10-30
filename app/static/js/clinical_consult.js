(() => {
  "use strict";
  const caseId = window.DIAGNO_CASE_ID;
  const $ = id => document.getElementById(id);
  const esc = value => String(value ?? "").replace(/[&<>"']/g, c => ({
    "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"
  })[c]);
  let busy = false;

  async function api(url, options = {}) {
    const response = await fetch(url, {credentials:"same-origin", ...options});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
    return data;
  }
  function setStatus(message) { $("consult-status").textContent = message || ""; }

  function renderRanking(items) {
    const box = $("disease-ranking");
    if (!items || !items.length) {
      box.innerHTML = '<p class="muted">No ranking available.</p>';
      return;
    }
    box.innerHTML = items.map(item => `
      <div class="score-row">
        <span>${esc(item.disease)}</span>
        <div class="score-track"><div class="score-fill" style="width:${Math.max(0,Math.min(100,Number(item.relative_score)||0))}%"></div></div>
        <strong>${Number(item.relative_score).toFixed(1)}%</strong>
      </div>`).join("");
  }

  function renderFindings(caseData) {
    const box = $("finding-list");
    const rows = [];
    Object.entries(caseData.symptoms || {}).forEach(([name,value]) => {
      if (value !== null) rows.push({name, kind:"Symptom", value});
    });
    Object.entries(caseData.tests || {}).forEach(([name,value]) => {
      if (value !== null) rows.push({name, kind:"Investigation", value});
    });
    if (!rows.length) {
      box.innerHTML = '<p class="muted">No findings recorded yet.</p>';
    } else {
      box.innerHTML = rows.map(item => `<p><span class="tag ${item.value ? "yes":"no"}">${esc(item.kind)} · ${esc(item.name.replaceAll("_"," "))}: ${item.value ? (item.kind==="Investigation"?"Positive":"Present") : (item.kind==="Investigation"?"Negative":"Absent")}</span></p>`).join("");
    }
    (caseData.skipped_tests || []).forEach(name => {
      box.insertAdjacentHTML("beforeend", `<p><span class="tag">Skipped test · ${esc(name.replaceAll("_"," "))}</span></p>`);
    });
    $("clinical-timeline").innerHTML = (caseData.timeline || []).map(item => `<li>${esc(item)}</li>`).join("");
  }

  function renderXaiChart(title, payload, method) {
    if (!payload || payload.status !== "ok") {
      return `<div class="xai-method"><h3>${method}</h3><p class="muted">${esc(payload?.message || "Explanation unavailable. Install the project requirements and retry.")}</p></div>`;
    }
    const items = (payload.features || []).slice(0,8).map(item => ({
      label: item.feature || item.description || "Feature",
      value: Number(item.contribution !== undefined ? item.contribution : item.weight) || 0
    }));
    const max = Math.max(0.0001, ...items.map(x => Math.abs(x.value)));
    const bars = items.map(item => {
      const width = Math.max(1, Math.abs(item.value) / max * 100);
      return `<div class="xai-row"><span>${esc(item.label)}</span><div class="xai-track"><div class="xai-fill ${item.value < 0 ? "negative":""}" style="width:${width}%"></div></div><strong>${item.value.toFixed(4)}</strong></div>`;
    }).join("");
    return `<div class="xai-method"><h3>${method}</h3><p class="muted">${esc(payload.note || "")}</p>${bars || '<p class="muted">No feature contributions returned.</p>'}</div>`;
  }

  async function explainFinal() {
    $("xai-visuals").innerHTML = '<p class="muted">Calculating SHAP and LIME explanations…</p>';
    try {
      const data = await api(`/api/xai/explain?case_id=${encodeURIComponent(caseId)}`);
      const custom = (data.top_positive_contributions || []).concat(data.top_negative_contributions || []);
      const customBlock = custom.length ? `<div class="xai-method"><h3>Linear Q-learning feature contributions</h3>${custom.map(x => `<p>${esc(x.feature)} · ${esc(x.finding)} · ${Number(x.contribution).toFixed(4)}</p>`).join("")}</div>` : "";
      $("xai-visuals").innerHTML =
        `<p class="muted">Explained action: ${esc(data.action?.label || "Final diagnosis")} · Q-score ${esc(data.q_score)}</p>` +
        renderXaiChart("SHAP explanation", data.shap, "SHAP") +
        renderXaiChart("LIME explanation", data.lime, "LIME") + customBlock;
    } catch (error) {
      $("xai-visuals").innerHTML = `<p class="muted">XAI error: ${esc(error.message)}. Check SHAP/LIME installation.</p>`;
    }
  }

  function renderAction(data) {
    $("step-count").textContent = `Step ${data.step || 0}`;
    renderRanking(data.top_diseases || []);
    const card = $("action-card");
    const prediction = $("prediction-card");
    card.innerHTML = "";
    prediction.innerHTML = "";
    if (data.status === "diagnosis") {
      card.innerHTML = '<p class="tag yes">RL diagnostic sequence completed</p>';
      prediction.innerHTML = `<div class="result"><h2>Leading RL prediction</h2><strong>${esc(data.prediction)}</strong><p class="muted">Experimental decision-support output. It is not a confirmed medical diagnosis.</p></div>`;
      $("whatif-panel").hidden = false;
      $("add-known").disabled = true;
      $("known-feature").disabled = true;
      $("known-value").disabled = true;
      explainFinal();
      setStatus("Final prediction generated. Review the explanation and then perform What-if analysis if required.");
      return;
    }
    const isTest = data.type === "test";
    const heading = isTest ? "RL recommends an investigation" : "RL selected a symptom inquiry";
    const prompt = isTest
      ? `The agent recommends: ${data.label}. Perform this investigation and enter its actual result.`
      : `Does the patient have ${data.label}?`;
    card.innerHTML = `<h3>${heading}</h3><p>${esc(prompt)}</p>
      <div style="display:flex;gap:8px;flex-wrap:wrap;">
        <button id="answer-yes" class="primary" type="button">${isTest ? "Record positive result":"Yes — present"}</button>
        <button id="answer-no" type="button">${isTest ? "Record negative result":"No — absent"}</button>
        ${isTest ? '<button id="skip-test" class="ghost" type="button">Skip investigation</button>':""}
      </div>`;
    $("answer-yes").addEventListener("click", () => answer(true));
    $("answer-no").addEventListener("click", () => answer(false));
    if (isTest) $("skip-test").addEventListener("click", skipTest);
    setStatus(isTest ? "Waiting for the doctor to enter the actual test result or skip the test." : "Waiting for the doctor's symptom response.");
  }

  async function refreshCase() {
    const c = await api(`/api/manual-cases/${encodeURIComponent(caseId)}`);
    renderFindings(c);
    return c;
  }
  async function next() {
    const data = await api(`/api/manual-cases/${encodeURIComponent(caseId)}/next`);
    await refreshCase();
    renderAction(data);
  }
  async function answer(value) {
    if (busy) return;
    busy = true;
    setStatus("Saving finding and updating the RL state…");
    try {
      await api(`/api/manual-cases/${encodeURIComponent(caseId)}/answer`, {
        method:"POST", headers:{"Content-Type":"application/json"},
        body:JSON.stringify({answer:value})
      });
      await next();
    } catch (error) { setStatus("Error: " + error.message); }
    finally { busy = false; }
  }
  async function skipTest() {
    if (busy) return;
    busy = true;
    setStatus("Recording skipped investigation…");
    try {
      await api(`/api/manual-cases/${encodeURIComponent(caseId)}/skip-test`, {method:"POST"});
      await next();
    } catch (error) { setStatus("Error: " + error.message); }
    finally { busy = false; }
  }

  async function loadKnownFeatures() {
    const data = await api("/api/whatif/features");
    $("known-feature").innerHTML = data.features.map(f =>
      `<option value="${esc(f.name)}">${esc(f.label)} (${esc(f.kind)})</option>`).join("");
  }
  $("add-known").addEventListener("click", async () => {
    if (busy) return;
    busy = true;
    try {
      await api(`/api/manual-cases/${encodeURIComponent(caseId)}/finding`, {
        method:"POST", headers:{"Content-Type":"application/json"},
        body:JSON.stringify({target:$("known-feature").value, value:Number($("known-value").value)})
      });
      await next();
    } catch (error) { setStatus("Could not add finding: " + error.message); }
    finally { busy = false; }
  });

  (async function init() {
    try {
      if (!caseId) throw new Error("Missing consultation ID.");
      await loadKnownFeatures();
      const existing = await api(`/api/manual-cases/${encodeURIComponent(caseId)}`);
      renderFindings(existing);
      if (existing.done) {
        const data = await api(`/api/manual-cases/${encodeURIComponent(caseId)}/next`);
        renderAction(data);
      } else {
        await next();
      }
    } catch (error) { setStatus("Could not load consultation: " + error.message); }
  })();
})();
