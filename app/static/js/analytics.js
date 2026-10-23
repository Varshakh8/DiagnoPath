(() => {
  "use strict";

  const content = document.getElementById("analytics-content");
  const refreshButton = document.getElementById("refresh-analytics");

  if (!content) {
    console.error("Analytics: #analytics-content was not found.");
    return;
  }

  const escapeHTML = (value) =>
    String(value ?? "").replace(/[&<>"']/g, (c) => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;"
    })[c]);

  const number = (value) => {
    const n = Number(value);
    return Number.isFinite(n)
      ? n.toLocaleString(undefined, { maximumFractionDigits: 2 })
      : "0";
  };

  const percent = (value) => {
    const n = Number(value);
    return Number.isFinite(n) ? `${n.toFixed(1)}%` : "0.0%";
  };

  function renderError(message) {
    content.innerHTML = `
      <p role="alert"><strong>Unable to load consultation analytics.</strong></p>
      <p>${escapeHTML(message)}</p>
      <p>Check that you are signed in and the server is running.</p>
    `;
  }

  function render(data) {
    const cases = Array.isArray(data.cases) ? data.cases : [];
    const stats = data.stats || {};

    const completed = Number(stats.completed_cases ?? cases.length) || 0;
    const correct = Number(stats.correct_cases ?? cases.filter(
      c => c.correct === true || c.correct === 1 || c.correct === "1"
    ).length) || 0;
    const incorrect = Math.max(0, completed - correct);
    const accuracy = Number(
      stats.accuracy_pct ??
      (completed ? (correct / completed) * 100 : 0)
    ) || 0;

    const avgCost = Number(stats.avg_cost) || 0;
    const avgSteps = Number(stats.avg_steps) || 0;

    const sumRecent = (key) => cases.reduce(
      (total, c) => total + (Number(c[key]) || 0),
      0
    );

    const recentCost = sumRecent("total_cost");
    const recentSymptoms = sumRecent("symptoms_asked");
    const recentTests = sumRecent("tests_ordered");

    const correctWidth = completed
      ? Math.min(100, Math.max(0, correct / completed * 100))
      : 0;
    const incorrectWidth = completed
      ? Math.min(100, Math.max(0, incorrect / completed * 100))
      : 0;

    const recentCosts = cases.slice(0, 8);
    const maxCost = Math.max(
      1,
      ...recentCosts.map(c => Number(c.total_cost) || 0)
    );

    const costChart = recentCosts.length
      ? recentCosts.map((c, i) => {
          const cost = Math.max(0, Number(c.total_cost) || 0);
          const width = Math.min(100, cost / maxCost * 100);
          const id = c.id ?? c.case_id ?? `Recent ${i + 1}`;

          return `
            <div style="margin:12px 0;">
              <div style="display:flex;justify-content:space-between;gap:12px;font-size:14px;">
                <span>${escapeHTML(`Case #${id}`)}</span>
                <strong>${escapeHTML(number(cost))}</strong>
              </div>
              <div style="height:10px;background:var(--line,#e5e7eb);border-radius:8px;overflow:hidden;margin-top:5px;">
                <div style="height:100%;width:${width}%;background:#287d8e;border-radius:8px;"></div>
              </div>
            </div>
          `;
        }).join("")
      : "<p>No consultation history is available yet.</p>";

    content.innerHTML = `
      <div class="stat-grid">
        <article>
          <span>Completed cases (all history)</span>
          <strong>${escapeHTML(number(completed))}</strong>
        </article>
        <article>
          <span>Correct diagnoses (all history)</span>
          <strong>${escapeHTML(number(correct))}</strong>
        </article>
        <article>
          <span>Diagnosis accuracy</span>
          <strong>${escapeHTML(percent(accuracy))}</strong>
        </article>
        <article>
          <span>Average diagnostic cost</span>
          <strong>${escapeHTML(number(avgCost))}</strong>
        </article>
        <article>
          <span>Average steps per case</span>
          <strong>${escapeHTML(number(avgSteps))}</strong>
        </article>
        <article>
          <span>Cost total (latest 40 cases)</span>
          <strong>${escapeHTML(number(recentCost))}</strong>
        </article>
        <article>
          <span>Symptoms asked (latest 40)</span>
          <strong>${escapeHTML(number(recentSymptoms))}</strong>
        </article>
        <article>
          <span>Tests ordered (latest 40)</span>
          <strong>${escapeHTML(number(recentTests))}</strong>
        </article>
      </div>

      <section style="margin:20px 0;">
        <h3>Diagnosis outcomes</h3>
        <p>Correct: <strong>${escapeHTML(number(correct))}</strong>
          &middot; Incorrect: <strong>${escapeHTML(number(incorrect))}</strong>
        </p>
        <div style="display:flex;height:22px;width:100%;border-radius:8px;overflow:hidden;background:var(--line,#e5e7eb);">
          <div title="Correct diagnoses" style="width:${correctWidth}%;background:#25866d;"></div>
          <div title="Incorrect diagnoses" style="width:${incorrectWidth}%;background:#c65b5b;"></div>
        </div>
        <p class="muted" style="font-size:13px;">
          Green represents correct diagnoses; red represents incorrect diagnoses.
          Accuracy uses all completed cases reported by the API.
        </p>
      </section>

      <section style="margin:20px 0;">
        <h3>Recent diagnostic costs</h3>
        <p class="muted">Up to 8 most recent cases. Bar lengths are scaled against the highest cost shown.</p>
        ${costChart}
      </section>

      <p class="muted" style="font-size:13px;">
        Overall statistics use the API's full-history aggregates.
        Cost, symptom, and test totals use the latest ${escapeHTML(number(cases.length))} returned records.
      </p>
    `;
  }

  async function loadAnalytics() {
    content.innerHTML = "<p>Loading consultation analytics...</p>";
    if (refreshButton) refreshButton.disabled = true;

    try {
      const response = await fetch("/api/history", {
        headers: { Accept: "application/json" },
        cache: "no-store",
        credentials: "same-origin"
      });

      if (!response.ok) {
        throw new Error(`History API returned HTTP ${response.status}.`);
      }

      const data = await response.json();

      if (!data || typeof data !== "object") {
        throw new Error("Unexpected response from the history API.");
      }

      render(data);
    } catch (error) {
      console.error("Analytics loading failed:", error);
      renderError(error.message || "Unexpected error.");
    } finally {
      if (refreshButton) refreshButton.disabled = false;
    }
  }

  if (refreshButton) {
    refreshButton.addEventListener("click", loadAnalytics);
  }

  loadAnalytics();
})();
