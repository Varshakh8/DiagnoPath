(() => {
  "use strict";

  const container = document.getElementById("whatif-content");
  if (!container) return;

  container.innerHTML = `
    <form id="whatif-form">
      <p class="muted">
        The baseline is loaded from your consultation history when available.
        Change the values to explore a hypothetical cost scenario.
      </p>

      <p>
        <label for="whatif-baseline">Baseline consultation cost</label><br>
        <input id="whatif-baseline" type="number" min="0" step="0.01"
          value="0" required>
      </p>

      <p>
        <label for="whatif-count">Hypothetical additional investigations</label><br>
        <input id="whatif-count" type="number" min="0" max="1000" step="1"
          value="1" required>
      </p>

      <p>
        <label for="whatif-unit-cost">Cost per additional investigation</label><br>
        <input id="whatif-unit-cost" type="number" min="0" step="0.01"
          value="3" required>
      </p>

      <button class="primary" type="submit">Calculate scenario</button>
      <button id="whatif-reset" type="button">Reset</button>
    </form>

    <p id="whatif-source" class="muted" aria-live="polite">
      Loading consultation history...
    </p>

    <div id="whatif-result" aria-live="polite">
      Enter the scenario values and calculate.
    </div>
  `;

  const form = document.getElementById("whatif-form");
  const baseline = document.getElementById("whatif-baseline");
  const count = document.getElementById("whatif-count");
  const unitCost = document.getElementById("whatif-unit-cost");
  const result = document.getElementById("whatif-result");
  const source = document.getElementById("whatif-source");
  const reset = document.getElementById("whatif-reset");

  let loadedAverage = null;

  const money = value => Number(value).toLocaleString(undefined, {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2
  });

  async function loadBaseline() {
    try {
      const response = await fetch("/api/history", {
        headers: { Accept: "application/json" },
        credentials: "same-origin",
        cache: "no-store"
      });

      if (!response.ok) {
        throw new Error(`History API returned HTTP ${response.status}.`);
      }

      const data = await response.json();
      const average = Number(data?.stats?.avg_cost);

      if (data?.stats?.avg_cost !== undefined &&
          data?.stats?.avg_cost !== null &&
          Number.isFinite(average) && average >= 0) {
        loadedAverage = average;
        baseline.value = average.toFixed(2);
        source.textContent =
          "Baseline loaded from the average cost of your completed consultations.";
      } else {
        source.textContent =
          "No valid historical average was returned. Enter a baseline cost manually.";
      }
    } catch (error) {
      console.error("What-if history loading failed:", error);
      source.textContent =
        "Could not load consultation history. You can still enter a baseline manually.";
    }
  }

  function calculate(event) {
    if (event) event.preventDefault();

    const base = Number(baseline.value);
    const investigations = Number(count.value);
    const perInvestigation = Number(unitCost.value);

    if (
      baseline.value.trim() === "" ||
      count.value.trim() === "" ||
      unitCost.value.trim() === "" ||
      !Number.isFinite(base) ||
      !Number.isInteger(investigations) ||
      !Number.isFinite(perInvestigation) ||
      base < 0 ||
      investigations < 0 ||
      investigations > 1000 ||
      perInvestigation < 0
    ) {
      result.innerHTML =
        "<p role='alert'>Enter valid non-negative costs and a whole number of investigations (0–1000).</p>";
      return;
    }

    const additional = investigations * perInvestigation;
    const projected = base + additional;
    const difference = projected - base;
    const maxValue = Math.max(base, projected, 1);
    const baseWidth = Math.min(100, base / maxValue * 100);
    const projectedWidth = Math.min(100, projected / maxValue * 100);

    result.innerHTML = `
      <h3>Scenario results</h3>

      <div class="stat-grid">
        <article>
          <span>Baseline cost</span>
          <strong>${money(base)}</strong>
        </article>
        <article>
          <span>Additional cost</span>
          <strong>${money(additional)}</strong>
        </article>
        <article>
          <span>Projected total</span>
          <strong>${money(projected)}</strong>
        </article>
        <article>
          <span>Cost increase</span>
          <strong>+${money(difference)}</strong>
        </article>
      </div>

      <h3>Cost comparison</h3>
      <div style="margin:12px 0;">
        <div style="display:flex;justify-content:space-between;gap:12px;">
          <span>Baseline</span><strong>${money(base)}</strong>
        </div>
        <div style="height:12px;background:var(--line,#e5e7eb);border-radius:8px;overflow:hidden;margin-top:5px;">
          <div style="height:100%;width:${baseWidth}%;background:#287d8e;border-radius:8px;"></div>
        </div>
      </div>

      <div style="margin:12px 0;">
        <div style="display:flex;justify-content:space-between;gap:12px;">
          <span>Hypothetical total</span><strong>${money(projected)}</strong>
        </div>
        <div style="height:12px;background:var(--line,#e5e7eb);border-radius:8px;overflow:hidden;margin-top:5px;">
          <div style="height:100%;width:${projectedWidth}%;background:#25866d;border-radius:8px;"></div>
        </div>
      </div>

      <p class="muted">
        This is a cost-only illustration based on the values entered.
        It does not recommend investigations or determine whether a test
        can be omitted. No saved consultation records were changed.
      </p>
    `;
  }

  form.addEventListener("submit", calculate);

  reset.addEventListener("click", () => {
    baseline.value = loadedAverage === null ? "0" : loadedAverage.toFixed(2);
    count.value = "1";
    unitCost.value = "3";
    result.textContent = "Enter the scenario values and calculate.";
  });

  loadBaseline();
})();
