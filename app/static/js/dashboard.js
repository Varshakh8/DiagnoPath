"use strict";
document.getElementById("start-consultation").addEventListener("click", async () => {
  const button = document.getElementById("start-consultation");
  const status = document.getElementById("start-status");
  button.disabled = true;
  status.textContent = "Creating consultation…";
  try {
    const response = await fetch("/api/manual-cases", {
      method: "POST",
      credentials: "same-origin",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({patient_ref: document.getElementById("patient-ref").value.trim()})
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
    window.location.href = `/app/manual-consult?case_id=${encodeURIComponent(data.case_id)}`;
  } catch (error) {
    status.textContent = "Could not start consultation: " + error.message;
    button.disabled = false;
  }
});
