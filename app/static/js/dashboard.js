
"use strict";

document.addEventListener("DOMContentLoaded", () => {
  const button = document.getElementById("start-consultation");
  const status = document.getElementById("start-status");
  const patientInput = document.getElementById("patient-ref");

  if (!button || !status || !patientInput) {
    console.error("Missing dashboard elements:", {
      button,
      status,
      patientInput
    });
    return;
  }

  button.addEventListener("click", async () => {
    button.disabled = true;
    status.textContent = "Creating consultation...";

    try {
      const response = await fetch("/api/manual-cases", {
        method: "POST",
        credentials: "same-origin",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          patient_ref: patientInput.value.trim()
        })
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || `HTTP ${response.status}`);
      }

      if (!data.case_id) {
        throw new Error("No consultation ID returned by the server.");
      }

      window.location.href =
        `/app/manual-consult?case_id=${encodeURIComponent(data.case_id)}`;

    } catch (error) {
      console.error("Consultation start failed:", error);
      status.textContent =
        "Could not start consultation: " + error.message;
      button.disabled = false;
    }
  });
});
