const tbody = document.getElementById("patient-rows");
const qInput = document.getElementById("patient-q");
const pageLabel = document.getElementById("page-label");
let offset = 0;
const limit = 12;

async function startCase(patientIndex) {
  const body = patientIndex == null ? {} : { patient_index: patientIndex };
  const res = await fetch("/api/cases", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok) {
    alert(data.error || "Could not open case");
    return;
  }
  window.location = `/app/consult?case_id=${data.case_id}`;
}

async function loadPatients() {
  const q = qInput.value.trim();
  const res = await fetch(`/api/patients?q=${encodeURIComponent(q)}&offset=${offset}&limit=${limit}`);
  const data = await res.json();
  if (!data.patients.length) {
    tbody.innerHTML = `<tr><td colspan="4">No matching patients.</td></tr>`;
  } else {
    tbody.innerHTML = data.patients
      .map(
        (p) => `<tr>
          <td><strong>${p.patient_id}</strong></td>
          <td>${p.index}</td>
          <td>${p.split}</td>
          <td><button class="ghost open-case" data-index="${p.index}">Open case</button></td>
        </tr>`
      )
      .join("");
  }
  pageLabel.textContent = `${data.offset + 1}–${Math.min(data.offset + limit, data.total)} of ${data.total}`;
  tbody.querySelectorAll(".open-case").forEach((btn) => {
    btn.addEventListener("click", () => startCase(Number(btn.dataset.index)));
  });
}

document.getElementById("random-case").addEventListener("click", () => startCase(null));
document.getElementById("prev-page").addEventListener("click", () => {
  offset = Math.max(0, offset - limit);
  loadPatients();
});
document.getElementById("next-page").addEventListener("click", () => {
  offset += limit;
  loadPatients();
});
qInput.addEventListener("input", () => {
  offset = 0;
  loadPatients();
});
loadPatients();
