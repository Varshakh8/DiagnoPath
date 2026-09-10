# DiagnoPath AI - Member 1: Clinical Environment + Doctor App

You own the **foundation** and the **usable clinic application**:

1. Multi-disease Kaggle dataset
2. Sequential diagnostic environment (`reset` / `step` / rewards)
3. Doctor login, backend APIs, and clinical frontend
4. Hooks so Members 2–4 can continue (RL, XAI, richer dashboard)

No built-in ML algorithms.

---

## Run the doctor application

```bash
cd "Module1"
pip install -r requirements.txt
python run_app.py
```

Open http://127.0.0.1:8000

**Demo login**

| Username | Password |
|----------|----------|
| `doctor` | `clinic123` |
| `dr.mehta` | `clinic123` |

Workflow: sign in → open a patient case → ask symptoms → order tests → diagnose. The disease is hidden until you submit a diagnosis.

---

## Dataset (Kaggle)

https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning

| Item | Count |
|------|------:|
| Patients | 4962 |
| Symptoms | 132 |
| Diseases | 41 |

Ask symptom cost = 1. Investigation cost = 3.

---

## What others plug in

| Member | Work | Integration point |
|--------|------|-------------------|
| **You (1)** | Data, environment, login, sequential diagnosis UI | Done |
| **2** | Custom RL agent | `GET /api/rl/suggest` and `encode_state_for_rl()` |
| **3** | Custom XAI | `GET /api/xai/explain` |
| **4** | Extra charts / what-if dashboard | Reuse `/api/cases` JSON |

### REST API (cookie session after login)

- `POST /login` — doctor authentication
- `GET /api/patients` — registry (disease hidden)
- `POST /api/cases` — start consultation (`{"patient_index": 150}` or `{}` for random)
- `POST /api/cases/{id}/actions` — `{"type":"ask"|"test"|"diagnose","target":"..."}`
- `GET /api/history` — completed cases for the logged-in doctor

---

## Folder structure

```
member1_clinical_env/
  app/                 # backend + frontend
  env/                 # diagnostic environment (Member 2 trains on this)
  data/patients.csv
  demo/meeting_demo.py
  run_app.py
```
