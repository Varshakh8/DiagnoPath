
"""
DiagnoPath AI - Doctor application + Member 2 manual RL consultation.

Existing dataset-based consultation endpoints are preserved.
Manual consultations and their clinical timelines are persisted in SQLite.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from uuid import uuid4
from app import xai_service
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from starlette.templating import Jinja2Templates

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import clinic, db, rl_service
from env.data_model import META
from env.diagnostic_env import (
    Action,
    ActionType,
    Observation,
    encode_state_for_rl,
)

# =========================================================
# CONFIGURATION
# =========================================================

APP_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))

MANUAL_DB = ROOT / "manual_consultations.sqlite3"

app = FastAPI(
    title="DiagnoPath AI",
    version="1.1.0",
)

app.add_middleware(
    SessionMiddleware,
    secret_key="diagnopath-member1-change-in-production",
    same_site="lax",
    https_only=False,
)

app.mount(
    "/static",
    StaticFiles(directory=str(APP_DIR / "static")),
    name="static",
)


# =========================================================
# PERSISTENT MANUAL CONSULTATION STORAGE
# =========================================================

def manual_db_connection():
    connection = sqlite3.connect(str(MANUAL_DB), timeout=10)
    connection.row_factory = sqlite3.Row
    return connection


def init_manual_db():
    with manual_db_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS manual_cases (
                case_id TEXT PRIMARY KEY,
                doctor_id INTEGER NOT NULL,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)


def save_manual_case(case_id: str, case: dict):
    payload = json.dumps(case)

    with manual_db_connection() as connection:
        connection.execute("""
            INSERT INTO manual_cases (case_id, doctor_id, payload)
            VALUES (?, ?, ?)
            ON CONFLICT(case_id) DO UPDATE SET
                doctor_id = excluded.doctor_id,
                payload = excluded.payload,
                updated_at = CURRENT_TIMESTAMP
        """, (
            case_id,
            case["doctor_id"],
            payload,
        ))


def load_manual_case(case_id: str, doctor_id: int):
    with manual_db_connection() as connection:
        row = connection.execute("""
            SELECT payload
            FROM manual_cases
            WHERE case_id = ? AND doctor_id = ?
        """, (case_id, doctor_id)).fetchone()

    if row is None:
        raise KeyError("Manual consultation not found")

    return json.loads(row["payload"])


# =========================================================
# STARTUP
# =========================================================

@app.on_event("startup")
def startup():
    db.init_db()
    clinic.dataset()
    init_manual_db()


# =========================================================
# AUTHENTICATION
# =========================================================

def current_doctor(request: Request):
    doctor_id = request.session.get("doctor_id")

    if not doctor_id:
        return None

    return db.get_doctor(int(doctor_id))


def require_doctor(request: Request):
    return current_doctor(request)


# =========================================================
# EXISTING PAGES
# =========================================================

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    destination = "/app" if current_doctor(request) else "/login"
    return RedirectResponse(destination, status_code=302)


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    if current_doctor(request):
        return RedirectResponse("/app", status_code=302)

    return templates.TemplateResponse(
        request=request,
        name="login.html",
        context={"request": request, "error": None},
    )


@app.post("/login", response_class=HTMLResponse)
def login_submit(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
):
    doctor = db.authenticate(username, password)

    if not doctor:
        return templates.TemplateResponse(
            request=request,
            name="login.html",
            context={
                "request": request,
                "error": "Invalid username or password.",
            },
            status_code=401,
        )

    request.session["doctor_id"] = doctor["id"]

    return RedirectResponse("/app", status_code=302)


@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=302)


@app.get("/app", response_class=HTMLResponse)
def dashboard(request: Request):
    doctor = current_doctor(request)

    if not doctor:
        return RedirectResponse("/login", status_code=302)

    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "request": request,
            "doctor": doctor,
            "catalog": clinic.catalog(),
            "stats": db.case_stats(doctor["id"]),
        },
    )


@app.get("/app/consult", response_class=HTMLResponse)
def consult_page(request: Request, case_id: str = ""):
    doctor = current_doctor(request)

    if not doctor:
        return RedirectResponse("/login", status_code=302)

    if not case_id:
        return RedirectResponse("/app", status_code=302)

    return templates.TemplateResponse(
        request=request,
        name="consult.html",
        context={
            "request": request,
            "doctor": doctor,
            "case_id": case_id,
        },
    )


@app.get("/app/history", response_class=HTMLResponse)
def history_page(request: Request):
    doctor = current_doctor(request)

    if not doctor:
        return RedirectResponse("/login", status_code=302)

    return templates.TemplateResponse(
        request=request,
        name="history.html",
        context={
            "request": request,
            "doctor": doctor,
            "cases": db.list_cases(doctor["id"]),
        },
    )


# =========================================================
# EXISTING JSON APIs
# =========================================================

@app.get("/api/me")
def api_me(request: Request):
    doctor = current_doctor(request)

    if not doctor:
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    return doctor


@app.get("/api/catalog")
def api_catalog(request: Request):
    if not current_doctor(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    return clinic.catalog()


@app.get("/api/patients")
def api_patients(
    request: Request,
    q: str = "",
    offset: int = 0,
    limit: int = 20,
):
    if not current_doctor(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    limit = max(1, min(limit, 50))
    return clinic.list_patients(q, offset, limit)


@app.post("/api/cases")
async def api_start_case(request: Request):
    doctor = current_doctor(request)

    if not doctor:
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    try:
        body = await request.json()
    except Exception:
        body = {}

    try:
        return clinic.start_case(
            doctor["id"],
            body.get("patient_index"),
        )
    except (IndexError, ValueError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


@app.get("/api/cases/{case_id}")
def api_get_case(request: Request, case_id: str):
    doctor = current_doctor(request)

    if not doctor:
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    try:
        return clinic.get_case(case_id, doctor["id"])
    except KeyError:
        return JSONResponse(
            {"error": "consultation not found"},
            status_code=404,
        )


@app.post("/api/cases/{case_id}/actions")
async def api_action(request: Request, case_id: str):
    doctor = current_doctor(request)

    if not doctor:
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    action_type = str(body.get("type", "")).lower()
    target = str(body.get("target", "")).strip()

    if not target:
        return JSONResponse({"error": "target is required"}, status_code=400)

    try:
        return clinic.apply_action(
            case_id,
            doctor["id"],
            action_type,
            target,
        )
    except KeyError:
        return JSONResponse(
            {"error": "consultation not found"},
            status_code=404,
        )
    except (RuntimeError, ValueError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


@app.get("/api/history")
def api_history(request: Request):
    doctor = current_doctor(request)

    if not doctor:
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    return {
        "cases": db.list_cases(doctor["id"]),
        "stats": db.case_stats(doctor["id"]),
    }


# =========================================================
# MEMBER 2 - EXISTING RL SUGGESTION API
# =========================================================

@app.get("/api/rl/suggest")
def api_rl_suggest(request: Request, case_id: str):
    doctor = current_doctor(request)

    if not doctor:
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    try:
        return rl_service.suggest_action(case_id, doctor["id"])
    except KeyError:
        return JSONResponse(
            {"error": "consultation not found"},
            status_code=404,
        )
    except FileNotFoundError as exc:
        return JSONResponse({"error": str(exc)}, status_code=503)


# =========================================================
# MEMBER 2 - MANUAL SEQUENTIAL RL CONSULTATION
# =========================================================

@app.get("/app/manual-consult", response_class=HTMLResponse)
def manual_consult_page(request: Request):
    if not current_doctor(request):
        return RedirectResponse("/login", status_code=302)

    return HTMLResponse("""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DiagnoPath Manual Consultation</title>
<style>
body {
    font-family: Arial, sans-serif;
    max-width: 780px;
    margin: 35px auto;
    padding: 20px;
    background: #f4f6fb;
    color: #222;
}
.panel {
    background: white;
    padding: 24px;
    border-radius: 12px;
    box-shadow: 0 2px 12px #0000000c;
}
button, a {
    display: inline-block;
    padding: 12px 18px;
    margin: 8px 8px 8px 0;
    cursor: pointer;
}
#question { font-size: 22px; }
#status, #details { white-space: pre-wrap; }
.small { color: #555; font-size: 14px; }
</style>
</head>
<body>
<h1>DiagnoPath AI</h1>
<p>Manual sequential consultation powered by the trained RL agent.</p>

<div class="panel">
    <p id="status">Starting consultation...</p>
    <h2 id="question"></h2>

    <div id="answers" hidden>
        <button id="yes">Yes / Positive</button>
        <button id="no">No / Negative</button>
    </div>

    <p id="details"></p>
    <button id="new-case">Start new consultation</button>
    <a href="/app">Dashboard</a>

    <p class="small">
    Experimental research output only. This is not a clinically
    validated diagnostic tool. Test results must come from actual
    measurements, not guesses.
    </p>
</div>

<script>
let caseId = null;
let busy = false;

async function api(url, options = {}) {
    const response = await fetch(url, options);
    const data = await response.json();

    if (!response.ok) {
        throw new Error(data.error || "Request failed");
    }
    return data;
}

function showError(error) {
    document.getElementById("status").textContent =
        "Error: " + error.message;
    document.getElementById("answers").hidden = true;
}

async function startNew() {
    if (busy) return;
    busy = true;

    document.getElementById("answers").hidden = true;
    document.getElementById("question").textContent = "";
    document.getElementById("details").textContent = "";
    document.getElementById("status").textContent =
        "Creating a persistent consultation...";

    try {
        const data = await api("/api/manual-cases", {
            method: "POST"
        });

        caseId = data.case_id;
        await nextAction();
    } catch (error) {
        showError(error);
    } finally {
        busy = false;
    }
}

async function nextAction() {
    const data = await api(
        "/api/manual-cases/" + encodeURIComponent(caseId) + "/next"
    );

    document.getElementById("answers").hidden = true;

    if (data.status === "diagnosis") {
        document.getElementById("status").textContent =
            "Consultation complete";

        document.getElementById("question").textContent =
            "Predicted disease: " + data.prediction;

        document.getElementById("details").textContent =
            "RL algorithm: Linear Q-Learning\\n" +
            "Answered symptoms: " + data.symptoms_answered +
            "\\nInvestigations recorded: " + data.tests_answered +
            "\\n\\nThe output is experimental and is not a medical diagnosis.";

        return;
    }

    document.getElementById("status").textContent =
        data.type === "ask"
            ? "RL agent selected: Ask symptom"
            : "RL agent selected: Recommend investigation";

    document.getElementById("question").textContent =
        data.type === "ask"
            ? "Does the patient have " + data.label + "?"
            : "What is the result of " + data.label + "?";

    document.getElementById("details").textContent =
        "Step: " + data.step +
        "\\nPreviously answered symptoms: " + data.symptoms_answered +
        "\\nInvestigations recorded: " + data.tests_answered;

    document.getElementById("answers").hidden = false;
}

async function answer(value) {
    if (busy || !caseId) return;
    busy = true;
    document.getElementById("answers").hidden = true;

    try {
        await api(
            "/api/manual-cases/" + encodeURIComponent(caseId) + "/answer",
            {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({answer: value})
            }
        );

        await nextAction();
    } catch (error) {
        showError(error);
    } finally {
        busy = false;
    }
}

document.getElementById("yes").addEventListener("click", () => answer(true));
document.getElementById("no").addEventListener("click", () => answer(false));
document.getElementById("new-case").addEventListener("click", startNew);

startNew();
</script>
</body>
</html>
""")


def _manual_case_for(case_id: str, doctor_id: int):
    return load_manual_case(case_id, doctor_id)


def _manual_observation(case: dict) -> Observation:
    return Observation(
        patient_id=case["patient_id"],
        known_symptoms=dict(case["symptoms"]),
        known_tests=dict(case["tests"]),
        step=case["step"],
        done=case["done"],
        last_reward=0.0,
        message="Manually entered patient findings",
        disease_options=list(META.diseases),
    )


@app.post("/api/manual-cases")
def start_manual_case(request: Request):
    doctor = current_doctor(request)

    if not doctor:
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    case_id = uuid4().hex

    case = {
        "doctor_id": doctor["id"],
        "patient_id": "manual-" + case_id[:8],
        "symptoms": {
            name: None for name in META.inquiry_symptoms
        },
        "tests": {
            name: None for name in META.investigation_features
        },
        "step": 0,
        "pending": None,
        "done": False,
        "prediction": None,
        "timeline": ["Consultation started."],
    }

    save_manual_case(case_id, case)

    return {
        "case_id": case_id,
        "status": "started",
        "message": "Manual consultation saved to SQLite.",
    }


@app.get("/api/manual-cases/{case_id}")
def get_manual_case(request: Request, case_id: str):
    doctor = current_doctor(request)

    if not doctor:
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    try:
        case = _manual_case_for(case_id, doctor["id"])
    except KeyError:
        return JSONResponse(
            {"error": "Manual consultation not found"},
            status_code=404,
        )

    return {
        "case_id": case_id,
        "patient_id": case["patient_id"],
        "step": case["step"],
        "done": case["done"],
        "prediction": case["prediction"],
        "symptoms": case["symptoms"],
        "tests": case["tests"],
        "pending": case["pending"],
        "timeline": case.get("timeline", ["Consultation started."]),
    }


@app.get("/api/manual-cases/{case_id}/next")
def next_manual_action(request: Request, case_id: str):
    doctor = current_doctor(request)

    if not doctor:
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    try:
        case = _manual_case_for(case_id, doctor["id"])
    except KeyError:
        return JSONResponse(
            {"error": "Manual consultation not found"},
            status_code=404,
        )

    if case["done"]:
        return {
            "status": "diagnosis",
            "prediction": case["prediction"],
            "step": case["step"],
            "symptoms_answered": sum(
                value is not None for value in case["symptoms"].values()
            ),
            "tests_answered": sum(
                value is not None for value in case["tests"].values()
            ),
        }

    # If an action is waiting for an answer, return the same action.
    if case["pending"] is not None:
        pending = case["pending"]

        return {
            "status": "question",
            "type": pending["type"],
            "target": pending["target"],
            "label": pending["target"].replace("_", " ").title(),
            "step": case["step"],
            "symptoms_answered": sum(
                value is not None for value in case["symptoms"].values()
            ),
            "tests_answered": sum(
                value is not None for value in case["tests"].values()
            ),
        }

    observation = _manual_observation(case)
    state = encode_state_for_rl(observation)

    actions = [
        Action(ActionType.ASK_SYMPTOM, name)
        for name, value in case["symptoms"].items()
        if value is None
    ]

    actions.extend(
        Action(ActionType.ORDER_TEST, name)
        for name, value in case["tests"].items()
        if value is None
    )

    # Preserve the existing minimum-step diagnosis rule.
    if case["step"] >= 3 or not actions:
        actions.extend(
            Action(ActionType.DIAGNOSE, disease)
            for disease in META.diseases
        )

    if not actions:
        return JSONResponse(
            {"error": "No available diagnostic actions"},
            status_code=400,
        )

    try:
        trained_agent = rl_service.agent()
        chosen = trained_agent.choose_action(
            state=state,
            actions=actions,
            training=False,
        )
    except FileNotFoundError as exc:
        return JSONResponse({"error": str(exc)}, status_code=503)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)

    if chosen.type == ActionType.DIAGNOSE:
        case["done"] = True
        case["prediction"] = chosen.target
        case["pending"] = None

        case.setdefault("timeline", ["Consultation started."])
        case["timeline"].append(
            f"RL agent predicted: {chosen.target}"
        )

        save_manual_case(case_id, case)

        return {
            "status": "diagnosis",
            "prediction": chosen.target,
            "step": case["step"],
            "symptoms_answered": sum(
                value is not None for value in case["symptoms"].values()
            ),
            "tests_answered": sum(
                value is not None for value in case["tests"].values()
            ),
        }

    action_type = (
        "ask"
        if chosen.type == ActionType.ASK_SYMPTOM
        else "test"
    )

    case["pending"] = {
        "type": action_type,
        "target": chosen.target,
    }

    save_manual_case(case_id, case)

    return {
        "status": "question",
        "type": action_type,
        "target": chosen.target,
        "label": chosen.target.replace("_", " ").title(),
        "step": case["step"],
        "symptoms_answered": sum(
            value is not None for value in case["symptoms"].values()
        ),
        "tests_answered": sum(
            value is not None for value in case["tests"].values()
        ),
    }


@app.post("/api/manual-cases/{case_id}/answer")
async def answer_manual_question(request: Request, case_id: str):
    doctor = current_doctor(request)

    if not doctor:
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    try:
        case = _manual_case_for(case_id, doctor["id"])
    except KeyError:
        return JSONResponse(
            {"error": "Manual consultation not found"},
            status_code=404,
        )

    if case["done"]:
        return JSONResponse(
            {"error": "Consultation is already complete"},
            status_code=400,
        )

    if case["pending"] is None:
        return JSONResponse(
            {"error": "There is no pending question"},
            status_code=400,
        )

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    value = body.get("answer")

    if not isinstance(value, bool):
        return JSONResponse(
            {"error": "answer must be true or false"},
            status_code=400,
        )

    pending = case["pending"]
    target = pending["target"]

    if pending["type"] == "ask":
        case["symptoms"][target] = int(value)
        result_text = "Present" if value else "Absent"
        event = f"Symptom recorded — {target.replace('_', ' ').title()}: {result_text}"

    elif pending["type"] == "test":
        case["tests"][target] = int(value)
        result_text = "Positive" if value else "Negative"
        event = f"Investigation recorded — {target.replace('_', ' ').title()}: {result_text}"

    else:
        return JSONResponse(
            {"error": "Unknown pending action type"},
            status_code=400,
        )

    case["step"] += 1
    case["pending"] = None

    case.setdefault("timeline", ["Consultation started."])
    case["timeline"].append(event)

    save_manual_case(case_id, case)

    return {
        "status": "recorded",
        "step": case["step"],
        "symptoms_answered": sum(
            value is not None for value in case["symptoms"].values()
        ),
        "tests_answered": sum(
            value is not None for value in case["tests"].values()
        ),
        "timeline": case["timeline"],
        "message": "Response and clinical timeline saved.",
    }


# =========================================================
# MEMBER 3 - XAI INTEGRATION POINT
# =========================================================


@app.get("/api/xai/explain")
def api_xai_explain(request: Request, case_id: str):
    """Explain a manual or live RL consultation."""

    doctor = current_doctor(request)

    if not doctor:
        return JSONResponse(
            {"error": "unauthorized"},
            status_code=401,
        )

    # First, check whether this is a manual RL consultation.
    try:
        manual_case = _manual_case_for(case_id, doctor["id"])
    except KeyError:
        manual_case = None

    if manual_case is not None:
        try:
            return xai_service.explain_manual_case(manual_case)
        except FileNotFoundError as exc:
            return JSONResponse(
                {"error": str(exc)},
                status_code=503,
            )
        except (ValueError, IndexError) as exc:
            return JSONResponse(
                {"error": str(exc)},
                status_code=500,
            )

    # Otherwise, preserve the existing live consultation flow.
    try:
        return xai_service.explain_action(case_id, doctor["id"])
    except KeyError:
        return JSONResponse(
            {"error": "Consultation not found in manual or live records."},
            status_code=404,
        )
    except FileNotFoundError as exc:
        return JSONResponse(
            {"error": str(exc)},
            status_code=503,
        )
