
"""
DiagnoPath AI - Single persistent RL-guided clinical consultation.

The dataset remains available for training/evaluation. The doctor-facing
workflow stores manually entered patient findings and clinical timelines
in SQLite, and uses the evaluated warm-start Linear Q-learning model.
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
@app.get("/app/manual-consult", response_class=HTMLResponse)
def consult_page(request: Request, case_id: str = ""):
    doctor = current_doctor(request)
    if not doctor:
        return RedirectResponse("/login", status_code=302)
    if not case_id:
        return RedirectResponse("/app", status_code=302)
    try:
        case = load_manual_case(case_id, doctor["id"])
    except KeyError:
        return RedirectResponse("/app", status_code=302)
    return templates.TemplateResponse(
        request=request,
        name="clinical_consult.html",
        context={
            "request": request,
            "doctor": doctor,
            "case_id": case_id,
            "patient_ref": case.get("patient_ref", "New patient"),
        },
    )


@app.get("/app/history", response_class=HTMLResponse)
def history_page(request: Request):
    doctor = current_doctor(request)

    if not doctor:
        return RedirectResponse("/login", status_code=302)

    # Existing dataset-based consultation history
    cases = list(db.list_cases(doctor["id"]))

    # Retrieve completed manual RL consultations
    with manual_db_connection() as connection:
        rows = connection.execute(
            """
            SELECT payload, created_at, updated_at
            FROM manual_cases
            WHERE doctor_id = ?
            """,
            (doctor["id"],),
        ).fetchall()

    for row in rows:
        manual_case = json.loads(row["payload"])
        if not manual_case.get("done"):
            continue

        performed_investigations = sum(
        value is not None
        for value in manual_case.get("tests", {}).values()
    )

        investigation_cost = (
        performed_investigations * META.investigation_cost
    )

        cases.append({
        "patient_id": manual_case.get("patient_ref")
            or manual_case.get("patient_id", "—"),
        "diagnosis": manual_case.get("prediction"),
        "true_label": None,
        "correct": None,
        "steps": manual_case.get("step", 0),
        "total_cost": investigation_cost,
        "finished_at": row["updated_at"],
    })

    # Display newest consultations first
    cases.sort(
        key=lambda c: str(c.get("finished_at") or ""),
        reverse=True,
    )

    return templates.TemplateResponse(
        request=request,
        name="history.html",
        context={
            "request": request,
            "doctor": doctor,
            "cases": cases,
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
async def start_manual_case(request: Request):
    doctor = current_doctor(request)

    if not doctor:
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    try:
        body = await request.json()
    except Exception:
        body = {}
    patient_ref = str(body.get("patient_ref", "")).strip()[:80]

    case_id = uuid4().hex

    case = {
        "doctor_id": doctor["id"],
        "patient_id": "manual-" + case_id[:8],
        "patient_ref": patient_ref or ("Patient " + case_id[:8].upper()),
        "symptoms": {
            name: None for name in META.inquiry_symptoms
        },
        "tests": {
            name: None for name in META.investigation_features
        },
        "skipped_tests": [],
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
        "patient_ref": case.get("patient_ref", case["patient_id"]),
        "skipped_tests": case.get("skipped_tests", []),
        "step": case["step"],
        "done": case["done"],
        "prediction": case["prediction"],
        "symptoms": case["symptoms"],
        "tests": case["tests"],
        "pending": case["pending"],
        "timeline": case.get("timeline", ["Consultation started."]),
    }


def _manual_disease_ranking(case: dict):
    """Return diagnosis Q-scores and relative normalized scores (not probabilities)."""
    observation = _manual_observation(case)
    state = encode_state_for_rl(observation)
    model = rl_service.agent()
    scores = [
        (disease, float(model.get_q(state, Action(ActionType.DIAGNOSE, disease))))
        for disease in META.diseases
    ]
    scores.sort(key=lambda item: item[1], reverse=True)
    # Stable softmax for a readable relative score display. It is NOT calibrated.
    max_score = scores[0][1] if scores else 0.0
    exp_scores = [__import__("math").exp(max(-50.0, min(0.0, score - max_score))) for _, score in scores]
    total = sum(exp_scores) or 1.0
    return [
        {"disease": disease, "q_score": round(score, 4),
         "relative_score": round(weight / total * 100, 2)}
        for (disease, score), weight in zip(scores, exp_scores)
    ]


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
            "top_diseases": _manual_disease_ranking(case)[:5],
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
            "top_diseases": _manual_disease_ranking(case)[:5],
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
        if value is None and name not in case.get("skipped_tests", [])
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
            "top_diseases": _manual_disease_ranking(case)[:5],
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
        "top_diseases": _manual_disease_ranking(case)[:5],
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




@app.post("/api/manual-cases/{case_id}/finding")
async def add_manual_finding(request: Request, case_id: str):
    doctor = current_doctor(request)
    if not doctor:
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    try:
        case = _manual_case_for(case_id, doctor["id"])
    except KeyError:
        return JSONResponse({"error": "Manual consultation not found"}, status_code=404)
    if case["done"]:
        return JSONResponse({"error": "Consultation is already complete"}, status_code=400)
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)
    target = str(body.get("target", ""))
    value = body.get("value")
    if value not in (0, 1, False, True):
        return JSONResponse({"error": "value must be 0 (absent/negative) or 1 (present/positive)"}, status_code=400)
    if target in case["symptoms"]:
        case["symptoms"][target] = int(bool(value))
        kind = "Symptom"
    elif target in case["tests"]:
        case["tests"][target] = int(bool(value))
        if target in case.get("skipped_tests", []):
            case["skipped_tests"].remove(target)
        kind = "Investigation"
    else:
        return JSONResponse({"error": "Unknown clinical feature"}, status_code=400)
    if case.get("pending") and case["pending"].get("target") == target:
        case["pending"] = None
    case["step"] += 1
    case.setdefault("timeline", []).append(
        f"{kind} added by doctor — {target.replace('_', ' ').title()}: {'Present / Positive' if value else 'Absent / Negative'}"
    )
    save_manual_case(case_id, case)
    return {"status": "recorded", "timeline": case["timeline"], "step": case["step"]}


@app.post("/api/manual-cases/{case_id}/skip-test")
async def skip_manual_test(request: Request, case_id: str):
    doctor = current_doctor(request)
    if not doctor:
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    try:
        case = _manual_case_for(case_id, doctor["id"])
    except KeyError:
        return JSONResponse({"error": "Manual consultation not found"}, status_code=404)
    if case["done"] or not case.get("pending") or case["pending"].get("type") != "test":
        return JSONResponse({"error": "There is no pending investigation to skip"}, status_code=400)
    target = case["pending"]["target"]
    case.setdefault("skipped_tests", [])
    if target not in case["skipped_tests"]:
        case["skipped_tests"].append(target)
    case["pending"] = None
    case["step"] += 1
    case.setdefault("timeline", []).append(
        f"Investigation skipped — {target.replace('_', ' ').title()}; no result was assumed."
    )
    save_manual_case(case_id, case)
    return {"status": "skipped", "timeline": case["timeline"], "step": case["step"]}


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


@app.get("/api/whatif/features")
def api_whatif_features(request: Request):
    if not current_doctor(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    return {
        "features": [
            {"name": name, "label": name.replace("_", " ").title(), "kind": kind}
            for kind, names in (
                ("symptom", META.inquiry_symptoms),
                ("test", META.investigation_features),
            )
            for name in names
        ]
    }


@app.post("/api/whatif/evaluate")
async def api_whatif_evaluate(request: Request):
    """Compare two hypothetical RL states without changing saved cases."""
    if not current_doctor(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    allowed = set(META.inquiry_symptoms) | set(META.investigation_features)

    def normalize(raw):
        if not isinstance(raw, dict):
            raise ValueError("Each scenario must be a feature-to-value object")
        result = {}
        for name, value in raw.items():
            if name not in allowed:
                raise ValueError(f"Unknown clinical feature: {name}")
            if value not in (None, 0, 1, False, True):
                raise ValueError(f"Invalid value for {name}; use unknown, absent or present")
            if value is not None:
                result[name] = int(bool(value))
        return result

    try:
        baseline = normalize(body.get("baseline", {}))
        modified = normalize(body.get("modified", {}))
        model = rl_service.agent()

        def evaluate(findings):
            observation = Observation(
                patient_id="WHAT_IF",
                known_symptoms={s: findings.get(s) for s in META.inquiry_symptoms},
                known_tests={t: findings.get(t) for t in META.investigation_features},
                step=len(findings), done=False, last_reward=0.0,
                message="Hypothetical patient state", disease_options=list(META.diseases),
            )
            state = encode_state_for_rl(observation)
            scored = sorted(
                ((d, model.get_q(state, Action(ActionType.DIAGNOSE, d))) for d in META.diseases),
                key=lambda pair: pair[1], reverse=True,
            )
            import math
            maximum = float(scored[0][1]) if scored else 0.0
            exp_scores = [
                math.exp(max(-50.0, min(0.0, float(score) - maximum)))
                for _, score in scored
            ]
            exp_total = sum(exp_scores) or 1.0
            actions = [
                Action(ActionType.ASK_SYMPTOM, s)
                for s in META.inquiry_symptoms if s not in findings
            ] + [
                Action(ActionType.ORDER_TEST, t)
                for t in META.investigation_features if t not in findings
            ]
            if len(findings) >= 2 or not actions:
                actions.extend(Action(ActionType.DIAGNOSE, d) for d in META.diseases)
            next_action = model.choose_action(state, actions, training=False)
            best_diagnosis = Action(ActionType.DIAGNOSE, scored[0][0])
            from app.xai_service import explain_findings
            explanation = explain_findings(findings, best_diagnosis)
            return {
                "known_findings": len(findings),
                "top_diseases": [
                    {"disease": d, "q_score": round(float(score), 4),
                     "relative_score": round(exp_scores[i] / exp_total * 100, 2)}
                    for i, (d, score) in enumerate(scored[:3])
                ],
                "best_disease": scored[0][0],
                "next_rl_action": {"type": next_action.type.value, "target": next_action.target},
                "explanation": explanation,
            }

        return {"status": "ok", "baseline": evaluate(baseline), "modified": evaluate(modified)}
    except (ValueError, KeyError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except FileNotFoundError as exc:
        return JSONResponse({"error": str(exc)}, status_code=503)
    except Exception as exc:
        return JSONResponse({"error": f"What-if evaluation failed: {exc}"}, status_code=500)
