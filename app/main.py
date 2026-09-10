"""
DiagnoPath AI - Member 1 doctor application.
Login-protected sequential diagnosis. RL/XAI endpoints are stubs for Members 2-3.
"""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import clinic, db


# ---------------------------------------------------------
# Paths and Templates
# ---------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent

templates = Jinja2Templates(
    directory=str(APP_DIR / "templates")
)


# ---------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------

app = FastAPI(
    title="DiagnoPath AI",
    version="1.0.0"
)


# ---------------------------------------------------------
# Session Middleware
# ---------------------------------------------------------

app.add_middleware(
    SessionMiddleware,
    secret_key="diagnopath-member1-change-in-production",
    same_site="lax",
    https_only=False,
)


# ---------------------------------------------------------
# Static Files
# ---------------------------------------------------------

app.mount(
    "/static",
    StaticFiles(directory=str(APP_DIR / "static")),
    name="static"
)


# ---------------------------------------------------------
# Startup
# ---------------------------------------------------------

@app.on_event("startup")
def startup() -> None:
    db.init_db()
    clinic.dataset()


# ---------------------------------------------------------
# Authentication Helpers
# ---------------------------------------------------------

def current_doctor(request: Request):
    doctor_id = request.session.get("doctor_id")

    if not doctor_id:
        return None

    return db.get_doctor(int(doctor_id))


def require_doctor(request: Request):
    doctor = current_doctor(request)

    if not doctor:
        return None

    return doctor


# =========================================================
# PAGES
# =========================================================

# ---------------------------------------------------------
# Home
# ---------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def home(request: Request):

    if current_doctor(request):
        return RedirectResponse(
            "/app",
            status_code=302
        )

    return RedirectResponse(
        "/login",
        status_code=302
    )


# ---------------------------------------------------------
# Login Page
# ---------------------------------------------------------

@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):

    if current_doctor(request):
        return RedirectResponse(
            "/app",
            status_code=302
        )

    return templates.TemplateResponse(
        request=request,
        name="login.html",
        context={
            "request": request,
            "error": None,
        },
    )


# ---------------------------------------------------------
# Login Submit
# ---------------------------------------------------------

@app.post("/login", response_class=HTMLResponse)
def login_submit(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
):

    doctor = db.authenticate(
        username,
        password
    )

    # Invalid login
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

    # Store doctor ID in session
    request.session["doctor_id"] = doctor["id"]

    return RedirectResponse(
        "/app",
        status_code=302
    )


# ---------------------------------------------------------
# Logout
# ---------------------------------------------------------

@app.get("/logout")
def logout(request: Request):

    request.session.clear()

    return RedirectResponse(
        "/login",
        status_code=302
    )


# ---------------------------------------------------------
# Doctor Dashboard
# ---------------------------------------------------------

@app.get("/app", response_class=HTMLResponse)
def dashboard(request: Request):

    doctor = current_doctor(request)

    if not doctor:
        return RedirectResponse(
            "/login",
            status_code=302
        )

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


# ---------------------------------------------------------
# Consultation Page
# ---------------------------------------------------------

@app.get("/app/consult", response_class=HTMLResponse)
def consult_page(
    request: Request,
    case_id: str = "",
):

    doctor = current_doctor(request)

    if not doctor:
        return RedirectResponse(
            "/login",
            status_code=302
        )

    if not case_id:
        return RedirectResponse(
            "/app",
            status_code=302
        )

    return templates.TemplateResponse(
        request=request,
        name="consult.html",
        context={
            "request": request,
            "doctor": doctor,
            "case_id": case_id,
        },
    )


# ---------------------------------------------------------
# History Page
# ---------------------------------------------------------

@app.get("/app/history", response_class=HTMLResponse)
def history_page(request: Request):

    doctor = current_doctor(request)

    if not doctor:
        return RedirectResponse(
            "/login",
            status_code=302
        )

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
# JSON API
# Members 2-4 consume these APIs
# =========================================================


# ---------------------------------------------------------
# Current Doctor
# ---------------------------------------------------------

@app.get("/api/me")
def api_me(request: Request):

    doctor = current_doctor(request)

    if not doctor:
        return JSONResponse(
            {
                "error": "unauthorized"
            },
            status_code=401,
        )

    return doctor


# ---------------------------------------------------------
# Symptom / Test Catalog
# ---------------------------------------------------------

@app.get("/api/catalog")
def api_catalog(request: Request):

    if not current_doctor(request):
        return JSONResponse(
            {
                "error": "unauthorized"
            },
            status_code=401,
        )

    return clinic.catalog()


# ---------------------------------------------------------
# Patient Registry
# ---------------------------------------------------------

@app.get("/api/patients")
def api_patients(
    request: Request,
    q: str = "",
    offset: int = 0,
    limit: int = 20,
):

    if not current_doctor(request):
        return JSONResponse(
            {
                "error": "unauthorized"
            },
            status_code=401,
        )

    # Keep the maximum number of returned patients reasonable
    limit = max(
        1,
        min(limit, 50)
    )

    return clinic.list_patients(
        q,
        offset,
        limit
    )


# ---------------------------------------------------------
# Start New Consultation
# ---------------------------------------------------------

@app.post("/api/cases")
async def api_start_case(request: Request):

    doctor = current_doctor(request)

    if not doctor:
        return JSONResponse(
            {
                "error": "unauthorized"
            },
            status_code=401,
        )

    body = {}

    try:
        body = await request.json()
    except Exception:
        body = {}

    patient_index = body.get(
        "patient_index"
    )

    try:

        return clinic.start_case(
            doctor["id"],
            patient_index
        )

    except IndexError as exc:

        return JSONResponse(
            {
                "error": str(exc)
            },
            status_code=400,
        )


# ---------------------------------------------------------
# Get Consultation Case
# ---------------------------------------------------------

@app.get("/api/cases/{case_id}")
def api_get_case(
    request: Request,
    case_id: str,
):

    doctor = current_doctor(request)

    if not doctor:
        return JSONResponse(
            {
                "error": "unauthorized"
            },
            status_code=401,
        )

    try:

        return clinic.get_case(
            case_id,
            doctor["id"]
        )

    except KeyError:

        return JSONResponse(
            {
                "error": "consultation not found"
            },
            status_code=404,
        )


# ---------------------------------------------------------
# Perform Consultation Action
# ask / test / diagnose
# ---------------------------------------------------------

@app.post("/api/cases/{case_id}/actions")
async def api_action(
    request: Request,
    case_id: str,
):

    doctor = current_doctor(request)

    if not doctor:
        return JSONResponse(
            {
                "error": "unauthorized"
            },
            status_code=401,
        )

    body = await request.json()

    action_type = str(
        body.get("type", "")
    ).lower()

    target = str(
        body.get("target", "")
    )

    if not target:

        return JSONResponse(
            {
                "error": "target is required"
            },
            status_code=400,
        )

    try:

        return clinic.apply_action(
            case_id,
            doctor["id"],
            action_type,
            target
        )

    except KeyError:

        return JSONResponse(
            {
                "error": "consultation not found"
            },
            status_code=404,
        )

    except RuntimeError as exc:

        return JSONResponse(
            {
                "error": str(exc)
            },
            status_code=400,
        )

    except ValueError as exc:

        return JSONResponse(
            {
                "error": str(exc)
            },
            status_code=400,
        )


# ---------------------------------------------------------
# Consultation History
# ---------------------------------------------------------

@app.get("/api/history")
def api_history(request: Request):

    doctor = current_doctor(request)

    if not doctor:
        return JSONResponse(
            {
                "error": "unauthorized"
            },
            status_code=401,
        )

    return {
        "cases": db.list_cases(
            doctor["id"]
        ),
        "stats": db.case_stats(
            doctor["id"]
        ),
    }


# =========================================================
# MEMBER 2 - REINFORCEMENT LEARNING HOOK
# =========================================================

@app.get("/api/rl/suggest")
def api_rl_stub(request: Request):

    """
    Member 2 integration point.

    Member 2 will replace this stub with the custom
    reinforcement-learning policy.

    Expected responsibility:
        - Receive the current diagnostic state
        - Encode the state
        - Select the next action
        - Return ask/test/diagnose recommendation
    """

    if not current_doctor(request):
        return JSONResponse(
            {
                "error": "unauthorized"
            },
            status_code=401,
        )

    return {
        "status": "not_implemented",
        "owner": "Member 2",
        "message": (
            "Connect the custom RL policy here "
            "to recommend the next ask/test/diagnose action."
        ),
    }


# =========================================================
# MEMBER 3 - XAI HOOK
# =========================================================

@app.get("/api/xai/explain")
def api_xai_stub(request: Request):

    """
    Member 3 integration point.

    Member 3 will replace this stub with the custom
    explainability implementation.
    """

    if not current_doctor(request):
        return JSONResponse(
            {
                "error": "unauthorized"
            },
            status_code=401,
        )

    return {
        "status": "not_implemented",
        "owner": "Member 3",
        "message": (
            "Connect custom feature-contribution "
            "explanations here."
        ),
    }
