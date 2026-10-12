"""
DiagnoPath AI - Member 2: LIVE (real-time) consultation.

In the offline consultation (clinic.py) the "patient" is a stored CSV
row, so when the agent asks "does the patient have fatigue?" the
program just looks the answer up.

Here nothing is stored. The DOCTOR answers for a real patient, one
question at a time, and the same trained agent decides what to ask
next, exactly as it does in rl/evaluate.py.

State handling is identical to training/evaluation:
    unknown = -1, absent = 0, present = 1  (encode_state_for_rl)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from env.data_model import META
from env.diagnostic_env import (
    Action,
    ActionType,
    Observation,
    encode_state_for_rl,
)

from . import db, rl_service


# =========================================================
# Settings
# =========================================================

# Same rule as the training environment: no diagnosis before
# at least 2 findings are known.
MIN_FINDINGS_BEFORE_DIAGNOSIS = 2

# Safety net for live use only: if the agent still has not
# committed to a diagnosis after this many findings, stop asking
# questions and let it pick its best diagnosis. (In evaluation,
# a few cases such as Hepatitis D/E ran out the 30-step limit
# without ever diagnosing.)
MAX_FINDINGS_BEFORE_FORCED_DIAGNOSIS = 20

# How many leading diagnoses to show.
TOP_K = 3


# =========================================================
# Feature table (every symptom / test the model knows)
# =========================================================

def _humanize(name: str) -> str:
    return " ".join(
        name.replace("_", " ").replace("-", " ").split()
    ).title()


def _feature_table() -> Dict[str, Dict[str, str]]:
    table: Dict[str, Dict[str, str]] = {}

    for name in META.inquiry_symptoms:
        table.setdefault(
            name,
            {
                "name": name,
                "label": _humanize(name),
                "kind": "symptom",
            },
        )

    for name in META.investigation_features:
        table.setdefault(
            name,
            {
                "name": name,
                "label": _humanize(name),
                "kind": "test",
            },
        )

    return table


FEATURES: Dict[str, Dict[str, str]] = _feature_table()


# =========================================================
# Sessions (in memory, like clinic.py's live cases)
# =========================================================

_sessions: Dict[str, Dict[str, Any]] = {}


def _require(session_id: str, doctor_id: int) -> Dict[str, Any]:
    rec = _sessions.get(session_id)

    if not rec or rec["doctor_id"] != doctor_id:
        raise KeyError("live session not found")

    return rec


# =========================================================
# Agent glue
# =========================================================

def _observation(known: Dict[str, int]) -> Observation:
    """Build the same Observation the environment would produce."""

    return Observation(
        patient_id="LIVE",
        known_symptoms={
            s: known.get(s)
            for s in META.inquiry_symptoms
        },
        known_tests={
            t: known.get(t)
            for t in META.investigation_features
        },
        step=len(known),
        done=False,
        last_reward=0.0,
        message="live snapshot",
        disease_options=list(META.diseases),
    )


def _candidate_actions(
    known: Dict[str, int],
    allow_questions: bool,
    allow_diagnosis: bool,
) -> List[Action]:

    actions: List[Action] = []

    if allow_questions:
        for name, feature in FEATURES.items():
            if name in known:
                continue

            kind = (
                ActionType.ASK_SYMPTOM
                if feature["kind"] == "symptom"
                else ActionType.ORDER_TEST
            )

            actions.append(Action(kind, name))

    if allow_diagnosis:
        for disease in META.diseases:
            actions.append(
                Action(ActionType.DIAGNOSE, disease)
            )

    return actions


def _suggest(known: Dict[str, int]) -> Dict[str, Any]:
    """Ask the trained agent what to do next, given what is known."""

    agent = rl_service.agent()

    n_known = len(known)

    state = encode_state_for_rl(_observation(known))

    # Current ranking of diagnoses (model's internal scores).
    features = agent.state_features(state)

    scored = sorted(
        (
            (
                disease,
                agent._q_from_features(
                    features,
                    Action(ActionType.DIAGNOSE, disease),
                ),
            )
            for disease in META.diseases
        ),
        key=lambda item: item[1],
        reverse=True,
    )

    top = [
        {"disease": d, "score": round(q, 2)}
        for d, q in scored[:TOP_K]
    ]

    forced = n_known >= MAX_FINDINGS_BEFORE_FORCED_DIAGNOSIS

    actions = _candidate_actions(
        known,
        allow_questions=not forced,
        allow_diagnosis=(
            n_known >= MIN_FINDINGS_BEFORE_DIAGNOSIS
        ),
    )

    if not actions:
        actions = _candidate_actions(
            known,
            allow_questions=False,
            allow_diagnosis=True,
        )

    chosen = agent.choose_action(
        state=state,
        actions=actions,
        training=False,
    )

    if chosen.type == ActionType.DIAGNOSE:
        kind = "diagnose"
        label = chosen.target
    elif chosen.type == ActionType.ORDER_TEST:
        kind = "test"
        label = FEATURES[chosen.target]["label"]
    else:
        kind = "ask"
        label = FEATURES[chosen.target]["label"]

    return {
        "type": kind,
        "target": chosen.target,
        "label": label,
        "top": top,
        "forced": forced,
    }


# =========================================================
# Public operations
# =========================================================

def _payload(
    session_id: str,
    rec: Dict[str, Any],
    include_features: bool = False,
) -> Dict[str, Any]:

    known: Dict[str, int] = rec["known"]

    data: Dict[str, Any] = {
        "session_id": session_id,
        "n_findings": len(known),
        "findings": [
            {
                "name": name,
                "label": FEATURES[name]["label"],
                "kind": FEATURES[name]["kind"],
                "present": bool(value),
            }
            for name, value in known.items()
        ],
        "suggestion": _suggest(known),
    }

    if include_features:
        data["features"] = list(FEATURES.values())

    return data


def start_session(doctor_id: int) -> Dict[str, Any]:
    session_id = uuid4().hex

    rec = {
        "doctor_id": doctor_id,
        "known": {},
    }

    _sessions[session_id] = rec

    return _payload(session_id, rec, include_features=True)


def get_session(session_id: str, doctor_id: int) -> Dict[str, Any]:
    rec = _require(session_id, doctor_id)

    return _payload(session_id, rec, include_features=True)


def record_answer(
    session_id: str,
    doctor_id: int,
    target: str,
    present: Any,
) -> Dict[str, Any]:
    """
    present=True  -> finding is present
    present=False -> finding is absent
    present=None  -> remove the finding (back to unknown)
    """

    rec = _require(session_id, doctor_id)

    if target not in FEATURES:
        raise ValueError(f"Unknown symptom/test: {target}")

    if present is None:
        rec["known"].pop(target, None)
    else:
        rec["known"][target] = 1 if present else 0

    return _payload(session_id, rec)


# =========================================================
# Routes
# =========================================================

APP_DIR = Path(__file__).resolve().parent

templates = Jinja2Templates(
    directory=str(APP_DIR / "templates")
)

router = APIRouter()


def _current_doctor(request: Request):
    doctor_id = request.session.get("doctor_id")

    if not doctor_id:
        return None

    return db.get_doctor(int(doctor_id))


def _run(request: Request, call):
    doctor = _current_doctor(request)

    if not doctor:
        return JSONResponse(
            {"error": "unauthorized"},
            status_code=401,
        )

    try:
        return call(doctor["id"])

    except KeyError:
        return JSONResponse(
            {"error": "live session not found"},
            status_code=404,
        )

    except ValueError as exc:
        return JSONResponse(
            {"error": str(exc)},
            status_code=400,
        )

    except FileNotFoundError as exc:
        return JSONResponse(
            {"error": str(exc)},
            status_code=503,
        )


@router.get("/app/live", response_class=HTMLResponse)
def live_page(request: Request):

    doctor = _current_doctor(request)

    if not doctor:
        return RedirectResponse(
            "/login",
            status_code=302,
        )

    return templates.TemplateResponse(
        request=request,
        name="live.html",
        context={
            "request": request,
            "doctor": doctor,
        },
    )


@router.post("/api/live/start")
def api_live_start(request: Request):
    return _run(
        request,
        lambda doctor_id: start_session(doctor_id),
    )


@router.get("/api/live/{session_id}")
def api_live_get(request: Request, session_id: str):
    return _run(
        request,
        lambda doctor_id: get_session(session_id, doctor_id),
    )


@router.post("/api/live/{session_id}/answer")
async def api_live_answer(request: Request, session_id: str):

    try:
        body = await request.json()
    except Exception:
        body = {}

    target = str(body.get("target", ""))

    if "present" not in body or not (
        body["present"] is None
        or isinstance(body["present"], bool)
    ):
        return JSONResponse(
            {"error": "'present' must be true, false or null"},
            status_code=400,
        )

    present = body["present"]

    return _run(
        request,
        lambda doctor_id: record_answer(
            session_id,
            doctor_id,
            target,
            present,
        ),
    )