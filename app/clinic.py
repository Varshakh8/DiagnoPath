"""Live consultation sessions wrapping the Member 1 diagnostic environment."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import uuid4

from env.diagnostic_env import Action, ActionType, DiagnosticEnvironment
from env.data_model import ClinicalDataset, default_data_path

from . import db

_dataset: Optional[ClinicalDataset] = None
_live: Dict[str, Dict[str, Any]] = {}


def dataset() -> ClinicalDataset:
    global _dataset
    if _dataset is None:
        _dataset = ClinicalDataset.from_csv(default_data_path())
    return _dataset


def catalog() -> Dict[str, Any]:
    data = dataset()
    return {
        "n_patients": len(data),
        "n_diseases": data.meta.n_diseases,
        "n_symptoms": len(data.meta.inquiry_symptoms),
        "n_investigations": len(data.meta.investigation_features),
        "diseases": list(data.meta.diseases),
        "ask_cost": data.meta.ask_cost,
        "investigation_cost": data.meta.investigation_cost,
        "source_name": data.meta.source_name,
    }


def list_patients(query: str = "", offset: int = 0, limit: int = 20) -> Dict[str, Any]:
    data = dataset()
    q = query.strip().lower()
    items = []
    for index, patient in enumerate(data.patients):
        if q and q not in patient.patient_id.lower() and q not in str(index):
            continue
        items.append(
            {
                "index": index,
                "patient_id": patient.patient_id,
                "split": patient.split,
            }
        )
    total = len(items)
    page = items[offset : offset + limit]
    return {"total": total, "offset": offset, "limit": limit, "patients": page}


def start_case(doctor_id: int, patient_index: Optional[int] = None) -> Dict[str, Any]:
    data = dataset()
    if patient_index is None:
        import random

        patient_index = random.randint(0, len(data) - 1)
    if patient_index < 0 or patient_index >= len(data):
        raise IndexError("patient_index out of range")

    env = DiagnosticEnvironment(data)
    env.reset(patient_index)
    case_id = uuid4().hex
    _live[case_id] = {
        "doctor_id": doctor_id,
        "patient_index": patient_index,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "saved": False,
        "env": env,
    }
    return _case_payload(case_id)


def get_case(case_id: str, doctor_id: int) -> Dict[str, Any]:
    rec = _require_case(case_id, doctor_id)
    return _case_payload(case_id, rec)

def get_env(case_id: str, doctor_id: int) -> DiagnosticEnvironment:
    """
    Public accessor for the live DiagnosticEnvironment behind a
    case, so other modules (e.g. app/rl_service.py) can inspect
    current state / available actions without reaching into the
    private _live dict directly.
    """
    rec = _require_case(case_id, doctor_id)
    return rec["env"]

def apply_action(case_id: str, doctor_id: int, action_type: str, target: str) -> Dict[str, Any]:
    rec = _require_case(case_id, doctor_id)
    env: DiagnosticEnvironment = rec["env"]
    mapping = {
        "ask": ActionType.ASK_SYMPTOM,
        "ask_symptom": ActionType.ASK_SYMPTOM,
        "test": ActionType.ORDER_TEST,
        "order_test": ActionType.ORDER_TEST,
        "diagnose": ActionType.DIAGNOSE,
    }
    if action_type not in mapping:
        raise ValueError("action must be ask, test, or diagnose")
    env.step(Action(mapping[action_type], target.strip()))
    payload = _case_payload(case_id, rec)
    if payload["state"]["done"] and not rec["saved"]:
        stats = payload["state"]["stats"]
        db.save_completed_case(
            doctor_id,
            {
                "patient_id": payload["patient_id"],
                "patient_index": rec["patient_index"],
                "started_at": rec["started_at"],
                "diagnosis": payload["state"]["predicted"],
                "true_label": payload["state"]["true_label"],
                "correct": payload["state"]["correct"],
                "steps": stats["steps"],
                "symptoms_asked": stats["symptoms_asked"],
                "tests_ordered": stats["tests_ordered"],
                "total_cost": stats["total_test_cost"],
                "history": payload["state"]["history"],
            },
        )
        rec["saved"] = True
    return payload


def _require_case(case_id: str, doctor_id: int) -> Dict[str, Any]:
    rec = _live.get(case_id)
    if not rec or rec["doctor_id"] != doctor_id:
        raise KeyError("consultation not found")
    return rec


def _case_payload(case_id: str, rec: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    rec = rec or _live[case_id]
    env: DiagnosticEnvironment = rec["env"]
    state = env.public_state()
    return {
        "case_id": case_id,
        "patient_id": state["patient_id"],
        "patient_index": rec["patient_index"],
        "started_at": rec["started_at"],
        "state": state,
    }
