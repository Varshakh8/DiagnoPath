"""SQLite storage for doctors and completed consultations."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .auth import DEMO_DOCTORS, hash_password, verify_password

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "clinic.db"


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS doctors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                full_name TEXT NOT NULL,
                specialty TEXT NOT NULL DEFAULT 'General Medicine',
                password_hash TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS cases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doctor_id INTEGER NOT NULL,
                patient_id TEXT NOT NULL,
                patient_index INTEGER NOT NULL,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                diagnosis TEXT,
                true_label TEXT,
                correct INTEGER,
                steps INTEGER DEFAULT 0,
                symptoms_asked INTEGER DEFAULT 0,
                tests_ordered INTEGER DEFAULT 0,
                total_cost INTEGER DEFAULT 0,
                history_json TEXT NOT NULL DEFAULT '[]',
                FOREIGN KEY (doctor_id) REFERENCES doctors(id)
            );
            """
        )
        for doctor in DEMO_DOCTORS:
            existing = conn.execute(
                "SELECT id FROM doctors WHERE username = ?", (doctor["username"],)
            ).fetchone()
            if existing:
                continue
            conn.execute(
                """
                INSERT INTO doctors (username, full_name, specialty, password_hash)
                VALUES (?, ?, ?, ?)
                """,
                (
                    doctor["username"],
                    doctor["full_name"],
                    doctor["specialty"],
                    hash_password(doctor["password"]),
                ),
            )


def authenticate(username: str, password: str) -> Optional[Dict[str, Any]]:
    with _connect() as conn:
        row = conn.execute(
            "SELECT id, username, full_name, specialty, password_hash FROM doctors WHERE username = ?",
            (username.strip(),),
        ).fetchone()
    if not row or not verify_password(password, row["password_hash"]):
        return None
    return {
        "id": row["id"],
        "username": row["username"],
        "full_name": row["full_name"],
        "specialty": row["specialty"],
    }


def get_doctor(doctor_id: int) -> Optional[Dict[str, Any]]:
    with _connect() as conn:
        row = conn.execute(
            "SELECT id, username, full_name, specialty FROM doctors WHERE id = ?",
            (doctor_id,),
        ).fetchone()
    return dict(row) if row else None


def save_completed_case(doctor_id: int, payload: Dict[str, Any]) -> int:
    finished = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO cases (
                doctor_id, patient_id, patient_index, started_at, finished_at,
                diagnosis, true_label, correct, steps, symptoms_asked,
                tests_ordered, total_cost, history_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                doctor_id,
                payload["patient_id"],
                payload["patient_index"],
                payload["started_at"],
                finished,
                payload.get("diagnosis"),
                payload.get("true_label"),
                1 if payload.get("correct") else 0,
                payload.get("steps", 0),
                payload.get("symptoms_asked", 0),
                payload.get("tests_ordered", 0),
                payload.get("total_cost", 0),
                json.dumps(payload.get("history", [])),
            ),
        )
        return int(cur.lastrowid)


def list_cases(doctor_id: int, limit: int = 40) -> List[Dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT id, patient_id, started_at, finished_at, diagnosis, true_label,
                   correct, steps, symptoms_asked, tests_ordered, total_cost
            FROM cases
            WHERE doctor_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (doctor_id, limit),
        ).fetchall()
    return [dict(row) for row in rows]


def case_stats(doctor_id: int) -> Dict[str, Any]:
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) AS correct,
                AVG(total_cost) AS avg_cost,
                AVG(steps) AS avg_steps
            FROM cases
            WHERE doctor_id = ?
            """,
            (doctor_id,),
        ).fetchone()
    total = int(row["total"] or 0)
    correct = int(row["correct"] or 0)
    return {
        "completed_cases": total,
        "correct_cases": correct,
        "accuracy_pct": round((correct / total) * 100, 1) if total else 0.0,
        "avg_cost": round(float(row["avg_cost"] or 0), 1),
        "avg_steps": round(float(row["avg_steps"] or 0), 1),
    }
