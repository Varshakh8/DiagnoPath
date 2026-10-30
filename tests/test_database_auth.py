
from app import auth, db


def sample_case_payload(patient_id="PATIENT-001", correct=True):
    return {
        "patient_id": patient_id,
        "patient_index": 0,
        "started_at": "2026-10-30T10:00:00+00:00",
        "diagnosis": "Disease A",
        "true_label": "Disease A",
        "correct": correct,
        "steps": 5,
        "symptoms_asked": 3,
        "tests_ordered": 2,
        "total_cost": 500,
        "history": [{"action": "ask", "target": "fever"}],
    }


# ---------------- AUTHENTICATION TESTS ----------------

def test_password_hash_is_not_plain_text():
    hashed = auth.hash_password("clinic123")

    assert hashed != "clinic123"
    assert "$" in hashed


def test_password_hash_uses_unique_salts():
    first = auth.hash_password("clinic123")
    second = auth.hash_password("clinic123")

    assert first != second


def test_verify_password_accepts_correct_password():
    hashed = auth.hash_password("clinic123")

    assert auth.verify_password("clinic123", hashed) is True


def test_verify_password_rejects_incorrect_password():
    hashed = auth.hash_password("clinic123")

    assert auth.verify_password("wrong-password", hashed) is False


def test_verify_password_rejects_malformed_hash():
    assert auth.verify_password("clinic123", "invalid-hash") is False


# ---------------- DATABASE TESTS ----------------

def test_database_initialization_is_idempotent(client):
    db.init_db()
    db.init_db()

    with db._connect() as conn:
        doctors = conn.execute(
            "SELECT COUNT(*) FROM doctors"
        ).fetchone()[0]

    assert doctors == len(auth.DEMO_DOCTORS)


def test_authenticate_returns_doctor_for_valid_credentials(client):
    doctor = db.authenticate("doctor", "clinic123")

    assert doctor is not None
    assert doctor["username"] == "doctor"
    assert doctor["full_name"] == "Dr. A. Sharma"
    assert "password_hash" not in doctor


def test_authenticate_accepts_whitespace_around_username(client):
    doctor = db.authenticate("  doctor  ", "clinic123")

    assert doctor is not None
    assert doctor["username"] == "doctor"


def test_authenticate_rejects_wrong_password(client):
    assert db.authenticate("doctor", "wrong-password") is None


def test_authenticate_rejects_unknown_username(client):
    assert db.authenticate("unknown-doctor", "clinic123") is None


def test_get_doctor_returns_existing_doctor(client):
    doctor = db.get_doctor(1)

    assert doctor is not None
    assert doctor["id"] == 1
    assert doctor["username"] == "doctor"
    assert doctor["specialty"] == "General Medicine"


def test_get_doctor_returns_none_for_unknown_id(client):
    assert db.get_doctor(999999) is None


def test_save_completed_case_returns_database_id(client):
    case_id = db.save_completed_case(
        1,
        sample_case_payload(),
    )

    assert isinstance(case_id, int)
    assert case_id > 0


def test_saved_case_can_be_retrieved(client):
    db.save_completed_case(
        1,
        sample_case_payload(),
    )

    cases = db.list_cases(1)

    assert len(cases) == 1
    assert cases[0]["patient_id"] == "PATIENT-001"
    assert cases[0]["diagnosis"] == "Disease A"
    assert cases[0]["correct"] == 1
    assert cases[0]["total_cost"] == 500


def test_case_history_is_saved_as_json(client):
    db.save_completed_case(
        1,
        sample_case_payload(),
    )

    with db._connect() as conn:
        row = conn.execute(
            "SELECT history_json FROM cases WHERE doctor_id = 1"
        ).fetchone()

    import json
    history = json.loads(row["history_json"])

    assert history == [{"action": "ask", "target": "fever"}]


def test_list_cases_only_returns_requested_doctors_cases(client):
    db.save_completed_case(
        1,
        sample_case_payload("PATIENT-001"),
    )
    db.save_completed_case(
        2,
        sample_case_payload("PATIENT-002"),
    )

    doctor_one_cases = db.list_cases(1)
    doctor_two_cases = db.list_cases(2)

    assert len(doctor_one_cases) == 1
    assert len(doctor_two_cases) == 1
    assert doctor_one_cases[0]["patient_id"] == "PATIENT-001"
    assert doctor_two_cases[0]["patient_id"] == "PATIENT-002"


def test_list_cases_respects_limit(client):
    for index in range(3):
        db.save_completed_case(
            1,
            sample_case_payload(f"PATIENT-{index}"),
        )

    cases = db.list_cases(1, limit=2)

    assert len(cases) == 2


def test_case_stats_for_empty_database(client):
    stats = db.case_stats(1)

    assert stats["completed_cases"] == 0
    assert stats["correct_cases"] == 0
    assert stats["accuracy_pct"] == 0.0
    assert stats["avg_cost"] == 0.0
    assert stats["avg_steps"] == 0.0


def test_case_stats_calculates_accuracy_and_averages(client):
    db.save_completed_case(
        1,
        sample_case_payload("PATIENT-001", correct=True),
    )

    second_payload = sample_case_payload(
        "PATIENT-002",
        correct=False,
    )
    second_payload["total_cost"] = 300
    second_payload["steps"] = 3

    db.save_completed_case(1, second_payload)

    stats = db.case_stats(1)

    assert stats["completed_cases"] == 2
    assert stats["correct_cases"] == 1
    assert stats["accuracy_pct"] == 50.0
    assert stats["avg_cost"] == 400.0
    assert stats["avg_steps"] == 4.0
