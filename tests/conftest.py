import pytest
from fastapi.testclient import TestClient

from app import clinic, db, main


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Isolate both databases from real application data.
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test_clinic.db")
    monkeypatch.setattr(main, "MANUAL_DB", tmp_path / "test_manual.db")

    # Supply a test identity without testing or submitting the login form.
    test_doctor = {
        "id": 1,
        "username": "pytest_doctor",
        "full_name": "Pytest Doctor",
        "specialty": "Testing",
    }
    monkeypatch.setattr(
        main,
        "current_doctor",
        lambda request: test_doctor,
    )

    clinic._live.clear()

    with TestClient(main.app) as test_client:
        yield test_client

    clinic._live.clear()