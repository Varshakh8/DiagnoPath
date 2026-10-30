from env.data_model import META


def test_catalog_returns_clinical_information(client):
    response = client.get("/api/catalog")

    assert response.status_code == 200

    data = response.json()
    assert data["n_patients"] > 0
    assert data["n_diseases"] > 0
    assert data["n_symptoms"] > 0
    assert data["n_investigations"] > 0
    assert isinstance(data["diseases"], list)


def test_patient_listing_returns_patients(client):
    response = client.get("/api/patients")

    assert response.status_code == 200

    data = response.json()
    assert data["total"] > 0
    assert len(data["patients"]) > 0
    assert "patient_id" in data["patients"][0]
    assert "index" in data["patients"][0]


def test_patient_listing_respects_limit(client):
    response = client.get("/api/patients?limit=2")

    assert response.status_code == 200
    assert len(response.json()["patients"]) <= 2


def test_patient_search_returns_matching_patient(client):
    initial = client.get("/api/patients?limit=1").json()
    patient_id = initial["patients"][0]["patient_id"]

    response = client.get("/api/patients", params={"q": patient_id})

    assert response.status_code == 200
    assert response.json()["total"] >= 1
    assert any(
        patient["patient_id"] == patient_id
        for patient in response.json()["patients"]
    )


def test_start_clinical_consultation(client):
    response = client.post(
        "/api/cases",
        json={"patient_index": 0},
    )

    assert response.status_code == 200

    data = response.json()
    assert data["case_id"]
    assert data["patient_id"]
    assert data["state"] is not None


def test_get_existing_clinical_consultation(client):
    created = client.post(
        "/api/cases",
        json={"patient_index": 0},
    ).json()

    response = client.get(f"/api/cases/{created['case_id']}")

    assert response.status_code == 200
    assert response.json()["case_id"] == created["case_id"]


def test_invalid_patient_index_is_rejected(client):
    response = client.post(
        "/api/cases",
        json={"patient_index": 999999},
    )

    assert response.status_code == 400


def test_nonexistent_consultation_returns_404(client):
    response = client.get("/api/cases/nonexistent-case")

    assert response.status_code == 404


def test_invalid_clinical_action_is_rejected(client):
    created = client.post(
        "/api/cases",
        json={"patient_index": 0},
    ).json()

    response = client.post(
        f"/api/cases/{created['case_id']}/actions",
        json={
            "type": "invalid_action",
            "target": "test_target",
        },
    )

    assert response.status_code == 400


def test_manual_consultation_can_be_created(client):
    response = client.post(
        "/api/manual-cases",
        json={"patient_ref": "PYTEST-PATIENT-001"},
    )

    assert response.status_code == 200

    data = response.json()
    assert data["case_id"]
    assert data["status"] == "started"


def test_manual_consultation_can_be_retrieved(client):
    created = client.post(
        "/api/manual-cases",
        json={"patient_ref": "PYTEST-PATIENT-002"},
    ).json()

    response = client.get(
        f"/api/manual-cases/{created['case_id']}"
    )

    assert response.status_code == 200

    data = response.json()
    assert data["patient_ref"] == "PYTEST-PATIENT-002"
    assert data["done"] is False
    assert data["step"] == 0
    assert data["pending"] is None
    assert isinstance(data["symptoms"], dict)
    assert isinstance(data["tests"], dict)


def test_manual_finding_accepts_valid_symptom(client):
    created = client.post(
        "/api/manual-cases",
        json={"patient_ref": "PYTEST-PATIENT-003"},
    ).json()

    symptom = META.inquiry_symptoms[0]

    response = client.post(
        f"/api/manual-cases/{created['case_id']}/finding",
        json={"target": symptom, "value": 1},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "recorded"
    assert response.json()["step"] == 1

    saved = client.get(
        f"/api/manual-cases/{created['case_id']}"
    ).json()

    assert saved["symptoms"][symptom] == 1


def test_manual_finding_rejects_unknown_feature(client):
    created = client.post("/api/manual-cases", json={}).json()

    response = client.post(
        f"/api/manual-cases/{created['case_id']}/finding",
        json={"target": "unknown_feature", "value": 1},
    )

    assert response.status_code == 400


def test_manual_finding_rejects_invalid_value(client):
    created = client.post("/api/manual-cases", json={}).json()

    symptom = META.inquiry_symptoms[0]

    response = client.post(
        f"/api/manual-cases/{created['case_id']}/finding",
        json={"target": symptom, "value": 5},
    )

    assert response.status_code == 400