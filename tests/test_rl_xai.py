from env.data_model import META


def create_live_case(client):
    response = client.post(
        "/api/cases",
        json={"patient_index": 0},
    )
    assert response.status_code == 200
    return response.json()["case_id"]


def create_manual_case(client):
    response = client.post(
        "/api/manual-cases",
        json={"patient_ref": "RL-TEST-PATIENT"},
    )
    assert response.status_code == 200
    return response.json()["case_id"]


def test_rl_suggest_returns_valid_action(client):
    case_id = create_live_case(client)

    response = client.get(
        "/api/rl/suggest",
        params={"case_id": case_id},
    )

    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"
    assert data["suggestion"]["type"] in ("ask", "test", "diagnose")
    assert data["suggestion"]["target"]


def test_rl_suggest_rejects_nonexistent_case(client):
    response = client.get(
        "/api/rl/suggest",
        params={"case_id": "invalid-case"},
    )

    assert response.status_code == 404


def test_manual_next_returns_valid_recommendation(client):
    case_id = create_manual_case(client)

    response = client.get(
        f"/api/manual-cases/{case_id}/next"
    )

    assert response.status_code == 200

    data = response.json()
    assert data["status"] in ("question", "diagnosis")

    if data["status"] == "question":
        assert data["type"] in ("ask", "test")
        assert data["target"]
    else:
        assert data["prediction"] in META.diseases


def test_manual_next_repeats_pending_question(client):
    case_id = create_manual_case(client)

    first = client.get(
        f"/api/manual-cases/{case_id}/next"
    ).json()

    if first["status"] != "question":
        return

    second = client.get(
        f"/api/manual-cases/{case_id}/next"
    )

    assert second.status_code == 200
    assert second.json()["status"] == "question"
    assert second.json()["target"] == first["target"]
    assert second.json()["type"] == first["type"]


def test_manual_answer_records_pending_question(client):
    case_id = create_manual_case(client)

    next_response = client.get(
        f"/api/manual-cases/{case_id}/next"
    ).json()

    if next_response["status"] != "question":
        return

    response = client.post(
        f"/api/manual-cases/{case_id}/answer",
        json={"answer": True},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "recorded"
    assert response.json()["step"] == 1

    saved = client.get(
        f"/api/manual-cases/{case_id}"
    ).json()

    target = next_response["target"]

    if next_response["type"] == "ask":
        assert saved["symptoms"][target] == 1
    else:
        assert saved["tests"][target] == 1


def test_manual_answer_rejects_non_boolean_value(client):
    case_id = create_manual_case(client)

    next_response = client.get(
        f"/api/manual-cases/{case_id}/next"
    ).json()

    if next_response["status"] != "question":
        return

    response = client.post(
        f"/api/manual-cases/{case_id}/answer",
        json={"answer": "yes"},
    )

    assert response.status_code == 400


def test_manual_answer_requires_pending_question(client):
    case_id = create_manual_case(client)

    response = client.post(
        f"/api/manual-cases/{case_id}/answer",
        json={"answer": True},
    )

    assert response.status_code == 400


def test_manual_skip_requires_pending_investigation(client):
    case_id = create_manual_case(client)

    response = client.post(
        f"/api/manual-cases/{case_id}/skip-test"
    )

    assert response.status_code == 400


def test_whatif_features_returns_feature_catalog(client):
    response = client.get("/api/whatif/features")

    assert response.status_code == 200

    features = response.json()["features"]
    names = {feature["name"] for feature in features}

    assert set(META.inquiry_symptoms).issubset(names)
    assert set(META.investigation_features).issubset(names)


def test_whatif_rejects_unknown_feature(client):
    response = client.post(
        "/api/whatif/evaluate",
        json={
            "baseline": {"unknown_feature": 1},
            "modified": {},
        },
    )

    assert response.status_code == 400
    assert "Unknown clinical feature" in response.json()["error"]


def test_whatif_rejects_invalid_feature_value(client):
    symptom = META.inquiry_symptoms[0]

    response = client.post(
        "/api/whatif/evaluate",
        json={
            "baseline": {symptom: 5},
            "modified": {},
        },
    )

    assert response.status_code == 400


def test_whatif_evaluates_two_valid_scenarios(client):
    symptom = META.inquiry_symptoms[0]

    response = client.post(
        "/api/whatif/evaluate",
        json={
            "baseline": {},
            "modified": {symptom: 1},
        },
    )

    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"

    for scenario in ("baseline", "modified"):
        result = data[scenario]
        assert "top_diseases" in result
        assert "best_disease" in result
        assert "next_rl_action" in result
        assert result["best_disease"] in META.diseases


def test_manual_xai_explanation(client):
    case_id = create_manual_case(client)

    # Obtain an RL-selected pending action first.
    next_response = client.get(
        f"/api/manual-cases/{case_id}/next"
    )
    assert next_response.status_code == 200

    response = client.get(
        "/api/xai/explain",
        params={"case_id": case_id},
    )

    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"
    assert data["explanation_type"] == "linear_q_feature_contributions"
    assert data["action"]["type"] in ("ask", "test", "diagnose")
    assert "top_positive_contributions" in data
    assert "top_negative_contributions" in data