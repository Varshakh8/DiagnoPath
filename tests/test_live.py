
import pytest

from app import live
from env.diagnostic_env import ActionType


@pytest.fixture(autouse=True)
def clear_live_sessions():
    """Prevent live sessions from leaking between tests."""
    live._sessions.clear()
    yield
    live._sessions.clear()


class FakeAgent:
    """Deterministic RL agent for testing live suggestions."""

    def state_features(self, state):
        return [0.0]

    def _q_from_features(self, features, action):
        return 1.0 if action.type == ActionType.DIAGNOSE else 0.5

    def choose_action(self, state, actions, training=False):
        assert actions
        return actions[0]


@pytest.fixture
def fake_agent(monkeypatch):
    agent = FakeAgent()
    monkeypatch.setattr(live.rl_service, "agent", lambda: agent)
    return agent


def feature_of_kind(kind):
    matches = [
        name for name, item in live.FEATURES.items()
        if item["kind"] == kind
    ]
    assert matches, f"No feature found for kind: {kind}"
    return matches[0]


def test_humanize():
    assert live._humanize("chest_pain") == "Chest Pain"
    assert live._humanize("blood-pressure") == "Blood Pressure"


def test_feature_table_contains_features():
    assert live.FEATURES
    assert all("name" in item for item in live.FEATURES.values())
    assert all("label" in item for item in live.FEATURES.values())
    assert all("kind" in item for item in live.FEATURES.values())


def test_observation_builds_symptoms_and_tests():
    symptom = feature_of_kind("symptom")
    test = feature_of_kind("test")

    observation = live._observation({
        symptom: 1,
        test: 0,
    })

    assert observation.patient_id == "LIVE"
    assert observation.known_symptoms[symptom] == 1
    assert observation.known_tests[test] == 0
    assert observation.step == 2
    assert observation.done is False


def test_candidate_actions_include_unknown_features():
    actions = live._candidate_actions(
        known={},
        skipped=[],
        allow_questions=True,
        allow_diagnosis=False,
    )

    assert actions
    assert all(
        action.type in (ActionType.ASK_SYMPTOM, ActionType.ORDER_TEST)
        for action in actions
    )


def test_candidate_actions_exclude_known_and_skipped_features():
    symptom = feature_of_kind("symptom")
    test = feature_of_kind("test")

    actions = live._candidate_actions(
        known={symptom: 1},
        skipped=[test],
        allow_questions=True,
        allow_diagnosis=False,
    )

    targets = {action.target for action in actions}
    assert symptom not in targets
    assert test not in targets


def test_candidate_actions_include_diagnoses_when_allowed():
    actions = live._candidate_actions(
        known={},
        skipped=[],
        allow_questions=False,
        allow_diagnosis=True,
    )

    assert actions
    assert all(action.type == ActionType.DIAGNOSE for action in actions)


def test_start_session_creates_session(fake_agent):
    result = live.start_session(doctor_id=1)

    assert result["session_id"]
    assert result["n_findings"] == 0
    assert result["n_skipped_tests"] == 0
    assert "features" in result
    assert result["suggestion"] is not None
    assert result["session_id"] in live._sessions


def test_get_session_returns_existing_session(fake_agent):
    created = live.start_session(doctor_id=1)

    result = live.get_session(created["session_id"], doctor_id=1)

    assert result["session_id"] == created["session_id"]
    assert result["n_findings"] == 0
    assert "features" in result


def test_get_session_rejects_wrong_doctor(fake_agent):
    created = live.start_session(doctor_id=1)

    with pytest.raises(KeyError):
        live.get_session(created["session_id"], doctor_id=2)


def test_get_session_rejects_unknown_session():
    with pytest.raises(KeyError):
        live.get_session("missing-session", doctor_id=1)


@pytest.mark.parametrize("answer", [True, False])
def test_record_answer_stores_boolean_answer(fake_agent, answer):
    created = live.start_session(doctor_id=1)
    target = feature_of_kind("symptom")

    result = live.record_answer(
        created["session_id"],
        doctor_id=1,
        target=target,
        present=answer,
    )

    assert result["n_findings"] == 1
    assert live._sessions[created["session_id"]]["known"][target] == int(answer)


def test_record_answer_none_removes_existing_answer(fake_agent):
    created = live.start_session(doctor_id=1)
    target = feature_of_kind("symptom")

    live.record_answer(
        created["session_id"], 1, target, True
    )

    result = live.record_answer(
        created["session_id"], 1, target, None
    )

    assert result["n_findings"] == 0
    assert target not in live._sessions[created["session_id"]]["known"]


def test_record_answer_rejects_unknown_feature(fake_agent):
    created = live.start_session(doctor_id=1)

    with pytest.raises(ValueError):
        live.record_answer(
            created["session_id"], 1, "invalid_feature", True
        )


def test_record_answer_rejects_wrong_doctor(fake_agent):
    created = live.start_session(doctor_id=1)
    target = feature_of_kind("symptom")

    with pytest.raises(KeyError):
        live.record_answer(
            created["session_id"], 2, target, True
        )


def test_skip_test_records_skipped_feature(fake_agent):
    created = live.start_session(doctor_id=1)
    target = feature_of_kind("test")

    result = live.skip_test(
        created["session_id"], 1, target
    )

    assert result["n_skipped_tests"] == 1
    assert target in live._sessions[created["session_id"]]["skipped"]


def test_skip_test_does_not_duplicate_skipped_feature(fake_agent):
    created = live.start_session(doctor_id=1)
    target = feature_of_kind("test")

    live.skip_test(created["session_id"], 1, target)
    result = live.skip_test(created["session_id"], 1, target)

    assert result["n_skipped_tests"] == 1


def test_skip_test_removes_existing_answer(fake_agent):
    created = live.start_session(doctor_id=1)
    target = feature_of_kind("test")

    live.record_answer(created["session_id"], 1, target, True)
    live.skip_test(created["session_id"], 1, target)

    assert target not in live._sessions[created["session_id"]]["known"]
    assert target in live._sessions[created["session_id"]]["skipped"]


def test_skip_test_rejects_unknown_feature(fake_agent):
    created = live.start_session(doctor_id=1)

    with pytest.raises(ValueError):
        live.skip_test(
            created["session_id"], 1, "invalid_test"
        )


def test_skip_test_rejects_wrong_doctor(fake_agent):
    created = live.start_session(doctor_id=1)
    target = feature_of_kind("test")

    with pytest.raises(KeyError):
        live.skip_test(created["session_id"], 2, target)


def test_require_rejects_missing_session():
    with pytest.raises(KeyError):
        live._require("missing-session", doctor_id=1)


def test_require_rejects_wrong_doctor():
    created = live.start_session(doctor_id=1)

    with pytest.raises(KeyError):
        live._require(created["session_id"], doctor_id=2)


def test_suggest_returns_expected_structure(fake_agent):
    suggestion = live._suggest({})

    assert suggestion["type"] in ("ask", "test", "diagnose")
    assert "target" in suggestion
    assert "label" in suggestion
    assert "top" in suggestion
    assert len(suggestion["top"]) <= live.TOP_K
    assert suggestion["forced"] is False


def test_suggest_forces_diagnosis_at_maximum_findings(fake_agent):
    known = {
        name: 1
        for name in list(live.FEATURES)[:live.MAX_FINDINGS_BEFORE_FORCED_DIAGNOSIS]
    }

    suggestion = live._suggest(known)

    assert suggestion["forced"] is True
    assert suggestion["type"] == "diagnose"


def test_payload_includes_features_when_requested(fake_agent):
    created = live.start_session(doctor_id=1)
    rec = live._sessions[created["session_id"]]

    payload = live._payload(
        created["session_id"],
        rec,
        include_features=True,
    )

    assert "features" in payload
    assert "findings" in payload
    assert "suggestion" in payload


def test_payload_omits_features_when_not_requested(fake_agent):
    created = live.start_session(doctor_id=1)
    rec = live._sessions[created["session_id"]]

    payload = live._payload(
        created["session_id"],
        rec,
        include_features=False,
    )

    assert "features" not in payload

