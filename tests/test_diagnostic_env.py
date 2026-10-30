
import pytest

from env.data_model import META, ClinicalDataset
from env.diagnostic_env import (
    Action,
    ActionType,
    DiagnosticEnvironment,
    encode_state_for_rl,
)


@pytest.fixture
def diagnostic_env():
    dataset = ClinicalDataset.from_csv()
    return DiagnosticEnvironment(dataset)


@pytest.fixture
def started_env(diagnostic_env):
    diagnostic_env.reset(patient_index=0)
    return diagnostic_env


def test_reset_initializes_patient_and_state(diagnostic_env):
    observation = diagnostic_env.reset(0)

    assert observation.patient_id
    assert observation.step == 0
    assert observation.done is False
    assert all(
        value is None
        for value in observation.known_symptoms.values()
    )
    assert all(
        value is None
        for value in observation.known_tests.values()
    )


def test_reset_rejects_negative_patient_index(diagnostic_env):
    with pytest.raises(IndexError):
        diagnostic_env.reset(-1)


def test_reset_rejects_out_of_range_patient_index(diagnostic_env):
    with pytest.raises(IndexError):
        diagnostic_env.reset(len(diagnostic_env.dataset))


def test_step_requires_reset(diagnostic_env):
    action = Action(
        ActionType.ASK_SYMPTOM,
        META.inquiry_symptoms[0],
    )

    with pytest.raises(RuntimeError):
        diagnostic_env.step(action)


def test_available_actions_before_reset_is_empty(diagnostic_env):
    assert diagnostic_env.available_actions() == []


def test_available_actions_include_symptoms_tests_and_diagnoses(started_env):
    actions = started_env.available_actions()

    action_types = {action.type for action in actions}

    assert ActionType.ASK_SYMPTOM in action_types
    assert ActionType.ORDER_TEST in action_types
    assert ActionType.DIAGNOSE in action_types


def test_asking_symptom_records_finding_and_cost(started_env):
    symptom = META.inquiry_symptoms[0]

    observation, reward, done, info = started_env.step(
        Action(ActionType.ASK_SYMPTOM, symptom)
    )

    patient = started_env.current_patient()
    expected_value = patient.symptoms[symptom]

    assert observation.known_symptoms[symptom] == expected_value
    assert started_env.episode_stats()["symptoms_asked"] == 1
    assert started_env.episode_stats()["total_test_cost"] == started_env.ask_cost
    assert done is False
    assert info["action"].target == symptom


def test_asking_unknown_symptom_gets_redundant_penalty(started_env):
    observation, reward, done, info = started_env.step(
        Action(ActionType.ASK_SYMPTOM, "unknown_symptom")
    )

    assert reward == DiagnosticEnvironment.REDUNDANT_PENALTY
    assert "Invalid symptom" in observation.message
    assert done is False
    assert started_env.episode_stats()["symptoms_asked"] == 0


def test_reasking_symptom_gets_redundant_penalty(started_env):
    symptom = META.inquiry_symptoms[0]
    action = Action(ActionType.ASK_SYMPTOM, symptom)

    started_env.step(action)
    observation, reward, done, info = started_env.step(action)

    assert reward == DiagnosticEnvironment.REDUNDANT_PENALTY
    assert "Already asked" in observation.message
    assert started_env.episode_stats()["symptoms_asked"] == 1


def test_ordering_investigation_records_finding_and_cost(started_env):
    test_name = META.investigation_features[0]

    observation, reward, done, info = started_env.step(
        Action(ActionType.ORDER_TEST, test_name)
    )

    patient = started_env.current_patient()
    expected_value = patient.test_values[test_name]

    assert observation.known_tests[test_name] == expected_value
    assert started_env.episode_stats()["tests_ordered"] == 1
    assert started_env.episode_stats()["total_test_cost"] == started_env.test_costs[test_name]
    assert done is False


def test_ordering_unknown_investigation_gets_penalty(started_env):
    observation, reward, done, info = started_env.step(
        Action(ActionType.ORDER_TEST, "unknown_test")
    )

    assert reward == DiagnosticEnvironment.REDUNDANT_PENALTY
    assert "Invalid investigation" in observation.message
    assert started_env.episode_stats()["tests_ordered"] == 0


def test_reordering_investigation_gets_penalty(started_env):
    test_name = META.investigation_features[0]
    action = Action(ActionType.ORDER_TEST, test_name)

    started_env.step(action)
    observation, reward, done, info = started_env.step(action)

    assert reward == DiagnosticEnvironment.REDUNDANT_PENALTY
    assert "Already ordered" in observation.message
    assert started_env.episode_stats()["tests_ordered"] == 1


def test_diagnosis_is_rejected_before_three_steps(started_env):
    disease = META.diseases[0]

    observation, reward, done, info = started_env.step(
        Action(ActionType.DIAGNOSE, disease)
    )

    assert reward == -5.0
    assert info["premature"] is True
    assert done is False
    assert started_env.current_patient() is not None


def test_unknown_disease_is_rejected_after_minimum_steps(started_env):
    for symptom in META.inquiry_symptoms[:3]:
        started_env.step(
            Action(ActionType.ASK_SYMPTOM, symptom)
        )

    observation, reward, done, info = started_env.step(
        Action(ActionType.DIAGNOSE, "unknown_disease")
    )

    assert reward == DiagnosticEnvironment.REDUNDANT_PENALTY
    assert info["correct"] is False
    assert done is False


def test_correct_diagnosis_completes_episode(started_env):
    for symptom in META.inquiry_symptoms[:2]:
        started_env.step(
            Action(ActionType.ASK_SYMPTOM, symptom)
        )

    true_disease = started_env.current_patient().true_label()

    observation, reward, done, info = started_env.step(
        Action(ActionType.DIAGNOSE, true_disease)
    )

    assert done is True
    assert observation.done is True
    assert reward == DiagnosticEnvironment.REWARD_CORRECT
    assert info["correct"] is True
    assert info["episode"].predicted == true_disease
    assert info["episode"].true_label == true_disease


def test_wrong_diagnosis_completes_episode_with_penalty(started_env):
    for symptom in META.inquiry_symptoms[:2]:
        started_env.step(
            Action(ActionType.ASK_SYMPTOM, symptom)
        )

    true_disease = started_env.current_patient().true_label()
    wrong_disease = next(
        disease for disease in META.diseases
        if disease != true_disease
    )

    observation, reward, done, info = started_env.step(
        Action(ActionType.DIAGNOSE, wrong_disease)
    )

    assert done is True
    assert reward == DiagnosticEnvironment.REWARD_WRONG
    assert info["correct"] is False
    assert info["episode"].predicted == wrong_disease


def test_step_is_rejected_after_episode_finishes(started_env):
    for symptom in META.inquiry_symptoms[:2]:
        started_env.step(
            Action(ActionType.ASK_SYMPTOM, symptom)
        )

    true_disease = started_env.current_patient().true_label()

    started_env.step(
        Action(ActionType.DIAGNOSE, true_disease)
    )

    with pytest.raises(RuntimeError):
        started_env.step(
            Action(ActionType.ASK_SYMPTOM, META.inquiry_symptoms[0])
        )


def test_public_state_hides_true_diagnosis_before_completion(started_env):
    state = started_env.public_state()

    assert state["true_label"] is None
    assert state["predicted"] is None
    assert state["done"] is False


def test_public_state_reveals_diagnosis_after_completion(started_env):
    for symptom in META.inquiry_symptoms[:2]:
        started_env.step(
            Action(ActionType.ASK_SYMPTOM, symptom)
        )

    true_disease = started_env.current_patient().true_label()

    started_env.step(
        Action(ActionType.DIAGNOSE, true_disease)
    )

    state = started_env.public_state()

    assert state["done"] is True
    assert state["true_label"] == true_disease
    assert state["predicted"] == true_disease
    assert state["correct"] is True


def test_current_observation_does_not_change_step_count(started_env):
    before = started_env.episode_stats()["steps"]

    observation = started_env.current_observation()

    after = started_env.episode_stats()["steps"]

    assert observation.patient_id
    assert before == after


def test_current_observation_requires_reset(diagnostic_env):
    with pytest.raises(RuntimeError):
        diagnostic_env.current_observation()


def test_episode_stats_tracks_actions(started_env):
    symptom = META.inquiry_symptoms[0]
    test_name = META.investigation_features[0]

    started_env.step(
        Action(ActionType.ASK_SYMPTOM, symptom)
    )
    started_env.step(
        Action(ActionType.ORDER_TEST, test_name)
    )

    stats = started_env.episode_stats()

    assert stats["steps"] == 2
    assert stats["symptoms_asked"] == 1
    assert stats["tests_ordered"] == 1
    assert stats["total_test_cost"] > 0
    assert stats["done"] is False


def test_get_history_returns_copy(started_env):
    history = started_env.get_history()
    original_length = len(history)

    history.append("External modification")

    assert len(started_env.get_history()) == original_length


def test_state_encoder_marks_unknown_findings_as_minus_one(started_env):
    observation = started_env.current_observation()

    encoded = encode_state_for_rl(observation)

    assert len(encoded) == (
        len(META.inquiry_symptoms)
        + len(META.investigation_features)
    )
    assert all(value == -1 for value in encoded)


def test_state_encoder_represents_positive_and_negative_findings(started_env):
    symptom = META.inquiry_symptoms[0]

    started_env.step(
        Action(ActionType.ASK_SYMPTOM, symptom)
    )

    observation = started_env.current_observation()
    encoded = encode_state_for_rl(observation)

    symptom_position = META.inquiry_symptoms.index(symptom)

    expected = observation.known_symptoms[symptom]
    assert encoded[symptom_position] == expected
    assert encoded[symptom_position] in (0, 1)


def test_reset_clears_previous_episode_state(started_env):
    symptom = META.inquiry_symptoms[0]

    started_env.step(
        Action(ActionType.ASK_SYMPTOM, symptom)
    )

    started_env.reset(1)

    stats = started_env.episode_stats()

    assert stats["steps"] == 0
    assert stats["symptoms_asked"] == 0
    assert stats["tests_ordered"] == 0
    assert stats["total_test_cost"] == 0
    assert stats["done"] is False


def test_humanize_formats_feature_names():
    from env.diagnostic_env import _humanize

    assert _humanize("high_blood-pressure") == "High Blood Pressure"


def test_timeout_ends_episode_at_thirty_steps(started_env):
    for _ in range(30):
        observation, reward, done, info = started_env.step(
            Action(ActionType.ASK_SYMPTOM, "unknown_symptom")
        )

    assert done is True
    assert info["timeout"] is True
    assert started_env.episode_stats()["steps"] == 30


def test_actions_are_unavailable_after_timeout(started_env):
    for _ in range(30):
        started_env.step(
            Action(ActionType.ASK_SYMPTOM, "unknown_symptom")
        )

    assert started_env.available_actions() == []
