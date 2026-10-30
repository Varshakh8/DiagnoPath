
"""DiagnoPath AI - Member 3 explainability service.

Explains the trained linear Q-learning agent's selected action
using learned feature contributions for manual and live cases.
"""

from __future__ import annotations

from .xai_methods import explain_with_shap_and_lime
from typing import Any, Dict, Optional

from env.diagnostic_env import (
    Action,
    ActionType,
    DiagnosticEnvironment,
    Observation,
    encode_state_for_rl,
)
from env.data_model import META

from . import clinic
from . import rl_service


_ACTION_TYPE_TO_API = {
    ActionType.ASK_SYMPTOM: "ask",
    ActionType.ORDER_TEST: "test",
    ActionType.DIAGNOSE: "diagnose",
}


def _humanize(name: str) -> str:
    return (
        name.replace("_", " ")
        .replace("-", " ")
        .strip()
        .title()
    )


def _manual_observation(case: Dict[str, Any]) -> Observation:
    """Build an RL observation from the saved manual-case findings."""

    return Observation(
        patient_id=case["patient_id"],
        known_symptoms=dict(case["symptoms"]),
        known_tests=dict(case["tests"]),
        step=case["step"],
        done=case["done"],
        last_reward=0.0,
        message="Recorded manual consultation findings",
        disease_options=list(META.diseases),
    )


def _manual_actions(case: Dict[str, Any]) -> list[Action]:
    """Reproduce the action choices used by the manual consultation."""

    actions = [
        Action(ActionType.ASK_SYMPTOM, name)
        for name, value in case["symptoms"].items()
        if value is None
    ]

    actions.extend(
        Action(ActionType.ORDER_TEST, name)
        for name, value in case["tests"].items()
        if value is None
    )

    if case["step"] >= 3 or not actions:
        actions.extend(
            Action(ActionType.DIAGNOSE, disease)
            for disease in META.diseases
        )

    return actions


def _contribution_payload(
    observation: Observation,
    chosen: Action,
) -> Dict[str, Any]:
    """Calculate feature contributions for a selected RL action."""

    state = encode_state_for_rl(observation)
    model = rl_service.agent()
    features = model.state_features(state)

    action_key = model.action_key(chosen)
    weights = model.weights[action_key]

    feature_names = (
        list(META.inquiry_symptoms)
        + list(META.investigation_features)
    )

    state_size = len(state)
    contributions = []

    for i, name in enumerate(feature_names):
        value = state[i]

        raw_index = 1 + i
        known_index = 1 + state_size + i
        positive_index = 1 + 2 * state_size + i
        negative_index = 1 + 3 * state_size + i

        contribution = (
            weights[raw_index] * features[raw_index]
            + weights[known_index] * features[known_index]
            + weights[positive_index] * features[positive_index]
            + weights[negative_index] * features[negative_index]
        )

        if value == -1:
            finding = "unknown"
        elif value == 1:
            finding = "present"
        else:
            finding = "absent"

        if value != -1:
            contributions.append({
                "feature": _humanize(name),
                "finding": finding,
                "contribution": round(contribution, 4),
            })

    positive = sorted(
        [
            item for item in contributions
            if item["contribution"] > 0
        ],
        key=lambda item: item["contribution"],
        reverse=True,
    )[:5]

    print("XAI known features:", contributions)

    negative = sorted(
        [
            item for item in contributions
            if item["contribution"] < 0
        ],
        key=lambda item: item["contribution"],
    )[:5]

    q_score = model._q_from_features(features, chosen)

    label = (
        chosen.target
        if chosen.type == ActionType.DIAGNOSE
        else _humanize(chosen.target)
    )

    # Existing custom linear Q-learning explanation.
    result = {
        "status": "ok",
        "explanation_type": "linear_q_feature_contributions",
        "action": {
            "type": _ACTION_TYPE_TO_API[chosen.type],
            "target": chosen.target,
            "label": label,
        },
        "q_score": round(q_score, 4),
        "top_positive_contributions": positive,
        "top_negative_contributions": negative,
        "note": (
            "Positive contributions increase the selected action's "
            "model score; negative contributions decrease it. "
            "These are model-score contributions, not disease "
            "probabilities or causal medical claims."
        ),
    }

    # Add SHAP and LIME explanations without removing the existing result.
    result.update(
        explain_with_shap_and_lime(state, chosen)
    )

    return result


def explain_manual_case(case: Dict[str, Any]) -> Dict[str, Any]:
    """Explain a manual consultation, including after completion."""

    observation = _manual_observation(case)

    if case["done"]:
        # Explain the diagnosis the agent actually selected.
        prediction = case.get("prediction")

        if not prediction:
            return {
                "status": "no_actions",
                "message": (
                    "The consultation is complete, but no final "
                    "prediction is available to explain."
                ),
            }

        chosen = Action(ActionType.DIAGNOSE, prediction)

    else:
        pending = case.get("pending")

        if pending:
            # Explain the action currently awaiting the doctor's answer.
            action_type = (
                ActionType.ASK_SYMPTOM
                if pending["type"] == "ask"
                else ActionType.ORDER_TEST
            )
            chosen = Action(action_type, pending["target"])
        else:
            actions = _manual_actions(case)

            if not actions:
                return {
                    "status": "no_actions",
                    "message": "No diagnostic actions are available.",
                }

            state = encode_state_for_rl(observation)
            chosen = rl_service.agent().choose_action(
                state=state,
                actions=actions,
                training=False,
            )

    return _contribution_payload(observation, chosen)


def explain_findings(known: Dict[str, int], chosen: Action) -> Dict[str, Any]:
    """Explain an action for a caller-supplied live/hypothetical RL state."""
    observation = Observation(
        patient_id="LIVE_OR_HYPOTHETICAL",
        known_symptoms={s: known.get(s) for s in META.inquiry_symptoms},
        known_tests={t: known.get(t) for t in META.investigation_features},
        step=len(known),
        done=chosen.type == ActionType.DIAGNOSE,
        last_reward=0.0,
        message="Doctor-entered or hypothetical findings",
        disease_options=list(META.diseases),
    )
    return _contribution_payload(observation, chosen)


def explain_action(case_id: str, doctor_id: int) -> Dict[str, Any]:
    """Explain the selected RL action for an existing live consultation."""

    env: DiagnosticEnvironment = clinic.get_env(case_id, doctor_id)
    observation = env.current_observation()
    public_state = env.public_state()

    if public_state["done"]:
        prediction = public_state.get("predicted")

        if not prediction:
            return {
                "status": "no_actions",
                "message": (
                    "The consultation is complete, but no final "
                    "prediction is available to explain."
                ),
            }

        chosen = Action(ActionType.DIAGNOSE, prediction)
        return _contribution_payload(observation, chosen)

    state = encode_state_for_rl(observation)
    actions = env.available_actions()

    if not actions:
        return {
            "status": "no_actions",
            "message": "No further actions are available.",
        }

    chosen = rl_service.agent().choose_action(
        state=state,
        actions=actions,
        training=False,
    )

    return _contribution_payload(observation, chosen)
