
"""SHAP and LIME explanations for DiagnoPath's Linear Q-learning agent."""

from __future__ import annotations

from functools import lru_cache
import random
from typing import Any

import numpy as np

from env.data_model import (
    ClinicalDataset,
    META,
    default_data_path,
)
from env.diagnostic_env import Action, encode_state_for_rl
from . import rl_service


FEATURE_NAMES = (
    list(META.inquiry_symptoms)
    + list(META.investigation_features)
)

# The state encoder and feature labels must have the same ordering.
STATE_SIZE = len(FEATURE_NAMES)


@lru_cache(maxsize=1)
def _training_patients():
    """Load the dataset once, rather than on every explanation request."""
    dataset = ClinicalDataset.from_csv(default_data_path())
    patients = [
        p for p in dataset.patients
        if p.split == "train"
    ]
    return patients or dataset.patients


def _patient_full_state(patient) -> list[int]:
    """Build a fully observed state in the RL encoder's feature order."""
    return (
        [int(patient.symptoms[s]) for s in META.inquiry_symptoms]
        + [int(patient.test_values[t])
           for t in META.investigation_features]
    )


def _background_states(
    current_state: tuple[int, ...],
    count: int = 12,
) -> np.ndarray:
    """
    Create realistic partial states from training patients.
    Each background state reveals roughly as many findings as
    the current consultation.
    """
    patients = _training_patients()
    if not patients:
        raise RuntimeError("No training patients are available.")

    known_count = sum(v != -1 for v in current_state)
    known_count = min(max(known_count, 1), STATE_SIZE)

    rng = random.Random(42)
    rows = []

    for _ in range(count):
        patient = rng.choice(patients)
        full_state = _patient_full_state(patient)

        revealed = set(rng.sample(range(STATE_SIZE), known_count))
        partial = [
            full_state[i] if i in revealed else -1
            for i in range(STATE_SIZE)
        ]
        rows.append(partial)

    return np.asarray(rows, dtype=int)


def _score_states(states, action: Action) -> np.ndarray:
    """Return the fixed action's Q-score for each supplied state."""
    model = rl_service.agent()
    scores = []

    for row in np.asarray(states):
        # Valid RL state values are -1 (unknown), 0 (absent), 1 (present).
        state = tuple(
            max(-1, min(1, int(round(float(v)))))
            for v in row
        )
        features = model.state_features(state)
        scores.append(model._q_from_features(features, action))

    return np.asarray(scores, dtype=float)


def _finding(value: int) -> str:
    return {-1: "unknown", 0: "absent", 1: "present"}.get(
        int(value), "unknown"
    )


def explain_with_shap(
    state: tuple[int, ...],
    action: Action,
    top_k: int = 10,
) -> dict[str, Any]:
    """Estimate feature contributions to the selected action's Q-score."""
    import shap

    if len(state) != STATE_SIZE:
        raise ValueError(
            f"State has {len(state)} features, expected {STATE_SIZE}."
        )

    background = _background_states(state, count=12)
    current = np.asarray([state], dtype=int)

    explainer = shap.KernelExplainer(
        lambda batch: _score_states(batch, action),
        background,
    )
    values = explainer.shap_values(current, nsamples=100)
    values = np.asarray(values, dtype=float).reshape(-1)

    indices = sorted(
        range(len(values)),
        key=lambda i: abs(values[i]),
        reverse=True,
    )[:top_k]

    return {
        "method": "SHAP",
        "explained_quantity": "selected_action_q_score",
        "base_value": float(np.asarray(explainer.expected_value).reshape(-1)[0]),
        "features": [
            {
                "feature": FEATURE_NAMES[i],
                "finding": _finding(state[i]),
                "contribution": round(float(values[i]), 5),
            }
            for i in indices
        ],
        "note": (
            "SHAP values estimate contributions to this action's Q-score "
            "relative to the background states; they are not disease probabilities."
        ),
    }


def explain_with_lime(
    state: tuple[int, ...],
    action: Action,
    top_k: int = 10,
) -> dict[str, Any]:
    """Fit a local surrogate to the selected action's Q-score."""
    from lime.lime_tabular import LimeTabularExplainer

    if len(state) != STATE_SIZE:
        raise ValueError(
            f"State has {len(state)} features, expected {STATE_SIZE}."
        )

    # LIME categorical values: 0=unknown, 1=absent, 2=present.
    def encode_for_lime(row):
        return [
            {-1: 0, 0: 1, 1: 2}.get(
                int(round(float(value))), 0
            )
            for value in row
        ]

    def decode_from_lime(batch):
        return np.asarray([
            [{0: -1, 1: 0, 2: 1}[int(round(float(v)))] for v in row]
            for row in batch
        ], dtype=int)

    background = _background_states(state, count=30)
    lime_background = np.asarray(
        [encode_for_lime(row) for row in background],
        dtype=int,
    )
    lime_instance = np.asarray(encode_for_lime(state), dtype=int)

    explainer = LimeTabularExplainer(
        training_data=lime_background,
        feature_names=FEATURE_NAMES,
        categorical_features=list(range(STATE_SIZE)),
        categorical_names={
            i: ["Unknown", "Absent", "Present"]
            for i in range(STATE_SIZE)
        },
        mode="regression",
        discretize_continuous=False,
        random_state=42,
    )

    explanation = explainer.explain_instance(
        lime_instance,
        lambda batch: _score_states(decode_from_lime(batch), action),
        num_features=min(top_k, STATE_SIZE),
        num_samples=500,
    )

    contributions = []
    for feature_description, weight in explanation.as_list():
        contributions.append({
            "description": feature_description,
            "weight": round(float(weight), 5),
        })

    return {
        "method": "LIME",
        "explained_quantity": "selected_action_q_score",
        "local_surrogate_r2": round(float(explanation.score), 5),
        "features": contributions,
        "note": (
            "LIME fits a local approximation near this consultation. "
            "Its weights describe the local surrogate, not medical causation."
        ),
    }


def explain_with_shap_and_lime(
    state: tuple[int, ...],
    action: Action,
) -> dict[str, Any]:
    """Run both explainers independently so one failure does not hide the other."""
    result = {}

    for name, function in (
        ("shap", explain_with_shap),
        ("lime", explain_with_lime),
    ):
        try:
            result[name] = {
                "status": "ok",
                **function(state, action),
            }
        except Exception as exc:
            # Keep the main explanation endpoint usable if an explainer fails.
            result[name] = {
                "status": "error",
                "message": f"{name.upper()} explanation failed: {exc}",
            }

    return result
