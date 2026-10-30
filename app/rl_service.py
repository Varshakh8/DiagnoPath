"""
DiagnoPath AI - Member 2 integration: RL suggestion service.

Loads the trained linear Q-learning agent once and exposes a
single function, suggest_action(), that the /api/rl/suggest route
in app/main.py calls for a live consultation case.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from env.diagnostic_env import (
    Action,
    ActionType,
    DiagnosticEnvironment,
    encode_state_for_rl,
)

from rl.agent import LinearQLearningAgent

from . import clinic


# =========================================================
# Model loading
# =========================================================

ROOT = Path(__file__).resolve().parent.parent

# Use the experimentally evaluated warm-start model (34/42 = 81.0% on
# the repository's fixed held-out test split). Keep all other model files intact.
MODEL_PATH = ROOT / "models" / "diagnostic_linear_q_warmstart_experiment.json"

_agent: Optional[LinearQLearningAgent] = None


def agent() -> LinearQLearningAgent:
    """
    Load the trained agent once (lazily, on first request) and
    reuse it for every subsequent call. epsilon is forced to 0.0
    so live suggestions are always the greedy (best-known) action.
    """

    global _agent

    if _agent is None:

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Trained model not found at {MODEL_PATH}. "
                f"Run `python -m rl.train` first."
            )

        _agent = LinearQLearningAgent.load(MODEL_PATH)
        _agent.epsilon = 0.0

    return _agent


# =========================================================
# Action type <-> API string mapping
# =========================================================

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


# =========================================================
# Suggestion
# =========================================================

def suggest_action(
    case_id: str,
    doctor_id: int,
) -> Dict[str, Any]:
    """
    Return the RL agent's suggested next action for a live case.
    Raises KeyError if the case doesn't exist / doesn't belong to
    this doctor.
    """

    env: DiagnosticEnvironment = clinic.get_env(
        case_id,
        doctor_id,
    )

    public_state = env.public_state()

    if public_state["done"]:
        return {
            "status": "done",
            "message": "This case is already complete.",
        }

    observation = env.current_observation()

    state = encode_state_for_rl(observation)

    actions = env.available_actions()

    if not actions:
        return {
            "status": "no_actions",
            "message": "No further actions available.",
        }

    chosen: Action = agent().choose_action(
        state=state,
        actions=actions,
        training=False,
    )

    api_type = _ACTION_TYPE_TO_API[chosen.type]

    label = (
        chosen.target
        if chosen.type == ActionType.DIAGNOSE
        else _humanize(chosen.target)
    )

    return {
        "status": "ok",
        "suggestion": {
            "type": api_type,
            "target": chosen.target,
            "label": label,
        },
    }