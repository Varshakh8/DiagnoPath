"""
DiagnoPath AI - Member 1: Diagnostic Environment

Models diagnosis as sequential decisions
(symptom inquiry -> investigations -> diagnose)
across multiple diseases from the Kaggle dataset.

No built-in ML / RL libraries.

FIX APPLIED (performance):
    encode_state_for_rl() previously called DatasetMeta.load()
    on every single invocation when no `meta` argument was
    passed. Since it's called twice per environment step (once
    for the current state, once for the next state), this meant
    re-reading and re-parsing dataset_meta.json from disk
    hundreds of thousands of times over a full training run --
    the actual cause of multi-hour training times, not the
    Q-learning itself. It now defaults to the module-level
    META object that env.data_model already loads exactly once
    at import time.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .data_model import (
    ASK_COST,
    DISEASES,
    MEDICAL_TESTS,
    META,
    SYMPTOMS,
    TEST_COSTS,
    ClinicalDataset,
    DatasetMeta,
    Patient,
    default_data_path,
)


class ActionType(str, Enum):
    ASK_SYMPTOM = "ask_symptom"
    ORDER_TEST = "order_test"
    DIAGNOSE = "diagnose"


@dataclass
class Action:
    type: ActionType
    target: str


@dataclass
class Observation:
    """What the agent currently knows about the patient."""

    patient_id: str
    known_symptoms: Dict[str, Optional[int]]
    known_tests: Dict[str, Optional[int]]
    step: int
    done: bool
    last_reward: float
    message: str
    disease_options: List[str]


@dataclass
class EpisodeResult:
    correct: bool
    predicted: str
    true_label: str
    symptoms_asked: int
    tests_ordered: int
    total_test_cost: int
    steps: int
    total_reward: float


class DiagnosticEnvironment:
    """
    Sequential clinical decision environment for multi-disease diagnosis.

    State:
        Partially observed patient.

    Actions:
        ask symptom
        order investigation
        make final diagnosis

    Reward:
        +15 correct diagnosis
        -25 wrong diagnosis
        -ask_cost for each symptom inquiry
        -investigation_cost for each investigation
        -0.5 for redundant ask/order
        +1.0 for discovering a positive finding

    Additional rule:
        The agent cannot diagnose during the first two steps.
        This prevents the trivial "guess immediately" strategy.
    """

    REWARD_CORRECT = 15.0
    REWARD_WRONG = -25.0
    REDUNDANT_PENALTY = -0.5
    POSITIVE_FINDING_BONUS = 1.0

    def __init__(
        self,
        dataset: Optional[ClinicalDataset] = None,
    ):
        self.dataset = (
            dataset
            or ClinicalDataset.from_csv(
                default_data_path()
            )
        )

        self.meta: DatasetMeta = self.dataset.meta

        self.symptoms = list(
            self.meta.inquiry_symptoms
        )

        self.tests = list(
            self.meta.investigation_features
        )

        self.diseases = list(
            self.meta.diseases
        )

        self.ask_cost = self.meta.ask_cost

        self.test_costs = {
            t: self.meta.investigation_cost
            for t in self.tests
        }

        self._patient: Optional[Patient] = None

        self._known_symptoms: Dict[
            str,
            Optional[int],
        ] = {}

        self._known_tests: Dict[
            str,
            Optional[int],
        ] = {}

        self._asked: set[str] = set()
        self._ordered: set[str] = set()

        self._step = 0
        self._done = False

        self._total_reward = 0.0
        self._total_test_cost = 0

        self._history: List[str] = []
        self._predicted: Optional[str] = None

    # =========================================================
    # Reset
    # =========================================================

    def reset(
        self,
        patient_index: int = 0,
    ) -> Observation:

        if (
            patient_index < 0
            or patient_index >= len(self.dataset)
        ):
            raise IndexError(
                f"patient_index out of range: "
                f"{patient_index}"
            )

        self._patient = self.dataset.get(
            patient_index
        )

        self._known_symptoms = {
            s: None
            for s in self.symptoms
        }

        self._known_tests = {
            t: None
            for t in self.tests
        }

        self._asked = set()
        self._ordered = set()

        self._step = 0
        self._done = False

        self._total_reward = 0.0
        self._total_test_cost = 0

        self._predicted = None

        self._history = [
            f"Case opened: "
            f"{self._patient.patient_id} "
            f"(split={self._patient.split})"
        ]

        return self._observe(
            "New patient case started. "
            "Begin sequential multi-disease diagnosis."
        )

    # =========================================================
    # Available actions
    # =========================================================

    def available_actions(
        self,
    ) -> List[Action]:

        if (
            self._done
            or self._patient is None
        ):
            return []

        actions: List[Action] = []

        # Unasked symptoms
        for symptom in self.symptoms:
            if symptom not in self._asked:
                actions.append(
                    Action(
                        ActionType.ASK_SYMPTOM,
                        symptom,
                    )
                )

        # Unordered investigations
        for test in self.tests:
            if test not in self._ordered:
                actions.append(
                    Action(
                        ActionType.ORDER_TEST,
                        test,
                    )
                )

        # Diagnosis actions
        for disease in self.diseases:
            actions.append(
                Action(
                    ActionType.DIAGNOSE,
                    disease,
                )
            )

        return actions

    # =========================================================
    # Environment step
    # =========================================================

    def step(
        self,
        action: Action,
    ) -> Tuple[
        Observation,
        float,
        bool,
        Dict[str, Any],
    ]:

        if self._patient is None:
            raise RuntimeError(
                "Call reset() before step()."
            )

        if self._done:
            raise RuntimeError(
                "Episode finished. "
                "Call reset() for a new patient."
            )

        self._step += 1

        info: Dict[str, Any] = {
            "action": action
        }

        if action.type == ActionType.ASK_SYMPTOM:

            reward, msg = self._ask_symptom(
                action.target
            )

        elif action.type == ActionType.ORDER_TEST:

            reward, msg = self._order_test(
                action.target
            )

        elif action.type == ActionType.DIAGNOSE:

            reward, msg, info_extra = (
                self._diagnose(
                    action.target
                )
            )

            info.update(info_extra)

        else:

            raise ValueError(
                f"Unknown action type: "
                f"{action.type}"
            )

        # Maximum episode length.
        # Only apply it if diagnosis did not
        # already finish the episode.
        if (
            self._step >= 30
            and not self._done
        ):
            self._done = True
            info["timeout"] = True

        self._total_reward += reward

        self._history.append(msg)

        obs = self._observe(msg)

        obs.last_reward = reward

        return (
            obs,
            reward,
            self._done,
            info,
        )

    # =========================================================
    # History
    # =========================================================

    def get_history(
        self,
    ) -> List[str]:

        return list(
            self._history
        )

    # =========================================================
    # Current patient
    # =========================================================

    def current_patient(
        self,
    ) -> Optional[Patient]:

        return self._patient

    # =========================================================
    # Public state for doctor UI
    # =========================================================

    def public_state(
        self,
    ) -> Dict[str, Any]:

        """
        JSON-safe view for the doctor UI.

        True disease label remains hidden until
        the episode is finished.
        """

        revealed_symptoms = [
            {
                "name": name,
                "label": _humanize(name),
                "present": bool(value),
                "cost": self.ask_cost,
            }
            for name, value
            in self._known_symptoms.items()
            if value is not None
        ]

        revealed_tests = [
            {
                "name": name,
                "label": _humanize(name),
                "present": bool(value),
                "cost": self.test_costs[name],
            }
            for name, value
            in self._known_tests.items()
            if value is not None
        ]

        pending_symptoms = [
            {
                "name": symptom,
                "label": _humanize(symptom),
                "cost": self.ask_cost,
            }
            for symptom in self.symptoms
            if symptom not in self._asked
        ]

        pending_tests = [
            {
                "name": test,
                "label": _humanize(test),
                "cost": self.test_costs[test],
            }
            for test in self.tests
            if test not in self._ordered
        ]

        patient = self._patient

        return {
            "patient_id": (
                patient.patient_id
                if patient
                else None
            ),

            "split": (
                patient.split
                if patient
                else None
            ),

            "done": self._done,

            "predicted": self._predicted,

            "true_label": (
                patient.true_label()
                if (
                    patient
                    and self._done
                )
                else None
            ),

            "correct": (
                self._predicted
                == patient.true_label()
                if (
                    patient
                    and self._done
                    and self._predicted
                )
                else None
            ),

            "stats": self.episode_stats(),

            "revealed_symptoms":
                revealed_symptoms,

            "revealed_tests":
                revealed_tests,

            "pending_symptoms":
                pending_symptoms,

            "pending_tests":
                pending_tests,

            "diseases":
                list(self.diseases),

            "history":
                self.get_history(),
        }
        # =========================================================
    # Live observation snapshot (Member 2 hook)
    # =========================================================

    def current_observation(
        self,
    ) -> Observation:

        """
        Return a live snapshot Observation of everything currently
        known about this case, WITHOUT stepping forward or
        mutating any state. Used by the RL suggestion endpoint to
        build encode_state_for_rl()'s input mid-consultation.
        """

        if self._patient is None:
            raise RuntimeError(
                "Call reset() before current_observation()."
            )

        return self._observe(
            "Live state snapshot."
        )

    # =========================================================
    # Episode statistics
    # =========================================================

    def episode_stats(
        self,
    ) -> Dict[str, Any]:

        return {
            "steps": self._step,

            "tests_ordered":
                len(self._ordered),

            "symptoms_asked":
                len(self._asked),

            "total_test_cost":
                self._total_test_cost,

            "total_reward":
                self._total_reward,

            "done":
                self._done,
        }

    # =========================================================
    # Ask symptom
    # =========================================================

    def _ask_symptom(
        self,
        symptom: str,
    ) -> Tuple[float, str]:

        assert self._patient is not None

        if symptom not in self.symptoms:

            return (
                self.REDUNDANT_PENALTY,
                f"Invalid symptom '{symptom}'.",
            )

        if symptom in self._asked:

            return (
                self.REDUNDANT_PENALTY,
                f"Already asked about {symptom}.",
            )

        value = self._patient.symptoms[
            symptom
        ]

        self._asked.add(symptom)

        self._known_symptoms[
            symptom
        ] = value

        self._total_test_cost += (
            self.ask_cost
        )

        reward = -float(
            self.ask_cost
        )

        if value == 1:
            reward += (
                self.POSITIVE_FINDING_BONUS
            )

        present = (
            "YES"
            if value == 1
            else "NO"
        )

        return (
            reward,
            (
                f"Asked symptom "
                f"'{symptom}': {present} "
                f"| cost={self.ask_cost} "
                f"(reward {reward:+.2f})"
            ),
        )

    # =========================================================
    # Order investigation
    # =========================================================

    def _order_test(
        self,
        test_name: str,
    ) -> Tuple[float, str]:

        assert self._patient is not None

        if test_name not in self.tests:

            return (
                self.REDUNDANT_PENALTY,
                (
                    f"Invalid investigation "
                    f"'{test_name}'."
                ),
            )

        if test_name in self._ordered:

            return (
                self.REDUNDANT_PENALTY,
                (
                    f"Already ordered "
                    f"{test_name}."
                ),
            )

        value = self._patient.test_values[
            test_name
        ]

        cost = self.test_costs[
            test_name
        ]

        self._ordered.add(
            test_name
        )

        self._known_tests[
            test_name
        ] = value

        self._total_test_cost += cost

        reward = -float(cost)

        if value == 1:
            reward += (
                self.POSITIVE_FINDING_BONUS
            )

        label = MEDICAL_TESTS.get(
            test_name,
            {},
        ).get(
            "label",
            test_name.replace(
                "_",
                " ",
            ).title(),
        )

        present = (
            "POSITIVE"
            if value == 1
            else "NEGATIVE"
        )

        return (
            reward,
            (
                f"Ordered investigation "
                f"'{label}': {present} "
                f"| cost={cost} "
                f"(reward {reward:+.2f})"
            ),
        )

    # =========================================================
    # Diagnose
    # =========================================================

    def _diagnose(
        self,
        disease: str,
    ) -> Tuple[
        float,
        str,
        Dict[str, Any],
    ]:

        assert self._patient is not None

        # -----------------------------------------------------
        # Prevent immediate diagnosis.
        #
        # The agent must first collect at least
        # two pieces of clinical information.
        # -----------------------------------------------------
        if self._step < 3:

            return (
                -5.0,
                (
                    "Diagnosis rejected: "
                    "gather more clinical "
                    "information first."
                ),
                {
                    "correct": False,
                    "premature": True,
                },
            )

        if disease not in self.diseases:

            return (
                self.REDUNDANT_PENALTY,
                (
                    f"Unknown disease label "
                    f"'{disease}'."
                ),
                {
                    "correct": False
                },
            )

        true_label = (
            self._patient.true_label()
        )

        correct = (
            disease == true_label
        )

        reward = (
            self.REWARD_CORRECT
            if correct
            else self.REWARD_WRONG
        )

        self._done = True
        self._predicted = disease

        result = EpisodeResult(
            correct=correct,
            predicted=disease,
            true_label=true_label,
            symptoms_asked=len(
                self._asked
            ),
            tests_ordered=len(
                self._ordered
            ),
            total_test_cost=(
                self._total_test_cost
            ),
            steps=self._step,
            total_reward=(
                self._total_reward
                + reward
            ),
        )

        verdict = (
            "CORRECT"
            if correct
            else "WRONG"
        )

        msg = (
            f"DIAGNOSIS: {disease} "
            f"-> {verdict} "
            f"(true={true_label}) "
            f"| reward {reward:+.2f} "
            f"| asks="
            f"{result.symptoms_asked}, "
            f"tests="
            f"{result.tests_ordered}, "
            f"cost="
            f"{result.total_test_cost}"
        )

        return (
            reward,
            msg,
            {
                "correct": correct,
                "episode": result,
            },
        )

    # =========================================================
    # Observation
    # =========================================================

    def _observe(
        self,
        message: str,
    ) -> Observation:

        assert self._patient is not None

        return Observation(
            patient_id=(
                self._patient.patient_id
            ),

            known_symptoms=dict(
                self._known_symptoms
            ),

            known_tests=dict(
                self._known_tests
            ),

            step=self._step,

            done=self._done,

            last_reward=0.0,

            message=message,

            disease_options=list(
                self.diseases
            ),
        )


# =============================================================
# Helpers
# =============================================================

def _humanize(
    name: str,
) -> str:

    return (
        name
        .replace("_", " ")
        .replace("-", " ")
        .strip()
        .title()
    )


# =============================================================
# RL state encoder
# =============================================================

def encode_state_for_rl(
    obs: Observation,
    meta: Optional[DatasetMeta] = None,
) -> Tuple[int, ...]:

    """
    Discrete state encoding helper for Member 2.

    Encoding:

        -1 = unknown
         0 = known negative
         1 = known positive

    PERF FIX: previously defaulted to `DatasetMeta.load()`,
    which re-reads and re-parses dataset_meta.json from disk
    on every single call. Since this is called at least twice
    per environment step, over a full training run that meant
    hundreds of thousands of redundant disk reads -- the actual
    cause of multi-hour training times. It now defaults to the
    module-level META object that env.data_model loads exactly
    once at import time. Pass an explicit `meta` only if you
    need a different vocabulary than the one currently loaded.
    """

    meta = meta or META

    symptom_tuple = tuple(
        (
            -1
            if obs.known_symptoms.get(
                symptom
            ) is None
            else int(
                obs.known_symptoms[
                    symptom
                ]
            )
        )
        for symptom
        in meta.inquiry_symptoms
    )

    test_tuple = tuple(
        (
            -1
            if obs.known_tests.get(
                test
            ) is None
            else int(
                obs.known_tests[
                    test
                ]
            )
        )
        for test
        in meta.investigation_features
    )

    return (
        symptom_tuple
        + test_tuple
    )


# =============================================================
# Backward-compatible exports
# =============================================================

__all__ = [
    "Action",
    "ActionType",
    "DiagnosticEnvironment",
    "EpisodeResult",
    "Observation",
    "encode_state_for_rl",
    "ASK_COST",
    "DISEASES",
    "MEDICAL_TESTS",
    "SYMPTOMS",
    "TEST_COSTS",
]