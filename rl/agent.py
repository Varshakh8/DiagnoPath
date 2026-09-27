from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

from env.diagnostic_env import Action, ActionType


State = Tuple[int, ...]


class LinearQLearningAgent:
    """
    Linear function-approximation Q-learning agent.

    Instead of storing one Q-value for every complete state,
    the agent learns a weight vector for each action.

    PERF FIX: the feature vector for a given state is identical
    across every action being compared (choose_action) or
    bootstrapped over (update's max over next_actions). The
    original code called state_features() fresh inside get_q()
    for every single action -- with ~100-130 available actions
    per step, that's ~100x more feature-vector construction than
    necessary, done in pure Python list comprehensions. Features
    are now computed once per state and reused across all actions
    that share that state.
    """

    def __init__(
        self,
        state_size: int = 132,
        learning_rate: float = 0.01,
        discount_factor: float = 0.90,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.9995,
        seed: int = 42,
        diagnose_lr_scale: float = 0.1,
    ) -> None:

        self.state_size = state_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # DIAGNOSE actions get updated at learning_rate *
        # diagnose_lr_scale during the RL phase (update()), not
        # the full learning_rate. Warm-started diagnose weights
        # already encode a good, hard-won discriminative pattern;
        # early in RL, epsilon is high and most diagnoses taken are
        # essentially random guesses, so full-rate TD updates on
        # DIAGNOSE actions would rapidly erode that initialization
        # with noise. ASK_SYMPTOM/ORDER_TEST actions are unaffected
        # and still learn at the full learning_rate, since those
        # start from zero and need to learn from scratch. This does
        # not apply to perceptron_step, which is only used during
        # the warm-start phase, before any RL episodes run.
        self.diagnose_lr_scale = diagnose_lr_scale

        self.random = random.Random(seed)

        # One weight vector for each action.
        self.weights: Dict[Tuple[str, str], List[float]] = {}

    # =========================================================
    # Feature engineering
    # =========================================================

    def state_features(self, state: State) -> List[float]:
        """
        Convert the 132-value clinical state into a richer
        feature vector.

        Original state:
            -1 = unknown
             0 = known negative
             1 = known positive

        Features:
            1 bias
            2 raw state values
            3 known indicators
            4 positive indicators
            5 negative indicators
            6 normalized aggregate counts
        """

        if len(state) != self.state_size:
            raise ValueError(
                f"Expected state size {self.state_size}, "
                f"got {len(state)}"
            )

        features: List[float] = [1.0]

        # Raw state.
        features.extend(float(value) for value in state)

        # Known indicator.
        features.extend(
            1.0 if value != -1 else 0.0
            for value in state
        )

        # Positive indicator.
        features.extend(
            1.0 if value == 1 else 0.0
            for value in state
        )

        # Negative indicator.
        features.extend(
            1.0 if value == 0 else 0.0
            for value in state
        )

        # Aggregate information.
        known_count = sum(
            1 for value in state if value != -1
        )

        positive_count = sum(
            1 for value in state if value == 1
        )

        negative_count = sum(
            1 for value in state if value == 0
        )

        denominator = max(1, self.state_size)

        features.extend(
            [
                known_count / denominator,
                positive_count / denominator,
                negative_count / denominator,
            ]
        )

        return features

    # =========================================================
    # Action helpers
    # =========================================================

    def action_key(
        self,
        action: Action,
    ) -> Tuple[str, str]:

        return (
            action.type.value,
            action.target,
        )

    def _ensure_action(
        self,
        action: Action,
    ) -> None:

        key = self.action_key(action)

        if key not in self.weights:
            feature_count = self.state_size * 4 + 4

            self.weights[key] = [
                0.0
                for _ in range(feature_count)
            ]

    # =========================================================
    # Q-value
    # =========================================================

    def _q_from_features(
        self,
        features: List[float],
        action: Action,
    ) -> float:
        """
        Compute Q(state, action) given an already-computed
        feature vector for that state. Use this instead of
        get_q() whenever you're evaluating multiple actions
        against the SAME state (choose_action, the max over
        next_actions in update) -- it avoids recomputing the
        feature vector redundantly for every action.
        """

        self._ensure_action(action)

        weights = self.weights[
            self.action_key(action)
        ]

        return sum(
            weight * feature
            for weight, feature in zip(
                weights,
                features,
            )
        )

    def get_q(
        self,
        state: State,
        action: Action,
    ) -> float:
        """
        Convenience single (state, action) lookup. Computes the
        feature vector fresh -- fine for one-off calls, but avoid
        calling this in a loop over multiple actions for the same
        state; use state_features() once + _q_from_features()
        instead (see choose_action and update below).
        """

        features = self.state_features(state)

        return self._q_from_features(
            features,
            action,
        )

    # =========================================================
    # Action selection
    # =========================================================

    def choose_action(
        self,
        state: State,
        actions: List[Action],
        training: bool = False,
    ) -> Action:

        if not actions:
            raise ValueError(
                "No available actions."
            )

        # Exploration during training.
        if (
            training
            and self.random.random() < self.epsilon
        ):
            return self.random.choice(actions)

        # Greedy action selection.
        # Compute the feature vector ONCE for this state and
        # reuse it across every candidate action.
        features = self.state_features(state)

        q_values = [
            self._q_from_features(features, action)
            for action in actions
        ]

        max_q = max(q_values)

        best_actions = [
            action
            for action, q_value in zip(
                actions,
                q_values,
            )
            if q_value == max_q
        ]

        return self.random.choice(
            best_actions
        )

    # =========================================================
    # Supervised warm-start update
    # =========================================================

    def supervised_update(
        self,
        features: List[float],
        action: Action,
        target: float,
    ) -> None:
        """
        Direct regression step: nudge Q(features, action) toward
        `target`, bypassing the TD bootstrap used in update().

        Used to warm-start the diagnose actions on fully-observed
        patients before RL episodes begin, so the agent starts
        sequential training from weights that already score the
        correct disease highest given full information, instead
        of learning 41-way discrimination from scratch via sparse
        end-of-episode reward alone.

        Caller is expected to have already computed `features`
        via state_features() once and reuse it across every
        diagnose action for the same patient.
        """

        self._ensure_action(action)

        current_q = self._q_from_features(
            features,
            action,
        )

        error = target - current_q

        weights = self.weights[
            self.action_key(action)
        ]

        for index in range(
            len(weights)
        ):

            weights[index] += (
                self.learning_rate
                * error
                * features[index]
            )

            weights[index] = max(
                -100.0,
                min(
                    100.0,
                    weights[index],
                ),
            )

    # =========================================================
    # Multi-class perceptron update (warm-start)
    # =========================================================

    def perceptron_step(
        self,
        features: List[float],
        true_action: Action,
        predicted_action: Action,
    ) -> None:
        """
        Classic multi-class perceptron update: push the TRUE
        action's weights toward this feature vector, and push the
        currently-predicted (wrong) action's weights away from it.
        Guaranteed to converge if the classes are linearly
        separable, and -- unlike independent per-action regression
        toward +correct/-wrong targets -- has no class-imbalance
        failure mode, since every update is exactly one push up and
        one push down regardless of how many other action classes
        exist. Used to warm-start the diagnose actions on
        fully-observed patients before RL episodes begin.

        No-op if predicted_action already equals true_action
        (already correctly classified, nothing to correct).
        """

        if (
            self.action_key(true_action)
            == self.action_key(predicted_action)
        ):
            return

        self._ensure_action(true_action)
        self._ensure_action(predicted_action)

        true_weights = self.weights[
            self.action_key(true_action)
        ]

        predicted_weights = self.weights[
            self.action_key(predicted_action)
        ]

        for index in range(len(features)):

            true_weights[index] += (
                self.learning_rate * features[index]
            )

            predicted_weights[index] -= (
                self.learning_rate * features[index]
            )

    # =========================================================
    # Q-learning update
    # =========================================================

    def update(
        self,
        state: State,
        action: Action,
        reward: float,
        next_state: State,
        next_actions: List[Action],
        done: bool,
    ) -> None:

        self._ensure_action(action)

        # Feature vector for the current state, computed once and
        # reused both for the current Q-value and the gradient step.
        features = self.state_features(state)

        current_q = self._q_from_features(
            features,
            action,
        )

        if done or not next_actions:

            target = reward

        else:

            # Feature vector for the next state, computed once and
            # reused across every candidate next action, instead of
            # recomputing it inside get_q() for each one.
            next_features = self.state_features(next_state)

            next_q = max(
                self._q_from_features(
                    next_features,
                    next_action,
                )
                for next_action in next_actions
            )

            target = (
                reward
                + self.discount_factor * next_q
            )

        error = target - current_q

        weights = self.weights[
            self.action_key(action)
        ]

        effective_lr = (
            self.learning_rate * self.diagnose_lr_scale
            if action.type == ActionType.DIAGNOSE
            else self.learning_rate
        )

        for index in range(
            len(weights)
        ):

            weights[index] += (
                effective_lr
                * error
                * features[index]
            )

            # Prevent unstable values.
            weights[index] = max(
                -100.0,
                min(
                    100.0,
                    weights[index],
                ),
            )

    # =========================================================
    # Model persistence
    # =========================================================

    def save(
        self,
        path: Path | str,
    ) -> None:

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        data = {
            "state_size": self.state_size,
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "diagnose_lr_scale": self.diagnose_lr_scale,
            "weights": {
                f"{action_type}|{target}": values
                for (
                    action_type,
                    target,
                ), values in self.weights.items()
            },
        }

        path.write_text(
            json.dumps(
                data,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls,
        path: Path | str,
    ) -> "LinearQLearningAgent":

        path = Path(path)

        data = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        agent = cls(
            state_size=data["state_size"],
            learning_rate=data["learning_rate"],
            discount_factor=data["discount_factor"],
            epsilon=data["epsilon"],
            epsilon_min=data["epsilon_min"],
            epsilon_decay=data["epsilon_decay"],
            diagnose_lr_scale=data.get(
                "diagnose_lr_scale",
                0.1,
            ),
        )

        for key, values in data[
            "weights"
        ].items():

            action_type, target = key.split(
                "|",
                1,
            )

            agent.weights[
                (
                    action_type,
                    target,
                )
            ] = [
                float(value)
                for value in values
            ]

        return agent

    # =========================================================
    # Statistics
    # =========================================================

    @property
    def number_of_actions(
        self,
    ) -> int:

        return len(
            self.weights
        )

    @property
    def number_of_parameters(
        self,
    ) -> int:

        return sum(
            len(weights)
            for weights in self.weights.values()
        )

    def decay_epsilon(self) -> None:
        """Reduce exploration after each training episode."""
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay,
        )