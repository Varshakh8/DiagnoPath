"""
DiagnoPath AI - Member 2
Train the custom Linear Q-learning diagnostic agent.

FIXES APPLIED:
1. EPISODES raised from 200 -> 30000, with epsilon_decay tuned so
   epsilon reaches epsilon_min around episode ~10,000.
2. Periodic held-out test evaluation (EVAL_EVERY) and a rolling-window
   training accuracy (WINDOW) so you can see real learning progress,
   not just a cumulative-since-start number dragged down by early
   near-random episodes.
3. Relies on the env/diagnostic_env.py fix to encode_state_for_rl()
   (no longer reloads dataset_meta.json from disk every call) and the
   rl/agent.py fix (feature vector computed once per state, not once
   per action) -- together these were the actual cause of multi-hour
   training times.
4. A supervised WARM-START pass on the diagnose actions before any
   RL episodes run. Plain sequential Q-learning here was trending
   *worse* than random as epsilon dropped (recent_acc: 0.030 -> 0.008
   over the first 8000 episodes) -- with 41 classes and a sparse
   terminal-only reward (a given disease is wrong ~97.6% of the time),
   the diagnose-action weights had no structured signal pulling them
   toward the right discriminating features.

   FIRST attempt at this warm-start used independent regression
   per action (+15 toward the correct disease, -25 toward all 40
   others, for every patient). That collapsed too: with 40 negative
   updates against every 1 positive update per action, the negative
   signal swamped the positive one and every action's weights just
   got pushed uniformly negative, with no discrimination. Full-info
   "accuracy" stayed flat at the ~2.4% random baseline through all
   10 epochs -- worse, since this is a dataset that's normally
   trivially separable (~95-100% with a plain linear classifier).

   FIX: replaced it with the classic multi-class PERCEPTRON update.
   For each patient, only two weight vectors are touched per
   example: the true disease's weights get nudged toward the
   feature vector, and whichever disease the model currently
   (wrongly) predicts highest gets nudged away from it. No
   imbalance is possible, since every update is exactly one push
   up and one push down regardless of class count, and this rule
   is provably guaranteed to converge on linearly separable data
   -- which this dataset is known to be. This alone got full-info
   training accuracy to 100% within 2 epochs.

   SECOND problem found: that 100% full-info accuracy did NOT
   transfer to real episodes. Test accuracy right after warm-start
   was only 4.8% -- because warm-start only ever showed the agent
   FULLY-revealed patients (all 132 features known), but at actual
   diagnosis time the agent has only asked about ~4-10 symptoms;
   the other ~120+ are unknown (-1). The learned weights had never
   seen that sparse regime and didn't generalize to it. Worse, the
   first 1000 RL episodes (run with epsilon still ~0.74, so mostly
   random actions) then eroded even that 4.8% down to 1.4%, because
   every wrong random diagnosis fed a full-strength TD update
   straight into the same warm-started weights.

   FIX: two changes.
     (a) Warm-start now trains on randomly-MASKED states (a random
         subset of features revealed, the rest set to -1 unknown),
         not fully-revealed ones -- so the weights actually learn
         to discriminate under the same sparse observability the
         agent faces for real. A short full-info phase still runs
         first purely as a fast, clean starting point.
     (b) DIAGNOSE actions now update at a much smaller learning
         rate during the RL phase (agent.diagnose_lr_scale, default
         0.1x) than ASK_SYMPTOM/ORDER_TEST actions. This protects
         the warm-started weights from being wiped out by noisy
         random-exploration mistakes early in training, while still
         letting them refine gradually as the ask/order sub-policy
         improves and epsilon drops.
"""

from __future__ import annotations

import random
from collections import deque
from pathlib import Path

from env.diagnostic_env import (
    Action,
    ActionType,
    DiagnosticEnvironment,
    encode_state_for_rl,
)

from env.data_model import (
    META,
    ClinicalDataset,
    DatasetMeta,
    Patient,
    default_data_path,
)

from rl.agent import LinearQLearningAgent


# =========================================================
# CONFIGURATION
# =========================================================

EPISODES = 30000

MAX_STEPS = 30

SEED = 42

MODEL_PATH = Path(
    "models/diagnostic_linear_q.json"
)

# How often (in episodes) to print training progress.
PROGRESS_EVERY = 1000

# How often (in episodes) to run a full greedy pass over the
# held-out test set and report accuracy. Set to 0 to disable.
EVAL_EVERY = 5000

# Size of the rolling window used for the "recent accuracy"
# metric printed alongside cumulative accuracy.
WINDOW = 1000

# Warm-start configuration.
# Phase A: a few epochs on fully-revealed patients, just to give
# the perceptron a clean, quickly-converging starting point.
WARM_START_FULL_EPOCHS = 3
# Phase B: many more epochs on randomly-masked (partially-revealed)
# patients -- this is the phase that actually matters, since it's
# the only one that trains the weights on the same kind of sparse,
# mostly-unknown feature vectors the agent will see at real
# diagnosis time.
WARM_START_PARTIAL_EPOCHS = 40
WARM_START_LEARNING_RATE = 0.05


# =========================================================
# WARM-START HELPERS
# =========================================================

def build_full_state(
    patient: Patient,
    meta: DatasetMeta,
) -> tuple:
    """
    Encode a patient as if every symptom and test were already
    known. Same tuple format as encode_state_for_rl, built
    directly from the patient record.
    """

    symptom_tuple = tuple(
        int(patient.symptoms[s])
        for s in meta.inquiry_symptoms
    )

    test_tuple = tuple(
        int(patient.test_values[t])
        for t in meta.investigation_features
    )

    return symptom_tuple + test_tuple


def build_partial_state(
    full_values: list[int],
    rng: random.Random,
    n_known: int,
) -> tuple:
    """
    Mask a fully-known feature list down to n_known randomly
    chosen positions (their real 0/1 value), with every other
    position set to -1 (unknown). This mirrors what the agent
    actually sees at diagnosis time during a real episode -- a
    small, randomly-ordered subset of symptoms/tests revealed,
    not the full patient record. Training the diagnose weights
    only on fully-known states (all features revealed) does not
    transfer to this sparse regime; the weights need to have
    actually seen sparse, mostly-unknown feature vectors during
    training to generalize to them.
    """

    n = len(full_values)
    n_known = max(0, min(n_known, n))

    revealed_indices = rng.sample(
        range(n),
        n_known,
    )

    state = [-1] * n

    for index in revealed_indices:
        state[index] = full_values[index]

    return tuple(state)


def warm_start_diagnose_weights(
    agent: LinearQLearningAgent,
    dataset: ClinicalDataset,
    train_indices: list[int],
) -> None:
    """
    Supervised pretraining pass over the diagnose actions only.
    See module docstring point 4 for why this exists, and why a
    plain full-info-only version failed to transfer to real
    (sparse) diagnosis-time states.
    """

    diagnose_actions = [
        Action(ActionType.DIAGNOSE, disease)
        for disease in META.diseases
    ]

    saved_lr = agent.learning_rate
    agent.learning_rate = WARM_START_LEARNING_RATE

    rng_local = random.Random(SEED)
    indices = list(train_indices)

    # Precompute each patient's fully-known feature list once;
    # both phases mask down from this.
    full_values_by_index = {
        patient_index: [
            int(dataset.get(patient_index).symptoms[s])
            for s in META.inquiry_symptoms
        ] + [
            int(dataset.get(patient_index).test_values[t])
            for t in META.investigation_features
        ]
        for patient_index in indices
    }

    n_features = META.n_symptoms

    def run_epoch(get_state, epoch_label: str) -> float:
        """
        One epoch of the multi-class perceptron update. `get_state`
        takes a patient_index and returns the (possibly masked)
        state tuple to train on for that example.
        """

        rng_local.shuffle(indices)

        correct = 0

        for patient_index in indices:

            patient = dataset.get(patient_index)

            state = get_state(patient_index)

            features = agent.state_features(state)

            predicted_action = None
            best_q = float("-inf")

            for action in diagnose_actions:
                q = agent._q_from_features(features, action)
                if q > best_q:
                    best_q = q
                    predicted_action = action

            true_action = Action(
                ActionType.DIAGNOSE,
                patient.disease,
            )

            if (
                predicted_action is not None
                and predicted_action.target == patient.disease
            ):
                correct += 1

            if predicted_action is not None:
                agent.perceptron_step(
                    features,
                    true_action,
                    predicted_action,
                )

        accuracy = correct / len(indices)

        print(
            f"  {epoch_label} | accuracy: {accuracy:.3f} "
            f"({accuracy * 100:.1f}%)"
        )

        return accuracy

    print("=" * 60)
    print("Warm-starting diagnose-action weights")
    print("=" * 60)

    print(
        f"Phase A: {WARM_START_FULL_EPOCHS} epochs, "
        f"fully-revealed patients (clean anchor)"
    )

    for epoch in range(1, WARM_START_FULL_EPOCHS + 1):

        run_epoch(
            lambda idx: tuple(full_values_by_index[idx]),
            f"Phase A epoch {epoch:2d}/{WARM_START_FULL_EPOCHS}",
        )

    print()

    print(
        f"Phase B: {WARM_START_PARTIAL_EPOCHS} epochs, "
        f"randomly-masked patients (matches real sparse "
        f"diagnosis-time states)"
    )

    for epoch in range(1, WARM_START_PARTIAL_EPOCHS + 1):

        def masked_state(idx, _rng=rng_local, _n=n_features):
            n_known = _rng.randint(1, _n)
            return build_partial_state(
                full_values_by_index[idx],
                _rng,
                n_known,
            )

        run_epoch(
            masked_state,
            f"Phase B epoch {epoch:2d}/{WARM_START_PARTIAL_EPOCHS}",
        )

    agent.learning_rate = saved_lr

    print("Warm-start complete.")
    print("=" * 60)
    print()


# =========================================================
# EVALUATION HELPER
# =========================================================

def evaluate_on_test(
    agent: LinearQLearningAgent,
    dataset: ClinicalDataset,
    test_indices: list[int],
) -> float:
    """
    Run the agent greedily (epsilon=0) over every test patient
    and return accuracy. Restores the agent's epsilon afterward.
    """

    saved_epsilon = agent.epsilon
    agent.epsilon = 0.0

    env = DiagnosticEnvironment(dataset)

    correct = 0

    for patient_index in test_indices:

        observation = env.reset(patient_index)

        done = False
        steps = 0
        predicted = ""
        true_label = dataset.get(patient_index).true_label()

        while not done and steps < MAX_STEPS:

            state = encode_state_for_rl(observation)

            actions = env.available_actions()

            if not actions:
                break

            action = agent.choose_action(
                state=state,
                actions=actions,
                training=False,
            )

            observation, reward, done, info = env.step(action)

            if "episode" in info:
                predicted = info["episode"].predicted

            steps += 1

        if predicted != "" and predicted == true_label:
            correct += 1

    agent.epsilon = saved_epsilon

    return correct / len(test_indices) if test_indices else 0.0


# =========================================================
# TRAINING
# =========================================================

def train():

    dataset = ClinicalDataset.from_csv(
        default_data_path()
    )

    train_indices = [
        index
        for index, patient
        in enumerate(dataset.patients)
        if patient.split == "train"
    ]

    test_indices = [
        index
        for index, patient
        in enumerate(dataset.patients)
        if patient.split == "test"
    ]

    if not train_indices:
        raise RuntimeError(
            "No training patients found."
        )

    rng = random.Random(
        SEED
    )

    agent = LinearQLearningAgent(
        state_size=132,

        learning_rate=0.005,

        discount_factor=0.90,

        epsilon=1.0,

        epsilon_min=0.05,

        epsilon_decay=0.9997,

        seed=SEED,
    )

    print("=" * 60)
    print(
        "DiagnoPath AI - Linear Q-Learning Training"
    )
    print("=" * 60)

    print(
        f"Total patients : {len(dataset)}"
    )

    print(
        f"Training cases : {len(train_indices)}"
    )

    print(
        f"Test cases     : {len(test_indices)}"
    )

    print(
        f"Episodes       : {EPISODES}"
    )

    print(
        f"Max steps      : {MAX_STEPS}"
    )

    print()

    # =====================================================
    # WARM START
    # =====================================================

    warm_start_diagnose_weights(
        agent,
        dataset,
        train_indices,
    )

    warm_start_test_accuracy = evaluate_on_test(
        agent,
        dataset,
        test_indices,
    )

    print(
        f"TEST accuracy right after warm-start "
        f"(before any RL episodes): "
        f"{warm_start_test_accuracy:.3f} "
        f"({warm_start_test_accuracy * 100:.1f}%)"
    )
    print()

    total_correct = 0
    total_reward = 0.0
    total_steps = 0

    total_tests = 0
    total_symptoms = 0

    # Rolling window of 0/1 correctness for the last WINDOW episodes.
    recent_correct = deque(maxlen=WINDOW)

    # =====================================================
    # EPISODES
    # =====================================================

    for episode in range(
        1,
        EPISODES + 1,
    ):

        patient_index = rng.choice(
            train_indices
        )

        env = DiagnosticEnvironment(
            dataset
        )

        observation = env.reset(
            patient_index
        )

        state = encode_state_for_rl(
            observation
        )

        episode_reward = 0.0

        episode_correct = False

        done = False

        # -------------------------------------------------
        # Episode steps
        # -------------------------------------------------

        for _ in range(
            MAX_STEPS
        ):

            actions = env.available_actions()

            if not actions:
                break

            action = agent.choose_action(
                state=state,
                actions=actions,
                training=True,
            )

            (
                next_observation,
                reward,
                done,
                info,
            ) = env.step(action)

            next_state = (
                encode_state_for_rl(
                    next_observation
                )
            )

            next_actions = (
                []
                if done
                else env.available_actions()
            )

            agent.update(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                next_actions=next_actions,
                done=done,
            )

            state = next_state

            episode_reward += reward

            if done:

                episode_correct = bool(
                    info.get(
                        "correct",
                        False,
                    )
                )

                break

        # -------------------------------------------------
        # Decay exploration
        # -------------------------------------------------

        agent.decay_epsilon()

        # -------------------------------------------------
        # Statistics
        # -------------------------------------------------

        stats = env.episode_stats()

        total_reward += episode_reward

        total_steps += stats[
            "steps"
        ]

        total_tests += stats[
            "tests_ordered"
        ]

        total_symptoms += stats[
            "symptoms_asked"
        ]

        if episode_correct:
            total_correct += 1

        recent_correct.append(
            1 if episode_correct else 0
        )

        # -------------------------------------------------
        # Progress
        # -------------------------------------------------

        if episode % PROGRESS_EVERY == 0:

            cumulative_accuracy = (
                total_correct
                / episode
            )

            recent_accuracy = (
                sum(recent_correct)
                / len(recent_correct)
            )

            avg_reward = (
                total_reward
                / episode
            )

            avg_steps = (
                total_steps
                / episode
            )

            avg_tests = (
                total_tests
                / episode
            )

            avg_symptoms = (
                total_symptoms
                / episode
            )

            print(
                f"Episode {episode:6d} | "
                f"epsilon={agent.epsilon:.3f} | "
                f"cum_acc={cumulative_accuracy:.3f} | "
                f"recent_acc(last {len(recent_correct)})="
                f"{recent_accuracy:.3f} | "
                f"reward={avg_reward:.2f} | "
                f"steps={avg_steps:.2f} | "
                f"symptoms={avg_symptoms:.2f} | "
                f"tests={avg_tests:.2f}"
            )

        # -------------------------------------------------
        # Periodic held-out test evaluation
        # -------------------------------------------------

        if (
            EVAL_EVERY
            and episode % EVAL_EVERY == 0
            and test_indices
        ):

            test_accuracy = evaluate_on_test(
                agent,
                dataset,
                test_indices,
            )

            print(
                f"           >>> "
                f"TEST accuracy at episode {episode}: "
                f"{test_accuracy:.3f} "
                f"({test_accuracy * 100:.1f}%)"
            )

    # =====================================================
    # SAVE
    # =====================================================

    agent.save(
        MODEL_PATH
    )

    # =====================================================
    # FINAL RESULTS
    # =====================================================

    print()
    print("=" * 60)
    print("Training complete")
    print("=" * 60)

    print(
        f"Training accuracy (cumulative) : "
        f"{total_correct / EPISODES:.3f}"
    )

    print(
        f"Training accuracy (last {WINDOW}): "
        f"{sum(recent_correct) / len(recent_correct):.3f}"
    )

    print(
        f"Average reward    : "
        f"{total_reward / EPISODES:.2f}"
    )

    print(
        f"Average steps     : "
        f"{total_steps / EPISODES:.2f}"
    )

    print(
        f"Average symptoms  : "
        f"{total_symptoms / EPISODES:.2f}"
    )

    print(
        f"Average tests     : "
        f"{total_tests / EPISODES:.2f}"
    )

    print(
        f"Learned actions   : "
        f"{agent.number_of_actions}"
    )

    print(
        f"Parameters        : "
        f"{agent.number_of_parameters}"
    )

    print(
        f"Final epsilon     : "
        f"{agent.epsilon:.3f}"
    )

    if test_indices:

        final_test_accuracy = evaluate_on_test(
            agent,
            dataset,
            test_indices,
        )

        print(
            f"Final TEST accuracy: "
            f"{final_test_accuracy:.3f} "
            f"({final_test_accuracy * 100:.1f}%)"
        )

    print(
        f"Model saved to    : "
        f"{MODEL_PATH}"
    )

    print("=" * 60)


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    train()