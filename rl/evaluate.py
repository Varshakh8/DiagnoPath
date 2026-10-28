from pathlib import Path
import random

from env.diagnostic_env import (
    DiagnosticEnvironment,
    encode_state_for_rl,
)
from rl.agent import LinearQLearningAgent


MODEL_PATH = Path("models/diagnostic_linear_q.json")
SEED = 42
MAX_STEPS = 30

DEBUG_LABELS = ("Hepatitis D", "Hepatitis E")
DEBUG_STEPS = (0, 2, 28)


def evaluate():
    random.seed(SEED)

    env = DiagnosticEnvironment()
    agent = LinearQLearningAgent.load(MODEL_PATH)

    # No exploration during evaluation.
    agent.epsilon = 0.0

    test_indices = [
        i
        for i, patient in enumerate(env.dataset.patients)
        if patient.split == "test"
    ]

    print("=" * 60)
    print("DiagnoPath AI - RL Test Evaluation")
    print("=" * 60)
    print(f"Model          : {MODEL_PATH}")
    print(f"Test cases     : {len(test_indices)}")
    print(f"Learned actions: {agent.number_of_actions}")
    print(f"Parameters     : {agent.number_of_parameters}")
    print("=" * 60)

    correct = 0

    total_reward = 0.0
    total_steps = 0
    total_symptoms = 0
    total_tests = 0
    total_cost = 0

    disease_results = {}

    for case_number, patient_index in enumerate(
        test_indices,
        start=1,
    ):
        observation = env.reset(patient_index)

        done = False
        episode_result = None
        steps = 0

        true_label = env.dataset.get(patient_index).true_label()

        while not done and steps < MAX_STEPS:
            state = encode_state_for_rl(observation)
            actions = env.available_actions()

            if not actions:
                break

            # Diagnostic debugging for Hepatitis D and E.
            # This only reads Q-scores; it does not update the model.
            if true_label in DEBUG_LABELS and steps in DEBUG_STEPS:
                features = agent.state_features(state)

                scored = [
                    (
                        agent._q_from_features(features, action),
                        action.type.value,
                        action.target,
                    )
                    for action in actions
                ]

                scored.sort(
                    key=lambda item: item[0],
                    reverse=True,
                )

                print(f"\n{true_label} | step {steps + 1}")

                print("Top 5 actions overall:")
                for score, action_type, target in scored[:5]:
                    print(
                        f"  {action_type:12s} "
                        f"{target:40s} "
                        f"Q={score:.3f}"
                    )

                diagnoses = [
                    item
                    for item in scored
                    if item[1] == "diagnose"
                ]

                diagnoses.sort(
                    key=lambda item: item[0],
                    reverse=True,
                )

                print("Top 5 diagnosis actions:")
                if diagnoses:
                    for score, action_type, target in diagnoses[:5]:
                        print(
                            f"  {action_type:12s} "
                            f"{target:40s} "
                            f"Q={score:.3f}"
                        )
                else:
                    print("  No diagnosis actions available.")

            action = agent.choose_action(
                state,
                actions,
                training=False,
            )

            observation, reward, done, info = env.step(action)

            # Print the selected action for the two debug cases.
            if true_label in DEBUG_LABELS:
                print(
                    f"  step={steps + 1}, "
                    f"action={action.type.value}, "
                    f"target={action.target}, "
                    f"reward={reward:.2f}, "
                    f"done={done}"
                )

            # The environment returns EpisodeResult as
            # info["episode"] when diagnosis happens.
            if "episode" in info:
                episode_result = info["episode"]

            steps += 1

        # -----------------------------------------------------
        # Episode result
        # -----------------------------------------------------

        if episode_result is not None:
            predicted = episode_result.predicted
            true_label = episode_result.true_label
        else:
            predicted = ""
            true_label = env.dataset.get(patient_index).true_label()

        is_correct = (
            predicted != ""
            and predicted == true_label
        )

        if is_correct:
            correct += 1

        # -----------------------------------------------------
        # Statistics
        # -----------------------------------------------------

        stats = env.episode_stats()

        total_reward += stats["total_reward"]
        total_steps += stats["steps"]
        total_symptoms += stats["symptoms_asked"]
        total_tests += stats["tests_ordered"]

        diagnostic_cost = (
            stats["symptoms_asked"] * env.ask_cost
            + stats["total_test_cost"]
        )

        total_cost += diagnostic_cost

        # -----------------------------------------------------
        # Disease results
        # -----------------------------------------------------

        if true_label not in disease_results:
            disease_results[true_label] = {
                "total": 0,
                "correct": 0,
            }

        disease_results[true_label]["total"] += 1

        if is_correct:
            disease_results[true_label]["correct"] += 1

        # -----------------------------------------------------
        # Case output
        # -----------------------------------------------------

        print(
            f"Case {case_number:2d}/{len(test_indices)} | "
            f"true={true_label} | "
            f"predicted={predicted} | "
            f"correct={is_correct} | "
            f"steps={stats['steps']}"
        )

    # ---------------------------------------------------------
    # Final metrics
    # ---------------------------------------------------------

    n = len(test_indices)

    accuracy = correct / n if n else 0.0
    avg_reward = total_reward / n if n else 0.0
    avg_steps = total_steps / n if n else 0.0
    avg_symptoms = total_symptoms / n if n else 0.0
    avg_tests = total_tests / n if n else 0.0
    avg_cost = total_cost / n if n else 0.0

    print()

    print("=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)

    print(f"Test cases              : {n}")
    print(f"Correct diagnoses       : {correct}")
    print(
        f"Test accuracy           : "
        f"{accuracy:.3f} "
        f"({accuracy * 100:.1f}%)"
    )
    print(f"Average reward          : {avg_reward:.2f}")
    print(f"Average steps           : {avg_steps:.2f}")
    print(f"Average symptoms asked  : {avg_symptoms:.2f}")
    print(f"Average tests ordered   : {avg_tests:.2f}")
    print(f"Average diagnostic cost : {avg_cost:.2f}")

    # ---------------------------------------------------------
    # Per-disease results
    # ---------------------------------------------------------

    print()

    print("=" * 60)
    print("PER-DISEASE RESULTS")
    print("=" * 60)

    for disease in sorted(disease_results):
        result = disease_results[disease]

        disease_accuracy = (
            result["correct"] / result["total"]
            if result["total"]
            else 0.0
        )

        print(
            f"{disease:35s} "
            f"{result['correct']}/"
            f"{result['total']} "
            f"({disease_accuracy * 100:.1f}%)"
        )

    print("=" * 60)


if __name__ == "__main__":
    evaluate()