
from pathlib import Path
import random

from env.diagnostic_env import (
    ActionType,
    DiagnosticEnvironment,
    encode_state_for_rl,
)
from rl.agent import LinearQLearningAgent


MODEL_PATH = Path("models/diagnostic_linear_q.json")
SEED = 42
MAX_STEPS = 30


def evaluate(hidden_symptom=None):
    random.seed(SEED)

    env = DiagnosticEnvironment()
    agent = LinearQLearningAgent.load(MODEL_PATH)
    agent.epsilon = 0.0

    # Use the exact same held-out test patients.
    test_indices = [
        i
        for i, patient in enumerate(env.dataset.patients)
        if patient.split == "test"
    ]

    # Temporarily prevent the agent from asking one symptom.
    if hidden_symptom is not None:
        if hidden_symptom not in env.symptoms:
            raise ValueError(
                f"Unknown symptom: {hidden_symptom!r}\n"
                f"Available symptoms: {env.symptoms}"
            )

        original_available_actions = env.available_actions

        def filtered_available_actions():
            return [
                action
                for action in original_available_actions()
                if not (
                    action.type == ActionType.ASK_SYMPTOM
                    and action.target == hidden_symptom
                )
            ]

        env.available_actions = filtered_available_actions

    correct = 0
    diagnosed = 0

    for patient_index in test_indices:
        observation = env.reset(patient_index)
        done = False
        predicted = ""
        steps = 0

        while not done and steps < MAX_STEPS:
            state = encode_state_for_rl(observation)
            actions = env.available_actions()

            if not actions:
                break

            action = agent.choose_action(
                state,
                actions,
                training=False,
            )

            observation, reward, done, info = env.step(action)

            if "episode" in info:
                predicted = info["episode"].predicted

            steps += 1

        true_label = env.dataset.get(
            patient_index
        ).true_label()

        if predicted:
            diagnosed += 1

        if predicted and predicted == true_label:
            correct += 1

    total = len(test_indices)

    return {
        "total": total,
        "correct": correct,
        "diagnosed": diagnosed,
        "accuracy": correct / total if total else 0.0,
    }


if __name__ == "__main__":
    # Show valid symptom names to avoid spelling mistakes.
    temp_env = DiagnosticEnvironment()

    print("\nAvailable symptoms:")
    for symptom in temp_env.symptoms:
        print(f" - {symptom}")

    hidden_symptom = input(
        "\nEnter the exact symptom to make unavailable: "
    ).strip()

    print("\nRunning baseline evaluation...")
    baseline = evaluate()

    print(f"\nMaking symptom unavailable: {hidden_symptom}")
    missing = evaluate(hidden_symptom)

    print("\n" + "=" * 55)
    print("MISSING-SYMPTOM ROBUSTNESS RESULTS")
    print("=" * 55)

    print(f"Test patients: {baseline['total']}")
    print(
        f"Baseline accuracy: "
        f"{baseline['accuracy'] * 100:.1f}% "
        f"({baseline['correct']}/{baseline['total']})"
    )
    print(
        f"Baseline diagnosed: "
        f"{baseline['diagnosed']}/{baseline['total']}"
    )

    print(
        f"\nAccuracy with '{hidden_symptom}' unavailable: "
        f"{missing['accuracy'] * 100:.1f}% "
        f"({missing['correct']}/{missing['total']})"
    )
    print(
        f"Diagnosed with symptom unavailable: "
        f"{missing['diagnosed']}/{missing['total']}"
    )

    change = (
        missing["accuracy"] - baseline["accuracy"]
    ) * 100

    print(f"\nAccuracy change: {change:+.1f} percentage points")
    print("=" * 55)
