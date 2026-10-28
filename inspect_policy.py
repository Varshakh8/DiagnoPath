
import random

from env.diagnostic_env import DiagnosticEnvironment, encode_state_for_rl
from rl.agent import LinearQLearningAgent

MODEL_PATH = "models/diagnostic_linear_q.json"
MAX_STEPS = 30


def main():
    random.seed(42)

    env = DiagnosticEnvironment()
    agent = LinearQLearningAgent.load(MODEL_PATH)
    agent.epsilon = 0.0

    patient_indices = [
        i for i, patient in enumerate(env.dataset.patients)
        if patient.split == "test"
        and patient.true_label() in ("Hepatitis D", "Hepatitis E")
    ]

    for patient_index in patient_indices:
        observation = env.reset(patient_index)
        true_label = env.dataset.get(patient_index).true_label()

        print("\n" + "=" * 70)
        print(f"Patient: {true_label}")
        print("=" * 70)

        done = False
        step = 0

        while not done and step < MAX_STEPS:
            state = encode_state_for_rl(observation)
            actions = env.available_actions()
            features = agent.state_features(state)

            scored = []
            for action in actions:
                q = agent._q_from_features(features, action)
                scored.append((q, action.type.value, action.target))

            scored.sort(reverse=True)

            if step in (0, 2, 5, 9, 14, 19, 24, 28):
                print(f"\nStep {step + 1}")
                for action_type in ("ask_symptom", "order_test", "diagnose"):
                    group = [
                        item for item in scored
                        if item[1] == action_type
                    ]
                    print(f"  Best {action_type} actions:")
                    for q, _, target in group[:3]:
                        print(f"    {target:40s} Q={q:8.3f}")

            action = agent.choose_action(state, actions, training=False)
            observation, reward, done, info = env.step(action)

            if action.type.value != "ask_symptom":
                print(
                    f"Selected at step {step + 1}: "
                    f"{action.type.value} {action.target}, "
                    f"reward={reward:.2f}"
                )

            step += 1

        if "episode" in info:
            print("Prediction:", info["episode"].predicted)
        else:
            print("No diagnosis before episode ended.")


if __name__ == "__main__":
    main()
