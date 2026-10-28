from pathlib import Path
import random
from collections import Counter

from env.diagnostic_env import (
    DiagnosticEnvironment,
    encode_state_for_rl,
)
from rl.agent import LinearQLearningAgent

MODEL_PATH = Path(
    "models/diagnostic_linear_q_warmstart_experiment.json"
)
MAX_STEPS = 30
DEBUG_LABELS = {"Hepatitis D", "Hepatitis E"}

random.seed(42)

env = DiagnosticEnvironment()
agent = LinearQLearningAgent.load(MODEL_PATH)
agent.epsilon = 0.0

test_indices = [
    i for i, patient in enumerate(env.dataset.patients)
    if patient.split == "test"
]

totals = Counter()
print("Model:", MODEL_PATH)
print("Test cases:", len(test_indices))

for patient_index in test_indices:
    observation = env.reset(patient_index)
    true_label = env.dataset.get(patient_index).true_label()
    done = False
    steps = 0
    episode_result = None
    action_counts = Counter()

    if true_label in DEBUG_LABELS:
        print(f"\n{'=' * 65}\nPatient: {true_label}")

    while not done and steps < MAX_STEPS:
        state = encode_state_for_rl(observation)
        actions = env.available_actions()

        if not actions:
            break

        features = agent.state_features(state)
        scored = [
            (agent._q_from_features(features, action), action)
            for action in actions
        ]

        best_by_type = {}
        for score, action in scored:
            kind = action.type.value
            if kind not in best_by_type or score > best_by_type[kind][0]:
                best_by_type[kind] = (score, action)

        selected = agent.choose_action(
            state, actions, training=False
        )

        selected_score = agent._q_from_features(features, selected)
        action_counts[selected.type.value] += 1

        if true_label in DEBUG_LABELS:
            print(f"\nStep {steps + 1}: selected "
                  f"{selected.type.value} -> {selected.target} "
                  f"(Q={selected_score:.3f})")
            for kind in ("ask_symptom", "order_test", "diagnose"):
                if kind in best_by_type:
                    score, action = best_by_type[kind]
                    print(f"  Best {kind:12s}: {action.target} "
                          f"(Q={score:.3f})")

        observation, reward, done, info = env.step(selected)
        steps += 1

        if "episode" in info:
            episode_result = info["episode"]

    totals.update(action_counts)

    if true_label in DEBUG_LABELS:
        if episode_result is None:
            print("\nResult: no completed diagnosis (timeout or stopped)")
        else:
            print(
                f"\nResult: predicted={episode_result.predicted}, "
                f"correct={episode_result.correct}, "
                f"steps={episode_result.steps}, "
                f"tests={episode_result.tests_ordered}"
            )

print("\n" + "=" * 65)
print("ACTION TOTALS ACROSS TEST PATIENTS")
for kind, count in sorted(totals.items()):
    print(f"{kind}: {count}")
