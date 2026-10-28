
import random

from env.data_model import ClinicalDataset, META, default_data_path
from env.diagnostic_env import (
    Action,
    ActionType,
    DiagnosticEnvironment,
    encode_state_for_rl,
)
from rl.agent import LinearQLearningAgent


MODEL_PATH = "models/diagnostic_linear_q.json"
SEED = 42


def main():
    dataset = ClinicalDataset.from_csv(default_data_path())
    test_indices = [
        i for i, patient in enumerate(dataset.patients)
        if patient.split == "test"
    ]

    random.seed(SEED)
    agent = LinearQLearningAgent.load(MODEL_PATH)
    agent.epsilon = 0.0

    rng = random.Random(SEED)
    correct = 0
    total = 0
    timed_out = 0
    symptoms_total = 0
    tests_total = 0

    env = DiagnosticEnvironment(dataset)

    for patient_index in test_indices:
        observation = env.reset(patient_index)
        done = False
        info = {}
        steps = 0
        episode_result = None
        while not done and steps < 30:
            state = encode_state_for_rl(observation)
            actions = env.available_actions()

            action = agent.choose_action(
                state=state,
                actions=actions,
                training=False,
            )

            observation, reward, done, info = env.step(action)
            if "episode" in info:
                episode_result = info["episode"]
            steps += 1

        total += 1
        stats = env.episode_stats()
        symptoms_total += stats["symptoms_asked"]
        tests_total += stats["tests_ordered"]

        if episode_result is not None:
            if episode_result.predicted == episode_result.true_label:
                correct += 1
        else:
            timed_out += 1

    print(f"Test cases: {total}")
    print(f"Correct: {correct}/{total} ({100 * correct / max(total, 1):.1f}%)")
    print(f"Timed out: {timed_out}")
    print(f"Average symptoms asked: {symptoms_total / max(total, 1):.2f}")
    print(f"Average tests ordered: {tests_total / max(total, 1):.2f}")


if __name__ == "__main__":
    main()
