
from env.diagnostic_env import DiagnosticEnvironment, encode_state_for_rl
from rl.agent import LinearQLearningAgent

MODEL_PATH = "models/diagnostic_linear_q.json"

agent = LinearQLearningAgent.load(MODEL_PATH)
agent.epsilon = 0.0

env = DiagnosticEnvironment()

for index, patient in enumerate(env.dataset.patients):
    if (
        patient.split == "test"
        and patient.true_label() in ("Hepatitis D", "Hepatitis E")
    ):
        print("\n" + "=" * 55)
        print("Disease:", patient.true_label())

        observation = env.reset(index)

        # Find the test action with the highest Q-value.
        actions = env.available_actions()
        test_actions = [
            action for action in actions
            if "ORDER_TEST" in str(action).upper()
        ]

        print("Available actions:", len(actions))
        print("Test actions found:", len(test_actions))

        if not test_actions:
            print("WARNING: No test actions matched the filter.")
            print("Sample actions:", actions[:10])
            continue

        state = encode_state_for_rl(observation)
        best_test = max(
            test_actions,
            key=lambda action: agent._q_from_features(
                agent.state_features(state), action
            )
        )

        print("Selected test action:", best_test)

        next_observation, reward, done, info = env.step(best_test)

        print("Reward:", reward)
        print("Done:", done)
        print("Step:", info.get("step", "not provided"))
        print("Updated observation:", next_observation)
