
from env.diagnostic_env import DiagnosticEnvironment, encode_state_for_rl
from rl.agent import LinearQLearningAgent

MODEL_PATH = "models/diagnostic_linear_q.json"

agent = LinearQLearningAgent.load(MODEL_PATH)
agent.epsilon = 0.0
env = DiagnosticEnvironment()

interesting_tests = [
    "yellowish_skin",
    "dark_urine",
    "yellowing_of_eyes",
    "high_fever",
    "acute_liver_failure",
    "coma",
    "stomach_bleeding",
]

for index, patient in enumerate(env.dataset.patients):
    if (
        patient.split == "test"
        and patient.true_label() in ("Hepatitis D", "Hepatitis E")
    ):
        observation = env.reset(index)
        state = encode_state_for_rl(observation)
        features = agent.state_features(state)

        print("\n" + "=" * 55)
        print("Disease:", patient.true_label())

        actions = env.available_actions()

        for test_name in interesting_tests:
            action = next(
                (
                    a for a in actions
                    if getattr(a, "target", None) == test_name
                    and "ORDER_TEST" in str(getattr(a, "type", "")).upper()
                ),
                None,
            )

            if action is not None:
                q = agent._q_from_features(features, action)
                actual = patient.test_values.get(test_name, "missing")
                print(
                    f"{test_name:25} "
                    f"actual={actual}  Q={q:.3f}"
                )

        best_symptom = max(
            (
                a for a in actions
                if "ASK_SYMPTOM" in str(getattr(a, "type", "")).upper()
            ),
            key=lambda a: agent._q_from_features(features, a),
            default=None,
        )

        best_test = max(
            (
                a for a in actions
                if "ORDER_TEST" in str(getattr(a, "type", "")).upper()
            ),
            key=lambda a: agent._q_from_features(features, a),
            default=None,
        )

        best_diagnosis = max(
            (
                a for a in actions
                if "DIAGNOSE" in str(getattr(a, "type", "")).upper()
            ),
            key=lambda a: agent._q_from_features(features, a),
            default=None,
        )

        for label, action in [
            ("Best symptom", best_symptom),
            ("Best test", best_test),
            ("Best diagnosis", best_diagnosis),
        ]:
            if action is not None:
                q = agent._q_from_features(features, action)
                print(f"{label:16}: {action} | Q={q:.3f}")
