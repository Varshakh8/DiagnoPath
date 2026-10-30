
import json

import pytest

from env.diagnostic_env import Action, ActionType
from rl.agent import LinearQLearningAgent


@pytest.fixture
def agent():
    return LinearQLearningAgent(
        state_size=3,
        learning_rate=0.1,
        discount_factor=0.9,
        epsilon=1.0,
        epsilon_min=0.1,
        epsilon_decay=0.5,
        seed=42,
        diagnose_lr_scale=0.1,
    )


@pytest.fixture
def actions():
    return [
        Action(ActionType.ASK_SYMPTOM, "fever"),
        Action(ActionType.ORDER_TEST, "blood_test"),
        Action(ActionType.DIAGNOSE, "disease_a"),
    ]


@pytest.fixture
def state():
    return (-1, 1, 0)


def test_state_features_have_expected_length(agent, state):
    features = agent.state_features(state)

    assert len(features) == 3 * 4 + 4


def test_state_features_contain_bias_and_raw_values(agent, state):
    features = agent.state_features(state)

    assert features[0] == 1.0
    assert features[1:4] == [-1.0, 1.0, 0.0]


def test_state_features_contain_known_indicators(agent, state):
    features = agent.state_features(state)

    assert features[4:7] == [0.0, 1.0, 1.0]


def test_state_features_contain_positive_indicators(agent, state):
    features = agent.state_features(state)

    assert features[7:10] == [0.0, 1.0, 0.0]


def test_state_features_contain_negative_indicators(agent, state):
    features = agent.state_features(state)

    assert features[10:13] == [0.0, 0.0, 1.0]


def test_state_features_contain_normalized_counts(agent, state):
    features = agent.state_features(state)

    assert features[13:] == pytest.approx(
        [2 / 3, 1 / 3, 1 / 3]
    )


def test_state_features_reject_wrong_state_size(agent):
    with pytest.raises(ValueError):
        agent.state_features((1, 0))


def test_q_learning_terminal_update_uses_reward_only(
    agent, state, actions
):
    action = actions[0]
    features = agent.state_features(state)

    agent.update(
        state=state,
        action=action,
        reward=5.0,
        next_state=(0, 0, 0),
        next_actions=actions,
        done=True,
    )

    expected_q = (
        agent.learning_rate
        * 5.0
        * sum(value ** 2 for value in features)
    )

    assert agent.get_q(state, action) == pytest.approx(expected_q)


def test_q_learning_non_terminal_update_uses_future_q_value(
    agent, state, actions
):
    current_action = actions[0]
    next_action = actions[1]
    features = agent.state_features(state)

    next_key = agent.action_key(next_action)
    agent.weights[next_key] = [2.0] + [0.0] * 15

    agent.update(
        state=state,
        action=current_action,
        reward=1.0,
        next_state=state,
        next_actions=[next_action],
        done=False,
    )

    next_q = 2.0
    target = 1.0 + agent.discount_factor * next_q

    expected_q = (
        agent.learning_rate
        * target
        * sum(value ** 2 for value in features)
    )

    assert agent.get_q(state, current_action) == pytest.approx(expected_q)


def test_q_learning_uses_reward_when_no_next_actions(
    agent, state, actions
):
    action = actions[0]
    features = agent.state_features(state)

    agent.update(
        state=state,
        action=action,
        reward=4.0,
        next_state=state,
        next_actions=[],
        done=False,
    )

    expected_q = (
        agent.learning_rate
        * 4.0
        * sum(value ** 2 for value in features)
    )

    assert agent.get_q(state, action) == pytest.approx(expected_q)


def test_diagnose_action_uses_scaled_learning_rate(
    agent, state, actions
):
    action = actions[2]
    features = agent.state_features(state)

    agent.update(
        state=state,
        action=action,
        reward=10.0,
        next_state=state,
        next_actions=[],
        done=True,
    )

    effective_lr = (
        agent.learning_rate * agent.diagnose_lr_scale
    )

    expected_q = (
        effective_lr
        * 10.0
        * sum(value ** 2 for value in features)
    )

    assert agent.get_q(state, action) == pytest.approx(expected_q)


def test_get_q_initially_returns_zero(agent, state, actions):
    assert agent.get_q(state, actions[0]) == 0.0


def test_ensure_action_creates_weight_vector(agent, actions):
    agent._ensure_action(actions[0])

    key = agent.action_key(actions[0])

    assert key in agent.weights
    assert len(agent.weights[key]) == 16
    assert all(value == 0.0 for value in agent.weights[key])


def test_q_value_changes_after_setting_weights(agent, state, actions):
    key = agent.action_key(actions[0])
    agent.weights[key] = [1.0] + [0.0] * 15

    assert agent.get_q(state, actions[0]) == 1.0


def test_choose_action_rejects_empty_action_list(agent, state):
    with pytest.raises(ValueError, match="No available actions"):
        agent.choose_action(state, [])


def test_greedy_selection_chooses_highest_q_value(
    agent, state, actions
):
    for action in actions:
        agent._ensure_action(action)

    agent.weights[agent.action_key(actions[0])][0] = 1.0
    agent.weights[agent.action_key(actions[1])][0] = 3.0
    agent.weights[agent.action_key(actions[2])][0] = 2.0

    selected = agent.choose_action(
        state,
        actions,
        training=False,
    )

    assert selected == actions[1]


def test_training_exploration_selects_available_action(
    agent, state, actions
):
    agent.epsilon = 1.0

    selected = agent.choose_action(
        state,
        actions,
        training=True,
    )

    assert selected in actions


def test_supervised_update_changes_weights(agent, actions):
    features = [1.0] + [0.0] * 15
    action = actions[2]

    agent.supervised_update(
        features,
        action,
        target=5.0,
    )

    assert agent.get_q((0, 0, 0), action) > 0.0


def test_supervised_update_moves_q_value_towards_target(
    agent, actions
):
    features = [1.0] + [0.0] * 15
    action = actions[0]

    agent.supervised_update(features, action, target=5.0)
    first_q = agent._q_from_features(features, action)

    agent.supervised_update(features, action, target=5.0)
    second_q = agent._q_from_features(features, action)

    assert second_q > first_q


def test_perceptron_step_updates_true_and_predicted_actions(
    agent, actions
):
    features = [1.0] + [0.0] * 15

    true_action = actions[0]
    predicted_action = actions[1]

    agent.perceptron_step(
        features,
        true_action,
        predicted_action,
    )

    true_weights = agent.weights[agent.action_key(true_action)]
    predicted_weights = agent.weights[
        agent.action_key(predicted_action)
    ]

    assert true_weights[0] > 0
    assert predicted_weights[0] < 0


def test_perceptron_step_does_nothing_for_correct_prediction(
    agent, actions
):
    action = actions[0]
    features = [1.0] + [0.0] * 15

    agent.perceptron_step(features, action, action)

    assert agent.number_of_actions == 0


def test_q_learning_terminal_update_uses_reward_only(
    agent, state, actions
):
    action = actions[0]

    agent.update(
        state=state,
        action=action,
        reward=5.0,
        next_state=(0, 0, 0),
        next_actions=actions,
        done=True,
    )

    assert agent.get_q(state, action) == pytest.approx(
    0.1 * 5.0 * sum(x ** 2 for x in agent.state_features(state))
)


def test_q_learning_non_terminal_update_uses_future_q_value(
    agent, state, actions
):
    current_action = actions[0]
    next_action = actions[1]

    next_key = agent.action_key(next_action)
    agent.weights[next_key] = [2.0] + [0.0] * 15

    agent.update(
        state=state,
        action=current_action,
        reward=1.0,
        next_state=state,
        next_actions=[next_action],
        done=False,
    )

    # Target = reward + discount * next Q
    #         = 1 + 0.9 * 2 = 2.8
    # Current Q starts at zero; bias update = 0.1 * 2.8
    assert agent.get_q(state, current_action) == pytest.approx(
    0.1 * 2.8 * sum(x ** 2 for x in agent.state_features(state))
)


def test_q_learning_uses_reward_when_no_next_actions(
    agent, state, actions
):
    action = actions[0]

    agent.update(
        state=state,
        action=action,
        reward=4.0,
        next_state=state,
        next_actions=[],
        done=False,
    )

    assert agent.get_q(state, action) == pytest.approx(
    0.1 * 4.0 * sum(x ** 2 for x in agent.state_features(state))
)


def test_diagnose_action_uses_scaled_learning_rate(
    agent, state, actions
):
    action = actions[2]

    agent.update(
        state=state,
        action=action,
        reward=10.0,
        next_state=state,
        next_actions=[],
        done=True,
    )

    # Effective learning rate = 0.1 * 0.1 = 0.01
    assert agent.get_q(state, action) == pytest.approx(
    0.01 * 10.0 * sum(x ** 2 for x in agent.state_features(state))
)


def test_weight_values_are_clipped(agent, state, actions):
    action = actions[0]
    key = agent.action_key(action)

    agent.weights[key] = [100.0] * 16

    agent.update(
        state=state,
        action=action,
        reward=10000.0,
        next_state=state,
        next_actions=[],
        done=True,
    )

    assert all(
        -100.0 <= weight <= 100.0
        for weight in agent.weights[key]
    )


def test_save_and_load_preserves_agent_state(
    agent, state, actions, tmp_path
):
    action = actions[0]

    agent.get_q(state, action)
    agent.epsilon = 0.4

    path = tmp_path / "models" / "agent.json"
    agent.save(path)

    loaded = LinearQLearningAgent.load(path)

    assert loaded.state_size == agent.state_size
    assert loaded.learning_rate == agent.learning_rate
    assert loaded.discount_factor == agent.discount_factor
    assert loaded.epsilon == agent.epsilon
    assert loaded.epsilon_min == agent.epsilon_min
    assert loaded.epsilon_decay == agent.epsilon_decay
    assert loaded.diagnose_lr_scale == agent.diagnose_lr_scale
    assert loaded.weights == agent.weights


def test_saved_model_is_valid_json(agent, actions, tmp_path):
    agent.get_q((0, 0, 0), actions[0])

    path = tmp_path / "agent.json"
    agent.save(path)

    data = json.loads(path.read_text(encoding="utf-8"))

    assert "state_size" in data
    assert "weights" in data


def test_load_supports_missing_diagnose_lr_scale(
    agent, tmp_path
):
    data = {
        "state_size": 3,
        "learning_rate": 0.1,
        "discount_factor": 0.9,
        "epsilon": 1.0,
        "epsilon_min": 0.1,
        "epsilon_decay": 0.5,
        "weights": {},
    }

    path = tmp_path / "old_agent.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    loaded = LinearQLearningAgent.load(path)

    assert loaded.diagnose_lr_scale == 0.1


def test_number_of_actions_and_parameters(agent, actions):
    agent.get_q((0, 0, 0), actions[0])
    agent.get_q((0, 0, 0), actions[1])

    assert agent.number_of_actions == 2
    assert agent.number_of_parameters == 32


def test_decay_epsilon_reduces_exploration(agent):
    agent.epsilon = 0.8

    agent.decay_epsilon()

    assert agent.epsilon == pytest.approx(0.4)


def test_decay_epsilon_does_not_go_below_minimum(agent):
    agent.epsilon = 0.15

    agent.decay_epsilon()

    assert agent.epsilon == 0.1
