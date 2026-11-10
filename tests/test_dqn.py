from pathlib import Path
import numpy as np
import pytest
import torch
from qprep.config import load_config
from qprep.contracts import Transition
from qprep.dqn import (
    QNetwork,
    ReplayBuffer,
    select_action,
    build_target_network,
    sync_target_network,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "demo1.json"


def make_transition(action_id, observation=None):
    if observation is None:
        observation = np.zeros(17, dtype=np.float32)

    return Transition(
        observation=observation,
        action_id=action_id,
        reward=-0.05,
        next_observation=np.ones(17, dtype=np.float32),
        terminated=False,
        truncated=False,
    )


def test_q_network_returns_one_value_per_action():
    network = QNetwork(load_config(CONFIG_PATH))
    q_values = network(torch.zeros(4, 17))
    assert q_values.shape == (4, 8)
    assert torch.isfinite(q_values).all()


def test_replay_batch_has_expected_types_and_independent_observations():
    buffer = ReplayBuffer(capacity=3, rng=np.random.default_rng(42))
    original = np.zeros(17, dtype=np.float32)
    buffer.add(make_transition(6, original))
    original[:] = 9

    batch = buffer.sample(1)

    assert batch["observations"].shape == (1, 17)
    assert batch["next_observations"].shape == (1, 17)
    assert batch["observations"].dtype == np.float32
    assert batch["next_observations"].dtype == np.float32
    assert batch["actions"].dtype == np.int64
    assert batch["rewards"].dtype == np.float32
    assert batch["terminated"].dtype == np.bool_
    assert batch["actions"].tolist() == [6]
    assert np.all(batch["observations"] == 0)

    batch["observations"][:] = 5
    assert np.all(buffer.sample(1)["observations"] == 0)


def test_replay_evicts_oldest_and_samples_reproducibly():
    first = ReplayBuffer(capacity=3, rng=np.random.default_rng(42))
    second = ReplayBuffer(capacity=3, rng=np.random.default_rng(42))

    for action_id in range(5):
        first.add(make_transition(action_id))
        second.add(make_transition(action_id))

    first_actions = first.sample(3)["actions"]
    second_actions = second.sample(3)["actions"]

    assert len(first) == 3
    assert sorted(first_actions) == [2, 3, 4]
    np.testing.assert_array_equal(first_actions, second_actions)


def test_replay_rejects_sampling_before_enough_transitions():
    buffer = ReplayBuffer(capacity=3, rng=np.random.default_rng(42))
    buffer.add(make_transition(0))

    with pytest.raises(ValueError, match="buffer size"):
        buffer.sample(2)


def test_greedy_action_uses_lowest_id_on_a_tie():
    network = QNetwork(load_config(CONFIG_PATH))
    observation = np.zeros(17, dtype=np.float32)
    rng = np.random.default_rng(42)
    untouched_rng = np.random.default_rng(42)

    # For a zero observation, biases set the highest Q-values.
    with torch.no_grad():
        network.layers[-1].weight.zero_()
        network.layers[-1].bias.zero_()
        network.layers[-1].bias[3] = 2.0
        network.layers[-1].bias[6] = 2.0

    assert select_action(network, observation, epsilon=0, rng=rng) == 3
    assert rng.random() == untouched_rng.random()


def test_exploration_is_seeded_and_selects_legal_actions():
    network = QNetwork(load_config(CONFIG_PATH))
    observation = np.zeros(17, dtype=np.float32)
    first_rng = np.random.default_rng(42)
    second_rng = np.random.default_rng(42)

    first = [
        select_action(network, observation, epsilon=1, rng=first_rng) for _ in range(20)
    ]
    second = [
        select_action(network, observation, epsilon=1, rng=second_rng)
        for _ in range(20)
    ]
    assert first == second
    assert all(0 <= action < 8 for action in first)
    assert len(set(first)) > 1


def test_target_network_changes_only_when_synchronised():
    online = QNetwork(load_config(CONFIG_PATH))
    target = build_target_network(online)
    observation = torch.zeros(1, 17)

    with torch.no_grad():
        original_values = target(observation)
        torch.testing.assert_close(original_values, online(observation))

        online.layers[-1].bias.add_(1.0)
        torch.testing.assert_close(target(observation), original_values)
        torch.testing.assert_close(online(observation), original_values + 1.0)

        sync_target_network(online, target)
        torch.testing.assert_close(target(observation), online(observation))

    assert all(not parameter.requires_grad for parameter in target.parameters())
