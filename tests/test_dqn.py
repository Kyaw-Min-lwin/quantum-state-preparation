from pathlib import Path
import numpy as np
import pytest
import torch
from qprep.config import load_config
from qprep.contracts import Transition
from qprep.dqn import QNetwork, ReplayBuffer


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