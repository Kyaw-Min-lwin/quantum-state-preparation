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
    compute_td_target,
    optimise_batch,
)
from qprep.checkpoint import checkpoint_state, load_checkpoint
from qprep.runtime import prepare_run

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


def test_td_target_uses_future_value_only_for_continuing_transitions():
    config = load_config(CONFIG_PATH)
    online = QNetwork(config)
    target = build_target_network(online)

    # Give every next observation a known maximum Q-value of 4.
    with torch.no_grad():
        target.layers[-1].weight.zero_()
        target.layers[-1].bias.zero_()
        target.layers[-1].bias[6] = 4.0

    batch = {
        "next_observations": np.zeros((3, 17), dtype=np.float32),
        "rewards": np.array([-0.05, -0.05, 9.95], dtype=np.float32),
        "terminated": np.array([False, True, True]),
    }

    targets = compute_td_target(target, batch, config["gamma"])

    assert targets[0].item() == pytest.approx(3.91)
    assert torch.equal(targets[1:], torch.from_numpy(batch["rewards"][1:]))
    assert not targets.requires_grad


def test_optimise_batch_updates_chosen_actions_and_keeps_target_fixed():
    config = load_config(CONFIG_PATH)
    online = QNetwork(config)

    with torch.no_grad():
        for parameter in online.parameters():
            parameter.zero_()

    target = build_target_network(online)
    optimizer = torch.optim.Adam(online.parameters(), lr=config["learning_rate"])
    batch = {
        "observations": np.zeros((2, 17), dtype=np.float32),
        "actions": np.array([0, 6], dtype=np.int64),
        "rewards": np.array([-0.05, 9.95], dtype=np.float32),
        "next_observations": np.zeros((2, 17), dtype=np.float32),
        "terminated": np.array([True, True]),
    }

    loss = optimise_batch(online, target, optimizer, batch, config["gamma"])

    assert np.isfinite(loss)
    assert loss > 0
    for parameter in online.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()

    observation = torch.zeros(17)
    with torch.no_grad():
        online_values = online(observation)
        target_values = target(observation)

    assert online_values[0] < 0
    assert online_values[6] > 0
    assert torch.count_nonzero(online_values) == 2
    torch.testing.assert_close(target_values, torch.zeros(8))


def test_checkpoint_restores_predictions_and_next_training_update(tmp_path):
    config = load_config(CONFIG_PATH)
    run = prepare_run(
        config,
        seed=config["seeds"][0],
        project_root=CONFIG_PATH.parents[1],
        output_root=tmp_path,
    )

    online = QNetwork(config)
    target = build_target_network(online)
    optimizer = torch.optim.Adam(online.parameters(), lr=config["learning_rate"])
    replay = ReplayBuffer(3, run.rngs["replay"])

    for action_id in range(5):
        replay.add(make_transition(action_id))

    # Give Adam some history and let the online weights diverge from the target
    optimise_batch(online, target, optimizer, replay.sample(2), config["gamma"])

    progress = {
        "environment_steps": 5,
        "gradient_steps": 1,
        "episodes_completed": 0,
        "epsilon": 0.8,
    }
    state = checkpoint_state(run, online, target, optimizer, replay, progress)
    path = run.run_dir / "checkpoint.pt"
    torch.save(state, path)

    observation = np.zeros(17, dtype=np.float32)
    observation_tensor = torch.from_numpy(observation)
    with torch.no_grad():
        expected_q = online(observation_tensor)
        expected_target_q = target(observation_tensor)

    expected_action = select_action(
        online, observation, epsilon=0, rng=run.rngs["exploration"]
    )

    # Continue the original run for one more update.
    replay.add(make_transition(6))
    expected_batch = replay.sample(3)
    expected_loss = optimise_batch(
        online, target, optimizer, expected_batch, config["gamma"]
    )

    restored_online = QNetwork(config)
    restored_target = build_target_network(restored_online)
    restored_optimizer = torch.optim.Adam(
        restored_online.parameters(), lr=config["learning_rate"]
    )
    restored_replay = ReplayBuffer(3, run.rngs["replay"])

    loaded = load_checkpoint(
        path,
        run,
        restored_online,
        restored_target,
        restored_optimizer,
        restored_replay,
    )

    for name, value in progress.items():
        assert loaded[name] == value

    with torch.no_grad():
        torch.testing.assert_close(restored_online(observation_tensor), expected_q)
        torch.testing.assert_close(
            restored_target(observation_tensor), expected_target_q
        )

    assert (
        select_action(
            restored_online, observation, epsilon=0, rng=run.rngs["exploration"]
        )
        == expected_action
    )

    restored_replay.add(make_transition(6))
    actual_batch = restored_replay.sample(3)
    np.testing.assert_array_equal(actual_batch["actions"], expected_batch["actions"])

    actual_loss = optimise_batch(
        restored_online,
        restored_target,
        restored_optimizer,
        actual_batch,
        config["gamma"],
    )
    assert actual_loss == pytest.approx(expected_loss)

    with torch.no_grad():
        torch.testing.assert_close(
            restored_online(observation_tensor), online(observation_tensor)
        )

    # A checkpoint with reordered action IDs must be rejected
    state["action_catalogue"].reverse()
    bad_path = run.run_dir / "incompatible.pt"
    torch.save(state, bad_path)

    with pytest.raises(ValueError, match="action catalogue"):
        load_checkpoint(
            bad_path,
            run,
            restored_online,
            restored_target,
            restored_optimizer,
            restored_replay,
        )
