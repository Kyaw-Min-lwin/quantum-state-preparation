from pathlib import Path
import pytest
from qprep.config import load_config
from qprep.environment import QuantumEnv

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "demo1.json"


def test_reset_starts_with_full_budget_and_bell_target():
    env = QuantumEnv(load_config(CONFIG_PATH))

    observation, info = env.reset(options={"target_id": "E"})

    assert env.action_space.n == 8
    assert observation.shape == (17,)
    assert env.observation_space.contains(observation)
    assert observation[-1] == 1.0
    assert info["target_id"] == "E"
    assert info["initial_fidelity"] == pytest.approx(0.5)
    assert info["remaining_gates"] == 8
    assert info["gate_count"] == 0
    assert info["end_reason"] is None
    assert not env.terminated


def test_zero_target_is_already_solved_and_next_reset_clears_it():
    env = QuantumEnv(load_config(CONFIG_PATH))
    observation, info = env.reset(options={"target_id": "zero"})

    assert env.observation_space.contains(observation)
    assert info["initial_fidelity"] == 1.0
    assert info["gate_count"] == 0
    assert info["remaining_gates"] == 8
    assert info["end_reason"] == "already_solved"
    assert env.terminated

    _, info = env.reset(options={"target_id": "E"})

    assert not env.terminated
    assert info["end_reason"] is None


def test_seeded_resets_reproduce_target_sequence():
    config = load_config(CONFIG_PATH)
    first = QuantumEnv(config)
    second = QuantumEnv(config)

    _, first_info = first.reset(seed=42)
    _, second_info = second.reset(seed=42)
    assert first_info["target_id"] == second_info["target_id"]

    for _ in range(10):
        _, first_info = first.reset()
        _, second_info = second.reset()
        assert first_info["target_id"] == second_info["target_id"]


def test_bell_target_succeeds_in_two_gates():
    env = QuantumEnv(load_config(CONFIG_PATH))
    env.reset(options={"target_id": "E"})

    obs, reward, terminated, truncated, info = env.step(0)  # H(q0)

    assert env.observation_space.contains(obs)
    assert obs[-1] == 7 / 8
    assert reward == pytest.approx(-0.05)
    assert not terminated
    assert not truncated
    assert info["fidelity"] == pytest.approx(0.25)
    assert info["gate_count"] == 1

    obs, reward, terminated, truncated, info = env.step(6)  # CX(q0, q1)

    assert obs[-1] == 6 / 8
    assert reward == pytest.approx(9.95)
    assert terminated
    assert not truncated
    assert info["fidelity"] == pytest.approx(1.0)
    assert info["end_reason"] == "success"
    assert env.action_ids == [0, 6]


@pytest.mark.parametrize(
    "last_action, expected_reason, expected_reward",
    [
        (0, "success", 9.95),  # H(q0) reaches target A.
        (2, "budget_exhausted", -0.05),  # S(q0) leaves |00> unchanged.
    ],
)
def test_eighth_gate_ends_episode(last_action, expected_reason, expected_reward):
    env = QuantumEnv(load_config(CONFIG_PATH))
    env.reset(options={"target_id": "A"})

    for _ in range(7):
        _, _, terminated, truncated, _ = env.step(2)  # S(q0)
        assert not terminated
        assert not truncated

    obs, reward, terminated, truncated, info = env.step(last_action)

    assert obs[-1] == 0.0
    assert reward == pytest.approx(expected_reward)
    assert terminated
    assert not truncated
    assert info["gate_count"] == 8
    assert info["end_reason"] == expected_reason

    with pytest.raises(RuntimeError, match="complete"):
        env.step(0)


def test_invalid_action_leaves_budget_and_log_unchanged():
    env = QuantumEnv(load_config(CONFIG_PATH))
    env.reset(options={"target_id": "E"})

    with pytest.raises(ValueError):
        env.step(-1)

    assert env.remaining_gates == 8
    assert env.action_ids == []


def test_step_requires_an_active_episode():
    env = QuantumEnv(load_config(CONFIG_PATH))
    with pytest.raises(RuntimeError, match="reset"):
        env.step(0)
    env.reset(options={"target_id": "zero"})
    with pytest.raises(RuntimeError, match="complete"):
        env.step(0)
    assert env.action_ids == []
