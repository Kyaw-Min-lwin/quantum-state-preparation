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
