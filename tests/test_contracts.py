from pathlib import Path

import numpy as np
import pytest

from qprep.config import load_config
from qprep.contracts import (
    ACTION_SPECS,
    EpisodeTrace,
    TraceStep,
    Transition,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "demo1.json"


def test_action_specs_match_configuration():
    config = load_config(CONFIG_PATH)

    labels = [
        "_".join([gate, *(str(qubit) for qubit in qubits)])
        for gate, qubits in ACTION_SPECS
    ]

    assert labels == config["action_labels"]
    assert len(set(ACTION_SPECS)) == 8
    assert ACTION_SPECS[6] == ("cx", (0, 1))
    assert ACTION_SPECS[7] == ("cx", (1, 0))


def test_transition_keeps_independent_observation_snapshots():
    observation = np.zeros(17, dtype=np.float32)
    next_observation = np.ones(17, dtype=np.float32)

    transition = Transition(
        observation=observation,
        action_id=0,
        reward=-0.05,
        next_observation=next_observation,
        terminated=False,
        truncated=False,
    )

    observation[:] = 7
    next_observation[:] = 9

    np.testing.assert_array_equal(transition.observation, np.zeros(17))
    np.testing.assert_array_equal(transition.next_observation, np.ones(17))

    with pytest.raises(ValueError):
        transition.observation[0] = 5


def test_transition_rejects_mismatched_observations():
    with pytest.raises(ValueError, match="shapes"):
        Transition(
            observation=np.zeros(17),
            action_id=0,
            reward=-0.05,
            next_observation=np.zeros(16),
            terminated=False,
            truncated=False,
        )


def test_already_solved_trace_has_zero_gates():
    trace = EpisodeTrace(
        target_id="zero",
        initial_fidelity=1.0,
        steps=(),
        end_reason="already_solved",
    )

    assert trace.success
    assert trace.gate_count == 0
    assert trace.final_fidelity == 1.0
    assert trace.total_reward == 0.0


def test_budget_failure_is_retained():
    steps = tuple(
        TraceStep(
            gate_count=index + 1,
            action_id=0,
            fidelity=0.5,
            reward=-0.05,
        )
        for index in range(8)
    )

    trace = EpisodeTrace(
        target_id="E",
        initial_fidelity=0.5,
        steps=steps,
        end_reason="budget_exhausted",
    )

    assert not trace.success
    assert trace.gate_count == 8
    assert trace.final_fidelity == 0.5
    assert trace.total_reward == pytest.approx(-0.4)
