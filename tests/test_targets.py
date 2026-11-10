from itertools import combinations
import numpy as np
from qprep.quantum import apply_gate, fidelity, validate_state, zero_state
from qprep.targets import build_target_catalogue


def test_targets_are_valid_distinct_and_reachable():
    targets = build_target_catalogue()

    assert list(targets) == ["A", "B", "C", "D", "E", "F"]

    for target in targets.values():
        validate_state(target.state, n=2)
        assert 1 <= len(target.reference_action_ids) <= 8

        replayed = zero_state(2)
        for action_id in target.reference_action_ids:
            replayed = apply_gate(replayed, action_id)

        assert fidelity(replayed, target.state) >= 1 - 1e-12

    for first, second in combinations(targets.values(), 2):
        assert fidelity(first.state, second.state) < 0.999999


def test_bell_targets_match_the_expected_amplitudes():
    targets = build_target_catalogue()

    np.testing.assert_allclose(
        targets["E"].state,
        np.array([1, 0, 0, 1]) / np.sqrt(2),
        atol=1e-12,
        rtol=0,
    )
    np.testing.assert_allclose(
        targets["F"].state,
        np.array([1, 0, 0, 1j]) / np.sqrt(2),
        atol=1e-12,
        rtol=0,
    )
