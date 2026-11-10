from itertools import combinations
import numpy as np
from qprep.quantum import apply_gate, fidelity, validate_state, zero_state
from qprep.targets import build_target_catalogue, sample_target
import pytest

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


def test_same_seed_reproduces_training_target_sequence():
    catalogue = build_target_catalogue()
    first_rng = np.random.default_rng(42)
    second_rng = np.random.default_rng(42)

    first = [sample_target(first_rng, catalogue).target_id for _ in range(20)]
    second = [sample_target(second_rng, catalogue).target_id for _ in range(20)]
    assert first == second
    assert set(first) <= set(catalogue)


def test_explicit_target_selection_preserves_the_random_stream():
    catalogue = build_target_catalogue()
    rng = np.random.default_rng(42)
    untouched_rng = np.random.default_rng(42)
    assert sample_target(rng, catalogue, target_id="E") is catalogue["E"]
    assert rng.random() == untouched_rng.random()


def test_unknown_target_id_is_rejected():
    catalogue = build_target_catalogue()
    rng = np.random.default_rng(42)
    with pytest.raises(ValueError, match="Unknown target ID"):
        sample_target(rng, catalogue, target_id="typo")
