import numpy as np
import pytest
from qprep.quantum import STATE_NORM_ATOL, validate_state, apply_gate, zero_state


@pytest.mark.parametrize(
    "psi, n",
    [
        ([1, 0], 1),
        ([1, 0, 0, 0], 2),
        ([2**-0.5, 0, 0, 2**-0.5], 2),
        ([2**-0.5, 0, 0, 1j * 2**-0.5], 2),
        ([0, 0, 0, 1j], 2),
    ],
)
def test_valid_states_are_accepted(psi, n):
    state = validate_state(psi, n)

    assert state.shape == (2**n,)
    assert state.dtype == np.complex128
    np.testing.assert_array_equal(state, np.asarray(psi, dtype=np.complex128))


def test_returns_independent_copy():
    original = np.array([1, 0, 0, 0], dtype=np.complex128)

    state = validate_state(original, 2)
    state[0] = 0

    assert original[0] == 1
    assert not np.shares_memory(state, original)


@pytest.mark.parametrize(
    "psi",
    [
        [1, 0],
        [1, 0, 0],
        [[1, 0, 0, 0]],
        [[1], [0], [0], [0]],
        1,
    ],
)
def test_wrong_shapes_are_rejected(psi):
    with pytest.raises(ValueError, match="shape"):
        validate_state(psi, 2)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf, complex(0, np.inf)])
def test_nonfinite_amplitudes_are_rejected(bad_value):
    with pytest.raises(ValueError, match="finite"):
        validate_state([bad_value, 0, 0, 0], 2)


@pytest.mark.parametrize(
    "psi",
    [
        [0, 0, 0, 0],
        [1, 0, 0, 1],
        [0.5, 0, 0, 0],
    ],
)
def test_non_normalised_states_are_rejected(psi):
    original = np.array(psi, dtype=np.complex128)
    before = original.copy()

    with pytest.raises(ValueError, match="normalised"):
        validate_state(original, 2)

    np.testing.assert_array_equal(original, before)


def test_small_norm_error_is_accepted_without_normalising():
    psi = np.array(
        [np.sqrt(1.0 + STATE_NORM_ATOL / 4), 0, 0, 0],
        dtype=np.complex128,
    )

    state = validate_state(psi, 2)

    np.testing.assert_array_equal(state, psi)
    assert state[0] != 1.0


def test_norm_error_above_tolerance_is_rejected():
    psi = [np.sqrt(1.0 + 10 * STATE_NORM_ATOL), 0, 0, 0]

    with pytest.raises(ValueError, match="normalised"):
        validate_state(psi, 2)


@pytest.mark.parametrize(
    "psi",
    [
        ["1", "0", "0", "0"],
        [True, False, False, False],
        np.array([1, 0, 0, 0], dtype=object),
    ],
)
def test_non_numeric_array_types_are_rejected(psi):
    with pytest.raises(TypeError, match="numeric"):
        validate_state(psi, 2)


@pytest.mark.parametrize("n", [True, 2.0, "2", None])
def test_invalid_qubit_count_types_are_rejected(n):
    with pytest.raises(TypeError, match="integer"):
        validate_state([1, 0, 0, 0], n)


@pytest.mark.parametrize("n", [0, -1])
def test_nonpositive_qubit_counts_are_rejected(n):
    with pytest.raises(ValueError, match="at least 1"):
        validate_state([1, 0, 0, 0], n)


def test_numpy_integer_qubit_count_is_accepted():
    state = validate_state([1, 0, 0, 0], np.int64(2))

    assert state.shape == (4,)


def test_finite_amplitudes_with_overflowing_norm_are_rejected():
    with pytest.raises(ValueError, match="finite"):
        validate_state([1e308, 0, 0, 0], 2)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_zero_state_is_the_first_basis_state(n):
    state = zero_state(n)
    assert state.dtype == np.complex128
    assert state.shape == (2**n,)
    assert state[0] == 1
    assert np.count_nonzero(state) == 1


@pytest.mark.parametrize(
    "action_id, expected",
    [
        (0, [1, 1, 0, 0]),
        (1, [1, 0, 1, 0]),
    ],
)
def test_h_placement_and_inverse(action_id, expected):
    start = zero_state(2)
    evolved = apply_gate(start, action_id)

    np.testing.assert_allclose(
        evolved, np.array(expected) / np.sqrt(2), atol=1e-12, rtol=0
    )
    np.testing.assert_allclose(
        apply_gate(evolved, action_id), start, atol=1e-12, rtol=0
    )


@pytest.mark.parametrize(
    "s_id, t_id, expected_s, expected_z",
    [
        (2, 4, [1, 1j, 1, 1j], [1, -1, 1, -1]),
        (3, 5, [1, 1, 1j, 1j], [1, 1, -1, -1]),
    ],
)
def test_phase_gates(s_id, t_id, expected_s, expected_z):
    start = np.full(4, 0.5, dtype=np.complex128)
    after_s = apply_gate(start, s_id)

    np.testing.assert_allclose(after_s, np.array(expected_s) / 2, atol=1e-12, rtol=0)
    np.testing.assert_allclose(
        apply_gate(after_s, s_id),
        np.array(expected_z) / 2,
        atol=1e-12,
        rtol=0,
    )
    np.testing.assert_allclose(
        apply_gate(apply_gate(start, t_id), t_id),
        after_s,
        atol=1e-12,
        rtol=0,
    )


@pytest.mark.parametrize(
    "action_id, destinations",
    [
        (6, [0, 3, 2, 1]),
        (7, [0, 1, 3, 2]),
    ],
)
def test_cx_direction_on_basis_states(action_id, destinations):
    basis = np.eye(4, dtype=np.complex128)

    for source, destination in enumerate(destinations):
        evolved = apply_gate(basis[source], action_id)
        np.testing.assert_array_equal(evolved, basis[destination])
        np.testing.assert_array_equal(apply_gate(evolved, action_id), basis[source])


def test_bell_preparation_preserves_input_and_norm():
    start = zero_state(2)
    bell = apply_gate(apply_gate(start, 0), 6)

    np.testing.assert_allclose(
        bell, np.array([1, 0, 0, 1]) / np.sqrt(2), atol=1e-12, rtol=0
    )
    np.testing.assert_array_equal(start, [1, 0, 0, 0])
    assert bell.dtype == np.complex128
    assert np.vdot(bell, bell).real == pytest.approx(1.0, abs=1e-12, rel=0)


@pytest.mark.parametrize("action_id", [-1, 8])
def test_apply_gate_rejects_out_of_range_ids(action_id):
    with pytest.raises(ValueError, match="between"):
        apply_gate(zero_state(2), action_id)
