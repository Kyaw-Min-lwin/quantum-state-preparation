import numpy as np
import pytest
from qprep.quantum import STATE_NORM_ATOL, validate_state


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
