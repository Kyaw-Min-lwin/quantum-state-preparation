from numbers import Integral
import numpy as np
from qprep.contracts import ComplexVector
from qiskit.circuit.library import CXGate, HGate, SGate, TGate
from qiskit.quantum_info import Statevector
from qprep.actions import get_action

# Absolute tolerance on sum(abs(amplitude) ** 2)
STATE_NORM_ATOL = 1e-10


def _validate_qubit_count(n: int) -> int:
    """
    Validate an n-qubit statevector.
    Return an independent complex128 array without normalising it
    """
    if isinstance(n, bool) or not isinstance(n, Integral):
        raise TypeError("n must be an integer, excluding bool.")

    n = int(n)

    if n < 1:
        raise ValueError("n must be at least 1.")

    return n


def validate_state(psi, n: int) -> ComplexVector:
    n = _validate_qubit_count(n)
    # Inspect the original array before converting its dtype.
    try:
        raw = np.asarray(psi)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "psi must be a rectangular one-dimensional numeric vector."
        ) from error

    expected_shape = (2**n,)

    if raw.shape != expected_shape:
        raise ValueError(
            f"Expected state shape {expected_shape}; received {raw.shape}."
        )

    # Reject strings, booleans, and object arrays.
    if raw.dtype.kind not in "iufc":
        raise TypeError("State amplitudes must be numeric values.")

    with np.errstate(over="ignore", invalid="ignore"):
        state = np.array(raw, dtype=np.complex128, copy=True)

    if not np.isfinite(state).all():
        raise ValueError("State amplitudes must be finite.")

    # <psi|psi> = sum(conj(psi_i) * psi_i).
    with np.errstate(over="ignore", invalid="ignore"):
        norm_squared = float(np.vdot(state, state).real)

    if not np.isfinite(norm_squared):
        raise ValueError("State squared norm must be finite.")

    if abs(norm_squared - 1.0) > STATE_NORM_ATOL:
        raise ValueError(
            "State must be normalised: "
            f"squared norm is {norm_squared:.16g}; "
            f"allowed absolute error is {STATE_NORM_ATOL}."
        )
    return state


def zero_state(n: int) -> ComplexVector:
    # Create the all-zero computational basis state for n qubits
    n = _validate_qubit_count(n)
    state = np.zeros(2**n, dtype=np.complex128)
    state[0] = 1.0
    return state


_DEMO1_N_QUBITS = 2
_QISKIT_GATES = {
    "h": HGate(),
    "s": SGate(),
    "t": TGate(),
    "cx": CXGate(),
}


def apply_gate(psi, action_id: int) -> ComplexVector:
    # Apply one Demo 1 action to a two-qubit state
    action = get_action(action_id)
    state = validate_state(psi, n=_DEMO1_N_QUBITS)

    evolved = Statevector(state).evolve(
        _QISKIT_GATES[action.gate],
        qargs=list(action.qubits),
    )
    return evolved.data
