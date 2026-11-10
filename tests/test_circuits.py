import numpy as np
import pytest
from qiskit.quantum_info import Statevector
from qprep.circuits import rebuild_circuit


@pytest.mark.parametrize("actions", [(0, 6), (1, 7)])
def test_bell_circuit_rebuilds_in_both_directions(actions):
    circuit = rebuild_circuit(actions)
    state = Statevector.from_instruction(circuit).data

    assert circuit.size() == 2
    np.testing.assert_allclose(
        state, np.array([1, 0, 0, 1]) / np.sqrt(2), atol=1e-12, rtol=0
    )


def test_empty_circuit_leaves_zero_state():
    circuit = rebuild_circuit(())
    state = Statevector.from_instruction(circuit).data

    assert circuit.size() == 0
    np.testing.assert_array_equal(state, [1, 0, 0, 0])
