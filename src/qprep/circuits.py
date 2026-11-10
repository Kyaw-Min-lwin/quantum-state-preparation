from qiskit import QuantumCircuit
from qprep.actions import get_action


def rebuild_circuit(action_ids) -> QuantumCircuit:
    # Build a fresh two-qubit circuit from a sequence of action IDs
    circuit = QuantumCircuit(2)

    for action_id in action_ids:
        action = get_action(action_id)

        if action.gate == "cx":
            control, target = action.qubits
            circuit.cx(control, target)
        else:
            qubit = action.qubits[0]

            if action.gate == "h":
                circuit.h(qubit)
            elif action.gate == "s":
                circuit.s(qubit)
            elif action.gate == "t":
                circuit.t(qubit)
            else:
                raise ValueError(f"Unsupported gate: {action.gate}")
    return circuit