from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, state_fidelity


def main():
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)

    state = Statevector.from_instruction(circuit)
    target = Statevector([2**-0.5, 0, 0, 2**-0.5])
    fidelity = state_fidelity(state, target)

    print(circuit.draw(output="text"))
    print("Statevector:", state.data)
    print(f"Bell-state fidelity: {fidelity:.12f}")

    if abs(fidelity - 1.0) > 1e-12:
        raise RuntimeError("Bell-state installation check failed.")


if __name__ == "__main__":
    main()