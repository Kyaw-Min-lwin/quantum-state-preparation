# Quantum State Preparation

Final-year project investigating goal-conditioned reinforcement learning
for quantum state preparation.

## Demo 1 scope

- Two noiseless qubits, starting from |00>.
- Qiskit basis ordering: |q1 q0>.
- Planned actions: H, S and T on either qubit, and CX in both directions.
- Planned gate budget: eight.
- Planned learner: DQN.
- Demo 1 does not establish generalisation or optimality.

## Current status

A3 project skeleton and installation check.
The quantum environment and learner are not implemented yet.

## Development setup

Use Python 3.12.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m qprep.smoke_check
```

In VS Code, select `.venv` as the Python interpreter and notebook kernel.

## Layout

- src/qprep/: reusable implementation
- tests/: correctness checks
- configs/: experiment configuration
- notebooks/: demonstration entry points
- runs/: generated experiment outputs, ignored by Git

## Dependencies

pyproject.toml declares dependencies.
requirements.txt captures the installed third-party versions from
the working local environment. Colab compatibility will be verified separately.

## Current status

installable package and Qiskit installation done.
validated Demo 1 configuration, derived dimensions,
and protected configuration snapshots.