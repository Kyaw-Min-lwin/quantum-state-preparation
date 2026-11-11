# Quantum State Preparation

My final-year project explores whether reinforcement learning can build a quantum circuit that prepares a target state using few gates. The project uses Qiskit for simulation and PyTorch for the Deep Q-Network (DQN).

## Demo 1

The agent starts at `|00⟩` and has up to eight gates to reach a two-qubit target. It can apply H, S, or T to either qubit, or CX in either direction.

| Policy | Successful attempts | Success rate |
| --- | ---: | ---: |
| DQN | 18/18 | 100.0% |
| Random gate selection | 142/600 | 23.7% |

Each of the three DQN training seeds solved all six training targets. All 18 evaluations matched after reloading the saved models. Successful DQN circuits used six gates on average, and 17 of 18 were longer than the known reference circuits. The next step is to evaluate on unseen targets.

See the [presentation notebook](notebooks/demo1_presentation.ipynb) for the results and circuit examples, or the [result note](docs/demo1_result_note.md) for the experiment details.

## Run locally

Use Python 3.12. From the project root in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
```

In VS Code, select `.venv` as the notebook kernel. To rerun the presentation notebook, point `experiment_dir` at the folder containing `selected_checkpoints.json`, `demo1_results.json`, and the checkpoint files. Generated run files are kept locally and ignored by Git.
