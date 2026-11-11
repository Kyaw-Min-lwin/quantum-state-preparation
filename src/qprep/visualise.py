# Circuit drawings, trace tables and plots for Demo 1

import json
from collections import deque
from statistics import mean
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter
from qprep.actions import get_action
from qprep.circuits import rebuild_circuit
from qprep.targets import REFERENCE_ACTIONS


def draw_circuit(trace):
    # Draw the gates recorded in an evaluation trace, in execution order
    action_ids = [step["action_id"] for step in trace["steps"]]
    return rebuild_circuit(action_ids).draw(output="mpl", fold=-1)


def step_table(trace):
    # Format initial fidelity and every recorded gate as a Markdown table
    lines = [
        "| Gates used | Operation | Fidelity | Reward |",
        "| ---: | :--- | ---: | ---: |",
        f"| 0 | Start | {trace['initial_fidelity']:.6f} | — |",
    ]
    for step in trace["steps"]:
        action = get_action(step["action_id"])
        if action.gate == "cx":
            control, target = action.qubits
            label = f"CX(q{control} → q{target})"
        else:
            label = f"{action.gate.upper()}(q{action.qubits[0]})"

        lines.append(
            f"| {step['gate_count']} | {label} | {step['fidelity']:.6f} "
            f"| {step['reward']:.2f} |"
        )

    return "\n".join(lines)


def plot_learning(experiment_dir, models, window=100):
    # Plot rolling training success for the log beside each final checkpoint
    fig, ax = plt.subplots(figsize=(9, 3.5), layout="constrained")

    for model in models:
        checkpoint_path = experiment_dir / model["checkpoint"]
        log_path = checkpoint_path.parent / "training.jsonl"
        if not log_path.is_file():
            continue

        outcomes = deque(maxlen=window)
        steps = []
        rates = []
        first_step = None

        with log_path.open(encoding="utf-8") as file:
            for line in file:
                record = json.loads(line)
                if first_step is None:
                    first_step = record["environment_steps"]

                if record["end_reason"] is None:
                    continue

                outcomes.append(record["end_reason"] == "success")
                if len(outcomes) == window:
                    steps.append(record["environment_steps"])
                    rates.append(mean(outcomes))

        if steps:
            label = f"Seed {model['seed']}"
            if first_step > 1:
                label += f" (log starts at step {first_step:,})"
            ax.plot(steps, rates, label=label, linewidth=1.4)

    ax.set(
        title=f"Training success over the last {window} completed episodes",
        xlabel="Environment steps",
        ylabel="Training success rate",
        ylim=(0, 1.02),
    )
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="y", alpha=0.25)
    if ax.lines:
        ax.legend()
    else:
        ax.text(
            0.5,
            0.5,
            "No training logs with enough completed episodes",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    return fig


def plot_comparison(results):
    # Compare task success and show gate counts for successful DQN attempts
    fig, (success_ax, gate_ax) = plt.subplots(
        1, 2, figsize=(11, 3.8), layout="constrained"
    )

    policies = [results["dqn"], results["random"]]
    bars = success_ax.bar(
        ["DQN", "Random"],
        [policy["success_rate"] for policy in policies],
        color=["#216b76", "#9a6049"],
    )
    for bar, policy in zip(bars, policies):
        success_ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.025,
            f"{policy['successes']}/{policy['attempts']}",
            ha="center",
        )

    success_ax.set(title="Success on Demo 1 targets", ylim=(0, 1.14))
    success_ax.set_yticks(np.linspace(0, 1, 6))
    success_ax.yaxis.set_major_formatter(PercentFormatter(1.0))

    targets = list(REFERENCE_ACTIONS)
    learned_counts = []
    for target in targets:
        counts = [
            row["successful_gate_count"]
            for row in results["attempts"]
            if row["target_id"] == target and row["success"]
        ]
        learned_counts.append(mean(counts) if counts else np.nan)

    x = np.arange(len(targets))
    gate_ax.bar(
        x - 0.18,
        learned_counts,
        0.36,
        label="DQN mean on success",
        color="#216b76",
    )
    gate_ax.bar(
        x + 0.18,
        [len(REFERENCE_ACTIONS[target]) for target in targets],
        0.36,
        label="Known reference",
        color="#ac8746",
    )
    gate_ax.set(
        title="Gates used to reach each target", xlabel="Target", ylabel="Gates"
    )
    gate_ax.set_xticks(x, targets)
    gate_ax.legend()
    for ax in (success_ax, gate_ax):
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)

    return fig
