import json
import math
from pathlib import Path

# gates the agent can use
EXPECTED_ACTIONS = [
    "h_0", "h_1",
    "s_0", "s_1",
    "t_0", "t_1",
    "cx_0_1", "cx_1_0",
]

REQUIRED_KEYS = {
    "experiment_name",
    "n_qubits",
    "basis_order",
    "action_labels",
    "max_gates",
    "success_fidelity",
    "gate_reward",
    "success_bonus",
    "target_ids",
    "seeds",
    "hidden_sizes",
    "learning_rate",
    "batch_size",
    "replay_capacity",
    "gamma",
    "warmup_steps",
    "target_update_interval",
    "epsilon_start",
    "epsilon_end",
    "epsilon_decay_steps",
    "training_steps",
    "eval_interval",
    "random_attempts_per_target",
}


def require(condition, message):
   # raise error when config fails
    if not condition:
        raise ValueError(message)

# exact type checks
def require_int(value, name, minimum=1):
    require(
        type(value) is int and value >= minimum,
        f"{name} must be an integer >= {minimum}.",
    )


def require_number(value, name):
    require(
        type(value) in (int, float) and math.isfinite(value),
        f"{name} must be a finite number.",
    )


def reject_duplicate_keys(pairs):
    # Prevent JSON from silently keeping the last duplicate key
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate configuration key: {key}")
        result[key] = value
    return result


def load_config(path):
    # Load and validate the fixed Demo 1 configuration.
    path = Path(path)

    with path.open("r", encoding="utf-8-sig") as file:
        config = json.load(file, object_pairs_hook=reject_duplicate_keys)

    require(type(config) is dict, "Configuration must be a JSON object.")

    missing = REQUIRED_KEYS - config.keys()
    unexpected = config.keys() - REQUIRED_KEYS

    require(not missing, f"Missing configuration keys: {sorted(missing)}")
    require(
        not unexpected,
        f"Unexpected configuration keys: {sorted(unexpected)}",
    )

    require(
        isinstance(config["experiment_name"], str)
        and bool(config["experiment_name"].strip()),
        "experiment_name must be a non-empty string.",
    )

    integer_fields = [
        "n_qubits",
        "max_gates",
        "batch_size",
        "replay_capacity",
        "warmup_steps",
        "target_update_interval",
        "epsilon_decay_steps",
        "training_steps",
        "eval_interval",
        "random_attempts_per_target",
    ]
    for name in integer_fields:
        require_int(config[name], name)

    number_fields = [
        "success_fidelity",
        "gate_reward",
        "success_bonus",
        "learning_rate",
        "gamma",
        "epsilon_start",
        "epsilon_end",
    ]
    for name in number_fields:
        require_number(config[name], name)

    
    require(config["n_qubits"] == 2, "Demo 1 requires n_qubits = 2.")
    require(config["basis_order"] == "q1q0", "Demo 1 uses basis_order = q1q0.")
    require(
        config["action_labels"] == EXPECTED_ACTIONS,
        "action_labels must match the eight Demo 1 actions in order.",
    )
    require(config["max_gates"] == 8, "Demo 1 requires max_gates = 8.")
    require(
        config["success_fidelity"] == 0.999999,
        "Demo 1 requires success_fidelity = 0.999999.",
    )
    require(
        config["gate_reward"] == -0.05,
        "Demo 1 requires gate_reward = -0.05.",
    )
    require(
        config["success_bonus"] == 10.0,
        "Demo 1 requires success_bonus = 10.0.",
    )
    require(
        config["target_ids"] == ["A", "B", "C", "D", "E", "F"],
        "Demo 1 requires target_ids A–F in order.",
    )

    seeds = config["seeds"]
    require(
        type(seeds) is list and len(seeds) == 3,
        "Demo 1 requires a list of three seeds.",
    )
    for seed in seeds:
        require_int(seed, "Each seed", minimum=0)
    require(len(set(seeds)) == 3, "The three seeds must be distinct.")

    hidden_sizes = config["hidden_sizes"]
    require(
        type(hidden_sizes) is list and len(hidden_sizes) > 0,
        "hidden_sizes must be a non-empty list.",
    )
    for width in hidden_sizes:
        require_int(width, "Each hidden layer width")

    require(config["learning_rate"] > 0, "learning_rate must be positive.")
    require(0 <= config["gamma"] <= 1, "gamma must be between 0 and 1.")
    require(
        0 <= config["epsilon_end"] <= config["epsilon_start"] <= 1,
        "Require 0 <= epsilon_end <= epsilon_start <= 1.",
    )
    require(
        config["batch_size"] <= config["replay_capacity"],
        "batch_size cannot exceed replay_capacity.",
    )
    require(
        config["batch_size"] <= config["warmup_steps"]
        < config["training_steps"],
        "Require batch_size <= warmup_steps < training_steps.",
    )
    require(
        config["eval_interval"] <= config["training_steps"],
        "eval_interval cannot exceed training_steps.",
    )

    # Derived values
    config["state_dim"] = 2 ** config["n_qubits"]
    config["observation_dim"] = 4 * config["state_dim"] + 1
    config["n_actions"] = len(config["action_labels"])

    return config


def save_resolved_config(config, run_dir):
    # Save a snapshot returned by load_config inside a run directory
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "config.resolved.json"

    # Exclusive creation to protect an existing run's configuration
    with path.open("x", encoding="utf-8") as file:
        json.dump(config, file, indent=2, allow_nan=False)
        file.write("\n")

    return path