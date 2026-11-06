import json
from pathlib import Path
import pytest
from qprep.config import load_config, save_resolved_config


CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "demo1.json"
)


def write_modified_config(tmp_path, changes):
    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8-sig"))
    data.update(changes)
    path = tmp_path / "modified.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_valid_demo1_config():
    config = load_config(CONFIG_PATH)
    assert config["state_dim"] == 4
    assert config["observation_dim"] == 17
    assert config["n_actions"] == 8


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"n_qubits": 3}, "n_qubits"),
        ({"max_gates": 0}, "max_gates"),
        ({"max_gates": 7}, "max_gates"),
        ({"max_gates": True}, "max_gates"),
        ({"success_fidelity": 0.95}, "success_fidelity"),
        ({"gate_reward": 0.05}, "gate_reward"),
        ({"action_labels": ["h_0"]}, "action_labels"),
        ({"seeds": [0, 0, 1]}, "distinct"),
        ({"learning_rate": -0.001}, "learning_rate"),
        ({"learning_rate": float("nan")}, "finite"),
        ({"gamma": 1.1}, "gamma"),
        ({"epsilon_end": 1.1}, "epsilon"),
        ({"replay_capacity": 32}, "replay_capacity"),
        ({"warmup_steps": 32}, "warmup_steps"),
        ({"eval_interval": 200000}, "eval_interval"),
        ({"hidden_sizes": [128, 0]}, "width"),
        ({"observation_dim": 99}, "Unexpected"),
        ({"learning_rates": 0.001}, "Unexpected"),
    ],
)
def test_invalid_settings_are_rejected(tmp_path, changes, message):
    path = write_modified_config(tmp_path, changes)

    with pytest.raises(ValueError, match=message):
        load_config(path)


def test_missing_setting_is_rejected(tmp_path):
    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8-sig"))
    del data["max_gates"]
    path = tmp_path / "missing.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="Missing"):
        load_config(path)


def test_duplicate_key_is_rejected(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"n_qubits": 2, "n_qubits": 3}', encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate"):
        load_config(path)


def test_resolved_config_is_saved_without_overwriting(tmp_path):
    config = load_config(CONFIG_PATH)
    run_dir = tmp_path / "example_run"

    saved_path = save_resolved_config(config, run_dir)
    saved = json.loads(saved_path.read_text(encoding="utf-8"))

    assert saved == config

    with pytest.raises(FileExistsError):
        save_resolved_config(config, run_dir)