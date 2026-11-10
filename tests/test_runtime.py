import json
from pathlib import Path
import numpy as np
import pytest
from qprep.config import load_config
from qprep.runtime import STREAM_NAMES, make_rngs, prepare_run
from qprep.actions import serialise_action_catalogue

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "demo1.json"


def test_same_seed_reproduces_each_stream():
    first = make_rngs(42)
    second = make_rngs(42)

    for name in STREAM_NAMES:
        np.testing.assert_array_equal(
            first[name].integers(0, 1000000, size=20),
            second[name].integers(0, 1000000, size=20),
        )


def test_exploration_does_not_advance_target_stream():
    first = make_rngs(42)
    second = make_rngs(42)

    first["exploration"].random(1000)

    np.testing.assert_array_equal(
        first["targets"].integers(0, 6, size=20),
        second["targets"].integers(0, 6, size=20),
    )


def test_run_records_settings_and_environment(tmp_path):
    config = load_config(CONFIG_PATH)

    run = prepare_run(
        config,
        seed=0,
        project_root=PROJECT_ROOT,
        output_root=tmp_path,
        device="cpu",
    )

    saved_config = json.loads(
        (run.run_dir / "config.resolved.json").read_text(encoding="utf-8")
    )
    metadata = json.loads((run.run_dir / "metadata.json").read_text(encoding="utf-8"))

    assert saved_config == config
    assert metadata["seed"] == 0
    assert metadata["device"] == "cpu"
    assert "qiskit" in metadata["packages"]
    assert "torch" in metadata["packages"]
    assert (run.run_dir / "packages.txt").is_file()

    config["hidden_sizes"][0] = 999
    assert run.config["hidden_sizes"][0] == 128

    saved_actions = json.loads(
        (run.run_dir / "action_catalogue.json").read_text(encoding="utf-8")
    )
    assert saved_actions == serialise_action_catalogue()


def test_each_run_gets_a_new_directory(tmp_path):
    config = load_config(CONFIG_PATH)
    arguments = {
        "seed": 0,
        "project_root": PROJECT_ROOT,
        "output_root": tmp_path,
    }

    first = prepare_run(config, **arguments)
    second = prepare_run(config, **arguments)

    assert first.run_dir != second.run_dir
    assert first.run_dir.exists()
    assert second.run_dir.exists()


def test_unconfigured_seed_is_rejected(tmp_path):
    config = load_config(CONFIG_PATH)

    with pytest.raises(ValueError, match="configured seeds"):
        prepare_run(
            config,
            seed=999,
            project_root=PROJECT_ROOT,
            output_root=tmp_path,
        )

    assert list(tmp_path.iterdir()) == []
