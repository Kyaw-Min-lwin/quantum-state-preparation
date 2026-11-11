from copy import deepcopy
from pathlib import Path
import numpy as np
import json
import pytest
import torch
from qprep.config import load_config
from qprep.evaluate import evaluate_policy
from qprep.runtime import prepare_run
from qprep.train import (
    Trainer,
    epsilon_at_step,
    restore_training_checkpoint,
    train_to_next_checkpoint,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "demo1.json"


def make_test_trainer(tmp_path):
    config = load_config(CONFIG_PATH)
    config.update(
        {
            "hidden_sizes": [16],
            "batch_size": 2,
            "replay_capacity": 10,
            "warmup_steps": 3,
            "target_update_interval": 2,
            "training_steps": 16,
            "eval_interval": 4,
            "epsilon_decay_steps": 4,
        }
    )

    run = prepare_run(
        config,
        seed=config["seeds"][0],
        project_root=PROJECT_ROOT,
        output_root=tmp_path,
    )
    return Trainer(run)


def test_epsilon_schedule():
    config = load_config(CONFIG_PATH)

    assert epsilon_at_step(config, 0) == pytest.approx(1.0)
    assert epsilon_at_step(config, 25000) == pytest.approx(0.525)
    assert epsilon_at_step(config, 50000) == pytest.approx(0.05)
    assert epsilon_at_step(config, 75000) == pytest.approx(0.05)


def test_warmup_updates_and_target_sync(tmp_path):
    trainer = make_test_trainer(tmp_path)

    assert trainer.train_step()["loss"] is None
    assert trainer.train_step()["loss"] is None

    third = trainer.train_step()
    assert np.isfinite(third["loss"])
    assert third["gradient_steps"] == 1

    fourth = trainer.train_step()
    assert fourth["environment_steps"] == 4
    assert fourth["gradient_steps"] == 2
    assert len(trainer.replay) == 4

    for online, target in zip(trainer.online.parameters(), trainer.target.parameters()):
        torch.testing.assert_close(online, target)


def test_evaluation_preserves_training_state(tmp_path):
    trainer = make_test_trainer(tmp_path)
    trainer.train_step()

    observation_before = trainer.observation.copy()
    actions_before = trainer.env.action_ids.copy()
    progress_before = trainer.progress.copy()
    replay_before = trainer.replay.state_dict()
    weights_before = deepcopy(trainer.online.state_dict())
    torch_rng_before = torch.get_rng_state()
    env_rng_before = deepcopy(trainer.env.np_random.bit_generator.state)
    rngs_before = {
        name: deepcopy(rng.bit_generator.state)
        for name, rng in trainer.run.rngs.items()
        if name != "evaluation"
    }

    results = evaluate_policy(
        trainer.online, trainer.config, trainer.run.rngs["evaluation"]
    )

    assert [row["target_id"] for row in results] == ["A", "B", "C", "D", "E", "F"]
    np.testing.assert_array_equal(trainer.observation, observation_before)
    assert trainer.env.action_ids == actions_before
    assert trainer.progress == progress_before
    assert trainer.replay.state_dict() == replay_before
    assert trainer.env.np_random.bit_generator.state == env_rng_before
    assert trainer.online.training
    assert torch.equal(torch.get_rng_state(), torch_rng_before)

    for name, state in rngs_before.items():
        assert trainer.run.rngs[name].bit_generator.state == state

    for name, weights in weights_before.items():
        torch.testing.assert_close(trainer.online.state_dict()[name], weights)


def test_resumed_training_matches_uninterrupted_training(tmp_path):
    original = make_test_trainer(tmp_path)
    path, _ = train_to_next_checkpoint(original)

    saved_observation = original.observation.copy()
    saved_actions = original.env.action_ids.copy()
    saved_progress = original.progress.copy()

    _, expected_evaluation = train_to_next_checkpoint(original)

    resumed = make_test_trainer(tmp_path)
    restore_training_checkpoint(resumed, path)

    np.testing.assert_array_equal(resumed.observation, saved_observation)
    assert resumed.env.action_ids == saved_actions
    assert resumed.progress == saved_progress

    _, actual_evaluation = train_to_next_checkpoint(resumed)

    original_log = original.run.run_dir / "training.jsonl"
    resumed_log = resumed.run.run_dir / "training.jsonl"

    expected = [
        json.loads(line)
        for line in original_log.read_text(encoding="utf-8").splitlines()[4:]
    ]
    actual = [
        json.loads(line)
        for line in resumed_log.read_text(encoding="utf-8").splitlines()
    ]
    assert actual == expected
    assert actual_evaluation == expected_evaluation