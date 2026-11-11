import json
from math import isfinite
import torch
from qprep.contracts import Transition
from qprep.dqn import (
    QNetwork,
    ReplayBuffer,
    build_target_network,
    optimise_batch,
    select_action,
    sync_target_network,
)
from qprep.environment import QuantumEnv
from qprep.checkpoint import checkpoint_state, load_checkpoint
from qprep.evaluate import evaluate_policy


def epsilon_at_step(config, step):
    # Decrease exploration linearly, stopping at epsilon_end
    fraction = min(step / config["epsilon_decay_steps"], 1.0)
    return config["epsilon_start"] + fraction * (
        config["epsilon_end"] - config["epsilon_start"]
    )


class Trainer:
    # Collect experience and update the DQN

    def __init__(self, run):
        self.run = run
        self.config = run.config

        self.online = QNetwork(self.config).to(run.device)
        self.target = build_target_network(self.online)
        self.optimizer = torch.optim.Adam(
            self.online.parameters(), lr=self.config["learning_rate"]
        )
        self.replay = ReplayBuffer(self.config["replay_capacity"], run.rngs["replay"])

        self.env = QuantumEnv(self.config)
        env_seed = int(run.rngs["targets"].integers(0, 2**32))
        self.observation, _ = self.env.reset(seed=env_seed)
        self.episode_done = self.env.terminated
        self.episode_return = 0.0

        self.progress = {
            "environment_steps": 0,
            "gradient_steps": 0,
            "episodes_completed": 0,
            "epsilon": epsilon_at_step(self.config, 0),
        }

    def train_step(self):
        # Take one environment step and, when ready, one learning update
        if self.episode_done:
            self.observation, _ = self.env.reset()
            self.episode_done = self.env.terminated
            self.episode_return = 0.0

        epsilon = self.progress["epsilon"]
        action = select_action(
            self.online,
            self.observation,
            epsilon,
            self.run.rngs["exploration"],
        )

        next_observation, reward, terminated, truncated, info = self.env.step(action)

        self.replay.add(
            Transition(
                observation=self.observation,
                action_id=action,
                reward=reward,
                next_observation=next_observation,
                terminated=terminated,
                truncated=truncated,
            )
        )

        self.observation = next_observation
        self.episode_return += reward
        self.episode_done = terminated or truncated
        self.progress["environment_steps"] += 1

        if self.episode_done:
            self.progress["episodes_completed"] += 1

        loss = None
        if (
            self.progress["environment_steps"] >= self.config["warmup_steps"]
            and len(self.replay) >= self.config["batch_size"]
        ):
            batch = self.replay.sample(self.config["batch_size"])
            loss = optimise_batch(
                self.online,
                self.target,
                self.optimizer,
                batch,
                self.config["gamma"],
            )
            if not isfinite(loss):
                raise RuntimeError("Training produced a non-finite loss.")

            self.progress["gradient_steps"] += 1

            if (
                self.progress["gradient_steps"] % self.config["target_update_interval"]
                == 0
            ):
                sync_target_network(self.online, self.target)

        self.progress["epsilon"] = epsilon_at_step(
            self.config, self.progress["environment_steps"]
        )

        return {
            "environment_steps": self.progress["environment_steps"],
            "gradient_steps": self.progress["gradient_steps"],
            "episodes_completed": self.progress["episodes_completed"],
            "epsilon": epsilon,
            "loss": loss,
            "target_id": info["target_id"],
            "fidelity": info["fidelity"],
            "end_reason": info["end_reason"],
            "episode_return": (self.episode_return if self.episode_done else None),
        }

    def train_loop(self, steps):
        # Train for a block of steps and append progress to the run log
        if steps < 1:
            raise ValueError("steps must be positive.")
        if self.progress["environment_steps"] + steps > self.config["training_steps"]:
            raise ValueError("This block would exceed training_steps.")

        log_path = self.run.run_dir / "training.jsonl"
        with log_path.open("a", encoding="utf-8") as file:
            for _ in range(steps):
                record = self.train_step()
                file.write(json.dumps(record, allow_nan=False) + "\n")


def save_training_checkpoint(trainer, evaluation):
    """Save the agent, active episode, and evaluation for this step."""
    env = trainer.env
    state = checkpoint_state(
        trainer.run,
        trainer.online,
        trainer.target,
        trainer.optimizer,
        trainer.replay,
        trainer.progress,
        environment_state={
            "target_id": env.target.target_id,
            "action_ids": env.action_ids.copy(),
            "rng_state": env.np_random.bit_generator.state,
        },
        rollout_state={
            "episode_done": trainer.episode_done,
            "episode_return": trainer.episode_return,
        },
    )
    state["evaluation"] = evaluation

    step = trainer.progress["environment_steps"]
    path = trainer.run.run_dir / f"checkpoint_{step:06d}.pt"
    if path.exists():
        raise FileExistsError(f"Checkpoint already exists: {path}")

    temporary_path = path.with_suffix(".tmp")
    torch.save(state, temporary_path)
    temporary_path.replace(path)
    return path


def restore_training_checkpoint(trainer, path):
    # Restore a trainer and rebuild its current observation
    state = load_checkpoint(
        path,
        trainer.run,
        trainer.online,
        trainer.target,
        trainer.optimizer,
        trainer.replay,
    )

    episode = state["environment_state"]
    observation, _ = trainer.env.reset(
        seed=0, options={"target_id": episode["target_id"]}
    )
    for action_id in episode["action_ids"]:
        observation, _, _, _, _ = trainer.env.step(action_id)

    trainer.env.np_random.bit_generator.state = episode["rng_state"]
    trainer.observation = observation
    trainer.episode_done = state["rollout_state"]["episode_done"]
    trainer.episode_return = state["rollout_state"]["episode_return"]

    for name in trainer.progress:
        trainer.progress[name] = state[name]


def train_to_next_checkpoint(trainer):
    # Train to the next evaluation step or the end of the run
    step = trainer.progress["environment_steps"]
    total = trainer.config["training_steps"]
    if step >= total:
        raise ValueError("Training is already complete.")

    interval = trainer.config["eval_interval"]
    next_step = min((step // interval + 1) * interval, total)
    trainer.train_loop(next_step - step)

    evaluation = evaluate_policy(
        trainer.online,
        trainer.config,
        trainer.run.rngs["evaluation"],
    )
    path = save_training_checkpoint(trainer, evaluation)
    return path, evaluation
