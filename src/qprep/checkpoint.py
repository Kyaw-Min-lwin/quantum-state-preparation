import json
import random
from copy import deepcopy
import torch
from qprep.actions import serialise_action_catalogue


def checkpoint_state(
    run,
    online,
    target,
    optimizer,
    replay,
    progress,
    *,
    environment_state=None,
    rollout_state=None,
):
    # Capture the state needed to continue training
    metadata = json.loads((run.run_dir / "metadata.json").read_text(encoding="utf-8"))

    return {
        "schema_version": 1,
        "config": deepcopy(run.config),
        "action_catalogue": serialise_action_catalogue(),
        "online_network_state": deepcopy(online.state_dict()),
        "target_network_state": deepcopy(target.state_dict()),
        "optimizer_state": deepcopy(optimizer.state_dict()),
        "replay_state": replay.state_dict(),
        "environment_steps": progress["environment_steps"],
        "gradient_steps": progress["gradient_steps"],
        "episodes_completed": progress["episodes_completed"],
        "epsilon": progress["epsilon"],
        "rng_states": {
            "numpy": {name: rng.bit_generator.state for name, rng in run.rngs.items()},
            "python": random.getstate(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": (
                torch.cuda.get_rng_state_all() if run.device.type == "cuda" else []
            ),
        },
        "environment_state": deepcopy(environment_state or {}),
        "rollout_state": deepcopy(rollout_state or {}),
        "provenance": metadata,
    }


def load_checkpoint(path, run, online, target, optimizer, replay):
    # Restore a Demo 1 training checkpoint and return its saved state
    state = torch.load(path, map_location="cpu", weights_only=True)

    if state["schema_version"] != 1:
        raise ValueError("Unsupported checkpoint version.")
    if state["action_catalogue"] != serialise_action_catalogue():
        raise ValueError("Checkpoint action catalogue does not match.")
    if state["config"] != run.config:
        raise ValueError("Checkpoint configuration does not match.")

    rng_states = state["rng_states"]
    if bool(rng_states["torch_cuda"]) != (run.device.type == "cuda"):
        raise ValueError("Resume training on the same device type as the checkpoint.")

    replay.load_state_dict(state["replay_state"])
    online.load_state_dict(state["online_network_state"])
    target.load_state_dict(state["target_network_state"])
    optimizer.load_state_dict(state["optimizer_state"])

    for name, rng in run.rngs.items():
        rng.bit_generator.state = rng_states["numpy"][name]

    random.setstate(rng_states["python"])
    torch.set_rng_state(rng_states["torch_cpu"])
    if rng_states["torch_cuda"]:
        torch.cuda.set_rng_state_all(rng_states["torch_cuda"])
    return state
