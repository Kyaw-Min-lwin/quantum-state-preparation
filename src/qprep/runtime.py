import copy
import importlib
import json
import platform
import random
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch

from qprep.config import save_resolved_config

STREAM_NAMES = (
    "targets",
    "exploration",
    "replay",
    "baseline",
    "evaluation",
)


@dataclass
class RunContext:
    run_dir: Path
    config: dict
    seed: int
    device: torch.device
    rngs: dict[str, np.random.Generator]


def make_rngs(seed):
    """Create reproducible generators for separate responsibilities."""
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer.")
    children = np.random.SeedSequence(seed).spawn(len(STREAM_NAMES))

    return {
        name: np.random.Generator(np.random.PCG64(child))
        for name, child in zip(STREAM_NAMES, children)
    }


def git_output(project_root, *arguments):
    """Return Git output, or None if Git information is unavailable."""
    try:
        result = subprocess.run(
            ["git", "-C", str(project_root), *arguments],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ):
        return None

    return result.stdout.strip()


def package_versions():
    """Check imports and record versions of direct dependencies."""
    packages = (
        "qiskit",
        "numpy",
        "gymnasium",
        "torch",
        "matplotlib",
        "pylatexenc",
    )

    versions = {}
    for package in packages:
        importlib.import_module(package)
        versions[package] = version(package)

    return versions


def write_json(path, data):
    """Create a JSON record without replacing an existing one."""
    with Path(path).open("x", encoding="utf-8") as file:
        json.dump(data, file, indent=2, allow_nan=False)
        file.write("\n")


def prepare_run(
    config,
    *,
    seed,
    project_root,
    output_root=None,
    device="cpu",
):
    if type(seed) is not int or seed not in config["seeds"]:
        raise ValueError("seed must be one of the configured seeds.")

    project_root = Path(project_root).resolve()
    if not (project_root / "pyproject.toml").is_file():
        raise ValueError("project_root must contain pyproject.toml.")

    if device not in ("cpu", "cuda"):
        raise ValueError("device must be 'cpu' or 'cuda'.")

    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable.")

    versions = package_versions()

    # Capture all installed package versions for this environment.
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--exclude-editable"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    ).stdout

    revision = git_output(project_root, "rev-parse", "HEAD")
    status = git_output(project_root, "status", "--porcelain", "--untracked-files=all")

    rngs = make_rngs(seed)

    # Global RNGs used by Python and PyTorch
    random.seed(seed)
    torch.manual_seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if output_root is None:
        output_root = project_root / "runs"
    else:
        output_root = Path(output_root).resolve()

    now = datetime.now(timezone.utc)
    run_name = f"{now:%Y%m%dT%H%M%S%fZ}" f"_seed{seed}_{uuid4().hex[:8]}"
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    #  this run's settings independent of the caller's dictionary
    run_config = copy.deepcopy(config)
    save_resolved_config(run_config, run_dir)

    metadata = {
        "schema_version": 1,
        "created_at_utc": now.isoformat(),
        "seed": seed,
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": versions,
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "gpu_name": (torch.cuda.get_device_name(0) if device == "cuda" else None),
        "git_revision": revision,
        "git_dirty": None if status is None else bool(status),
        "git_status": status,
        "rng_stream_names": list(STREAM_NAMES),
        "rng_bit_generator": "PCG64",
        "rng_spawn_keys": {name: [index] for index, name in enumerate(STREAM_NAMES)},
        "torch_deterministic_algorithms": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }

    write_json(run_dir / "metadata.json", metadata)

    with (run_dir / "packages.txt").open("x", encoding="utf-8") as file:
        file.write(freeze)

    return RunContext(
        run_dir=run_dir,
        config=run_config,
        seed=seed,
        device=torch.device(device),
        rngs=rngs,
    )
