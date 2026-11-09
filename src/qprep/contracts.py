from dataclasses import dataclass
from typing import Any, Literal, TypedDict

import numpy as np
from numpy.typing import NDArray


ComplexVector = NDArray[np.complex128]
Observation = NDArray[np.float32]

GateName = Literal["h", "s", "t", "cx"]

EndReason = Literal[
    "already_solved",
    "success",
    "budget_exhausted",
    "external_truncation",
]


# List position is the action ID.
# For CX, qubits are ordered as (control, target).
ACTION_SPECS: tuple[tuple[GateName, tuple[int, ...]], ...] = (
    ("h", (0,)),
    ("h", (1,)),
    ("s", (0,)),
    ("s", (1,)),
    ("t", (0,)),
    ("t", (1,)),
    ("cx", (0, 1)),
    ("cx", (1, 0)),
)

OBSERVATION_BLOCKS = (
    "current_real",
    "current_imag",
    "target_real",
    "target_imag",
    "remaining_fraction",
)


@dataclass(frozen=True)
class Action:
    # One gate placed on specified qubits

    action_id: int
    gate: GateName
    qubits: tuple[int, ...]


@dataclass(frozen=True)
class Target:
    # A target state and its evaluation-only reference solution

    target_id: str
    state: ComplexVector
    reference_action_ids: tuple[int, ...]


@dataclass(frozen=True)
class Transition:
    # One interaction stored for DQN learning

    observation: Observation
    action_id: int
    reward: float
    next_observation: Observation
    terminated: bool
    truncated: bool

    def __post_init__(self):
        # Store independent snapshots, not references to environment buffers.
        observation = np.array(
            self.observation, dtype=np.float32, copy=True
        )
        next_observation = np.array(
            self.next_observation, dtype=np.float32, copy=True
        )

        if observation.ndim != 1:
            raise ValueError("observation must be a one-dimensional vector.")

        if next_observation.shape != observation.shape:
            raise ValueError("Observation shapes must match.")

        if not np.isfinite(observation).all():
            raise ValueError("observation must contain finite values.")

        if not np.isfinite(next_observation).all():
            raise ValueError("next_observation must contain finite values.")

        observation.setflags(write=False)
        next_observation.setflags(write=False)

        object.__setattr__(self, "observation", observation)
        object.__setattr__(self, "next_observation", next_observation)


@dataclass(frozen=True)
class TraceStep:
    # Diagnostic information after exactly one applied gate

    gate_count: int
    action_id: int
    fidelity: float
    reward: float


@dataclass(frozen=True)
class EpisodeTrace:
    # One complete attempt, including zero-action attempts

    target_id: str
    initial_fidelity: float
    steps: tuple[TraceStep, ...]
    end_reason: EndReason

    @property
    def gate_count(self) -> int:
        return len(self.steps)

    @property
    def final_fidelity(self) -> float:
        if not self.steps:
            return self.initial_fidelity
        return self.steps[-1].fidelity

    @property
    def total_reward(self) -> float:
        return sum(step.reward for step in self.steps)

    @property
    def success(self) -> bool:
        return self.end_reason in ("already_solved", "success")


class Checkpoint(TypedDict):
    # Required fields for a future resumable training checkpoint

    schema_version: int
    config: dict[str, Any]
    action_catalogue: list[dict[str, Any]]

    online_network_state: dict[str, Any]
    target_network_state: dict[str, Any]
    optimizer_state: dict[str, Any]
    replay_state: dict[str, Any]

    environment_steps: int
    gradient_steps: int
    episodes_completed: int
    epsilon: float

    rng_states: dict[str, Any]
    environment_state: dict[str, Any]
    rollout_state: dict[str, Any]
    provenance: dict[str, Any]