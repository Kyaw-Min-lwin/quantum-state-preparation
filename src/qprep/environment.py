import gymnasium as gym
import numpy as np
from qprep.encoding import encode_observation
from qprep.quantum import fidelity, zero_state
from qprep.targets import build_target_catalogue, sample_target


class QuantumEnv(gym.Env):
    # Two-qubit state-preparation environment for Demo 1

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.max_gates = config["max_gates"]
        self.catalogue = build_target_catalogue()
        self.action_space = gym.spaces.Discrete(config["n_actions"])

        low = np.full(config["observation_dim"], -1.0, dtype=np.float32)
        high = np.ones(config["observation_dim"], dtype=np.float32)
        low[-1] = 0.0  # The remaining-budget fraction is in [0, 1].
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        self.state = None
        self.target = None
        self.remaining_gates = 0
        self.action_ids = []
        self.initial_fidelity = None
        self.terminated = False
        self.end_reason = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        target_id = options.get("target_id") if options is not None else None
        target = sample_target(self.np_random, self.catalogue, target_id)
        state = zero_state(self.config["n_qubits"])
        initial_fidelity = fidelity(state, target.state)

        observation = encode_observation(
            state, target.state, self.max_gates, self.max_gates
        )

        self.state = state
        self.target = target
        self.remaining_gates = self.max_gates
        self.action_ids = []
        self.initial_fidelity = initial_fidelity
        self.terminated = initial_fidelity >= self.config["success_fidelity"]
        self.end_reason = "already_solved" if self.terminated else None

        info = {
            "target_id": target.target_id,
            "initial_fidelity": initial_fidelity,
            "remaining_gates": self.remaining_gates,
            "gate_count": 0,
            "end_reason": self.end_reason,
        }
        return observation, info
