import numpy as np
from torch import nn
from qprep.contracts import Transition
import torch
from copy import deepcopy


class QNetwork(nn.Module):
    # Predict one action value for each legal action
    def __init__(self, config):
        super().__init__()
        self.n_actions = config["n_actions"]
        layers = []
        input_size = config["observation_dim"]

        for hidden_size in config["hidden_sizes"]:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size

        layers.append(nn.Linear(input_size, self.n_actions))
        self.layers = nn.Sequential(*layers)

    def forward(self, observation):
        return self.layers(observation)


class ReplayBuffer:
    # Keep recent transitions and sample batches for training

    def __init__(self, capacity, rng):
        if capacity < 1:
            raise ValueError("Replay capacity must be positive.")

        self.capacity = capacity
        self.rng = rng
        self._items = []
        self._next_index = 0

    def __len__(self):
        return len(self._items)

    def add(self, transition: Transition):
        if len(self._items) < self.capacity:
            self._items.append(transition)
        else:
            self._items[self._next_index] = transition
        self._next_index = (self._next_index + 1) % self.capacity

    def sample(self, batch_size):
        if not 1 <= batch_size <= len(self):
            raise ValueError("Batch size must be between 1 and the buffer size.")

        indices = self.rng.choice(len(self), size=batch_size, replace=False)
        batch = [self._items[index] for index in indices]

        return {
            "observations": np.stack([item.observation for item in batch]),
            "actions": np.array([item.action_id for item in batch], dtype=np.int64),
            "rewards": np.array([item.reward for item in batch], dtype=np.float32),
            "next_observations": np.stack([item.next_observation for item in batch]),
            "terminated": np.array([item.terminated for item in batch], dtype=np.bool_),
        }


def select_action(network, observation, epsilon, rng):
    # Choose an action using epsilon-greedy exploration
    if not 0 <= epsilon <= 1:
        raise ValueError("Epsilon must be between 0 and 1.")
    if epsilon > 0 and rng.random() < epsilon:
        return int(rng.integers(network.n_actions))

    device = next(network.parameters()).device
    observation_tensor = torch.tensor(observation, dtype=torch.float32, device=device)

    with torch.no_grad():
        q_values = network(observation_tensor)
        return int(q_values.argmax().item())


def build_target_network(online_network):
    # Create a separate copy of the online network
    target_network = deepcopy(online_network)
    target_network.requires_grad_(False)
    target_network.eval()
    return target_network


def sync_target_network(online_network, target_network):
    # Copy the current online weights into the target network
    target_network.load_state_dict(online_network.state_dict())
