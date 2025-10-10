import pufferlib
import torch
from torch import nn
from torch.distributions import Categorical

import numpy as np

from typing import List, Tuple, Union, Callable

from .VQCBackbones import (
    SeparableBackbone,
    EntangledBackbone,
    EntangledBackboneTrainableCRZ,
    EntangledBackboneTrainableIsingZZ,
    EntangledBackboneTrainableCP,
)

BACKBONES = {
    "entangled": EntangledBackbone,
    "entangled_trainable_crz": EntangledBackboneTrainableCRZ,
    "separable": SeparableBackbone,
    "entangled_trainable_rzz": EntangledBackboneTrainableIsingZZ,
    "entangled_trainable_cp": EntangledBackboneTrainableCP,
}

class PongClassicalCritic(nn.Module):
    def __init__(self, input_dim: int = 3, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
    
class PongClassicalPolicy(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class PongHybridAgent(nn.Module):
    """
    A hybrid agent with a classical critic and a quantum policy.
    """
    def __init__(self, agent_type, env, agent_args:dict):
        """
        :param agent_type: Type of the agent (e.g., "ghz", "graph_state", "separable").
        :param env: The environment instance.
        :param agent_args: Additional arguments for the agent (n_layers, post_select, edge_list)
        """
        super().__init__()
        assert agent_type in BACKBONES, f"Unknown agent type: {agent_type}"
        self.single_action_dim = env.single_action_space.n
        self.observation_dim = env.single_observation_space.shape[0]
        self.backbone = BACKBONES[agent_type](
            output_dim=8,
            observation_space=self.observation_dim,
            **agent_args,
        )
        self.critic = PongClassicalCritic(input_dim=8)
        self.actor = PongClassicalPolicy(
            input_dim=8,
            output_dim=self.single_action_dim,
        )
    
    def get_representation(self, x):
        hidden = self.backbone(x)
        return hidden
    
    def get_value(self, x):
        hidden = self.get_representation(x)
        return self.critic(hidden)

    def get_action_and_value(self, x, action=None):
        hidden = self.get_representation(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

class PongClassicalAgent(nn.Module):
    """
    A classical agent with a classical critic.
    """
    def __init__(self, env):
        super().__init__()
        self.single_action_dim = env.single_action_space.n
        self.observation_dim = env.single_observation_space.shape[0]
        self.backbone = nn.Sequential(
            nn.Linear(self.observation_dim, 4),
            nn.ReLU(),
            nn.Linear(4, 8)  # Output dimension for the critic
        )
        self.actor = PongClassicalPolicy(
            input_dim=8,
            output_dim=self.single_action_dim
        )
        self.critic = PongClassicalCritic(input_dim=8)

    def get_representations(self, x):
        hidden = self.backbone(x)
        return hidden

    def get_value(self, x):
        hidden = self.get_representations(x)
        return self.critic(hidden)

    def get_action_and_value(self, x, action=None):
        hidden = self.get_representations(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

if __name__ == "__main__":
    from pufferlib import vector
    from pufferlib.ocean import env_creator
    import time

    agent_type = "entangled_trainable_crz"  # "entangled" or "separable"
    agent_args = {
        "n_layers": 6
    }

    env_name = 'puffer_pong'
    env_creator = env_creator(env_name)
    vecenv = vector.make(env_creator, num_envs=2, num_workers=2, batch_size=1,
        backend=vector.Multiprocessing, env_kwargs={'num_envs': 4096})
    
    obs, _ = vecenv.reset()
    agent = PongHybridAgent(agent_type, vecenv, agent_args)
    
    test_count = 0
    while True:
        action, log_prob, entropy, value = agent.get_action_and_value(torch.tensor(obs))
        obs, reward, terminated, truncated, info = vecenv.step(action.numpy())
        print("Observation shape:", obs.shape)
        print("Action shape:", action.shape)
        time.sleep(1)
        test_count += 1
        if test_count > 10:
            break
    vecenv.close()