import pufferlib
import torch
from torch import nn
from torch.distributions import Categorical

import numpy as np

from typing import List, Tuple, Union, Callable

from .VQCAgent import (
    GHZAgent,
    GraphStateAgent,
    SeparableAgent,
    WStateAgent
)

AGENTS = {
    "ghz": GHZAgent,
    "graph_state": GraphStateAgent,
    "separable": SeparableAgent,
    "w_state": WStateAgent
}

# EDGE_LIST = [(3, 0), (2, 0), (6, 0), (4, 0), (5, 0), (3, 1), (2, 1), (4, 1), (5, 1), (7, 1), (0, 1)]

class PongClassicalCritic(nn.Module):
    def __init__(self, input_dim: int = 8, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

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
        assert agent_type in AGENTS, f"Unknown agent type: {agent_type}"
        self.single_action_dim = env.single_action_space.n
        self.observation_dim = env.single_observation_space.shape[0]
        self.actor = AGENTS[agent_type](
            action_space=self.single_action_dim,
            observation_space=self.observation_dim,
            **agent_args
        )
        self.critic = PongClassicalCritic(input_dim=self.observation_dim)
    
    def get_value(self, x):
        return self.critic(x)
    
    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


if __name__ == "__main__":
    from pufferlib import vector
    from pufferlib.ocean import env_creator
    import time

    agent_type = "graph_state"
    agent_args = {
        "n_layers": 6,
        "post_select": False,
        "edge_list": [(3, 0), (2, 0), (6, 0), (4, 0), (5, 0), (3, 1), (2, 1), (4, 1), (5, 1), (7, 1), (0, 1)]
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