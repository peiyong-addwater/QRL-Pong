import pufferlib
import torch
from torch import nn
from torch.distributions import Categorical

import numpy as np

from typing import List, Tuple, Union, Callable

from .VQCBackbones2 import (
    GHZBackbone,
    GraphStateBackbone,
    SeparableBackbone,
    WStateBackbone
)

BACKBONES = {
    "ghz": GHZBackbone,
    "graph_state": GraphStateBackbone,
    "separable": SeparableBackbone,
    "w_state": WStateBackbone
}

# EDGE_LIST = [(3, 0), (2, 0), (6, 0), (4, 0), (5, 0), (3, 1), (2, 1), (4, 1), (5, 1), (7, 1), (0, 1)]

class ElementwiseScaleShift(nn.Module):
    """
    Element-wise affine transform: y = x * scale + shift

    Parameters are shaped for broadcasting against the input. Typical usage is to
    pass the feature dimension size so the parameters have shape (features,) and
    apply along the last dimension, but any broadcastable shape works.

    Args:
        shape: int or tuple defining the parameter shape (broadcastable to inputs).
        init_scale: initial value for scale (default 1.0).
        init_shift: initial value for shift (default 0.0).
        learnable: if True, scale/shift are learnable parameters; otherwise buffers.
    """

    def __init__(
        self,
        shape: Union[int, Tuple[int, ...]],
        init_scale: float = 1.0,
        init_shift: float = 0.0,
        learnable: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(shape, int):
            param_shape: Tuple[int, ...] = (shape,)
        else:
            param_shape = tuple(shape)

        if learnable:
            self.scale = nn.Parameter(torch.full(param_shape, float(init_scale)))
            self.shift = nn.Parameter(torch.full(param_shape, float(init_shift)))
        else:
            self.register_buffer("scale", torch.full(param_shape, float(init_scale)))
            self.register_buffer("shift", torch.full(param_shape, float(init_shift)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale + self.shift

    def extra_repr(self) -> str:
        return f"shape={tuple(self.scale.shape)}"

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
            output_dim=3,
            observation_space=self.observation_dim,
            **agent_args,
        )
        # Element-wise affine scaling of the quantum backbone output
        self.affine = ElementwiseScaleShift(shape=3)
        self.critic = PongClassicalCritic(input_dim=3)
        self.actor = PongClassicalPolicy(
            input_dim=3,
            output_dim=self.single_action_dim,
        )
    
    def get_representation(self, x):
        hidden = self.backbone(x)
        hidden = self.affine(hidden)
        return hidden
    
    def get_value(self, x):
        x = x * torch.pi * 2 # Scale inputs to [0, 2π]
        hidden = self.get_representation(x)
        return self.critic(hidden)

    def get_action_and_value(self, x, action=None):
        x = x * torch.pi * 2 # Scale inputs to [0, 2π]
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
            nn.Linear(self.observation_dim, 2**8),
            nn.ReLU(),
            nn.Linear(2**8, 3)  # Output dimension for the critic
        )
        self.actor = PongClassicalPolicy(
            input_dim=3,
            output_dim=self.single_action_dim
        )
        self.critic = PongClassicalCritic(input_dim=3)

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

    agent_type = "graph_state"
    agent_args = {
        "n_layers": 6,
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