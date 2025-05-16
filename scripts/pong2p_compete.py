import os
import random
import time
import datetime
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from model import (
    Backbone2P,
    EntangledPPOAgent,
    SeparablePPOAgent
)

from pettingzoo.atari import pong_v3
import supersuit as ss

@dataclass
class Args:
    exp_name: str = "Pong2PCompete"
    """The base name of the experiment."""
    seed: int = 1
    """The random seed."""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_entity: str = "addwater0315-csiro"
    """the entity (team) of wandb's project"""
    num_episodes: int = 1000
    """the number of episodes to run"""
    
    # Agent settings
    backbone_out_dim: int = 12
    """the output dimension of the backbone network"""
    clamp_actor_weights: bool = True
    """if toggled, the quantum actor weights will be clamped to [-pi, pi]. Only used for quantum-classical hybrid agents"""

if __name__ == "__main__":
    args = tyro.cli(Args)
    args.backbone_out_dim = 12
    args.n_layers = int(args.backbone_out_dim**2/(args.backbone_out_dim/3*3))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    run_name = f"{args.exp_name}__ActorParamClamped__{args.clamp_actor_weights}__Dim__{12}__Seed__{args.seed}__{int(time.time())}"

    base_pretrained_model_dir = os.path.join("trained_models", "Pong2PModels")

    env = pong_v3.parallel_env(render_mode="human")
    env = ss.frame_skip_v0(env, 4)
    env = ss.clip_reward_v0(env, lower_bound=-1, upper_bound=1)
    env = ss.color_reduction_v0(env, mode="B")
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, 4)
    env.single_action_space = env.action_space('first_0')
    env.single_observation_space = env.observation_space('first_0')
    env = ss.agent_indicator_v0(env, type_only=False)

    separableAgent = SeparablePPOAgent(env, n_layers=args.n_layers, backbone_out_dim=args.backbone_out_dim, pretrained_backbone=False, backbone = Backbone2P(out_dim = args.backbone_out_dim)).to(device) # 'first_0'
    entangledAgent = EntangledPPOAgent(env, n_layers=args.n_layers, backbone_out_dim=args.backbone_out_dim, pretrained_backbone=False, backbone = Backbone2P(out_dim = args.backbone_out_dim)).to(device) # 'second_0'

    if args.clamp_actor_weights:
        entangled_pretrained_model_path = os.path.join(base_pretrained_model_dir, "Pong2P__entangled__ActorParamClamped__True__Dim__12__Seed__1__1746905426.pt")
        separable_pretrained_model_path = os.path.join(base_pretrained_model_dir, "Pong2P__separable__ActorParamClamped__True__Dim__12__Seed__1__1746905426.pt")

        entangledAgent.load_state_dict(torch.load(entangled_pretrained_model_path))
        separableAgent.load_state_dict(torch.load(separable_pretrained_model_path))
    else:
        entangled_pretrained_model_path = os.path.join(base_pretrained_model_dir, "Pong2P__entangled__ActorParamClamped__False__Dim__12__Seed__1__1746905426.pt")
        separable_pretrained_model_path = os.path.join(base_pretrained_model_dir, "Pong2P__separable__ActorParamClamped__False__Dim__12__Seed__1__1746905426.pt")

        entangledAgent.load_state_dict(torch.load(entangled_pretrained_model_path))
        separableAgent.load_state_dict(torch.load(separable_pretrained_model_path))
    


