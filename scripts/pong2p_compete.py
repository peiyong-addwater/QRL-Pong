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
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_entity: str = "addwater0315-csiro"
    """the entity (team) of wandb's project"""
    num_episodes: int = 1000
    """the number of episodes to run"""
    num_tests: int = 100
    
    # Agent settings
    backbone_out_dim: int = 12
    """the output dimension of the backbone network"""
    clamp_actor_weights: bool = True
    """if toggled, the quantum actor weights will be clamped to [-pi, pi]. Only used for quantum-classical hybrid agents"""

if __name__ == "__main__":
    args = tyro.cli(Args)
    args.backbone_out_dim = 12
    args.num_envs = 1
    args.n_layers = int(args.backbone_out_dim**2/(args.backbone_out_dim/3*3))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    run_name = f"{args.exp_name}__ActorParamClamped__{args.clamp_actor_weights}__Dim__{12}__Seed__{args.seed}__{int(time.time())}"

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=False,
            save_code=False,
        )
    
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    base_pretrained_model_dir = os.path.join("trained-models", "Pong2PModels")

    env = pong_v3.parallel_env(render_mode="human")
    env = ss.frame_skip_v0(env, 4)
    env = ss.clip_reward_v0(env, lower_bound=-1, upper_bound=1)
    env = ss.color_reduction_v0(env, mode="B")
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, 4)
    env = ss.agent_indicator_v0(env, type_only=False)
    env.single_action_space = env.action_space('first_0')
    env.single_observation_space = env.observation_space('first_0')

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
    

    # We set the entangled agent to be 'first_0' and the separable agent to be 'second_0'
    obs_sep = torch.zeros((args.num_episodes, args.num_envs) + env.single_observation_space.shape).to(device)
    actions_sep = torch.zeros((args.num_episodes, args.num_envs) + env.single_action_space.shape).to(device)
    
    rewards_sep = torch.zeros((args.num_episodes, args.num_envs)).to(device)
    dones_sep = torch.zeros((args.num_episodes, args.num_envs)).to(device)
    
    # storage for the entangled agent
    obs_ent = torch.zeros((args.num_episodes, args.num_envs) + env.single_observation_space.shape).to(device)
    actions_ent = torch.zeros((args.num_episodes, args.num_envs) + env.single_action_space.shape).to(device)
    
    rewards_ent = torch.zeros((args.num_episodes, args.num_envs)).to(device)
    dones_ent = torch.zeros((args.num_episodes, args.num_envs)).to(device)

    next_obs, _ = env.reset(seed=args.seed)

    total_episodic_return = {'first_0':[], 'second_0':[]}
    episodic_length = 0

    next_obs_S = torch.Tensor(next_obs['first_0']).to(device).unsqueeze(0)
    next_obs_E = torch.Tensor(next_obs['second_0']).to(device).unsqueeze(0)

    next_done_S = torch.zeros(args.num_envs).to(device)
    next_done_E = torch.zeros(args.num_envs).to(device)

    with torch.no_grad():
        for step in range(args.num_episodes):
            obs_sep[step] = next_obs_S
            obs_ent[step] = next_obs_E

            dones_sep[step] = next_done_S
            dones_ent[step] = next_done_E

            # Action logic
            action_sep, logprob_sep, _, value_sep = separableAgent.get_action_and_value(next_obs_S)
            action_ent, logprob_ent, _, value_ent = entangledAgent.get_action_and_value(next_obs_E)
            actions_sep[step] = action_sep
            actions_ent[step] = action_ent

             # execute the game and log data.
            while env.agents:
                actions = {
                    'first_0':action_sep.cpu().numpy().item(),
                    'second_0':action_ent.cpu().numpy().item()
                }
                next_obs, rewards, terminations, truncations, infos = env.step(actions)
                break

            total_episodic_return['first_0'].append(rewards['first_0'].item())
            total_episodic_return['second_0'].append(rewards['second_0'].item())
            episode_over = (terminations['first_0'] or terminations['second_0']) or (truncations['first_0'] or truncations['second_0'])
            print(episode_over)




