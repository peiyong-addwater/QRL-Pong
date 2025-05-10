# Based on CleanRL's PPO implementation
# https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_ataripy
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
    exp_name: str = "Pong2P-SharedParam"
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

    # Agent settings
    agent_type: str = "entangled"
    """the type of the agent. Can be either 'entangled' or 'separable'"""
    backbone_out_dim: int = 12
    """the output dimension of the backbone network"""
    model_save_path: str = None
    """the path to save the model"""
    clamp_actor_weights: bool = True
    """if toggled, the quantum actor weights will be clamped to [-pi, pi]. Only used for quantum-classical hybrid agents"""
    num_envs: int = 8
    """the number of parallel environments"""

    # Algorithm specific arguments
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.n_layers = int(args.backbone_out_dim**2/(args.backbone_out_dim/3*3))
    args.batch_size = int(args.num_envs * args.num_steps)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    args.wandb_project_name = 'TestPong2P'#f"Pong2P__Dim__{args.backbone_out_dim}"

    assert args.backbone_out_dim / 3 == int(args.backbone_out_dim / 3), "backbone_out_dim must be a multiple of 3"

    run_name = f"Pong2P__{args.agent_type}__Dim__{args.backbone_out_dim}__Seed__{args.seed}__{int(time.time())}"

    print(args)

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

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    env = pong_v3.parallel_env(num_players=2)
    env = ss.frame_skip_v0(env, 4)
    env = ss.clip_reward_v0(env, lower_bound=-1, upper_bound=1)
    env = ss.color_reduction_v0(env, mode="B")
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, 4)
    env = ss.agent_indicator_v0(env, type_only=False)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    envs = ss.concat_vec_envs_v1(
        env, args.num_envs // 2, num_cpus=0, base_class="gymnasium"
    )
    envs.single_observation_space = envs.observation_space
    envs.single_action_space = envs.action_space
    envs.is_vector_env = True
    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"

    if args.agent_type == "entangled":
        agent = EntangledPPOAgent(envs, n_layers=args.n_layers, backbone_out_dim=args.backbone_out_dim, pretrained_backbone=False, backbone = Backbone2P(out_dim = args.backbone_out_dim)).to(device)
    elif args.agent_type == "separable":
        agent = SeparablePPOAgent(envs, n_layers=args.n_layers, backbone_out_dim=args.backbone_out_dim, pretrained_backbone=False, backbone = Backbone2P(out_dim = args.backbone_out_dim)).to(device)
    else:
        raise ValueError("agent_type must be either 'entangled' or 'separable'")
    
    print("Agents created...\n")

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)


    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)


    for iteration in range(1, args.num_iterations + 1):
        # track the histogram of parameters
        for param_tensor in agent.state_dict():
            #print(param_tensor, "\t", agent.state_dict()[param_tensor].size())
            writer.add_histogram(f"3-Model-Params/{param_tensor}", agent.state_dict()[param_tensor], global_step)
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        total_episodic_rewards = np.zeros((args.num_envs,), dtype=np.float32)
        total_episodic_lengths = np.zeros((args.num_envs,), dtype=np.float32)

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1) # has shape (128, 8)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            # manully log the rewards and episodic lengths
            total_episodic_rewards += reward
            total_episodic_lengths += 1

            for i in range(args.num_envs):
                player_idx = i % 2
                if next_done[i] > 0:
                    print(f"Player {player_idx} Return {total_episodic_rewards[i]}")
            