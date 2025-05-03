# Based on CleanRL's PPO implementation
# https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_ataripy
import os
import random
import time
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
    exp_name: str = "Pong2P"
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

    # Agent settings
    backbone_out_dim: int = 12
    """the output dimension of the backbone network"""
    model_save_path: str = None
    """the path to save the model"""
    clamp_actor_weights: bool = False
    """if toggled, the quantum actor weights will be clamped to [-pi, pi]. Only used for quantum-classical hybrid agents"""

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
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

if __name__ == "__main__":
    args = tyro.cli(Args)
    args.num_envs = 1
    args.n_layers = int(args.backbone_out_dim**2/(args.backbone_out_dim/3*3))
    args.batch_size = int(args.num_envs * args.num_steps)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    args.wandb_project_name = f"Pong2P__Dim__{args.backbone_out_dim}"

    assert args.backbone_out_dim / 3 == int(args.backbone_out_dim / 3), "backbone_out_dim must be a multiple of 3"

    run_name = f"Pong2P__Dim__{args.backbone_out_dim}__Seed__{args.seed}__{int(time.time())}"

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
    env = pong_v3.parallel_env(render_mode="human")
    env = ss.frame_skip_v0(env, 4)
    env = ss.clip_reward_v0(env, lower_bound=-1, upper_bound=1)
    env = ss.color_reduction_v0(env, mode="B")
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, 4)
    env.single_action_space = env.action_space('first_0')
    env.single_observation_space = env.observation_space('first_0')

    # agents
    separableAgent = SeparablePPOAgent(env, n_layers=args.n_layers, backbone_out_dim=args.backbone_out_dim, pretrained_backbone=False, backbone = Backbone2P(out_dim = args.backbone_out_dim)).to(device) # 'first_0'
    entangledAgent = EntangledPPOAgent(env, n_layers=args.n_layers, backbone_out_dim=args.backbone_out_dim, pretrained_backbone=False, backbone = Backbone2P(out_dim = args.backbone_out_dim)).to(device) # 'second_0'
    print("Agents created...\n")

    # optimiser
    optimizerS = optim.Adam(separableAgent.parameters(), lr=args.learning_rate, eps=1e-5)
    optimizerE = optim.Adam(entangledAgent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    ## storage for the separable agent
    obs_sep = torch.zeros((args.num_steps, args.num_envs) + env.single_observation_space.shape).to(device)
    actions_sep = torch.zeros((args.num_steps, args.num_envs) + env.single_action_space.shape).to(device)
    logprobs_sep = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards_sep = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones_sep = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values_sep = torch.zeros((args.num_steps, args.num_envs)).to(device)
    # storage for the entangled agent
    obs_ent = torch.zeros((args.num_steps, args.num_envs) + env.single_observation_space.shape).to(device)
    actions_ent = torch.zeros((args.num_steps, args.num_envs) + env.single_action_space.shape).to(device)
    logprobs_ent = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards_ent = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones_ent = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values_ent = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # Start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = env.reset(seed = args.seed)

    next_obs_S = torch.Tensor(next_obs['first_0']).to(device).unsqueeze(0)
    next_obs_E = torch.Tensor(next_obs['second_0']).to(device).unsqueeze(0)

    # print(next_obs_S.shape)

    next_done_S = torch.zeros(args.num_envs).to(device)
    next_done_E = torch.zeros(args.num_envs).to(device)

    # Training loop
    while env.agents:
        for iteration in range(1, args.num_iterations + 1):
            # Annealing the rate if instructed to do so.
            if args.anneal_lr:
                frac = 1.0 - (iteration - 1) / args.num_iterations
                lrnow = frac * args.learning_rate
                optimizerS.param_groups[0]["lr"] = lrnow
                optimizerE.param_groups[0]["lr"] = lrnow
            for step in range(0, args.num_steps):
                global_step += args.num_envs
                obs_sep[step] = next_obs_S
                obs_ent[step] = next_obs_E

                dones_sep[step] = next_done_S
                dones_ent[step] = next_done_E

                # Action logic
                with torch.no_grad():
                    action_sep, logprob_sep, _, value_sep = separableAgent.get_action_and_value(next_obs_S)
                    action_ent, logprob_ent, _, value_ent = entangledAgent.get_action_and_value(next_obs_E)
                actions_sep[step] = action_sep
                actions_ent[step] = action_ent
                logprobs_sep[step] = logprob_sep
                logprobs_ent[step] = logprob_ent

                # execute the game and log data.
                actions = {'first_0': action_sep.cpu().numpy().item(), 'second_0': action_ent.cpu().numpy().item()}
                next_obs, rewards, terminations, truncations, infos = env.step(actions)

                next_done_S = np.array([np.logical_or(terminations['first_0'], truncations['first_0'])])
                next_done_S = torch.Tensor(next_done_S).to(device)
                next_done_E = np.array([np.logical_or(terminations['second_0'], truncations['second_0'])])
                next_done_E = torch.Tensor(next_done_E).to(device)

                next_obs_S = torch.Tensor(next_obs['first_0']).to(device).unsqueeze(0)
                next_obs_E = torch.Tensor(next_obs['second_0']).to(device).unsqueeze(0)
                
                #print(rewards_sep.shape)
                #print(torch.Tensor([rewards['first_0']]).to(device).view(-1).shape)
                rewards_sep[step] = torch.Tensor([rewards['first_0']]).to(device).view(-1)
                rewards_ent[step] = torch.Tensor([rewards['second_0']]).to(device).view(-1)

                infos_S = infos['first_0']
                infos_E = infos['second_0']



        # after all iterations finished, break the while loop
        break







    