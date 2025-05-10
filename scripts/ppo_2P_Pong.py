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
    exp_name: str = "Pong2P"
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
    capture_video: bool = True
    """if toggled, the video will be captured"""

    # Agent settings
    backbone_out_dim: int = 12
    """the output dimension of the backbone network"""
    model_save_path: str = None
    """the path to save the model"""
    clamp_actor_weights: bool = True
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
    args.wandb_project_name = 'TestPong2P'#f"Pong2P__Dim__{args.backbone_out_dim}"

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
    # global_step = 0
    start_time = time.time()
    # Training loop
    for iteration in range(1, args.num_iterations + 1):
        print(f"----Iteration {iteration} of {args.num_iterations}...")
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizerS.param_groups[0]["lr"] = lrnow
            optimizerE.param_groups[0]["lr"] = lrnow
            
        next_obs, _ = env.reset(seed=None)

        total_episodic_return = {'first_0':0, 'second_0':0}
        episodic_length = 0

        next_obs_S = torch.Tensor(next_obs['first_0']).to(device).unsqueeze(0)
        next_obs_E = torch.Tensor(next_obs['second_0']).to(device).unsqueeze(0)

        # print(next_obs_S.shape)

        next_done_S = torch.zeros(args.num_envs).to(device)
        next_done_E = torch.zeros(args.num_envs).to(device)

        with torch.no_grad():
            for step in range(0, args.num_steps):

                # global_step += args.num_envs

                obs_sep[step] = next_obs_S
                obs_ent[step] = next_obs_E

                dones_sep[step] = next_done_S
                dones_ent[step] = next_done_E

                # Action logic
                action_sep, logprob_sep, _, value_sep = separableAgent.get_action_and_value(next_obs_S)
                action_ent, logprob_ent, _, value_ent = entangledAgent.get_action_and_value(next_obs_E)
                actions_sep[step] = action_sep
                actions_ent[step] = action_ent
                logprobs_sep[step] = logprob_sep
                logprobs_ent[step] = logprob_ent

                # execute the game and log data.

                while env.agents:
                    actions = {'first_0': action_sep.cpu().numpy().item(), 'second_0': action_ent.cpu().numpy().item()}
                    # print(actions)
                    next_obs, rewards, terminations, truncations, infos = env.step(actions)
                    break
                #print("actions taken")

                total_episodic_return['first_0'] += rewards['first_0'].item()
                total_episodic_return['second_0'] += rewards['second_0'].item()
                episodic_length += 1

                episode_over = (terminations['first_0'] or terminations['second_0']) or (truncations['first_0'] or truncations['second_0'])


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

                # infos has nothing since in the original code (https://github.com/Farama-Foundation/PettingZoo/blob/master/pettingzoo/atari/base_atari_env.py) it is implemented as
                # infos = {agent: {} for agent in self.possible_agents if agent in self.agents}
                # infos_S = infos['first_0']
                # infos_E = infos['second_0']

                if episode_over:
                    break
            
            print(f"Iteration={iteration}, Episodic length={episodic_length}, SepAgent episodic return={total_episodic_return['first_0']}, EntAgent episodic return={total_episodic_return['second_0']}")
            writer.add_scalar("0-Episodic-Stats/SepAgentEpisodicReturn", total_episodic_return['first_0'], iteration)
            writer.add_scalar("0-Episodic-Stats/EntAgentEpisodicReturn", total_episodic_return['second_0'], iteration)

            # bootstrap value if not done
            next_value_S = separableAgent.get_value(next_obs_S).reshape(1, -1)
            next_value_E = entangledAgent.get_value(next_obs_E).reshape(1, -1)

            advantages_S = torch.zeros_like(rewards_sep).to(device)
            advantages_E = torch.zeros_like(rewards_ent).to(device)

            lastgaelam_S = 0
            lastgaelam_E = 0

            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal_S = 1.0 - next_done_S
                    nextnonterminal_E = 1.0 - next_done_E
                    nextvalues_S = next_value_S
                    nextvalues_E = next_value_E
                else:
                    nextnonterminal_S = 1.0 - dones_sep[t + 1]
                    nextnonterminal_E = 1.0 - dones_ent[t + 1]
                    nextvalues_S = values_sep[t + 1]
                    nextvalues_E = values_ent[t + 1]

                delta_S = rewards_sep[t] + args.gamma * nextvalues_S * nextnonterminal_S - values_sep[t]
                advantages_S[t] = lastgaelam_S = delta_S + args.gamma * args.gae_lambda * nextnonterminal_S * lastgaelam_S

                delta_E = rewards_ent[t] + args.gamma * nextvalues_E * nextnonterminal_E - values_ent[t]
                advantages_E[t] = lastgaelam_E = delta_E + args.gamma * args.gae_lambda * nextnonterminal_E * lastgaelam_E
            
            returns_S = advantages_S + values_sep
            returns_E = advantages_E + values_ent
        
        # flatten the batch
        b_obs_S = obs_sep.reshape((-1,) + env.single_observation_space.shape)
        b_obs_E = obs_ent.reshape((-1,) + env.single_observation_space.shape)
        b_actions_S = actions_sep.reshape((-1,) + env.single_action_space.shape)
        b_actions_E = actions_ent.reshape((-1,) + env.single_action_space.shape)
        b_logprobs_S = logprobs_sep.reshape(-1)
        b_logprobs_E = logprobs_ent.reshape(-1)
        b_returns_S = returns_S.reshape(-1)
        b_returns_E = returns_E.reshape(-1)
        b_advantages_S = advantages_S.reshape(-1)
        b_advantages_E = advantages_E.reshape(-1)
        b_values_S = values_sep.reshape(-1)
        b_values_E = values_ent.reshape(-1)
        
        b_inds = np.arange(args.batch_size)

        # optimizing the separable agent
        clipfracs_S = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                _, newlogprob_S, entropy_S, newvalue_S = separableAgent.get_action_and_value(b_obs_S[mb_inds], b_actions_S[mb_inds])
                logratio_S = newlogprob_S - b_logprobs_S[mb_inds]
                ratio_S = logratio_S.exp()
                with torch.no_grad():
                    approx_kl_S = (logprobs_sep[mb_inds] - newlogprob_S).mean()
                    clipfracs_S += [((ratio_S - 1.0).abs() > args.clip_coef).float().mean().item()]
                mb_advantages_S = b_advantages_S[mb_inds]
                if args.norm_adv:
                    mb_advantages_S = (mb_advantages_S - mb_advantages_S.mean()) / (mb_advantages_S.std() + 1e-8)
                
                # policy loss
                pg_loss1_S = -mb_advantages_S * ratio_S
                pg_loss2_S = -mb_advantages_S * torch.clamp(ratio_S, 1.0 - args.clip_coef, 1.0 + args.clip_coef)
                pg_loss_S = torch.max(pg_loss1_S, pg_loss2_S).mean()
                # value loss
                newvalue_S = newvalue_S.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped_S = (newvalue_S - b_returns_S[mb_inds]) ** 2
                    v_clipped_S = b_values_S[mb_inds] + torch.clamp(newvalue_S - b_values_S[mb_inds], -args.clip_coef, args.clip_coef)
                    v_loss_clipped_S = (v_clipped_S - b_returns_S[mb_inds]) ** 2
                    v_loss_max_S = torch.max(v_loss_unclipped_S, v_loss_clipped_S)
                    v_loss_S = 0.5 * v_loss_max_S.mean()
                else:
                    v_loss_S = 0.5 * ((newvalue_S - b_returns_S[mb_inds]) ** 2).mean()
                
                entropy_loss_S = entropy_S.mean()
                loss_S = pg_loss_S - args.ent_coef * entropy_loss_S + args.vf_coef * v_loss_S
                
                #old_param = separableAgent.state_dict()['backbone.network.7.weight']
                #print("old_param:", old_param)
                optimizerS.zero_grad()
                loss_S.backward()
                nn.utils.clip_grad_norm_(separableAgent.parameters(), args.max_grad_norm)
                optimizerS.step()
                #new_param = separableAgent.state_dict()['backbone.network.7.weight']
                #print("new_param:", new_param)

                if args.clamp_actor_weights:
                    separableAgent.state_dict()["actor.0.q_params"].data.clamp_(-np.pi, np.pi)
        
        # calculate the explained variance of the separable agent
        y_pred_S, y_true_S = b_values_S.cpu().numpy(), b_returns_S.cpu().numpy()
        var_y_S = np.var(y_true_S)
        explained_var_S = np.nan if var_y_S == 0 else 1 - np.var(y_true_S - y_pred_S) / var_y_S

        # log the data of the separable agent
        writer.add_scalar("1-Training-Stats/SeparableAgent/ExplainedVariance", explained_var_S, iteration)
        writer.add_scalar("1-Training-Stats/SeparableAgent/PolicyLoss", pg_loss_S.item(), iteration)
        writer.add_scalar("1-Training-Stats/SeparableAgent/ValueLoss", v_loss_S.item(), iteration)
        writer.add_scalar("1-Training-Stats/SeparableAgent/EntropyLoss", entropy_loss_S.item(), iteration)
        writer.add_scalar("1-Training-Stats/SeparableAgent/ApproxKL", approx_kl_S.item(), iteration)
        writer.add_scalar("1-Training-Stats/SeparableAgent/ClipFrac", np.mean(clipfracs_S), iteration)

        # optimizing the entangled agent
        clipfracs_E = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                _, newlogprob_E, entropy_E, newvalue_E = entangledAgent.get_action_and_value(b_obs_E[mb_inds], b_actions_E[mb_inds])
                logratio_E = newlogprob_E - b_logprobs_E[mb_inds]
                ratio_E = logratio_E.exp()
                with torch.no_grad():
                    approx_kl_E = (logprobs_ent[mb_inds] - newlogprob_E).mean()
                    clipfracs_E += [((ratio_E - 1.0).abs() > args.clip_coef).float().mean().item()]
                mb_advantages_E = b_advantages_E[mb_inds]
                if args.norm_adv:
                    mb_advantages_E = (mb_advantages_E - mb_advantages_E.mean()) / (mb_advantages_E.std() + 1e-8)
                
                # policy loss
                pg_loss1_E = -mb_advantages_E * ratio_E
                pg_loss2_E = -mb_advantages_E * torch.clamp(ratio_E, 1.0 - args.clip_coef, 1.0 + args.clip_coef)
                pg_loss_E = torch.max(pg_loss1_E, pg_loss2_E).mean()
                # value loss
                newvalue_E = newvalue_E.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped_E = (newvalue_E - b_returns_S[mb_inds]) ** 2
                    v_clipped_E = b_values_E[mb_inds] + torch.clamp(newvalue_E - b_values_E[mb_inds], -args.clip_coef, args.clip_coef)
                    v_loss_clipped_E = (v_clipped_E - b_returns_S[mb_inds]) ** 2
                    v_loss_max_E = torch.max(v_loss_unclipped_E, v_loss_clipped_E)
                    v_loss_E = 0.5 * v_loss_max_E.mean()
                else:
                    v_loss_E = 0.5 * ((newvalue_E - b_returns_S[mb_inds]) ** 2).mean()
                
                entropy_loss_E = entropy_E.mean()
                loss_E = pg_loss_E - args.ent_coef * entropy_loss_E + args.vf_coef * v_loss_E

                #old_param = entangledAgent.state_dict()['backbone.network.7.weight']
                #print("old_param:", old_param)
                optimizerE.zero_grad()
                loss_E.backward()
                nn.utils.clip_grad_norm_(entangledAgent.parameters(), args.max_grad_norm)
                optimizerE.step()
                #new_param = entangledAgent.state_dict()['backbone.network.7.weight']
                #print("new_param:", new_param)
                if args.clamp_actor_weights:
                    entangledAgent.state_dict()["actor.0.q_params"].data.clamp_(-np.pi, np.pi)
            
        # calculate the explained variance of the entangled agent
        y_pred_E, y_true_E = b_values_E.cpu().numpy(), b_returns_S.cpu().numpy()
        var_y_E = np.var(y_true_E)
        explained_var_E = np.nan if var_y_E == 0 else 1 - np.var(y_true_E - y_pred_E) / var_y_E
        # log the data of the entangled agent
        writer.add_scalar("1-Training-Stats/EntangledAgent/ExplainedVariance", explained_var_E, iteration)
        writer.add_scalar("1-Training-Stats/EntangledAgent/PolicyLoss", pg_loss_E.item(), iteration)
        writer.add_scalar("1-Training-Stats/EntangledAgent/ValueLoss", v_loss_E.item(), iteration)
        writer.add_scalar("1-Training-Stats/EntangledAgent/EntropyLoss", entropy_loss_E.item(), iteration)
        writer.add_scalar("1-Training-Stats/EntangledAgent/ApproxKL", approx_kl_E.item(), iteration)
        writer.add_scalar("1-Training-Stats/EntangledAgent/ClipFrac", np.mean(clipfracs_E), iteration)
                    
        
        
        # Time estimation
        iter_avg_time = (time.time() - start_time) / iteration
        print(f"------Single Iteration Time: {iter_avg_time:.4f} seconds, time remaining: {str(datetime.timedelta(seconds=iter_avg_time*(args.num_iterations - iteration)))}")
            







    