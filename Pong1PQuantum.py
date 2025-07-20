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

from PongAgents import PongHybridAgent

@dataclass
class Args:
    exp_name: str = "Pong1PQuantum" #os.path.basename(__file__)[: -len(".py")]
    """the base name of the experiment"""
    seed: int = 1
    """random seed"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_entity: str = "addwater0315-csiro"
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    num_workers: int = 1
    """the number of workers to use for the vectorized environment"""

    # Agent specific arguments
    agent_type: str = "ghz"
    """the type of the agent, choose from ["ghz", "graph_state", "separable"] """
    post_select: bool = True
    """whether to use post-selection in the quantum agent"""
    n_layers: int = 2
    """the number of layers in the quantum actor network"""
    model_save_path: str = None
    """the path to save the model"""

    # Algorithm specific arguments
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 2
    """the number of parallel game environments"""
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

# the edge list for the graph state agent
EDGE_LIST = [(3, 0), (2, 0), (6, 0), (4, 0), (5, 0), (3, 1), (2, 1), (4, 1), (5, 1), (7, 1), (0, 1)]

if __name__ == "__main__":
    from pufferlib import vector
    from pufferlib.ocean import env_creator

    args = tyro.cli(Args)

    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    args.wandb_project_name = f"Pong1PQuantumOnly"

    run_name = f"Pong1P_{args.agent_type}_QLayers_{args.n_layers}_PostSel_{args.post_select}___seed_{args.seed}_{int(time.time())}"

    if args.model_save_path is None:
        if not os.path.exists("trained-models"):
            os.makedirs("trained-models")
            if not os.path.exists(os.path.join("trained-models", "Pong1PModels")):
                os.makedirs(os.path.join("trained-models", "Pong1PModels"))
        args.model_save_path = f"trained-models/Pong1PModels/{run_name}.pt"
    
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

    print(f"Using device: {device}")

    # env setup
    env_name = 'puffer_pong'
    env_creator = env_creator(env_name)
    envs = vector.make(env_creator, num_envs=2, num_workers=2, batch_size=1, backend=vector.Multiprocessing, env_kwargs={'num_envs': args.num_envs, 'log_interval':1})

    agent = PongHybridAgent(
        agent_type=args.agent_type,
        env=envs,
        agent_args={
            "n_layers": args.n_layers,
            "post_select": args.post_select,
            "edge_list": EDGE_LIST if args.agent_type == "graph_state" else None
        }
    ).to(device)

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # Start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        print("Episode start.............................................")
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
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

        print("Episode end...............................................")
        print(torch.sum(rewards, dim=0))
        print(torch.sum(dones, dim=0))


