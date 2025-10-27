# Collects observations from the environment.
# Use the 4096P-model for action selection.

import torch
import numpy as np
import os

from PongAgents import PongClassicalAgent4096PBackbone

from pufferlib import vector
from pufferlib.ocean import env_creator

# some hyperparameters
NUM_ENVS = 1024
NUM_STEPS = 512 # number of policy rollout steps per environment
TRAINED_MODEL_FILENAME = "Pong1PCB2L4096P__seed_0_1760930165.pt"
MODEL_FOLDER = os.path.join("trained-models", "Pong1PClassicalBaseline")
MODEL_PATH = os.path.join(MODEL_FOLDER, TRAINED_MODEL_FILENAME)
SAVE_FOLDER = "collected_obs"
SAVE_PATH = os.path.join(SAVE_FOLDER, "obs_collection_Classical4096P.npy")

# check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# env setup
env_name = 'puffer_pong'
env_creator = env_creator(env_name)
envs = vector.make(env_creator, num_envs=1, num_workers=1, batch_size=1, backend=vector.Multiprocessing, env_kwargs={'num_envs': NUM_ENVS, 'log_interval':1})

# initialize agent
agent = PongClassicalAgent4096PBackbone(env=envs).to(device)
# load trained model from the saved state_dict
agent.load_state_dict(torch.load(MODEL_PATH, map_location=device))
print(f"Loaded trained model from {MODEL_PATH}")

# storage setup
obs = torch.zeros((NUM_STEPS, NUM_ENVS) + envs.single_observation_space.shape).to(device)

# start the game
next_obs, _ = envs.reset(seed=0)
next_obs = torch.Tensor(next_obs).to(device)
next_done = False

for step in range(NUM_STEPS):
    # store the observation
    obs[step] = next_obs
    with torch.no_grad():
        action, logprob, _, value = agent.get_action_and_value(next_obs)
    next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
    next_done = np.logical_or(terminations, truncations)
    next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)
    print(f"Step {step+1}/{NUM_STEPS}", end='\r')

# close environments
envs.close()
# save the collected observations
os.makedirs(SAVE_FOLDER, exist_ok=True)
# print shape of saved observations
print(f"Collected observations shape: {obs.shape}")
# flatten the first two dimensions (NUM_STEPS, NUM_ENVS)
obs = obs.reshape(-1, 8)  # assuming observation shape is (8,)
print(f"Flattened saved observations shape: {obs.shape}")
np.save(SAVE_PATH, obs.cpu().numpy())