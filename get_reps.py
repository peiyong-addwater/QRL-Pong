import os
import re
import torch
import numpy as np

from pufferlib import vector
from pufferlib.ocean import env_creator

from PongAgents import PongHybridAgent

# device configuration
device = torch.device("cpu")

# path to the collected observations
OBS_PATH = os.path.join("collected_obs", "obs_collection_Classical4096P.npy")

# load the collected observations from npy file
observations = np.load(OBS_PATH)
observations = torch.Tensor(observations).to(device)
print(f"Loaded collected observations from {OBS_PATH} with shape {observations.shape}")

# path to save the representations
REPS_FOLDER = "collected_reps"
os.makedirs(REPS_FOLDER, exist_ok=True)

# path to the trained quantum models
MODEL_FOLDER = os.path.join("trained-models", "Pong1PModels")

# get the files in the model folder
model_files = os.listdir(MODEL_FOLDER)

# dummy environment to get observation shape
env_name = 'puffer_pong'
env_creator = env_creator(env_name)
envs = vector.make(env_creator, num_envs=1, num_workers=1, batch_size=1, backend=vector.Multiprocessing, env_kwargs={'num_envs': 128, 'log_interval':1})


# generate representations for each model
for model_file in model_files:
    # load the trained model
    model_path = os.path.join(MODEL_FOLDER, model_file)
    # extract agent type from the model file name
    # model file name format: Pong1PQFM_XObs_<AgentType>_QLayers_<num_layers>___seed_<seed>_<timestamp>.pth
    # possible <AgentType>: separable, entangled, entangled_trainable_rzz
    match = re.search(r'Pong1PQFM_XObs_([a-z_]+)_QLayers_\d+___seed_\d+_', model_file)
    if match:
        agent_type = match.group(1)
        num_layers = int(re.search(r'_QLayers_(\d+)', model_file).group(1))
        print(f"Processing model: {model_file} | Agent Type: {agent_type} | Num Layers: {num_layers}")
        agent = PongHybridAgent(agent_type=agent_type, env = envs, agent_args = {"n_layers": num_layers}).to(device)
        print(f"Loading model from {model_path}")
        agent.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded successfully.")
        with torch.no_grad():
            # get representations
            reps = agent.get_representation(observations)
            print(f"Generated representations with shape: {reps.shape}")


