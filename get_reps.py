import os
import re
import torch
import numpy as np
import json

from pufferlib import vector
from pufferlib.ocean import env_creator

from PongAgents import PongHybridAgent, PongClassicalAgent4096PBackbone, PongClassicalAgent64PBackbone, PongClassicalAgent128PBackbone, PongClassicalAgent256PBackbone, PongClassicalAgent336PBackbone

# device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
CLASSICAL_MODEL_FOLDER = os.path.join("trained-models", "Pong1PClassicalBaseline")

# get the files in the model folder
model_files = os.listdir(MODEL_FOLDER)
classical_model_files = os.listdir(CLASSICAL_MODEL_FOLDER)

# dummy environment to get observation shape
env_name = 'puffer_pong'
env_creator = env_creator(env_name)
envs = vector.make(env_creator, num_envs=1, num_workers=1, batch_size=1, backend=vector.Multiprocessing, env_kwargs={'num_envs': 128, 'log_interval':1})

# a dictionary to store the path to the representations for each model
reps_paths_dict = {}
reps_paths_dict["separable"] = {}
reps_paths_dict["entangled"] = {}
reps_paths_dict["entangled_trainable_rzz"] = {}
reps_paths_dict["classical"] = {}
reps_paths_dict["classical"]["64P"] = []
reps_paths_dict["classical"]["4096P"] = []
reps_paths_dict["classical"]["128P"] = []
reps_paths_dict["classical"]["256P"] = []
reps_paths_dict["classical"]["336P"] = []

# generate representations for each classical model
file_count = 0
for model_file in classical_model_files:
    # load the trained model
    model_path = os.path.join(CLASSICAL_MODEL_FOLDER, model_file)

    # filename has the format:
    # Pong1PCB2L<64P or 4096P>__seed_<seed>_<timestamp>.pth
    # extract the seed from the file name
    match = re.search(r'Pong1PCB2L(64P|128P|256P|336P|4096P)__seed_(\d+)_', model_file)
    if match:
        print(f"======File Count: {file_count+1}/{len(classical_model_files)} Processing Classical model: {model_file}======")
        if "64P" in model_file:
            agent = PongClassicalAgent64PBackbone(env=envs).to(device)
            print("Loading 64P Classical Backbone model")
            agent.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Model loaded successfully.")
            with torch.no_grad():
                # get representations
                reps = agent.get_representations(observations)
                print(f"Generated representations with shape: {reps.shape}")
                # save representations
                reps_path = os.path.join(REPS_FOLDER, f"Pong1PCReps_64P_seed_{match.group(2)}.npy")
                np.save(reps_path, reps.cpu().numpy())
                print(f"Saved representations to {reps_path}")
                reps_paths_dict["classical"]["64P"].append(reps_path)
                print("---------------------------------------------------")
        elif "4096P" in model_file:
            agent = PongClassicalAgent4096PBackbone(env=envs).to(device)
            print("Loading 4096P Classical Backbone model")
            agent.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Model loaded successfully.")
            with torch.no_grad():
                # get representations
                reps = agent.get_representations(observations)
                print(f"Generated representations with shape: {reps.shape}")
                # save representations
                reps_path = os.path.join(REPS_FOLDER, f"Pong1PCReps_4096P_seed_{match.group(2)}.npy")
                np.save(reps_path, reps.cpu().numpy())
                print(f"Saved representations to {reps_path}")
                reps_paths_dict["classical"]["4096P"].append(reps_path)
                print("---------------------------------------------------")
        elif "128P" in model_file:
            agent = PongClassicalAgent128PBackbone(env=envs).to(device)
            print("Loading 128P Classical Backbone model")
            agent.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Model loaded successfully.")
            with torch.no_grad():
                # get representations
                reps = agent.get_representations(observations)
                print(f"Generated representations with shape: {reps.shape}")
                # save representations
                reps_path = os.path.join(REPS_FOLDER, f"Pong1PCReps_128P_seed_{match.group(2)}.npy")
                np.save(reps_path, reps.cpu().numpy())
                print(f"Saved representations to {reps_path}")
                reps_paths_dict["classical"]["128P"].append(reps_path)
                print("---------------------------------------------------")
        elif "256P" in model_file:
            agent = PongClassicalAgent256PBackbone(env=envs).to(device)
            print("Loading 256P Classical Backbone model")
            agent.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Model loaded successfully.")
            with torch.no_grad():
                # get representations
                reps = agent.get_representations(observations)
                print(f"Generated representations with shape: {reps.shape}")
                # save representations
                reps_path = os.path.join(REPS_FOLDER, f"Pong1PCReps_256P_seed_{match.group(2)}.npy")
                np.save(reps_path, reps.cpu().numpy())
                print(f"Saved representations to {reps_path}")
                reps_paths_dict["classical"]["256P"].append(reps_path)
                print("---------------------------------------------------")
        elif "336P" in model_file:
            agent = PongClassicalAgent336PBackbone(env=envs).to(device)
            print("Loading 336P Classical Backbone model")
            agent.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Model loaded successfully.")
            with torch.no_grad():
                # get representations
                reps = agent.get_representations(observations)
                print(f"Generated representations with shape: {reps.shape}")
                # save representations
                reps_path = os.path.join(REPS_FOLDER, f"Pong1PCReps_336P_seed_{match.group(2)}.npy")
                np.save(reps_path, reps.cpu().numpy())
                print(f"Saved representations to {reps_path}")
                reps_paths_dict["classical"]["336P"].append(reps_path)
                print("---------------------------------------------------")
        print()
    file_count += 1




# generate representations for each quantum model
file_count = 0
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
        seed = int(re.search(r'___seed_(\d+)_', model_file).group(1))
        
        if num_layers not in reps_paths_dict[agent_type].keys():
            reps_paths_dict[agent_type][num_layers] = []

        print(f"======File Count: {file_count+1}/{len(model_files)} Processing model: {model_file} | Agent Type: {agent_type} | Num Layers: {num_layers}======")
        agent = PongHybridAgent(agent_type=agent_type, env = envs, agent_args = {"n_layers": num_layers}).to(device)
        
        print(f"Loading model from {model_path}")
        agent.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded successfully.")
        
        with torch.no_grad():
            # get representations
            reps = agent.get_representation(observations)
            print(f"Generated representations with shape: {reps.shape}")
            # save representations
            reps_path = os.path.join(REPS_FOLDER, f"Pong1PQFMXObsReps_{agent_type}_layers_{num_layers}_seed_{seed}.npy")
            np.save(reps_path, reps.cpu().numpy())
            print(f"Saved representations to {reps_path}")
            reps_paths_dict[agent_type][num_layers].append(reps_path)
            print("---------------------------------------------------")
        print()
        
    file_count += 1

# save the paths to the representations in a json file at the root of the project
with open("reps_paths_dict.json", "w") as f:
    json.dump(reps_paths_dict, f, indent=4)

print("Saved representation paths to reps_paths_dict.json")
envs.close()

