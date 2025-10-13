import math
import os
import shlex
import subprocess
import uuid
from dataclasses import dataclass
from typing import List, Optional
import re

import requests
import tyro

def run_experiment(command: str):
    command_list = shlex.split(command)
    print(f"running {command}")

    # Use subprocess.PIPE to capture the output
    fd = subprocess.Popen(command_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, errors = fd.communicate()

    return_code = fd.returncode
    assert return_code == 0, f"Command failed with error: {errors.decode('utf-8')}"

    # Convert bytes to string and strip leading/trailing whitespaces
    return output.decode("utf-8").strip()

seeds_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

n_layers_list = [1, 2]

command_list = []


separable_finished = set()
entangled_finished = set()
entangled_trainable_crz_finished = set()
entangled_trainable_rzz_finished = set()
entangled_trainable_cp_finished = set()

# the (QLayers, seed) combo that has been finished 
trained_models_dir = os.path.join("trained-models", "Pong1PModels")
if not os.path.exists(trained_models_dir):
    os.makedirs(trained_models_dir)
for trained_model_file in os.listdir(trained_models_dir):
    # process entangled backbone models
    if "entangled" in trained_model_file:
        # find teh number of layers and seed with regex
        match = re.search(r"QLayers_(\d+)___seed_(\d+)_", trained_model_file)
        if match:
            n_layers = int(match.group(1))
            seed = int(match.group(2))
            entangled_finished.add((n_layers, seed))
            print(f"Found entangled actor model: n_layers={n_layers}, seed={seed}. Excluding from commands.")
    # process separable backbone models
    if "separable" in trained_model_file:
        match = re.search(r"QLayers_(\d+)___seed_(\d+)_", trained_model_file)
        if match:
            n_layers = int(match.group(1))
            seed = int(match.group(2))
            separable_finished.add((n_layers, seed))
            print(f"Found separable actor model: n_layers={n_layers}, seed={seed}. Excluding from commands.")
    # process entangled backbone with trainable ZZ models
    if "entangled_trainable_crz" in trained_model_file:
        match = re.search(r"QLayers_(\d+)___seed_(\d+)_", trained_model_file)
        if match:
            n_layers = int(match.group(1))
            seed = int(match.group(2))
            entangled_trainable_crz_finished.add((n_layers, seed))
            print(f"Found entangled_trainable_zz actor model: n_layers={n_layers}, seed={seed}. Excluding from commands.")
    # process entangled backbone with trainable RZZ models
    if "entangled_trainable_rzz" in trained_model_file:
        match = re.search(r"QLayers_(\d+)___seed_(\d+)_", trained_model_file)
        if match:
            n_layers = int(match.group(1))
            seed = int(match.group(2))
            entangled_trainable_rzz_finished.add((n_layers, seed))
            print(f"Found entangled_trainable_rzz actor model: n_layers={n_layers}, seed={seed}. Excluding from commands.")
    # process entangled backbone with trainable CP models
    if "entangled_trainable_cp" in trained_model_file:
        match = re.search(r"QLayers_(\d+)___seed_(\d+)_", trained_model_file)
        if match:
            n_layers = int(match.group(1))
            seed = int(match.group(2))
            entangled_trainable_cp_finished.add((n_layers, seed))
            print(f"Found entangled_trainable_cp actor model: n_layers={n_layers}, seed={seed}. Excluding from commands.")

for n_layers in n_layers_list:
    for seed in seeds_list:
        if (n_layers, seed) not in entangled_finished:
            command_list.append(
                f"uv run Pong1PQuantum.py --cuda_device 0 --num_envs 32 --num_steps 128 --agent_type 'entangled' --seed {seed} --n_layers {n_layers}"
            )
        if (n_layers, seed) not in separable_finished:
            command_list.append(
                f"uv run Pong1PQuantum.py --cuda_device 0 --num_envs 32 --num_steps 128 --agent_type 'separable' --seed {seed} --n_layers {n_layers}"
            )
        if (n_layers, seed) not in entangled_trainable_crz_finished:
            command_list.append(
                f"uv run Pong1PQuantum.py --cuda_device 0 --num_envs 32 --num_steps 128 --agent_type 'entangled_trainable_crz' --seed {seed} --n_layers {n_layers}"
            )
        if (n_layers, seed) not in entangled_trainable_rzz_finished:
            command_list.append(
                f"uv run Pong1PQuantum.py --cuda_device 0 --num_envs 32 --num_steps 128 --agent_type 'entangled_trainable_rzz' --seed {seed} --n_layers {n_layers}"
            )
        if (n_layers, seed) not in entangled_trainable_cp_finished:
            command_list.append(
                f"uv run Pong1PQuantum.py --cuda_device 0 --num_envs 32 --num_steps 128 --agent_type 'entangled_trainable_cp' --seed {seed} --n_layers {n_layers}"
            )

if __name__ == "__main__":
    from concurrent.futures import ThreadPoolExecutor

    NUM_CPUS = 25

    n_workers = len(command_list) if len(command_list) < NUM_CPUS else NUM_CPUS

    print("======= commands to run:")
    for command in command_list:
        print(command)
    print("======= number of workers: ", n_workers)
    print("======= number of commands: ", len(command_list))

    executor = ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="Pong1P-worker-")

    for command in command_list:
        executor.submit(run_experiment, command)
    executor.shutdown(wait=True)
    print("======= all commands finished")
