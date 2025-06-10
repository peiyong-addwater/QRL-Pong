import math
import os
import shlex
import subprocess
import uuid
from dataclasses import dataclass
from typing import List, Optional

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

command_list = [
    # from scratch
    # "uv run ppo_1P_Pong.py --agent_type 'entangledActor' --from_scratch --clamp_actor_weights",
    "uv run ppo_1P_Pong.py --agent_type 'entangledActor' --from_scratch --no-clamp_actor_weights --seed 0",
    "uv run ppo_1P_Pong.py --agent_type 'entangledActor' --from_scratch --no-clamp_actor_weights --seed 1",
    "uv run ppo_1P_Pong.py --agent_type 'entangledActor' --from_scratch --no-clamp_actor_weights --seed 2",
    "uv run ppo_1P_Pong.py --agent_type 'entangledActor' --from_scratch --no-clamp_actor_weights --seed 3",
    "uv run ppo_1P_Pong.py --agent_type 'entangledActor' --from_scratch --no-clamp_actor_weights --seed 4",
    # "uv run ppo_1P_Pong.py --agent_type 'separableActor' --from_scratch --clamp_actor_weights",
    "uv run ppo_1P_Pong.py --agent_type 'separableActor' --from_scratch --no-clamp_actor_weights --seed 0",
    "uv run ppo_1P_Pong.py --agent_type 'separableActor' --from_scratch --no-clamp_actor_weights --seed 1",
    "uv run ppo_1P_Pong.py --agent_type 'separableActor' --from_scratch --no-clamp_actor_weights --seed 2",
    "uv run ppo_1P_Pong.py --agent_type 'separableActor' --from_scratch --no-clamp_actor_weights --seed 3",
    "uv run ppo_1P_Pong.py --agent_type 'separableActor' --from_scratch --no-clamp_actor_weights --seed 4",
]

if __name__ == "__main__":
    from concurrent.futures import ThreadPoolExecutor

    n_workers = len(command_list) if len(command_list) < 30 else 30

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


