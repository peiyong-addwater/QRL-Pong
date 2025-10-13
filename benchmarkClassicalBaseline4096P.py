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

command_list = []

for seed in seeds_list:
    command_list.append(
        f"uv run Pong1PClassicalBaseline4096P.py --num_envs 32 --num_steps 128 --seed {seed}"
    )

if __name__ == "__main__":
    from concurrent.futures import ThreadPoolExecutor
    import torch

    

    n_workers = len(command_list) if len(command_list) < 10 else 10

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


