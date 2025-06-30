import math
import os
import shlex
import subprocess
import uuid
from dataclasses import dataclass
from typing import List, Optional

import requests
import tyro


@dataclass
class Args:
    env_ids: List[str]
    """the ids of the environment to compare"""
    command: str
    """the command to run"""
    num_seeds: int = 3
    """the number of random seeds"""
    start_seed: int = 1
    """the number of the starting seed"""
    workers: int = 0
    """the number of workers to run benchmark experimenets"""


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



if __name__ == "__main__":
    args = tyro.cli(Args)
    commands = []
    for seed in range(0, args.num_seeds):
        for env_id in args.env_ids:
            commands += [" ".join([args.command, "--env-id", env_id, "--seed", str(args.start_seed + seed)])]

    print("======= commands to run:")
    for command in commands:
        print(command)

    if args.workers > 0 and args.slurm_template_path is None:
        from concurrent.futures import ThreadPoolExecutor

        executor = ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="cleanrl-benchmark-worker-")
        for command in commands:
            executor.submit(run_experiment, command)
        executor.shutdown(wait=True)
    else:
        print("not running the experiments because --workers is set to 0; just printing the commands to run")