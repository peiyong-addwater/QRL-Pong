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
    "uv run ppo_1P_Pong.py --agent_type 'entangledActor' --from_scratch --clamp_actor_weights",
    "uv run ppo_1P_Pong.py --agent_type 'entangledActor' --from_scratch --no-clamp_actor_weights",
    "uv run ppo_1P_Pong.py --agent_type 'separableActor' --from_scratch --clamp_actor_weights",
    "uv run ppo_1P_Pong.py --agent_type 'separableActor' --from_scratch --no-clamp_actor_weights",
]