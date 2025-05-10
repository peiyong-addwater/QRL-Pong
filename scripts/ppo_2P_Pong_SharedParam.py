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

