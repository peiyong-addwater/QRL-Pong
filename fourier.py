import pennylane as qml
import pandas as pd
from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from typing import List, Tuple, Union, Callable
from functools import partial


plt.style.use(['science','nature'])
# set legnd font size globally
plt.rcParams['legend.fontsize'] = 10
# set axes label font size globally
plt.rcParams['axes.labelsize'] = 12
# set title font size globally
plt.rcParams['axes.titlesize'] = 12
# set the legend location globally: upper left
plt.rcParams['legend.loc'] = 'upper left'

