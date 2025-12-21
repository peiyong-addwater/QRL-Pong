import pennylane as qml
import pandas as pd
from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from typing import List, Tuple, Union, Callable
from functools import partial

from PongAgents.VQCBackbones import (
    single_qubit_U3_layer,
)

plt.style.use(['science','nature'])
# set legnd font size globally
plt.rcParams['legend.fontsize'] = 10
# set axes label font size globally
plt.rcParams['axes.labelsize'] = 12
# set title font size globally
plt.rcParams['axes.titlesize'] = 12
# set the legend location globally: upper left
plt.rcParams['legend.loc'] = 'upper left'


def make_separable_circ(n_layers:int, obs_idx: int)-> Callable:

    assert obs_idx in range(8), "obs_idx must be between 0 and 7"

    device = qml.device("default.qubit", wires = 8)
    @qml.qnode(device, interface='torch')
    def circuit(parames, x):
        """
        input x has shape (, 8)
        params has shape (n_layers, 8, 6)
        """
        for l in range(n_layers):
            # single-qubit U3 gates with data-encoding
            for i in range(8):
                w_0, w_1, w_2, b_0, b_1, b_2 = params[l][i]
                single_qubit_U3_layer(x[...,i], w_0, w_1, w_2, b_0, b_1, b_2, i)
            # entanglement layer - none for separable circuit

        # measure only one qubit
        return qml.expval(qml.PauliX(obs_idx))
    
    return circuit

if __name__ == "__main__":

    # test the circuit
    n_layers = 2
    circuit = make_separable_circ(n_layers, obs_idx=0)

    # random input data
    x = torch.rand(8) * 2 * np.pi  # batch of 4 samples, each with 8 features


    # random parameters
    params = torch.rand((n_layers, 8, 6)) * 2 * np.pi

    # execute the circuit
    res = qml.fourier.coefficients(partial(circuit, params), 8, 2)
    print(res)