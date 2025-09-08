import pennylane as qml

import torch
import torch.nn as nn

import numpy as np

from typing import List, Tuple, Union, Callable

ket0bra0 = np.outer(np.array([1., 0.]), np.array([1., 0.]))


def single_qubit_U3_layer(x_i, w_0, w_1, w_2, b_0, b_1, b_2, qubit):
    """
    Single-qubit U3 layer for encoding linear-transformed single feature x_i.
    """
    qml.U3(w_0 * x_i + b_0, w_1 * x_i + b_1, w_2 * x_i + b_2, wires=qubit)

def make_separable_circ(n_layers: int)->Callable:
    """
    Creates a parameterised circuit with no entanglement.
    """
    
    assert n_layers > 1, "Number of layers must be greater than 1. Got: {}".format(n_layers)

    device = qml.device("default.qubit", wires = 8)
    
    def circuit(x, params):
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

        # measure all qubits in Z basis
        return [qml.expval(qml.PauliZ(i)) for i in range(8)]
    
    compiled_circuit = qml.compile(circuit)
    qnode = qml.QNode(compiled_circuit, device, interface='torch')
    
    def qfunc(x, params):
        """
        QNode function that takes input x and parameters params.
        """
        assert x.shape[-1] == 8, "Input x must have shape (..., 8). Got: {}".format(x.shape)
        assert params.shape == (n_layers, 8, 6), "Parameters must have shape (n_layers, 8, 6). Got: {}".format(params.shape)
        circ_out = qnode(x, params)
        circ_out = torch.stack(circ_out)
        circ_out = torch.einsum("ij->ji", circ_out)
        return circ_out
    
    return qfunc




























class SeparableBackbone(nn.Module):
    def __init__(self, n_layers, output_dim, observation_space, edge_list=None):
        super().__init__()
        self.output_dim = output_dim
        self.observation_dim = observation_space
        assert self.output_dim == 8 # only 8 output dim
        assert self.observation_dim == 8 # paddly_yl, paddle_yr, ball_x, ball_y, ball_vx, ball_vy, score_l, score_r

        self.qfunc = make_separable_circ(n_layers)
        self.params = nn.Parameter(
            torch.rand((n_layers, 8, 6), requires_grad=True)
            )

    def forward(self, x):
        x = x * torch.pi
        out = self.qfunc(x, self.params).to(x.dtype)
        return out

if __name__ == "__main__":
    from pufferlib import vector
    from pufferlib.ocean import env_creator

    # Example usage
    n_layers = 6
    circuit = make_separable_circ(n_layers)
    
    x = torch.tensor(np.random.rand(12, 8))
    params = torch.tensor(np.random.rand(n_layers, 8, 6), requires_grad=True)
    result = circuit(x, params)
    print("Circuit output shape:", result.shape)
    print("Circuit output:\n", result)
