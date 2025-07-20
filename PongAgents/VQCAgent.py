import pennylane as qml

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import numpy as np

from typing import List, Tuple, Union, Callable

ket0bra0 = np.outer(np.array([1., 0.]), np.array([1., 0.]))

def create_GHZ_state(qubit_list: List[int]):
    """
    Creates a GHZ state on the specified qubits.
    """
    qml.Hadamard(qubit_list[0])
    for qubit in qubit_list[1:]:
        qml.CNOT(wires=[qubit_list[0], qubit])

def create_graph_state(qubit_list: List[int], edges: List[Tuple[int, int]]):
    """
    Creates a graph state on the specified qubits with given edges.
    """
    for qubit in qubit_list:
        qml.Hadamard(qubit)
    for edge in edges:
        node_1, node_2 = edge
        assert node_1 in qubit_list and node_2 in qubit_list, "Both nodes must be in the qubit list. Got: {}, {}".format(node_1, node_2)
        qml.CZ(wires=[node_1, node_2])

def V_i_l(x_i, theta_0, theta_1, theta_2, qubit):
    """
    Single-qubit PQC for data-encoding and parameterized gates.
    """
    qml.U3(theta_0, theta_1, theta_2, wires=qubit)
    qml.RX(x_i, wires=qubit)
    qml.RY(x_i, wires=qubit)
    qml.RZ(x_i, wires=qubit)

def U3Layer(thetas, qubit_list: List[int]):
    """
    Applies a layer of U3 gates to the specified qubits.
    """
    for i, qubit in enumerate(qubit_list):
        theta_0, theta_1, theta_2 = thetas[i]
        qml.U3(theta_0, theta_1, theta_2, wires=qubit)

def make_entangled_circ_ghz(n_layers: int, post_select = True)->Callable:
    """
    Creates a parameterised circuit starting with a GHZ state.
    """

    assert n_layers > 1, "Number of layers must be greater than 1. Got: {}".format(n_layers)

    device = qml.device("default.qubit", wires = 8)
    qubit_list = [1, 0, 2, 3, 4, 5, 6, 7] # qubit #1 is the control qubit for GHZ state since it is the right paddle qubit
    # measurement observables
    if post_select:
        # project all the qubits other than the second qubit onto |0>
        meas_x = qml.Hermitian(ket0bra0, wires=0)@qml.PauliX(1)@qml.Hermitian(ket0bra0, wires=2)@qml.Hermitian(ket0bra0, wires=3)@qml.Hermitian(ket0bra0, wires=4)@qml.Hermitian(ket0bra0, wires=5)@qml.Hermitian(ket0bra0, wires=6)@qml.Hermitian(ket0bra0, wires=7)
        meas_y = qml.Hermitian(ket0bra0, wires=0)@qml.PauliY(1)@qml.Hermitian(ket0bra0, wires=2)@qml.Hermitian(ket0bra0, wires=3)@qml.Hermitian(ket0bra0, wires=4)@qml.Hermitian(ket0bra0, wires=5)@qml.Hermitian(ket0bra0, wires=6)@qml.Hermitian(ket0bra0, wires=7)
        meas_z = qml.Hermitian(ket0bra0, wires=0)@qml.PauliZ(1)@qml.Hermitian(ket0bra0, wires=2)@qml.Hermitian(ket0bra0, wires=3)@qml.Hermitian(ket0bra0, wires=4)@qml.Hermitian(ket0bra0, wires=5)@qml.Hermitian(ket0bra0, wires=6)@qml.Hermitian(ket0bra0, wires=7)
    else:
        meas_x = qml.PauliX(1)
        meas_y = qml.PauliY(1)
        meas_z = qml.PauliZ(1)

    def circuit(x, params):
        """
        input x has shape (, 8)
        params has shape (n_layers, 8, 3)
        """
        create_GHZ_state(qubit_list)

        for l in range(n_layers - 1):
            for i in range(8):
                V_i_l(x[...,i], params[l][i][0], params[l][i][1], params[l][i][2], i)
        
        U3Layer(params[l-1], qubit_list)

        qml.adjoint(create_GHZ_state)(qubit_list)

        return [qml.expval(meas_x), qml.expval(meas_y), qml.expval(meas_z)]

    compiled_circuit = qml.compile(circuit)
    qnode = qml.QNode(compiled_circuit, device, interface='torch')

    def qfunc(x, params):
        """
        QNode function that takes input x and parameters params.
        """
        assert x.shape[-1] == 8, "Input x must have shape (..., 8). Got: {}".format(x.shape)
        assert params.shape == (n_layers, 8, 3), "Parameters must have shape (n_layers, 8, 3). Got: {}".format(params.shape)
        
        circ_out = qnode(x, params)
        circ_out = torch.stack(circ_out)
        circ_out = torch.einsum("ij->ji", circ_out)
        return circ_out
    
    return qfunc

class GHZAgent(nn.Module):
    def __init__(self, n_layers, post_select, env):
        super().__init__()
        self.single_action_dim = env.action_space.n
        self.observation_dim = env.single_observation_space.shape[0]
        assert self.single_action_dim == 3 # only 3 actions: up, down, no action
        assert self.observation_dim == 8 # paddly_yl, paddle_yr, ball_x, ball_y, ball_vx, ball_vy, score_l, score_r

        self.qfunc = make_entangled_circ_ghz(n_layers, post_select=post_select)
        self.params = nn.Parameter(
            torch.rand((n_layers, 8, 3), requires_grad=True)
            )
    
    def forward(self, x):
        x = x * torch.pi
        out = self.qfunc(x, self.params).to(x.dtype)
        return out



if __name__ == "__main__":
    # Example usage
    n_layers = 6
    circuit = make_entangled_circ_ghz(n_layers, post_select=False)
    
    x = torch.tensor(np.random.rand(12, 8))
    params = torch.tensor(np.random.rand(n_layers, 8, 3), requires_grad=True)
    result = circuit(x, params)
    print("Circuit output shape:", result.shape)
    print("Circuit output:\n", result)
