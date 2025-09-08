import pennylane as qml

import torch
import torch.nn as nn

import numpy as np

from typing import List, Tuple, Union, Callable



def single_qubit_U3_layer(x_i, w_0, w_1, w_2, b_0, b_1, b_2, qubit):
    """
    Single-qubit U3 layer for encoding linear-transformed single feature x_i.
    """
    qml.U3(w_0 * x_i + b_0, w_1 * x_i + b_1, w_2 * x_i + b_2, wires=qubit)

def make_separable_circ(n_layers: int)->Callable:
    """
    Creates a parameterised circuit with no entanglement.
    """
    
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


def make_entangled_circ(n_layers: int)->Callable:
    """
    Creates a parameterised circuit with entanglement.
    The only difference from the separable circuit is the addition of CZ gates
    after each layer of single-qubit U3 gates, in a ring topology.
    """

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
            # entanglement layer - CZ gates in a ring topology
            for i in range(8):
                qml.CZ(wires=[i, (i+1)%8])

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


class ElementwiseScaleShift(nn.Module):
    """
    Element-wise affine transform: y = x * scale + shift

    Parameters are shaped for broadcasting against the input. Typical usage is to
    pass the feature dimension size so the parameters have shape (features,) and
    apply along the last dimension, but any broadcastable shape works.

    Args:
        shape: int or tuple defining the parameter shape (broadcastable to inputs).
        init_scale: initial value for scale (default 1.0).
        init_shift: initial value for shift (default 0.0).
        learnable: if True, scale/shift are learnable parameters; otherwise buffers.
    """

    def __init__(
        self,
        shape: Union[int, Tuple[int, ...]],
        init_scale: float = 1.0,
        init_shift: float = 0.0,
        learnable: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(shape, int):
            param_shape: Tuple[int, ...] = (shape,)
        else:
            param_shape = tuple(shape)

        if learnable:
            self.scale = nn.Parameter(torch.full(param_shape, float(init_scale)))
            self.shift = nn.Parameter(torch.full(param_shape, float(init_shift)))
        else:
            self.register_buffer("scale", torch.full(param_shape, float(init_scale)))
            self.register_buffer("shift", torch.full(param_shape, float(init_shift)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale + self.shift

    def extra_repr(self) -> str:
        return f"shape={tuple(self.scale.shape)}"

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
        self.affine = ElementwiseScaleShift(shape=8)

    def forward(self, x):
        x = x * torch.pi * 2 # Scale inputs to [0, 2π]
        out = self.qfunc(x, self.params).to(x.dtype)
        out = self.affine(out)
        return out

if __name__ == "__main__":
    from pufferlib import vector
    from pufferlib.ocean import env_creator

    # Example usage
    n_layers = 6
    sep_circuit = make_separable_circ(n_layers)
    
    x = torch.tensor(np.random.rand(12, 8))
    params = torch.tensor(np.random.rand(n_layers, 8, 6), requires_grad=True)
    result = sep_circuit(x, params)
    print("Separable circuit output shape:", result.shape)
    print("Separable circuit output:\n", result)

    sep_backbone = SeparableBackbone(n_layers, 8, 8)
    output = sep_backbone(x)
    print("SeparableBackbone output shape:", output.shape)
    print("SeparableBackbone output:\n", output)

    ent_circ = make_entangled_circ(n_layers)
    ent_result = ent_circ(x, params)
    print("Entangled circuit output shape:", ent_result.shape)
    print("Entangled circuit output:\n", ent_result)

