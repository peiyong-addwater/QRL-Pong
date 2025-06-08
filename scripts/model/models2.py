import numpy as np

import json
from pathlib import Path
import datetime
from collections import deque
import os
import unittest

import math



# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


import pennylane as qml

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    if layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ScaleToPi(nn.Module):
    """
    Scale the input to [0, 2*pi]
    """
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.sigmoid(x)
        return torch.pi * x * 2

class SinusoidalActivation(nn.Module):
    """
    Sinusoidal activation function
    for the classical placeholder
    since the output of the quantum circuit
    are also periodic and in the range of [-1,1].
    """
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.sin(x)

class Backbone512(nn.Module):
    """
    The backbone classical NN for extracting features from the Atari game screen.
    It takes grey scale images of size 84x84 as input.
    """
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 16, 8, stride=4)), # out: 20 * 20
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, 4, stride=2)), # out: 9 * 9
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(32*9*9, 512)),
            # nn.ReLU(),
            # layer_init(nn.Linear(512, out_dim)),
            # nn.ReLU(),
            # ScaleToPi()
        )
    
    def forward(self, x):
        # expected input shape:
        # (num_envs, num_stacked_frames, 84, 84)
        return self.network(x)

class Backbone2P512(nn.Module):
    """
    The backbone classical NN for extracting features from the Atari game screen.
    It takes grey scale images of size 84x84 as input.
    """
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(6, 16, 8, stride=4)), # out: 20 * 20
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, 4, stride=2)), # out: 9 * 9
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(32*9*9, 512)),
            # nn.ReLU(),
            # layer_init(nn.Linear(512, out_dim)),
            # nn.ReLU(),
            # ScaleToPi()
        )
    
    def forward(self, x):
        x = x.clone()
        x[:, :, :, [0, 1, 2, 3]] /= 255.0
        return self.network(x.permute((0, 3, 1, 2)))

class Critic512Input(nn.Module):
    """
    The critic network for the PPO agent.
    It takes the output of the backbone as input.
    """
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Linear(512, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 1), std=1)
        )
    
    def forward(self, x):
        return self.network(x)



class Placeholder4VQC(nn.Module):
    """
    A placeholder linear layer to replace the VQC system.
    With sinusoidal activation function.
    """
    def __init__(self, in_out_dim, n_layers=None):
        super().__init__()
        self.placeholder = nn.Sequential(
            layer_init(nn.Linear(in_out_dim, in_out_dim, bias=False)),
            SinusoidalActivation()
        )
        
    def forward(self, x):
        return self.placeholder(x)

class Placeholder4VQCSineless(nn.Module):
    """
    A placeholder linear layer to replace the VQC system.
    Without sinusoidal activation function.
    """
    def __init__(self, in_out_dim, n_layers=None):
        super().__init__()
        self.placeholder = nn.Sequential(
            layer_init(nn.Linear(in_out_dim, in_out_dim, bias=False))
        )
        
    def forward(self, x):
        return self.placeholder(x)

class ClassicalPPOAgentWithPlaceholderSineless(nn.Module):
    def __init__(self, envs, n_layers=None, backbone_out_dim = 30, backbone = None , pretrained_backbone = False):
        super().__init__()
        n_actions = envs.single_action_space.n
        assert backbone is not None, "backbone is not provided"
        assert backbone.out_dim == backbone_out_dim, f"backbone output dimension {backbone.out_dim} does not match {backbone_out_dim}"
        assert pretrained_backbone == False, "pretrained_backbone is not supported"
        self.circ_qubits = math.ceil(backbone_out_dim/3)
        self.circ_out_dim = self.circ_qubits * 3
        if not pretrained_backbone:
            self.backbone = backbone
        else:
            self.backbone = backbone
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.actor = nn.Sequential(
            Placeholder4VQCSineless(backbone_out_dim, n_layers),
            layer_init(nn.Linear(self.circ_out_dim, n_actions), std=0.01) # classical post-processing
            )
        self.critic = layer_init(nn.Linear(backbone_out_dim, 1), std=1)
    
    def get_value(self, x):
        return self.critic(self.backbone(x/255.0))

    def get_action_and_value(self, x, action=None):
        hidden = self.backbone(x/255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)
    
    def get_backbone(self):
        return self.backbone


class ClassicalPPOAgentWithPlaceholder(nn.Module):
    def __init__(self, envs, n_layers=None, backbone_out_dim = 30, backbone = None , pretrained_backbone = False):
        super().__init__()
        n_actions = envs.single_action_space.n
        assert backbone is not None, "backbone is not provided"
        assert backbone.out_dim == backbone_out_dim, f"backbone output dimension {backbone.out_dim} does not match {backbone_out_dim}"
        assert pretrained_backbone == False, "pretrained_backbone is not supported"
        self.circ_qubits = math.ceil(backbone_out_dim/3)
        self.circ_out_dim = self.circ_qubits * 3
        if not pretrained_backbone:
            self.backbone = backbone
        else:
            self.backbone = backbone
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.actor = nn.Sequential(
            Placeholder4VQC(backbone_out_dim, n_layers),
            layer_init(nn.Linear(self.circ_out_dim, n_actions), std=0.01) # classical post-processing
            )
        self.critic = layer_init(nn.Linear(backbone_out_dim, 1), std=1)
    
    def get_value(self, x):
        return self.critic(self.backbone(x/255.0))

    def get_action_and_value(self, x, action=None):
        hidden = self.backbone(x/255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)
    
    def get_backbone(self):
        return self.backbone

def make_separable_vqc_sys(n_layers, n_qubits)->callable:
    """
    Construct n single-qubit VQC systems, each with n_layers.
    """
    device = qml.device("default.qubit", wires = 1)
    @qml.qnode(device, interface="torch")
    def single_qubit_circ(features, weights):
        """
        features has shape (..., 3) since we are using 3-parameter single-qubit gates to encode the data.
        weights has shape (..., n_layers, 3)  since the trainable gates are also single-qubit with 3 parameters.
        the circuit starts from the |+> state.
        """
        assert weights.shape[-2] == n_layers, f"weights shape {weights.shape} does not match n_layers {n_layers}"
        qml.Hadamard(wires=0)
        for i in range(n_layers):
            weights_layer_i = weights[...,i,:]
            # encode the data with the qml.U3 gate
            qml.Rot(features[...,0], features[...,1], features[...,2], wires=0)
            # trainable weights
            qml.U3(weights_layer_i[...,0], weights_layer_i[...,1], weights_layer_i[...,2], wires=0)
        
        return qml.expval(qml.PauliX(0)), qml.expval(qml.PauliY(0)), qml.expval(qml.PauliZ(0))
    
    single_qubit_circ_func = lambda features, weights: torch.stack(single_qubit_circ(features, weights))
    # vmap along the number of qubits
    vmap_single_qubit_circ_func = torch.vmap(single_qubit_circ_func, in_dims=(-2, -3))

    def circuit(features, weights):
        """
        features has shape (..., n_qubits, 3)
        weights has shape (..., n_qubits, n_layers, 3)
        reuse the single-qubit circuit to avoid large state vectors.
        """
        assert features.shape[-1] == 3, f"features shape {features.shape} does not match 3"
        assert weights.shape[-2] == n_layers, f"weights shape {weights.shape} does not match n_layers {n_layers}"
        assert weights.shape[-3] == n_qubits, f"weights shape {weights.shape} does not match n_qubits {n_qubits}"

        circ_out = vmap_single_qubit_circ_func(features, weights)
        circ_out = torch.einsum("ijk->kij", circ_out)
        return circ_out
    
    return circuit

class SeparableVQC(nn.Module):
    def __init__(self, in_dim, n_layers):
        super().__init__()
        self.n_qubits = math.ceil(in_dim/3)
        
        self.padded_input_dim = self.n_qubits * 3

        self.circ_out_dim = self.n_qubits * 3
        
        self.q_params = nn.Parameter(
            torch.rand((self.n_qubits, n_layers, 3), requires_grad=True)
        )
        self.circuit = make_separable_vqc_sys(n_layers, self.n_qubits)

    
    def forward(self, x):
        x = F.pad(x, (0, self.padded_input_dim - x.shape[-1]))
        x = x.reshape(-1, self.n_qubits, 3)
        out = self.circuit(x, self.q_params).to(x.dtype)
        # flatten the output
        out = out.flatten(start_dim=1)
        return out

class SeparablePPOAgent(nn.Module):
    def __init__(self, envs, n_layers=5, backbone_out_dim=30, backbone=None, pretrained_backbone=True):
        super().__init__()
        n_actions = envs.single_action_space.n
        assert backbone is not None, "backbone is not provided"
        assert backbone.out_dim == backbone_out_dim, f"backbone output dimension {backbone.out_dim} does not match {backbone_out_dim}"
        self.circ_qubits = math.ceil(backbone_out_dim/3)
        self.circ_out_dim = self.circ_qubits * 3
        self.circ_in_dim = self.circ_qubits * 3
        if not pretrained_backbone:
            self.backbone = backbone
        else:
            self.backbone = backbone
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.actor = nn.Sequential(
            layer_init(nn.Linear(512, backbone_out_dim), std=0.01), # classical pre-processing
            ScaleToPi(),
            SeparableVQC(self.circ_in_dim, n_layers),
            layer_init(nn.Linear(self.circ_out_dim, n_actions), std=0.01) # classical post-processing
            )
        self.critic = Critic512Input()
    
    def get_value(self, x):
        return self.critic(self.backbone(x/255.0))

    def get_action_and_value(self, x, action=None):
        hidden = self.backbone(x/255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)
    
def make_entangled_vqc_sys(n_layers, n_qubits)->callable:
    """
    Construct a single VQC system with n_layers and n_qubits.
    Entanglement is introduced by the CNOT gates in a ring topology.
    """
    device = qml.device("default.qubit", wires = n_qubits)
    
    def entangled_circ(features, weights):
        """
        features has shape (...,n_qubits, 3) since we are using 3-parameter single-qubit gates to encode the data.
        weights has shape (..., n_qubits, n_layers, 3)  since the trainable gates are also single-qubit with 3 parameters.
        """
        assert weights.shape[-2] == n_layers, f"weights shape {weights.shape} does not match n_layers {n_layers}"
        assert weights.shape[-3] == n_qubits, f"weights shape {weights.shape} does not match n_qubits {n_qubits}"

        for i in range(n_qubits):
            qml.Hadamard(wires=i)
        for i in range(n_layers):
            weights_layer_i = weights[...,:,i,:] # has shape (..., n_qubits, 3)
            # encode the data with the qml.Rot gate
            for j in range(n_qubits):
                qml.Rot(features[...,j,0], features[...,j,1], features[...,j,2], wires=j)
            # trainable weights
            for j in range(n_qubits):
                qml.U3(weights_layer_i[...,j,0], weights_layer_i[...,j,1], weights_layer_i[...,j,2], wires=j)
            # entanglement
            # UPDATE: Changed the simple circular entanglement to an all-to-all entanglement
            #for j in range(n_qubits):
            #    qml.CNOT(wires=[j, (j+1)%n_qubits])
            for j in range(n_qubits):
                for k in range(n_qubits):
                    if j > k:
                        qml.CZ(wires=[j, k])
        return [[qml.expval(qml.PauliX(i)), qml.expval(qml.PauliY(i)), qml.expval(qml.PauliZ(i))] for i in range(n_qubits)]
    
    # not sure whether compiling the circuit will cause trouble
    # it does flattened the measurement results
    # i.e. changed the (stacked) shape of the output from (n_qubits, 3, batch_size) to (3*n_qubits, batch_size)
    compiled_circuit = qml.compile(entangled_circ)
    qnode = qml.QNode(compiled_circuit, device, interface="torch")

    def circuit(features, weights):
        """
        features has shape (..., n_qubits, 3)
        weights has shape (..., n_qubits, n_layers, 3)
        """
        assert features.shape[-1] == 3, f"features shape {features.shape} does not match 3"
        assert weights.shape[-2] == n_layers, f"weights shape {weights.shape} does not match n_layers {n_layers}"
        assert weights.shape[-3] == n_qubits, f"weights shape {weights.shape} does not match n_qubits {n_qubits}"
        circ_out = qnode(features, weights)
        circ_out = torch.stack(circ_out)
        circ_out = torch.einsum("ij->ji", circ_out)
        return circ_out

    return circuit

class EntangledVQC(nn.Module):
    def __init__(self, in_dim, n_layers):
        super().__init__()
        self.n_qubits = math.ceil(in_dim/3)
        
        self.padded_input_dim = self.n_qubits * 3

        self.circ_out_dim = self.n_qubits * 3
        
        self.q_params = nn.Parameter(
            torch.rand((self.n_qubits, n_layers, 3), requires_grad=True)
        )

        self.circuit = make_entangled_vqc_sys(n_layers, self.n_qubits)
    
    def forward(self, x):
        x = F.pad(x, (0, self.padded_input_dim - x.shape[-1]))
        x = x.reshape(-1, self.n_qubits, 3)
        out = self.circuit(x, self.q_params).to(x.dtype)
        # flatten the output
        out = out.flatten(start_dim=1)
        return out

class EntangledPPOAgent(nn.Module):
    def __init__(self, envs, n_layers=18, backbone_out_dim=18, backbone=None, pretrained_backbone=True):
        super().__init__()
        n_actions = envs.single_action_space.n
        assert backbone is not None, "backbone is not provided"
        assert backbone.out_dim == backbone_out_dim, f"backbone output dimension {backbone.out_dim} does not match {backbone_out_dim}"
        self.circ_qubits = math.ceil(backbone_out_dim/3)
        self.circ_out_dim = self.circ_qubits * 3
        self.circ_in_dim = self.circ_qubits * 3
        if not pretrained_backbone:
            self.backbone = backbone
        else:
            self.backbone = backbone
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.actor = nn.Sequential(
            layer_init(nn.Linear(512, backbone_out_dim), std=0.01), # classical pre-processing
            ScaleToPi(),
            EntangledVQC(backbone_out_dim, n_layers),
            layer_init(nn.Linear(self.circ_out_dim, n_actions), std=0.01) # classical post-processing
            )
        self.critic = Critic512Input()
    
    def get_value(self, x):
        return self.critic(self.backbone(x/255.0))

    def get_action_and_value(self, x, action=None):
        hidden = self.backbone(x/255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

if __name__ == "__main__":
    # test the single-qubit VQC system
    n_layers = 18
    n_qubits = 6
    features = torch.randn(10 ,n_qubits, 3)
    weights = torch.randn(n_qubits, n_layers, 3)
    circuit = make_entangled_vqc_sys(n_layers, n_qubits)
    out = circuit(features, weights)
    print(out.shape)
    x = torch.randn(2, 18)
    model = EntangledVQC(x.shape[1], n_layers)
    out = model(x)
    print(out.shape)




