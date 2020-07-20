import torch
torch.manual_seed(42)
import torch.nn as nn
import numpy as np
from typing import Tuple, List


class MLP(nn.Module):
    def __init__(self, M1_dim: Tuple[int,int], M2_dim: Tuple[int, int], hiddens: List[int], activation = "ReLU"):
        super(MLP, self).__init__()
        assert M1_dim[1] == M2_dim[0], f"dimensions of the matrices do not work for mat mul, M1: {M1_dim}, M2: {M2_dim}"
        # calculate dimensions
        M1_flat = np.product(M1_dim)
        M2_flat = np.product(M2_dim)
        input_dim = M1_flat + M2_flat
        out_dim = np.product([M1_dim[0], M2_dim[1]])
        self.fc_layers = nn.ModuleList()
        last_out = input_dim
        for hidden in hiddens:
            self.fc_layers.append(nn.Linear(last_out, hidden))
            last_out = hidden
        self.out = nn.Linear(last_out, out_dim)
        if activation.lower() == "relu":
            self.activation = nn.ReLU()
        elif activation.lower() == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation.lower() == "tanh":
            self.activation = nn.Tanh()
        else:
            print(f"your activation function {activation} is not supported, using Identity (i.e. no activation)")
            self.activation = nn.Identity()
    def forward(self, x: torch.Tensor):
        for layer in self.fc_layers:
            x = layer(x)
            x = self.activation(x)
        out = self.out(x) # no activation on output (regression problem)
        return out

# defines a nn.Module like nn.Linear, but where our function is the product of (wi * xi) instead of affine / sum
# in nn.Linear
class ProdLayer(nn.Module):
    def __init__(self, in_features: int, out_featuers: int, bias: bool = False):
        super(ProdLayer, self).__init__()
        self.bias = bias
        self.weights = nn.Parameter(nn.init.xavier_uniform_(torch.zeros((in_features, out_featuers)))).unsqueeze(0)
        if self.bias:
            self.bias_weights = nn.Parameter(torch.zeros(out_featuers))
    def forward(self, x: torch.Tensor):
        #breakpoint()
        # x: B x in_d -> B x in_d x 1
        # weights: 1 x id_d x out_d
        x = torch.mul(x.unsqueeze(2),  self.weights)
        # x: B x in_d x out_d
        x = torch.prod(x, dim =1)
        # x: B x out_d
        if self.bias:
            x = x + self.bias_weights
        return x

# all hidden layers are product layers
# the output layer is an affine layer
class ProdMLP(nn.Module):
    def __init__(self, M1_dim: Tuple[int,int], M2_dim: Tuple[int, int], hiddens: List[int], activation = "ReLU"):
        super(ProdMLP, self).__init__()
        assert M1_dim[1] == M2_dim[0], f"dimensions of the matrices do not work for mat mul, M1: {M1_dim}, M2: {M2_dim}"
        M1_flat = np.product(M1_dim)
        M2_flat = np.product(M2_dim)
        input_dim = M1_flat + M2_flat
        out_dim = np.product([M1_dim[0], M2_dim[1]])
        self.fc_layers = nn.ModuleList()
        last_out = input_dim
        for hidden in hiddens:
            self.fc_layers.append(ProdLayer(last_out, hidden))
            last_out = hidden
        self.out = nn.Linear(last_out, out_dim)
        if activation.lower() == "relu":
            self.activation = nn.ReLU()
        elif activation.lower() == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation.lower() == "tanh":
            self.activation = nn.Tanh()
        else:
            print(f"your activation function {activation} is not supported, using Identity (i.e. no activation)")
            self.activation = nn.Identity()

    def forward(self, x: torch.Tensor):
        #breakpoint()
        for layer in self.fc_layers:
            x = layer(x)
            x = self.activation(x)
        out = self.out(x)  # no activation on output (regression problem)
        return out

