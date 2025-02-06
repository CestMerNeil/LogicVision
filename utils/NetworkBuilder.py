import torch
import torch.nn as nn

# This file is to build the network for the Logic Tensor Networks model.
# Output of each Builder are the possible 0-1 values of the logic tensor.

def _build_cnn(self, input_dim: int, channels: list) -> nn.Module:
    layers = []
    in_channels = input_dim
    for out_channels in channels:
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=self.conv_kernel_size, padding=1))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(kernel_size=self.pool_size))
        leyers.append(nn.Sigmoid())
    return nn.Sequential(*layers)

def _build_mlp(self, input_dim: int, hidden_dims: list, dropout: float) -> nn.Module:
    layers = []
    for h_dim in hidden_dims:
        layers.append(nn.Linear(input_dim, h_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        input_dim = h_dim
    layers.append(nn.Linear(hidden_dims[-1], 1))
    layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)