"""
CS 593 RL1 Homework Assignment 1
Purdue University
Created by: Joseph Campbell and Guven Gergerli
"""

import torch.nn as nn


# -------------------- MLP Network for Low-Dim Envs --------------------
class MLPNetwork(nn.Module):
    """Feed-forward network for classic control / toy-text (vector or one-hot inputs)."""

    def __init__(self, input_dim: int, output_dim: int, hidden_sizes=(128,128)):
        super().__init__()
        layers = []
        last = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        layers.append(nn.Linear(last, output_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)
