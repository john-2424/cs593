"""
CS 593 RL1 Homework Assignment 2
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


# -------------------- Dueling DQN Network --------------------
class DuelingMLPNetwork(nn.Module):
    """Dueling DQN architecture with separate value and advantage streams"""
    
    def __init__(self, input_dim, output_dim, hidden_size=128):
        super(DuelingMLPNetwork, self).__init__()
        
        # Feature extractor (shared layers)
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Value head:
        # This head estimates state value: how good the state is
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Advantage head
        # This head estimates advantage of each action: how much better taking a specific action is compared to others
        self.advantage_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_dim)
        )
        
    def forward(self, x):
        features = self.feature_layer(x)

        #TODO: Implement value and advantage streams
        value = self.value_head(features)
        advantages = self.advantage_head(features)

        # TODO: Now combine value and advantage using the dueling formula:
        # Q value is equal to the state value + advantages substracted by the mean advantage across all actions
        # This ensures the advantage head has zero mean. HINT: use keepdim=True in the mean calculation
        return value + advantages - advantages.mean(dim=1, keepdim=True)
    





