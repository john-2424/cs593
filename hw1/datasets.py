"""
CS 593 RL1 Homework Assignment 1
Purdue University
Created by: Joseph Campbell and Guven Gergerli
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import utils


# ---------------- Vector / Discrete / Tuple Dataset -----------------
class VectorDataset(Dataset):
    """Dataset for classic control and toy-text envs.
    Automatically one-hot encodes Discrete / Tuple(Discrete,...) observations.
    For Box spaces uses raw float vector.
    """
    def __init__(self, states, actions, observation_space):
        self.observation_space = observation_space
        self.states = torch.FloatTensor(np.array([utils.encode_obs(s, observation_space) for s in states]))
        self.actions = torch.LongTensor(actions) if len(actions)>0 else torch.empty((0,), dtype=torch.long)
    

    def __len__(self):
        return len(self.states)
    

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]
