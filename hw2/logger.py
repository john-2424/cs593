"""
CS 593 RL1 Homework Assignment 2
Purdue University
Created by: Joseph Campbell and Guven Gergerli
"""

import numpy as np
import os
import time

from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, env_tag, env_name, action_dim, source_tag, num_params):
        """Configure TensorBoard logging."""

        log_dir = os.path.join('runs', env_tag, f"{source_tag}_{int(time.time())}")
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)
        
        # Log configuration information
        self.writer.add_text(
            'run/config', 
            f'env={env_name}, action_dim={action_dim}, source={source_tag}, params={num_params}'
        )

    def add_value(self, name, y, x):
        """Add a single scalar value to Tensorboard"""

        self.writer.add_scalar(name, y, x)
        self.writer.flush()

    def add_frames(self, name, frames, x, fps=10):
        """Add rendered frames from a list (should be list of lists) of episodes to Tensorboard"""

        video = np.array([np.transpose(frame, [0, 3, 1, 2]) for frame in frames])

        self.writer.add_video('{}'.format(name), video, x, fps=fps)