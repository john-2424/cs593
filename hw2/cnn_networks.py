"""
CS 593 RL1 Homework Assignment 2
Purdue University
Created by: Joseph Campbell and Guven Gergerli
"""

import torch.nn as nn
import torch

# -------------------- CNN Network for Image-Based Envs --------------------
class CNNNetwork(nn.Module):
    """CNN network for processing image observations (like CarRacing)"""
    
    def __init__(self, input_shape, output_dim, hidden_size=512):
        super(CNNNetwork, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # (C, H, W) - channels, height, width
        c, h, w = input_shape
        
        # CNN feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(c, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
        )
        
        # size of the feature maps after CNN layers
        feature_size = self._get_conv_output(input_shape)
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim)
        )
        
    def _get_conv_output(self, shape):
        """Helper function to calculate the size of CNN output"""
        batch_size = 1
        input = torch.zeros(batch_size, *shape)
        output = self.features(input)
        return int(torch.prod(torch.tensor(output.size()[1:])))
        
    def forward(self, x):
        # Expected x shape: (batch_size, channels, height, width)
        features = self.features(x)
        features = features.view(features.size(0), -1)  # Flatten
        return self.fc(features)


# -------------------- Dueling CNN Network --------------------
class DuelingCNNNetwork(nn.Module):
    """Dueling architecture with CNN for image-based observations"""
    
    def __init__(self, input_shape, output_dim, hidden_size=512):
        super(DuelingCNNNetwork, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Expected input shape: (C, H, W) - channels, height, width
        c, h, w = input_shape
        
        # Shared CNN feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(c, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
        )
        
        # size of the feature maps after CNN layers
        feature_size = self._get_conv_output(input_shape)
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim)
        )
        
    def _get_conv_output(self, shape):
        batch_size = 1
        input = torch.zeros(batch_size, *shape)
        output = self.features(input)
        return int(torch.prod(torch.tensor(output.size()[1:])))
    
    def forward(self, x):
        features = self.features(x)
        # Flatten
        features = features.view(features.size(0), -1)  
        
        #TODO: Implement value and advantage streams
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)

        # TODO: Now combine value and advantage using the dueling formula:
        # Q value is equal to the state value + advantages substracted by the mean advantage across all actions
        # This ensures the advantage head has zero mean. HINT: use keepdim=True in the mean calculation
        return value + advantages - advantages.mean(dim=1, keepdim=True)
    


