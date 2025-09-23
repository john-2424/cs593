"""
CS 593 RL1 Homework Assignment 1
Purdue University
Created by: Joseph Campbell and Guven Gergerli
"""

import gymnasium as gym
import os
import torch
import torch.nn as nn
import torch.optim as optim
import utils

from logger import Logger
from networks import MLPNetwork
from torch.utils.data import DataLoader


class ImitationLearner:
    """
    Implements behavior cloning to learn policies from demonstration data.
    """
    
    def __init__(self, env_name = "CartPole-v1", source_tag = "unknown", epochs = 20, batch_size = 64, save_interval = 5, lr=1e-3):
        """
        Initialize the imitation learning agent.
        
        Args:
            env_name: Name of the Gymnasium environment
            source_tag: Identifier for the demonstration data source
            epochs: Number of training epochs
            batch_size: Batch size for training
            save_interval: How frequently to log progress, videos, and save model checkpoint (in terms of epochs)
            lr: Learning rate for training
        """

        # Create environment and get dimensions
        env = gym.make(env_name)
        action_dim = env.action_space.n
        
        # Determine observation dimension based on space type
        space = env.observation_space
        obs_dim = utils.get_observation_dim(space)

        env.close()
        
        # Store environment information
        self.env_name = env_name
        self.action_dim = action_dim
        self.source_tag = source_tag
        self.obs_space = space

        self.epochs = epochs
        self.batch_size = batch_size
        self.save_interval = save_interval
        
        # Initialize policy network and training components
        # TODO: Initialize the policy network using MLPNetwork with appropriate input and output dimensions
        # HINT: The network should map from observation dimension to action dimension
        self.policy = MLPNetwork(input_dim=obs_dim, output_dim=action_dim)

        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.training_history = {'loss': [], 'accuracy': []}
        
        # Set up logging
        env_tag = env_name.lower().replace('-', '_').split('/')[-1]
        self.env_tag = env_tag
        self.logger = Logger(env_tag, self.env_name, self.action_dim, self.source_tag, sum(p.numel() for p in self.policy.parameters()))
    

    def train(self, dataset):
        """
        Train the policy network on demonstration data.
        
        Args:
            dataset: Dataset containing state-action pairs
            epochs: Number of training epochs
            batch_size: Batch size for training
            save_interval: Interval for printing progress and saving checkpoints
        """

        # Prepare data and loss function with action weighting
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.loss = nn.CrossEntropyLoss()
        
        # Create directory for saved models
        os.makedirs('models', exist_ok=True)
        
        # Training loop
        for epoch in range(0, self.epochs + 1):
            # Skip training on the first epoch so we can collect eval metrics for the randomly initialized policy
            if epoch > 0:
                # Train for one epoch
                avg_loss, avg_acc = self._train_epoch(dataloader)
            else:
                avg_loss = 0; avg_acc = 0
            
            # Evaluate current policy
            avg_eval_reward = utils.evaluate_policy(self.policy, self.obs_space, self.env_name, episodes=10, return_frames=False)

            # Log training metrics
            self.logger.add_value('train-imitation/loss', avg_loss, epoch)
            self.logger.add_value('train-imitation/acc', avg_acc, epoch)
            self.logger.add_value('train-imitation/eval_mean_reward', avg_eval_reward, epoch)
            
            # Print progress at intervals
            if epoch % self.save_interval == 0:
                # Collect a single additional rollout for a video and log it
                _, frames = utils.evaluate_policy(self.policy, self.obs_space, self.env_name, episodes=1, return_frames=True, max_length=100)
                self.logger.add_frames('train-imitation/eval_video', frames, epoch)

                print(f"Epoch {epoch}/{self.epochs} - Training Loss: {avg_loss:.4f}. Training Acc: {avg_acc:.4f}. Evaluation Reward: {avg_eval_reward:.3f}.")
        
        # Save final model
        self.save_policy(f'models/{self.env_tag}_imitation_final_{self.source_tag}.pth')
    

    def _train_epoch(self, dataloader):
        """Train for one epoch and return metrics."""

        epoch_loss = 0.0
        epoch_acc = 0.0
        batch_count = 0
        
        self.policy.train()
        for states, actions in dataloader:
            # Forward pass
            logits = self.policy(states)

            # TODO: Calculate the loss using self.loss. Keep in mind it takes two parameters: the model logits and the demonstration targets. Refer to: https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
            loss = self.loss(logits, actions)
            
            # Backward pass and optimization
            # TODO: Perform backpropagation 
            # HINT: Don't forget to zero the gradients
            # Refer to: https://docs.pytorch.org/tutorials/recipes/recipes/zeroing_out_gradients.html
            self.optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)

            # TODO: Perform an optimization step using self.optimizer
            # Refer to: https://docs.pytorch.org/tutorials/recipes/recipes/zeroing_out_gradients.html
            self.optimizer.step()

            # TODO: Calculate prediction accuracy by comparing predictions with true actions
            # HINT: First get predictions using torch.argmax, then compare with actions
            predictions = torch.argmax(logits, dim=1)
            accuracy = (predictions == actions).float().mean().item()
            
            # Accumulate metrics
            epoch_loss += loss.item()
            epoch_acc += accuracy
            batch_count += 1
        
        # Calculate averages
        avg_loss = epoch_loss / max(1, batch_count)
        avg_acc = epoch_acc / max(1, batch_count)
        
        # Store history
        self.training_history['loss'].append(avg_loss)
        self.training_history['accuracy'].append(avg_acc)
        
        return avg_loss, avg_acc
    

    def save_policy(self, path):
        """
        Save the trained policy and related information.
        
        Args:
            path: Path where to save the policy
        """

        torch.save({
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history,
            'env_name': self.env_name,
            'action_dim': self.action_dim,
            'source': self.source_tag
        }, path)
