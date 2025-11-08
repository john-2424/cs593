"""
CS 593 RL1 Homework Assignment 3
Purdue University
Created by: Joseph Campbell and Guven Gergerli
"""

from Cython import returns
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import gymnasium as gym
import os
import torch
import torch.nn as nn
import torch.optim as optim
import utils

import numpy as np

from logger import Logger



class MLPNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim) - 0.5)

    def forward(self, state):
        network_features = self.net(state)
        mean = self.mean(network_features)
        # Constrain std for stable actions
        std = torch.exp(self.log_std.clamp(-5, 0))
        return mean, std

    

class REINFORCEAgent:
    '''REINFORCE Agent on continuous action spaces'''

    def __init__(self, env_name, lr=1e-3, num_episodes=1000, batch_size=64, gamma=0.99, save_interval=50):
        '''Initialize REINFORCE agent'''

        self.env_name = env_name
        self.env = gym.make(env_name)

        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        print(f"State dim: {state_dim}, Action dim: {action_dim}")
        print(f"Action space: {self.env.action_space.low} to {self.env.action_space.high}")

        # hyper parameters
        self.lr = lr
        self.num_episodes = num_episodes
        self.batch_size = batch_size
        self.gamma = gamma
        self.save_interval = save_interval

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # initialize policy network
        self.policy = MLPNetwork(state_dim, action_dim).to(self.device)
        # initialize optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # Logger
        self.env_tag = env_name.lower().replace('-', '_').split('/')[-1]
        self.num_params = sum(p.numel() for p in self.policy.parameters())
        self.variant_tag = "reinforce"
        self.logger = Logger(self.env_tag, self.env_name, self.variant_tag, self.num_params)



    def select_action(self, state):
        '''Select action according to current policy'''

        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)

        # TODO: complete action selection:

        # Forward pass through the policy network
        # to get the action distribution parameters
        mean, std = self.policy(state_tensor)

        # create the normal distribution with the mean and std
        # HINT: For continuous actions, we often use a Gaussian distribution
        dist = torch.distributions.Normal(mean, std)

        # sample an action from the distribution
        # HINT: torch.distributions has methods to sample from distributions
        raw_action = dist.sample()

        # get the log probability of the action
        # HINT: the log probability is the log of the probability density function,
        # which is basically the log of how likely the action is under the current policy.
        # we just use this to weight the policy gradient update later
        log_prob = dist.log_prob(raw_action).sum(dim=-1)

        # Squash to [-1, 1] range for stability
        # squashed_action = torch.tanh(raw_action)

        # Scale action to the environment's action space (for stability)
        low = torch.tensor(self.env.action_space.low, dtype=torch.float32, device=self.device)
        high = torch.tensor(self.env.action_space.high, dtype=torch.float32, device=self.device)
        scaled_action = low + (0.5 * (raw_action + 1.0) * (high - low))
        
        return scaled_action.detach().cpu().numpy(), log_prob
        



    def calculate_discounted_returns(self, rewards):
        '''Update policy network using REINFORCE algorithm'''

        # initialize returns array:
        # return G_t is the discounted sum of future rewards
        # so we will calculate it in reverse order and accumulate into the returns array
        # this way, rewards will be calculated each timestep discounted with gamma

        returns = np.zeros_like(rewards, dtype=np.float32)
        running_return = 0

        # for each time step t in reverse order:
        for t in reversed(range(len(rewards))):

            # TODO: calculate the return G_t at time t 
            # HINT: using immediate reward and the discounted return at time t+1.
            running_return = rewards[t] + self.gamma * running_return

            # accumulate the return into the returns array
            returns[t] = running_return

        return returns



    def train_reinforce(self):
        """
        Train the agent using REINFORCE algorithm with logging
        """

        # Training loop
        for i_episode in range(self.num_episodes):
            state, _ = self.env.reset()
            log_probs, rewards = [], []
            total_reward = 0
            
            # Episode loop
            done = False
            while not done:
                action, log_prob = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                log_probs.append(log_prob)
                rewards.append(reward)
                total_reward += reward
                state = next_state
            
            # Calculate returns and update policy
            discounted_returns = self.calculate_discounted_returns(rewards)
            
            # Track loss for logging
            returns_tensor = torch.tensor(discounted_returns, dtype=torch.float32, device=self.device)
            returns_normalized = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)

            # TODO: Calculate policy loss
            # HINT1: REINFORCE policy loss is the negative sum of log probabilities
            # weighted by the corresponding discounted returns.
            # HINT2: use `torch.stack` function to combine the log probabilities.
            # HINT3: you don't need to use torch.stack for returns 
            policy_loss = -torch.sum(torch.stack(log_probs) * returns_normalized)
            
            # Update policy
            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()
            
            # Log episode statistics
            self.logger.add_value('train/episode_reward', total_reward, i_episode)
            self.logger.add_value('train/policy_loss', policy_loss.item(), i_episode)
            
            # Save model and evaluate at intervals
            if i_episode % self.save_interval == 0:
                if not os.path.exists('checkpoints'):
                    os.makedirs('checkpoints')
                torch.save(self.policy.state_dict(), f'checkpoints/{self.variant_tag}_{self.env_name}_episode_{i_episode}.pth')

                # Evaluate policy and log video
                avg_eval_reward = utils.evaluate_policy(self.policy, self.env.observation_space, self.env_name, episodes=3, return_frames=False)
                self.logger.add_value('train/eval_mean_reward', avg_eval_reward, i_episode)
                
                # Log evaluation video
                _, frames = utils.evaluate_policy(self.policy, self.env.observation_space, self.env_name, episodes=1, return_frames=True, max_length=1000)
                self.logger.add_frames('train/eval_video', frames, i_episode)
            
            print(f"Episode {i_episode}/{self.num_episodes} - Total Reward: {total_reward:.2f} - Policy Loss: {policy_loss.item():.4f}")
        
        print('Training complete')
        self.env.close()

