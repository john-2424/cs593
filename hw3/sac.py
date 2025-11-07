"""
CS 593 RL1 Homework Assignment 3
Purdue University
Created by: Joseph Campbell and Guven Gergerli
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gymnasium as gym
import os
import random
from collections import namedtuple, deque
from logger import Logger
import utils


# Replay Buffer for off-policy learning
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)
    
    def __len__(self):
        return len(self.buffer)



# Actor Network (Policy)
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp()
        
        return mean, std
    
    def sample(self, state):
        mean, std = self.forward(state)
        normal = torch.distributions.Normal(mean, std)
        
        # TODO: Reparameterization trick
        # https://spinningup.openai.com/en/latest/algorithms/sac.html

        # HINT: rsample is a method that is used to sample from a distribution
        # in a way that allows for gradients to be backpropagated through the sampling process.
        # The reason that we use rsample instead of sample is that
        # sample does not allow for gradients to be backpropagated
        # but we need to be able to backpropagate through the action sampling,
        # in order to update the policy network in SAC.
        # HINT2: After sampling, apply the tanh function to the sampled action
        # to ensure the action is in the range [-1, 1].
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        
        # Calculate log_prob
        log_prob = normal.log_prob(x_t)
        
        # Apply correction for tanh squashing
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob, torch.tanh(mean)



# Critic Network (Q-Value)
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(QNetwork, self).__init__()
        
        # Q1 architecture
        self.linear1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1 = nn.Linear(hidden_dim, 1)
        
        # Q2 architecture
        self.linear3 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, hidden_dim)
        self.q2 = nn.Linear(hidden_dim, 1)
    
    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        
        # Q1
        q1 = F.relu(self.linear1(x))
        q1 = F.relu(self.linear2(q1))
        q1 = self.q1(q1)
        
        # Q2
        q2 = F.relu(self.linear3(x))
        q2 = F.relu(self.linear4(q2))
        q2 = self.q2(q2)
        
        return q1, q2




class SACAgent:
    '''Soft Actor-Critic Agent for continuous action spaces'''

    def __init__(self, env_name, lr=3e-4, num_episodes=1000, batch_size=256, gamma=0.99, 
                 tau=0.005, alpha=0.2, buffer_size=1000000, updates_per_step=1, hidden_size=256,
                 save_interval=50, start_steps=10000):
        '''Initialize SAC agent'''

        self.env_name = env_name
        self.env = gym.make(env_name)
        self.is_cnn = False

        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        print(f"State dim: {state_dim}, Action dim: {action_dim}")
        print(f"Action space: {self.env.action_space.low} to {self.env.action_space.high}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # hyper parameters
        self.lr = lr
        self.num_episodes = num_episodes
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.alpha = torch.tensor(alpha, device=self.device)  # Entropy regularization coefficient
        self.updates_per_step = updates_per_step
        self.start_steps = start_steps  # Random exploration steps before using policy
        self.save_interval = save_interval
        self.hidden_size = hidden_size

        # Initialize networks
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim=hidden_size).to(self.device)
        self.critic = QNetwork(state_dim, action_dim, hidden_dim=hidden_size).to(self.device)
        self.critic_target = QNetwork(state_dim, action_dim, hidden_dim=hidden_size).to(self.device)

        # Initialize target network with same weights
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
            
        # Initialize optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # Initialize replay buffer
        self.memory = ReplayBuffer(buffer_size)
        
        # Automatic entropy tuning
        self.target_entropy = -torch.prod(torch.Tensor(self.env.action_space.shape).to(self.device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)

        # Logger
        self.env_tag = env_name.lower().replace('-', '_').split('/')[-1]
        self.num_params = sum(p.numel() for p in self.policy.parameters()) + sum(p.numel() for p in self.critic.parameters())
        self.variant_tag = "sac"
        self.logger = Logger(self.env_tag, self.env_name, self.variant_tag, self.num_params)
        
        # Step counter
        self.total_steps = 0



    def select_action(self, state):
        '''Select action according to current policy'''

        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)

        action, _, _ = self.policy.sample(state)
            
        return action.detach().cpu().numpy()[0]



    def update_parameters(self):
        '''Update the networks using SAC update rules'''

        if len(self.memory) < self.batch_size:
            return
            
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(self.batch_size)
        
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        done_batch = torch.FloatTensor(done_batch).to(self.device).unsqueeze(1)
        
        with torch.no_grad():
            # To compute target Q value, sample next action from the policy
            next_action, next_log_prob, _ = self.policy.sample(next_state_batch)

            # get target Q values from target networks
            target_q1, target_q2 = self.critic_target(next_state_batch, next_action)


            # TODO: Target Q value:
            # Compute the target Q value with entropy term
            # calculate the target Q value using the minimum of the two target Q-values
            # and subtracting the alpha-weighted log probability of the next action
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob

            # immediate reward + discounted target Q value
            # HINT: use (1 - done_batch) to zero out target Q values for terminal states
            target_q = reward_batch + (1.0 - done_batch) * self.gamma * target_q
        


        # TODO: Critic Loss
        # https://spinningup.openai.com/en/latest/algorithms/sac.html
        # Compute current Q estimates and critic loss
        current_q1, current_q2 = self.critic(state_batch, action_batch)

        # sum both Q losses
        # HINT1: The critic loss is the mean squared error between
        # the current Q estimates and the target Q values.
        # HINT2: Use MSE loss (F.mse_loss) between current Q estimates and target Q values
        # HINT3: Sum the losses from both Q-networks to get the total critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        

        # Update policy
        new_actions, log_prob, _ = self.policy.sample(state_batch)
        q1, q2 = self.critic(state_batch, new_actions)


        # TODO: Policy Loss
        # https://spinningup.openai.com/en/latest/algorithms/sac.html

        # HINT1: For the Q-value that we will use, use the minimum of the two Q-values
        # to mitigate positive bias in the policy improvement step.
        q = torch.min(q1, q2)

        # HINT2: The policy loss is the expected value of the alpha-weighted log
        # probability of the action minus the Q-value.
        policy_loss = (self.alpha * log_prob - q).mean()
        

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        
        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        self.alpha = self.log_alpha.exp()
        

        # Soft update target networks
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
            
        return critic_loss.item(), policy_loss.item(), alpha_loss.item()



    def train_sac(self):
        """
        Train the agent using Soft Actor-Critic algorithm with logging
        """

        for i_episode in range(self.num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            episode_steps = 0
            done = False
            
            critic_loss = 0
            policy_loss = 0
            alpha_loss = 0
            
            while not done:
                # Select action
                if self.total_steps < self.start_steps:
                    # Random initial exploration
                    action = self.env.action_space.sample()
                else:
                    action = self.select_action(state)
                

                # Take step in environment
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_steps += 1
                self.total_steps += 1
                episode_reward += reward
                

                # Store the transition in memory
                self.memory.push(state, action, reward, next_state, float(done))
                

                # Move to the next state
                state = next_state
                

                # Update parameters if enough samples in buffer
                if len(self.memory) > self.batch_size:
                    for _ in range(self.updates_per_step):
                        critic_loss, policy_loss, alpha_loss = self.update_parameters()
            

            # Log episode statistics
            self.logger.add_value('train/episode_reward', episode_reward, i_episode)
            self.logger.add_value('train/critic_loss', critic_loss, i_episode)
            self.logger.add_value('train/policy_loss', policy_loss, i_episode)
            self.logger.add_value('train/alpha_loss', alpha_loss, i_episode)
            self.logger.add_value('train/alpha', self.alpha.item(), i_episode)
            
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
            
            print(f"Episode {i_episode}/{self.num_episodes} - Total Reward: {episode_reward:.2f} - Critic Loss: {critic_loss:.4f} - Policy Loss: {policy_loss:.4f}")
        
        print('Training complete')
        self.env.close()




