"""
CS 593 RL1 Homework Assignment 3
Purdue University
Created by: Joseph Campbell and Guven Gergerli
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import os
from collections import deque
from logger import Logger
import utils


# Actor Network
class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        
        # Policy head (actor)
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim) - 0.5)

    def forward(self, state):
        features = self.network(state)
        mean = self.mean(features)
        # Constrain std for stable actions
        std = torch.exp(self.log_std.clamp(-5, 0))
        return mean, std

# Critic Network
class CriticNetwork(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # Value output
        )

    def forward(self, state):
        return self.network(state)




class PPOAgent:
    '''PPO Agent for continuous action spaces'''

    def __init__(self, env_name, lr=3e-4, num_episodes=1000, 
                 gamma=0.99, gae_lambda=0.95, clip_ratio=0.2,
                 value_coef=0.5, entropy_coef=0.01, 
                 update_epochs=10, batch_size=64, save_interval=50):
        '''Initialize PPO agent'''

        self.env_name = env_name
        self.env = gym.make(env_name)

        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        print(f"State dim: {state_dim}, Action dim: {action_dim}")
        print(f"Action space: {self.env.action_space.low} to {self.env.action_space.high}")

        # PPO hyperparameters
        self.lr = lr
        self.num_episodes = num_episodes
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.save_interval = save_interval

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize actor-critic network
        self.actor = ActorNetwork(state_dim, action_dim).to(self.device)
        self.critic = CriticNetwork(state_dim).to(self.device)
        # Initialize optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # Logger
        self.env_tag = env_name.lower().replace('-', '_').split('/')[-1]
        self.num_params = sum(p.numel() for p in self.actor.parameters()) + sum(p.numel() for p in self.critic.parameters())
        self.variant_tag = "ppo"
        self.logger = Logger(self.env_tag, self.env_name, self.variant_tag, self.num_params)



    def select_action(self, state):
        '''Select action according to current policy and compute log probability'''
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
            
            # Get action distribution parameters from actor
            mean, std = self.actor(state_tensor)
            
            # Create normal distribution
            dist = torch.distributions.Normal(mean, std)
            
            # Sample action and compute log probability
            raw_action = dist.sample()
            log_prob = dist.log_prob(raw_action).sum(-1)
            
            # Get entropy for exploration bonus
            # dist.entropy() returns entropy per action dimension
            # with formula H(X) = 0.5 * log(2 * pi * e * sigma^2)
            entropy = dist.entropy().sum(-1)
            
            # Get value estimate from critic
            value = self.critic(state_tensor).item()
            
            # Scale action to environment's action space
            low = torch.tensor(self.env.action_space.low, dtype=torch.float32, device=self.device)
            high = torch.tensor(self.env.action_space.high, dtype=torch.float32, device=self.device)
            scaled_action = low + (0.5 * (raw_action + 1.0) * (high - low))
            
            return scaled_action.cpu().numpy().flatten(), log_prob.cpu().item(), value, entropy.cpu().item(), raw_action.cpu().numpy().flatten()



    def compute_gae(self, rewards, values, dones, next_value):
        '''Compute Generalized Advantage Estimation'''
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        running_return = next_value
        running_advantage = 0
        
        # Compute GAE in reverse order
        for t in reversed(range(len(rewards))):
            # If done, the next value is 0
            next_val = next_value if t == len(rewards) - 1 else values[t + 1]
            next_non_terminal = 1.0 - dones[t]
            
            # TODO: Calculate TD error
            # HINT1: TD error is calculated as: immediate reward + discounted next value - current value estimate
            # HINT2: use next_non_terminal to handle episode ends while calculating next value
            delta = rewards[t] + self.gamma * next_non_terminal * next_val - values[t]
            
            # TODO: Calculate GAE
            # HINT1: GAE is accumulated TD errors discounted by gamma and lambda
            # HINT2: use next_non_terminal to handle episode ends while accumulating advantage
            running_advantage = delta + self.gamma * self.gae_lambda * next_non_terminal * running_advantage
            advantages[t] = running_advantage
            
            # TODO: Calculate returns (for value function training)
            # HINT1: returns are calculated as immediate reward + discounted next return
            # HINT2: again use next_non_terminal to handle episode ends while accumulating return
            running_return = rewards[t] + self.gamma * next_non_terminal * running_return
            returns[t] = running_return
            
        return advantages, returns



    def update_policy(self, states, actions, old_log_probs, advantages, returns, old_values):
        '''Update policy using PPO clipped objective with decoupled actor and critic'''
        # Convert to tensors and move to device
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        old_values = torch.FloatTensor(old_values).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Calculate number of mini-batches
        dataset_size = len(states)
        indices = np.arange(dataset_size)
        
        actor_loss_epoch = 0
        critic_loss_epoch = 0
        entropy_loss_epoch = 0
        
        # Training for multiple epochs
        for _ in range(self.update_epochs):
            # Shuffle for mini-batches so that each mini-batch is different
            np.random.shuffle(indices)
            
            # Process mini-batches
            for start_idx in range(0, dataset_size, self.batch_size):
                end_idx = min(start_idx + self.batch_size, dataset_size)
                batch_indices = indices[start_idx:end_idx]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # ACTOR UPDATE
                # Get current action distribution
                mean, std = self.actor(batch_states)
                dist = torch.distributions.Normal(mean, std)
                curr_log_probs = dist.log_prob(batch_actions).sum(-1)
                entropy = dist.entropy().sum(-1)
                

                # TODO: PPO surrogate clipped objective function
                # HINT: https://huggingface.co/learn/deep-rl-course/unit8/clipped-surrogate-objective

                # HINT: compute ratio between current and old policy log probabilities
                # using exponentiation of their difference.
                # HINT2: The reason we do this is because 
                # log(a/b) = log(a) - log(b)  =>  a/b = exp(log(a) - log(b))
                ratio = torch.exp(curr_log_probs - batch_old_log_probs)
                
                # Then for PPO clipped objective, compute unclipped and clipped surrogate losses
                # and take the minimum of these two for the final loss
                # for surrogate loss 1, use the ratio directly multiplied by advantages
                surr1 = ratio * batch_advantages

                # for surrogate loss 2, we will clamp the ratio to be within [1-epsilon, 1+epsilon]
                # where epsilon is the clip ratio hyperparameter
                # then multiply the clipped ratio by advantages
                # the reason for clamping is to avoid large policy updates that can destabilize training
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * batch_advantages

                # Now the actor loss is the negative mean of the minimum of the two surrogate losses
                # HINT: negative mean because we want to maximize the objective
                actor_loss = -torch.min(surr1, surr2).mean()

                entropy_loss = -entropy.mean()
                
                # Actor network update
                self.actor_optimizer.zero_grad()
                actor_total_loss = actor_loss + self.entropy_coef * entropy_loss
                actor_total_loss.backward()
                self.actor_optimizer.step()
                

                # CRITIC UPDATE
                # Current value estimates
                curr_values = self.critic(batch_states).squeeze()
                
                # Value loss
                critic_loss = 0.5 * ((curr_values - batch_returns) ** 2).mean()
                
                # Critic network update
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()
                
                actor_loss_epoch += actor_loss.item()
                critic_loss_epoch += critic_loss.item()
                entropy_loss_epoch += entropy_loss.item()
        

        num_updates = self.update_epochs * (dataset_size // self.batch_size + int(dataset_size % self.batch_size > 0))
        return actor_loss_epoch/num_updates, critic_loss_epoch/num_updates, entropy_loss_epoch/num_updates



    def train_ppo(self):
        """Train the agent using PPO algorithm with logging"""
        
        for i_episode in range(self.num_episodes):
            state, _ = self.env.reset()
            
            # Storage for episode data
            states, actions, rewards, log_probs, values, dones, entropies, raw_actions = [], [], [], [], [], [], [], []
            total_reward = 0
            
            # Episode loop
            done = False
            while not done:
                # Select action
                action, log_prob, value, entropy, raw_action = self.select_action(state)
                
                # Step environment
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # Store data
                states.append(state)
                actions.append(raw_action)  # Store unscaled actions
                rewards.append(reward)
                log_probs.append(log_prob)
                values.append(value)
                dones.append(float(done))
                entropies.append(entropy)
                
                total_reward += reward
                state = next_state
            
            # Compute next state value (0 if done)
            if done:
                next_value = 0
            else:
                state_tensor = torch.FloatTensor(state).to(self.device)
                if self.is_cnn:
                    state_tensor = state_tensor.unsqueeze(0)
                next_value = self.critic(state_tensor).item()
            
            # Compute advantages and returns
            advantages, returns = self.compute_gae(np.array(rewards), np.array(values), np.array(dones), next_value)
            
            # Update policy: states, actions, old_log_probs, advantages, returns, old_values
            actor_loss, value_loss, entropy_loss = self.update_policy(
                states=np.array(states), 
                actions=np.array(actions), 
                old_log_probs=np.array(log_probs), 
                advantages=advantages, 
                returns=returns, 
                old_values=np.array(values)
            )
            
            # Log episode statistics
            self.logger.add_value('train/episode_reward', total_reward, i_episode)
            self.logger.add_value('train/actor_loss', actor_loss, i_episode)
            self.logger.add_value('train/critic_loss', value_loss, i_episode)
            self.logger.add_value('train/entropy_loss', entropy_loss, i_episode)
            
            # Save model and evaluate at intervals
            if i_episode % self.save_interval == 0:
                if not os.path.exists('checkpoints'):
                    os.makedirs('checkpoints')
                torch.save({
                    'actor': self.actor.state_dict(),
                    'critic': self.critic.state_dict()
                }, f'checkpoints/{self.variant_tag}_{self.env_name}_episode_{i_episode}.pth')
                
                # Evaluate policy and log video
                avg_eval_reward = utils.evaluate_policy(self.actor, self.env.observation_space, self.env_name, episodes=3, return_frames=False)
                self.logger.add_value('train/eval_mean_reward', avg_eval_reward, i_episode)
                
                # Log evaluation video
                _, frames = utils.evaluate_policy(self.actor, self.env.observation_space, self.env_name, episodes=1, return_frames=True, max_length=1000)
                self.logger.add_frames('train/eval_video', frames, i_episode)
            
            print(f"Episode {i_episode}/{self.num_episodes} - Total Reward: {total_reward:.2f} - Actor Loss: {actor_loss:.4f} - Critic Loss: {value_loss:.4f}")
        
        print('Training complete')
        self.env.close()





