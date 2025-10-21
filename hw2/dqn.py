"""
CS 593 RL1 Homework Assignment 2
Purdue University
Created by: Joseph Campbell and Guven Gergerli
"""

import gymnasium as gym
import os
import torch
import torch.nn as nn
import torch.optim as optim
import utils
from collections import namedtuple, deque
import random

import math
from itertools import count
import numpy as np

from logger import Logger
from networks import MLPNetwork, DuelingMLPNetwork
from cnn_networks import CNNNetwork, DuelingCNNNetwork
# from torch.utils.data import DataLoader
from wrapper import FrameStackResize


# Transition tuple for replay memory consisting of (state, action, next_state, reward)
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    """Buffer to store environment transitions."""

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Saves a transition"""
        # TODO: Add the transition to the memory 
        # HINT: no need to return just add it to the memory.
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        # TODO: Implement random sampling from replay memory
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class PrioritizedReplayMemory:
    """Prioritized Experience Replay Buffer"""

    def __init__(self, capacity, alpha=0.6, beta=0.4):
        """
        alpha: How much prioritization to use (0 = no prioritization, 1 = full prioritization)
        beta: Importance sampling weight factor (0 = no correction, 1 = full correction)
        """
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent
        self.beta = beta    # Importance sampling exponent
        self.memory = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.eps = 1e-5  # Small positive constant to avoid zero priorities
        
    def push(self, *args):
        """Save a transition with max priority"""
        # If memory not full, add new memory
        if len(self.memory) < self.capacity:
            self.memory.append(None)
            
        # Find max priority (for new transitions)
        max_priority = np.max(self.priorities) if self.memory[0] is not None else 1.0
        
        # Store transition
        self.memory[self.position] = Transition(*args)
        self.priorities[self.position] = max_priority
        
        # Update position (circular buffer)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        """
        Sample a batch of transitions based on priorities;
        We will convert priorities to probabilities and sample accordingly.
        Also compute importance sampling weights for bias correction.
        """
        # Get current priorities (only filled part of the buffer)
        if len(self.memory) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]
            
        # TODO: Calculate sampling probabilities:
        # alpha determines how much prioritization is used 
        # (0 = uniform, 1 = full prioritization)
        probs = (priorities + self.eps) ** self.alpha
        probs /= np.sum(probs)
        
        # TODO: Sample indices based on priorities and get corresponding transitions
        indices = np.random.choice(len(priorities), batch_size, p=probs)
        samples = [self.memory[idx] for idx in indices]
        
        # TODO: Calculate importance sampling weights:
        # The sampling is biased towards high-priority samples since they are sampled more frequently.
        # This can result on Q value overestimation towards high-priority samples.
        # To correct for this bias, we use importance sampling weights.
        # Beta here controls how strongly you correct the bias using importance sampling (0 = no correction, 1 = full correction).
        weights = (len(self.memory) * probs[indices]) ** (-self.beta)
        weights /= np.max(weights)  # Normalize weights
        weights = torch.tensor(weights, dtype=torch.float32)
        
        return samples, indices, weights
    
    def update_priorities(self, indices, td_errors):
        """Update priorities based on TD errors"""
        for idx, td_error in zip(indices, td_errors):
            # TODO: Priority is measured as absolute TD error plus small constant to avoid zero priority
            self.priorities[idx] = np.abs(td_error) + self.eps
    
    def __len__(self):
        return len(self.memory)
    


class DQNAgent:
    """Implements DQN Agent to train an off-policy model given an environment 
    with discrete action space using Replay Memory buffer."""

    def __init__(self, env_name = "CartPole-v1", episodes = 20, batch_size = 32, save_interval = 5, 
                 lr=1e-3, hidden_size = 128, replay_size = 10000, epsilon_start = 0.9, epsilon_end = 0.05, 
                 epsilon_decay = 500, gamma = 0.99, target_update = 10, tau = 0.005,
                 alpha=0.6, beta=0.4,
                 use_double_dqn=False, use_per=False, dueling_dqn=False):
        
        # Create environment
        self.env_name = env_name
        if self.env_name == 'CarRacing-v2':
            self.env = gym.make(env_name, continuous=False)
            self.env = FrameStackResize(self.env, num_stack=4, resize_shape=(84, 84))
            self.is_cnn = True # CNN for image-based observations
        else:
            self.env = gym.make(env_name)
            self.is_cnn = False

        # Determine observation dimension based on space type
        space = self.env.observation_space

        if self.is_cnn:
            # For CarRacing
            # self.input_shape = (3, space.shape[0], space.shape[1])
            self.input_shape = space.shape
            self.obs_dim = space.shape
        else:
            self.obs_dim = utils.get_observation_dim(space)
            self.input_shape = None
        

        self.action_space = self.env.action_space


        # Hyperparameters
        self.episodes = episodes
        self.batch_size = batch_size
        self.lr = lr
        self.replay_size = replay_size
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.target_update = target_update
        self.tau = tau
        self.eps_threshold = self.epsilon_start
        self.alpha = alpha
        self.beta = beta


        # Save interval for validation
        self.save_interval = save_interval
        # Model hidden layer size
        self.hidden_size = hidden_size
        # Use Double DQN
        self.use_double_dqn = use_double_dqn
        # Use Prioritized Experience Replay (PER)
        self.use_per = use_per
        # Use Dueling DQN
        self.dueling_dqn = dueling_dqn

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        # Initialize policy and target networks
        # NOTE: we use the name policy_net for the main Q-network to be consistent with the DQN paper,
        # in reality, this is the Q-network that gets updated every step and is used to select actions,
        # but there is no policy in DQN, instead the actions are selected via epsilon-greedy from the Q-values.
        if self.dueling_dqn:
            if not self.is_cnn:
                self.policy_net = DuelingMLPNetwork(input_dim=self.obs_dim, output_dim=self.action_space.n).to(self.device)
                self.target_net = DuelingMLPNetwork(input_dim=self.obs_dim, output_dim=self.action_space.n).to(self.device)
            else:
                self.policy_net = DuelingCNNNetwork(input_shape=self.input_shape, output_dim=self.action_space.n).to(self.device)
                self.target_net = DuelingCNNNetwork(input_shape=self.input_shape, output_dim=self.action_space.n).to(self.device)
        else:
            if not self.is_cnn:
                self.policy_net = MLPNetwork(input_dim=self.obs_dim, output_dim=self.action_space.n).to(self.device)
                self.target_net = MLPNetwork(input_dim=self.obs_dim, output_dim=self.action_space.n).to(self.device)
            else:
                self.policy_net = CNNNetwork(input_shape=self.input_shape, output_dim=self.action_space.n).to(self.device)
                self.target_net = CNNNetwork(input_shape=self.input_shape, output_dim=self.action_space.n).to(self.device)
        # Target network, which is used for stable Q-value targets gets updated less frequently
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)


        # Initialize replay memory buffer
        if self.use_per:
            self.memory = PrioritizedReplayMemory(replay_size, self.alpha, self.beta)
            print("Using Prioritized Experience Replay")
        else:
            self.memory = ReplayMemory(replay_size)
            print("Using Standard Replay Memory")

        self.steps_done = 0


        # Logger
        self.env_tag = env_name.lower().replace('-', '_').split('/')[-1]
        self.num_params = sum(p.numel() for p in self.policy_net.parameters())
        self.variant_tag = "dqn"
        if self.use_double_dqn:
            self.variant_tag += "_double"
        if self.dueling_dqn:
            self.variant_tag += "_dueling"
        if self.use_per:
            self.variant_tag += "_PERreplay"
        self.logger = Logger(self.env_tag, self.env_name, self.action_space.n, self.variant_tag, self.num_params)



    def select_action(self, state):
        """Select action using epsilon-greedy policy."""
        # sample a random action that will be used to select between exploration and exploitation
        sample = random.random()

        # compute epsilon threshold
        # HINT: epsilon threshold decays from epsilon_start to epsilon_end 
        # via negative exponential wrt steps_done and epsilon_decay
        self.eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-1. * self.steps_done / self.epsilon_decay)
        
        # increment step count for epsilon decay
        self.steps_done += 1

        if sample > self.eps_threshold:
            # return greedy action from policy network. This is exploitation
            with torch.no_grad():
                if self.is_cnn:
                    # For image observations, preprocess and keep dimensions
                    state_processed = utils.preprocess_image(state)
                    state_tensor = torch.from_numpy(state_processed).float().unsqueeze(0).to(self.device)
                else:
                    # For vector observations, encode and flatten
                    state_encoded = utils.encode_obs(state, self.env.observation_space)
                    state_tensor = torch.from_numpy(state_encoded).float().unsqueeze(0).to(self.device)

                # TODO: Return the greedy action from the policy network
                # HINT: You need to get the action with the highest Q-value from the policy.
                return self.policy_net(state_tensor).argmax(1).item()
        else:
            # TODO: return the return random action. 
            # HINT: This is exploration
            return self.env.action_space.sample()
        

    def optimize_model(self):
        """Optimize the model via sampling from the replay memory"""
        if len(self.memory) < self.batch_size:
            return
        
        # sample a batch of transitions. A transition is ('state', 'action', 'next_state', 'reward')
        # based on the memory type
        if self.use_per:
            # * PER Implementation
            transitions, indices, weights = self.memory.sample(self.batch_size)
            # Convert weights to device
            weights = weights.to(self.device)
        else:
            # * Standard Replay Implementation
            transitions = self.memory.sample(self.batch_size)
            weights = None  # Uniform weights for standard replay

        batch = Transition(*zip(*transitions))


        # Create masks for non-final states (to make sure we don't use final next states in target Q-value computation)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)

        if self.is_cnn:
            # For image observations:
            non_final_next_states = torch.cat([torch.from_numpy(utils.preprocess_image(s)).float().unsqueeze(0) 
                                    for s in batch.next_state if s is not None]).to(self.device)
            state_batch = torch.cat([torch.from_numpy(utils.preprocess_image(s)).float().unsqueeze(0) 
                                    for s in batch.state]).to(self.device)
        else:
            # For vector observations:
            non_final_next_states = torch.cat([torch.from_numpy(utils.encode_obs(s, self.env.observation_space)).float().unsqueeze(0) 
                                    for s in batch.next_state if s is not None]).to(self.device)
            state_batch = torch.cat([torch.from_numpy(utils.encode_obs(s, self.env.observation_space)).float().unsqueeze(0) 
                                    for s in batch.state]).to(self.device)

        action_batch = torch.tensor([[a] for a in batch.action], device=self.device)
        reward_batch = torch.cat(batch.reward)


        # Compute Q(s_t, a) for current states.
        #   The model computes Q(s_t), then we select the columns of actions taken. 
        #   These are the actions which would've been taken for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)


        # Compute V(s_t+1) for all next states.
        # We compute expected values of action for next states based on the "older" target_net.
        # Select their best reward with max(1).values from the target_net.
        # We will have either the expected state value or 0 in case the state was final (via the mask)
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            if self.use_double_dqn:
                # * Double DQN Implementation
                # TODO: Get the actions that maximize the Q values from the policy network (argmax)
                best_actions = 0#TODO
                # TODO: Get Q-values from target network using the actions from policy network. The target network only evaluates the chosen action's Q-value
                best_actions = self.policy_net(non_final_next_states).max(1).indices.unsqueeze(1)
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).gather(1, best_actions).squeeze(1)
            else:
                # * DQN Implementation
                # TODO: Take all Q-values from the target network and pick the max values directly again with the target network
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values


        # Compute the expected Q values (reward + discounted max next state value)
        expected_state_action_values = reward_batch + (next_state_values * self.gamma)

        # Compute the TD errors for PER, so the weights can be updated
        with torch.no_grad():
            td_errors = expected_state_action_values.unsqueeze(1) - state_action_values


        # Compute Huber loss (similar to MSE loss but less sensitive to outliers (outlier = large TD error))

        # We compute the loss between the state-action values and the expected state-action values
        if self.use_per:
            # * PER Implementation
            # Multiply the loss by the importance sampling weights that we get from the PER buffer
            criterion = nn.SmoothL1Loss(reduction='none')
            losses = (criterion(state_action_values, expected_state_action_values.unsqueeze(1)))
            loss = (weights * losses).mean()
        else:
            # * Standard Replay Implementation
            criterion = nn.SmoothL1Loss()
            loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))


        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)

        self.optimizer.step()


        # Update priorities in PER buffer based on the new TD errors
        if self.use_per:
            td_errors = td_errors.detach().cpu().numpy()
            self.memory.update_priorities(indices, td_errors)

        return loss.item()


    def train(self):
        """Main training loop for DQN agent."""

        for i_episode in range(self.episodes):
            state, info = self.env.reset()
            total_reward = 0.0
            episode_loss = 0.0
            optimization_steps = 0

            for t in count():
                # Select action via the epsilon-greedy policy
                action = self.select_action(state)
                # Step the environment
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                # Accumulate reward
                total_reward += reward

                reward = torch.tensor([reward], device=self.device)

                if terminated or truncated:
                    next_state = None

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                loss = self.optimize_model()

                if loss is not None:
                    episode_loss += loss
                    optimization_steps += 1

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                if self.tau > 0 and t % self.target_update == 0:  # Soft update every target_update steps
                    target_net_state_dict = self.target_net.state_dict()
                    policy_net_state_dict = self.policy_net.state_dict()
                    for key in policy_net_state_dict:
                        target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
                    self.target_net.load_state_dict(target_net_state_dict)
                elif self.tau == 0 and t % self.target_update == 0:  # Hard update every target_update steps
                    self.target_net.load_state_dict(self.policy_net.state_dict())

                if terminated or truncated:
                    break

            # Log episode statistics
            self.logger.add_value('train-dqn/episode_reward', total_reward, i_episode)

            if optimization_steps > 0:
                avg_episode_loss = episode_loss / optimization_steps
                self.logger.add_value('train-dqn/avg_loss', avg_episode_loss, i_episode)

            self.logger.add_value('train-dqn/epsilon', self.eps_threshold, i_episode)

            # Save model and evaluate at intervals
            if i_episode % self.save_interval == 0:
                if not os.path.exists('checkpoints'):
                    os.makedirs('checkpoints')
                torch.save(self.policy_net.state_dict(), f'checkpoints/{self.variant_tag}_{self.env_name}_episode_{i_episode}.pth')
                
                # Evaluate policy and log video
                avg_eval_reward = utils.evaluate_policy(self.policy_net, self.env.observation_space, self.env_name, episodes=3, return_frames=False)
                self.logger.add_value('train-dqn/eval_mean_reward', avg_eval_reward, i_episode)
                
                # Log evaluation video
                _, frames = utils.evaluate_policy(self.policy_net, self.env.observation_space, self.env_name, episodes=1, return_frames=True, max_length=1000)
                self.logger.add_frames('train-dqn/eval_video', frames, i_episode)

            print(f"Episode {i_episode} - Total Reward: {total_reward:.2f}, Epsilon: {self.eps_threshold:.3f}")

        print('Training complete')
        self.env.close()







