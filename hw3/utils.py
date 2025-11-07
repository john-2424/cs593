"""
CS 593 RL1 Homework Assignment 3
Purdue University
Created by: Joseph Campbell and Guven Gergerli
"""

import gymnasium as gym
import numpy as np
import os
import torch


def get_observation_dim(space):
    """Determine observation dimension based on space type."""

    if isinstance(space, gym.spaces.Box):
        return int(np.prod(space.shape))
    elif isinstance(space, gym.spaces.Discrete):
        return space.n
    elif isinstance(space, gym.spaces.Tuple):
        return sum(s.n for s in space.spaces if isinstance(s, gym.spaces.Discrete))
    else:
        raise ValueError('Unsupported observation space')


def encode_obs(obs, space):
    """
    Convert observations to flat vector representation based on space type.
    
    Args:
        obs: Raw observation from environment
        space: The observation space
        
    Returns:
        Encoded observation as flat numpy array
    """

    # if isinstance(space, gym.spaces.Box):
    #     return np.array(obs, dtype=np.float32).reshape(-1)
    if isinstance(space, gym.spaces.Box):
        if len(space.shape) >= 3:
            # For image observations, return as is (don't flatten)
            return np.array(obs, dtype=np.float32)
        else:
            return np.array(obs, dtype=np.float32).reshape(-1)


    if isinstance(space, gym.spaces.Discrete):
        vec = np.zeros(space.n, dtype=np.float32)
        vec[int(obs)] = 1.0
        return vec
        
    if isinstance(space, gym.spaces.Tuple):
        parts = []
        for sub, val in zip(space.spaces, obs):
            if isinstance(sub, gym.spaces.Discrete):
                one_hot = np.zeros(sub.n, dtype=np.float32)
                one_hot[int(val)] = 1.0
                parts.append(one_hot)
            else:
                raise ValueError('Unsupported subspace in Tuple')
        return np.concatenate(parts).astype(np.float32)
        
    raise ValueError('Unsupported space')


def preprocess_image(obs):
    # """
    # Preprocess image observations for CNN input
    # - Convert from (H, W, C) to (C, H, W) format
    # - Normalize values to [0, 1]
    # """
    # # Convert from HWC to CHW format (height, width, channels) -> (channels, height, width)
    # if len(obs.shape) == 3 and obs.shape[2] == 3:  # RGB image
    #     obs = np.transpose(obs, (2, 0, 1))
    
    # Normalize to [0, 1]
    obs = obs.astype(np.float32) / 255.0
    
    return obs


def evaluate_policy(policy, obs_space, env_name, episodes=5, return_frames=False, max_length=1000):
    """
    Evaluates the policy in the given environment
    
    Args:
        policy: The policy to evaluate
        obs_space: Observation space for preprocessing
        env_name: Name of the environment to create
        episodes: Number of episodes to evaluate
        return_frames: Whether to return frames for video logging
        max_length: Maximum length of each episode
        
    Returns:
        mean_reward: Mean reward across episodes
        frames: List of frames if return_frames=True
    """
    env = gym.make(env_name, render_mode="rgb_array" if return_frames else None)
    is_cnn = False

    policy.eval()
    rewards = []
    frames = []
    
    device = next(policy.parameters()).device

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0
        t = 0
        episode_frames = []

        if return_frames:
            try:
                if is_cnn:
                    frame = env.env.render()
                else:
                    frame = env.render()
                if frame is not None:
                    episode_frames.append(frame)
                else:
                    episode_frames.append(np.zeros((100, 100, 3), dtype=np.uint8))
            except Exception as e:
                print(f"Warning: Failed to render initial frame: {e}")
                episode_frames.append(np.zeros((100, 100, 3), dtype=np.uint8))

        while not done:
            # Process observation and select action
            if is_cnn:
                obs_vec = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            else:
                # For vector observations, encode normally
                obs_vec = torch.from_numpy(encode_obs(obs, obs_space)).float().unsqueeze(0).to(device)
            
            # Get mean and std from policy
            with torch.no_grad():
                mean, std = policy(obs_vec)
                dist = torch.distributions.Normal(mean, std)
                raw_action = dist.sample()
                
                # Scale action to the environment's action space
                low = torch.tensor(env.action_space.low, dtype=torch.float32)
                high = torch.tensor(env.action_space.high, dtype=torch.float32)
                scaled_action = low + (0.5 * (raw_action + 1.0) * (high - low))
            
            # Execute action - ensure proper shape
            action_np = scaled_action.cpu().numpy()
            if len(action_np.shape) > 1 and action_np.shape[0] == 1:
                action_np = action_np.flatten()
                
            obs, reward, terminated, truncated, _ = env.step(action_np)
            
            # Handle reward if it's an array
            if isinstance(reward, np.ndarray):
                reward = float(reward.item()) if reward.size == 1 else float(reward.sum())

            if return_frames:
                try:
                    frame = env.render()
                    if frame is not None:
                        episode_frames.append(frame)
                    else:
                        if episode_frames:
                            episode_frames.append(episode_frames[-1])
                        else:
                            episode_frames.append(np.zeros((100, 100, 3), dtype=np.uint8))
                except Exception as e:
                    print(f"Warning: Failed to render frame: {e}")
                    if episode_frames:
                        episode_frames.append(episode_frames[-1])
                    else:
                        episode_frames.append(np.zeros((100, 100, 3), dtype=np.uint8))

            episode_reward += reward
            done = terminated or truncated
            t += 1

            if max_length is not None and t >= max_length:
                break
        
        rewards.append(episode_reward)
        if return_frames:
            frames.append(episode_frames)
    
    env.close()

    avg_reward = float(np.mean(rewards)) if rewards else 0.0

    policy.train()

    if return_frames:
        return avg_reward, frames
    
    return avg_reward
    
    



def get_source_tag(filename):
    """Determine the source type from the demonstration filename"""

    filename = filename.lower()
    if 'human' in filename:
        return 'human'
    elif 'random' in filename:
        return 'random'
    elif 'policy' in filename:
        return 'policy'
    else:
        return 'unknown'
    

def select_demonstration_file(data_dir):
    """Helper to select a demonstration file from the data directory"""

    files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]
    
    if not files:
        print(f"No .pkl demonstration files found in {data_dir}.")
        print("Run phase 1 to collect demonstrations first.")
        return None
        
    print("Available demonstration files:")
    for i, filename in enumerate(files, 1):
        print(f"  {i}. {filename}")
        
    choice = input(f"Select file number (1-{len(files)}) or enter name: ").strip()
    
    if choice.isdigit() and 1 <= int(choice) <= len(files):
        return files[int(choice) - 1]
        
    # Handle free-form name input
    if not choice.endswith('.pkl'):
        choice += '.pkl'
        
    if choice in files:
        return choice
    else:
        print(f"File '{choice}' not found in data directory.")
        return None

