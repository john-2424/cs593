"""
CS 593 RL1 Homework Assignment 2
Purdue University
Created by: Joseph Campbell and Guven Gergerli
"""

import gymnasium as gym
import numpy as np
import os
import torch
from wrapper import FrameStackResize


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


def evaluate_policy(policy, obs_space, env_name, episodes = 5, return_frames = False, max_length = None):
    """
    Evaluate the current policy on the environment.
    
    Args:
        episodes: Number of evaluation episodes
        env_name: The gym environment name
        episodes: The number of episodes to evaluate for
        return_frames: Whether to return rendered frames (adds an additional return value)
        max_length: The maximum number of steps to evaluate a single episode for
        
    Returns:
        Average reward across episodes
        (Optional) Rendered frames from each episode
    """

    if env_name == 'CarRacing-v2':
        env = gym.make(env_name, render_mode="rgb_array", continuous=False)
        env = FrameStackResize(env, num_stack=4, resize_shape=(84, 84))
        is_cnn = True
    else:
        env = gym.make(env_name, render_mode="rgb_array")
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

        if return_frames:
            # For CarRacing render the unwrapped env to get the RGB image
            if is_cnn:
                frames.append([env.env.render()])
            else:
                frames.append([env.render()])
        
        while not done:

            # Process observation and select action
            if is_cnn:
                # For image observations, preprocess to channel-first format
                # obs_processed = preprocess_image(obs)
                # obs_vec = torch.from_numpy(obs_processed).float().unsqueeze(0)
                obs_vec = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            else:
                # For vector observations, encode normally
                obs_vec = torch.from_numpy(encode_obs(obs, obs_space)).float().unsqueeze(0).to(device)


            with torch.no_grad():
                logits = policy(obs_vec)
                # TODO: Select the action with highest probability from logits
                action = logits.argmax(dim=-1).item()#
            
            # Execute action in environment
            # TODO: Step the environment using the selected action and accumulate reward
            obs, reward, terminated, truncated, _ = env.step(action)#

            if return_frames:
                frames[-1].append(env.render())

            episode_reward += reward
            done = terminated or truncated

            t += 1

            if max_length is not None and t >= max_length:
                break
            
        rewards.append(episode_reward)
        
    env.close()

    avg_reward = float(np.mean(rewards)) if rewards else 0.0

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

