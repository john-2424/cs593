#!/usr/bin/env python3
"""
CS 593 RL1 Homework Assignment 1
Purdue University
Created by: Joseph Campbell and Guven Gergerli

This script provides a command-line interface for a complete RL workflow:
1. Collect (expert?) demonstrations with keyboard input
2. Train imitation learning models using behavioral cloning

Examples:
    python main.py --env CartPole-v1 
    python main.py --env Blackjack-v1

    python main.py --env CartPole-v1 --collect
    python main.py --env Blackjack-v1 --collect
"""

import os
import argparse
import gymnasium as gym
import utils

from datasets import VectorDataset
from collector import Collector
from imitation import ImitationLearner


DATA_DIR = "data"

        
def collect_demonstrations(env_name):
    """Collect demonstrations using keyboard inputs"""

    print("\n==== Collecting Demonstrations ====")
    collector = Collector(env_name=env_name)
    
    try:
        states, actions, filename = None, None, None
        
        num_episodes = int(input("Number of episodes (default 50): ") or '50')
        print(f"Collecting {num_episodes} human demonstrations. Follow on-screen instructions...")
        states, actions = collector.collect_human_demonstrations(num_episodes)
        filename = 'human_demonstrations.pkl'
        
        # Save collected demonstrations
        if states and actions and filename:
            save_path = collector.save_demonstrations(states, actions, filename)
            print(f"Successfully saved {len(states)} state-action pairs to {save_path}")
            print("Next, run without --collect to train an imitation model using the newly collected data")
    
    except Exception as e:
        print(f"Error collecting demonstrations: {e}")
    

def train_imitation(env_name, lr, epochs, batch_size, save_interval):
    """Train an imitation learning model using behavioral cloning"""

    print("\n==== Training Imitation Model ====")
    
    # Check for demonstration data
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory '{DATA_DIR}' not found. Collect data first.")
        return
        
    # Select demonstration file
    filename = utils.select_demonstration_file(DATA_DIR)
    if not filename:
        return
        
    # TODO: Modify the below code to load_demonstrations
    # HINT: Use Collector to load states and actions

    collector = Collector(env_name=env_name)
    states, actions = collector.load_demonstrations(filename=filename)

    # Create dataset
    env_tmp = gym.make(env_name)
    dataset = VectorDataset(states, actions, env_tmp.observation_space)
    env_tmp.close()
    
    if len(dataset) == 0:
        print("Error: Loaded dataset is empty. Collect demonstrations first.")
        return
        
    # Determine source type
    source_tag = utils.get_source_tag(filename)
    
    # Initialize learner
    learner = ImitationLearner(
        env_name=env_name,
        source_tag=source_tag,
        lr=lr,
        epochs=epochs, 
        batch_size=batch_size, 
        save_interval=save_interval
    )
    
    # Train the model
    print(f"Starting imitation training:")
    print(f"  - File: {filename}")
    print(f"  - Dataset size: {len(dataset)} examples")
    print(f"  - Epochs: {epochs}, Batch size: {batch_size}")
    
    learner.train(
        dataset
    )
    
    print("\nTraining complete!")
    print("  - Check runs/ directory for TensorBoard logs")
        



def main():
    """Entry point for HW 1."""

    parser = argparse.ArgumentParser(
        description='CS 593 RL1: HW1',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--collect', 
        action="store_true",
        help='Collect demonstration data using keyboard input'
    )
    
    parser.add_argument(
        '--env', 
        type=str, 
        default='CartPole-v1',
        help='Gym environment (CartPole-v1, Blackjack-v1)'
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=1e-5,
        help='Learning rate'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size'
    )

    parser.add_argument(
        '--num_epochs',
        type=int,
        default=20,
        help='Number of training epochs'
    )

    parser.add_argument(
        '--save_interval',
        type=int,
        default=5,
        help='How frequently to log progress, videos, and save models'
    )
    
    args = parser.parse_args()

    # Ensure directories exist
    os.makedirs(DATA_DIR, exist_ok=True)

    if args.collect:
        collect_demonstrations(args.env)
    else:
        train_imitation(args.env, args.lr, args.num_epochs, args.batch_size, args.save_interval)


if __name__ == '__main__':
    main()

