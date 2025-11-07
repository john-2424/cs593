#!/usr/bin/env python3
"""
CS 593 RL1 Homework Assignment 3
Purdue University
Created by: Joseph Campbell and Guven Gergerli
"""

import os
import argparse
from reinforce import REINFORCEAgent
from sac import SACAgent
from ppo import PPOAgent

DATA_DIR = "data"


def main():
    """Entry point for HW 3."""

    parser = argparse.ArgumentParser(
        description='CS 593 RL1: HW3',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--reinforce', 
        action="store_true",
        help='Use REINFORCE agent'
    )

    parser.add_argument(
        '--sac', 
        action="store_true",
        help='Use Soft Actor-Critic (SAC) agent'
    )

    parser.add_argument(
        '--ppo',
        action="store_true",
        help='Use Proximal Policy Optimization (PPO) agent'
    )

    parser.add_argument(
        '--env', 
        type=str, 
        default='Pendulum-v1',
        help='Gym environment (Pendulum-v1, BipedalWalker-v3, MountainCarContinuous-v0, CarRacing-v2)'
    )

    # Common hyperparameters
    parser.add_argument(
        '--lr',
        type=float,
        default=5e-5,
        help='Learning rate'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=256,
        help='Batch size'
    )

    parser.add_argument(
        '--num_epochs',
        type=int,
        default=3000,
        help='Number of training epochs/episodes'
    )

    parser.add_argument(
        '--save_interval',
        type=int,
        default=50,
        help='How frequently to log progress, videos, and save models'
    )

    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='Discount factor for future rewards'
    )

    # SAC specific hyperparameters
    parser.add_argument(
        '--hidden_size',
        type=int,
        default=128,
        help='Hidden layer size for neural networks'
    )

    parser.add_argument(
        '--replay_size',
        type=int,
        default=10000,
        help='Size of the replay buffer'
    )

    parser.add_argument(
        '--tau',
        type=float,
        default=0.005,
        help='Soft update parameter for target network'
    )

    parser.add_argument(
        '--alpha',
        type=float,
        default=0.2,
        help='Entropy regularization coefficient for SAC'
    )
    
    parser.add_argument(
        '--updates_per_step',
        type=int,
        default=1,
        help='Number of updates per environment step for SAC'
    )
    
    parser.add_argument(
        '--start_steps',
        type=int,
        default=10000,
        help='Number of initial random exploration steps for SAC'
    )

    # PPO specific hyperparameters
    parser.add_argument(
        '--gae_lambda',
        type=float,
        default=0.95,
        help='GAE lambda parameter for PPO'
    )
    
    parser.add_argument(
        '--clip_ratio',
        type=float,
        default=0.2,
        help='PPO clip ratio/epsilon'
    )
    
    parser.add_argument(
        '--value_coef',
        type=float,
        default=0.5,
        help='Value function loss coefficient for PPO'
    )
    
    parser.add_argument(
        '--entropy_coef',
        type=float,
        default=0.01,
        help='Entropy loss coefficient for PPO'
    )
    
    parser.add_argument(
        '--update_epochs',
        type=int,
        default=10,
        help='Number of epochs to update policy for each batch in PPO'
    )


    args = parser.parse_args()

    # Ensure directories exist
    os.makedirs(DATA_DIR, exist_ok=True)

    if args.reinforce:
        print("Training REINFORCE agent...")
        reinforce_agent = REINFORCEAgent(
            env_name=args.env,
            lr=args.lr,
            num_episodes=args.num_epochs,
            batch_size=args.batch_size,
            gamma=args.gamma,
            save_interval=args.save_interval
        )
        reinforce_agent.train_reinforce()

    elif args.sac:
        print("Training SAC agent...")
        sac_agent = SACAgent(
            env_name=args.env,
            lr=args.lr,
            num_episodes=args.num_epochs,
            batch_size=args.batch_size,
            gamma=args.gamma,
            tau=args.tau,
            hidden_size=args.hidden_size,
            buffer_size=args.replay_size,
            alpha=args.alpha,
            updates_per_step=args.updates_per_step,
            start_steps=args.start_steps,
            save_interval=args.save_interval
        )
        sac_agent.train_sac()

    elif args.ppo:
        print("Training PPO agent...")
        ppo_agent = PPOAgent(
            env_name=args.env,
            lr=args.lr,
            num_episodes=args.num_epochs,
            batch_size=args.batch_size,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_ratio=args.clip_ratio,
            value_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
            update_epochs=args.update_epochs,
            save_interval=args.save_interval
        )
        ppo_agent.train_ppo()
        
    else:
        print("Please specify an agent to use: --reinforce --ppo --sac")



if __name__ == '__main__':
    main()




