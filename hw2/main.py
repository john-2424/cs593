#!/usr/bin/env python3
"""
CS 593 RL1 Homework Assignment 2
Purdue University
Created by: Joseph Campbell and Guven Gergerli

Examples:
    python main.py --env CartPole-v1 
    python main.py --use_double_dqn --use_per --env Blackjack-v1

"""

import os
import argparse
from dqn import DQNAgent

DATA_DIR = "data"


def main():
    """Entry point for HW 2."""

    parser = argparse.ArgumentParser(
        description='CS 593 RL1: HW2',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--use_double_dqn', 
        action="store_true",
        help='Use Double DQN agent'
    )

    parser.add_argument(
        '--use_dueling_dqn',
        action="store_true",
        help='Use Dueling DQN agent'
    )

    parser.add_argument(
        '--use_per',
        action="store_true",
        help='Use Prioritized Experience Replay'
    )

    parser.add_argument(
        '--env', 
        type=str, 
        default='CartPole-v1',
        help='Gym environment (CartPole-v1, Blackjack-v1, LunarLander-v2, CarRacing-v2)'
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
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
        default=100,
        help='Number of training epochs'
    )

    parser.add_argument(
        '--save_interval',
        type=int,
        default=20,
        help='How frequently to log progress, videos, and save models'
    )

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
        '--epsilon_start',
        type=float,
        default=0.9,
        help='Starting value for epsilon in epsilon-greedy strategy'
    )

    parser.add_argument(
        '--epsilon_end',
        type=float,
        default=0.05,
        help='Final value for epsilon in epsilon-greedy strategy'
    )

    parser.add_argument(
        '--epsilon_decay',
        type=int,
        default=15000,
        help='Rate of decay for epsilon in epsilon-greedy strategy'
    )

    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='Discount factor for future rewards'
    )

    parser.add_argument(
        '--target_update',
        type=int,
        default=2,
        help='How often to update the target network'
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
        default=0.6,
        help='Prioritization exponent for PER'
    )

    parser.add_argument(
        '--beta',
        type=float,
        default=0.4,
        help='Importance sampling exponent for PER'
    )

    args = parser.parse_args()

    # Ensure directories exist
    os.makedirs(DATA_DIR, exist_ok=True)


    dqn_agent = DQNAgent(
        env_name=args.env,
        episodes=args.num_epochs,
        batch_size=args.batch_size,
        save_interval=args.save_interval,
        lr=args.lr,
        hidden_size=args.hidden_size,
        replay_size=args.replay_size,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        gamma=args.gamma,
        target_update=args.target_update,
        tau=args.tau,
        alpha=args.alpha,
        beta=args.beta,
        use_double_dqn=args.use_double_dqn,
        use_per=args.use_per,
        dueling_dqn=args.use_dueling_dqn
    )
    
    dqn_agent.train()


if __name__ == '__main__':
    main()

