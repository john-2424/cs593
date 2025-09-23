"""
CS 593 RL1 Homework Assignment 1
Purdue University
Created by: Joseph Campbell and Guven Gergerli
"""

import os
import time
import pickle
import gymnasium as gym

class Collector:
    """Collects demonstrations from keyboard input:
      - CartPole: left/right via keys
      - Blackjack: pygame UI or terminal fallback
    """

    def __init__(self, env_name = "CartPole-v1"):
        self.env_name = env_name


    # ---------------------- Human Collection Entry ----------------------
    def collect_human_demonstrations(self, num_episodes= 5, frame_delay= 0.05):
        if 'Blackjack' in self.env_name:
            return self._collect_human_blackjack(num_episodes, frame_delay=frame_delay)
        if 'CartPole' in self.env_name:
            return self._collect_human_cartpole(num_episodes, frame_delay=frame_delay)
        raise ValueError('Unsupported env for human collection')


    # ---------------------- Environment Specific Human Control ----------------------
    def _collect_human_cartpole(self, num_episodes, frame_delay= 0.05):
        try:
            import pygame
        except ImportError:
            print("pygame missing; install pygame for human CartPole control (pip install pygame). Falling back to random.")
            return self.collect_random_demonstrations(num_episodes)
        
        print("Controls (CartPole):")
        print("  Left Arrow / A : Action 0 (push cart left)")
        print("  Right Arrow / D: Action 1 (push cart right)")
        print("  ESC            : End current episode")
        input("Press Enter to start...")
        pygame.init()
        env = gym.make(self.env_name, render_mode='human')
        states, actions = [], []
        key_left = [getattr(__import__('pygame'), 'K_LEFT'), getattr(__import__('pygame'), 'K_a')]
        key_right = [getattr(__import__('pygame'), 'K_RIGHT'), getattr(__import__('pygame'), 'K_d')]
        try:
            for ep in range(num_episodes):
                state, _ = env.reset(); done=False; current_action=0; ep_reward=0.0
                while not done:
                    env.render()
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            done=True
                        elif event.type == pygame.KEYDOWN:
                            if event.key == getattr(__import__('pygame'), 'K_ESCAPE'):
                                done=True
                            elif event.key in key_left:
                                current_action = 0
                            elif event.key in key_right:
                                current_action = 1
                    states.append(state); actions.append(current_action)
                    state, r, term, trunc, _ = env.step(current_action); ep_reward += r
                    done = term or trunc
                    if frame_delay > 0:
                        time.sleep(frame_delay)
                print(f"[Human CartPole] Episode {ep+1}/{num_episodes} reward={ep_reward:.2f}")
        finally:
            env.close(); pygame.quit()
        return states, actions


    def _collect_human_blackjack(self, num_episodes, frame_delay = 0.05):
        try:
            import pygame
            use_pygame = True
        except ImportError:
            use_pygame = False

        if not use_pygame:
            print("pygame missing; using terminal Blackjack mode.")
            print("Controls (Blackjack): h=Hit(1), s=Stick(0)")
            states, actions = [], []
            env = gym.make(self.env_name)
            for ep in range(num_episodes):
                state, _ = env.reset(); done=False; ep_reward=0.0
                while not done:
                    print(f"State: {state}")
                    act_in = input("Action [h=hit / s=stick / q=quit episode]: ").strip().lower()
                    if act_in == 'q':
                        print("Ending episode early."); break
                    if act_in not in ['h','s']:
                        print("Invalid input. Use h or s."); continue
                    action = 1 if act_in == 'h' else 0
                    states.append(state); actions.append(action)
                    state, r, term, trunc, _ = env.step(action); done = term or trunc
                    ep_reward += r
                print(f"[Human Blackjack-Terminal] Episode {ep+1}/{num_episodes} reward={ep_reward:.2f}")
            env.close(); return states, actions
        print("Blackjack pygame controls: H=Hit, S=Stand, ESC=Abort episode")


        pygame.init()
        font = pygame.font.SysFont(None, 32)
        screen = pygame.display.set_mode((540, 260))
        env = gym.make(self.env_name)
        states, actions = [], []
        try:
            for ep in range(num_episodes):
                state, _ = env.reset(); done=False; last_reward=0.0; ep_reward=0.0
                while not done:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            done=True
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_ESCAPE:
                                done=True
                            elif event.key == pygame.K_h:
                                action=1; states.append(state); actions.append(action)
                                state, last_reward, term, trunc, _ = env.step(action); ep_reward += last_reward; done = term or trunc
                            elif event.key == pygame.K_s:
                                action=0; states.append(state); actions.append(action)
                                state, last_reward, term, trunc, _ = env.step(action); ep_reward += last_reward; done = term or trunc
                    screen.fill((25,25,30))
                    p_sum, dealer_card, usable_ace = state
                    lines = [
                        f'Episode {ep+1}/{num_episodes}',
                        f'Player sum: {p_sum}',
                        f'Dealer showing: {dealer_card}',
                        f'Usable ace: {usable_ace}',
                        f'Last reward: {last_reward}',
                        'H=Hit  S=Stand  ESC=Abort'
                    ]
                    y=25
                    for ln in lines:
                        surf = font.render(ln, True, (220,220,220))
                        screen.blit(surf, (20,y)); y+=34
                    pygame.display.flip()
                    if frame_delay>0: time.sleep(frame_delay)
                print(f"[Human Blackjack] Episode {ep+1}/{num_episodes} reward={ep_reward:.2f}")
                if ep < num_episodes-1: time.sleep(0.6)
        finally:
            env.close(); pygame.quit()
        return states, actions


    # ---------------------- Save / Load ----------------------
    def save_demonstrations(self, states, actions, filename):
        os.makedirs('data', exist_ok=True)
        env_tag = self.env_name.lower().replace('-', '_').split('/')[-1]
        # prepend env tag if not already included
        if not filename.startswith(env_tag):
            filename = f'{env_tag}_{filename}'
        path = os.path.join('data', filename)
        with open(path, 'wb') as f:
            pickle.dump({'states': states,'actions': actions,'env_name': self.env_name,'num_transitions': len(states),'collection_time': time.time()}, f)
        return path


    def load_demonstrations(self, filename):
        path = os.path.join('data', filename) if not filename.startswith('data/') else filename
        with open(path, 'rb') as f: data = pickle.load(f)
        return data['states'], data['actions']
