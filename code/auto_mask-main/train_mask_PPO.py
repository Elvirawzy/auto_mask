from Maze_continuous import Gridworld
import pickle
import random
import zlib
import numpy as np
from PIL import Image
import time

import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[0]))
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter

from agents.DQN_agent import DQN
from agents.DQN_mask_agent import MaskDQN

log_time = str(int(time.time()))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(eps, render, environment, algorithm, media_path):
    done = False
    total_reward = 0
    state = environment.reset()
    fig_path = None

    if render:
        fig_path = 'episode' + str(eps)
        media_path = os.path.join(media_path, str(eps))
        if not os.path.exists(fig_path):
            os.mkdir(fig_path)

    while not done:
        if render:
            environment.render(fig_path)
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action, _, mask = algorithm.get_action(state)
        next_obs, reward, done, info = environment.step(action[0])
        total_reward += reward
        state = next_obs

    if render:
        environment.save2gif(fig_path, media_path)
        print(f'episode:{eps}, reward:{total_reward}')
        print(f'mask:{mask}')
        print(torch.sum(mask[0]))

    return total_reward


def train(args):
    print("action directions:", args.action_space)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
    else:
        torch.manual_seed(0)
    FILE_PATH = str(Path(__file__).resolve().parents[0])
    log_path = os.path.join(FILE_PATH, args.log_path, args.env,
                            args.algo + '_' + str(args.action_space) + '_' + log_time + '_reward')
    writer = SummaryWriter(log_path)
    media_path = os.path.join(FILE_PATH, args.media_path, args.env,
                              args.algo + '_' + str(args.action_space) + '_' + log_time)
    save_path = os.path.join(FILE_PATH, args.save_path, args.env,
                             args.algo + '_' + str(args.action_space) + '_' + log_time)
    os.mkdir(media_path)
    os.mkdir(save_path)

    env = Gridworld(n_actions=args.action_space, debug=False)
    if args.algo == "DQN":
        agent = DQN(
            obs_n=env.observation_space.shape[0],
            act_n=env.action_space.n,
            device=device,
            policy_lr=args.policy_lr,
            gamma=args.gamma,
            batch_size=args.batch_size,
            replay_memory_size=args.replay_memory_size,
            reload_freq=args.reload_freq
        )
    elif args.algo == "MaskDQN":
        agent = MaskDQN(
            load_flag=args.load_flag,
            obs_n=env.observation_space.shape[0],
            act_n=env.action_space.n,
            motion=env.motions,
            device=device,
            n_hidden=256,
            policy_lr=args.policy_lr,
            mask_lr=args.mask_lr,
            gamma=args.gamma,
            batch_size=args.batch_size,
            replay_memory_size=args.replay_memory_size,
            reload_freq=args.reload_freq,
            update_freq=args.update_freq,
        )
    else:
        agent = None

    if args.load_flag:
        agent.load_models(args.load_dir)

    for episode in range(args.num_episodes):
        # epsilon = args.final_epsilon + (max(num_decay_epochs - episode, 0) * (
        # 		args.initial_epsilon - args.final_epsilon) / num_decay_epochs)
        epsilon = 0.3

        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        done = False
        total_reward = 0
        transition_loss, reward_loss, mask_loss = 0, 0, 0

        while not done:
            sample = random.random()
            if sample > epsilon:
                with torch.no_grad():
                    action, _, _ = agent.get_action(state)
            else:
                if args.load_flag:
                    _, _, mask = agent.get_action(state)
                    choices = np.where(mask[0] == 1)[0]
                    action = torch.tensor(np.array([random.choice(choices)]))
                else:
                    action = torch.tensor(np.array([env.action_space.sample()]), device=device, dtype=torch.int64)
            next_obs, reward, done, info = env.step(action[0])
            total_reward += reward

            reward = torch.tensor([reward], device=device, dtype=torch.float32)
            next_state = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)

            agent.replay_memory.push(state, action.unsqueeze(0), next_state, reward, not bool(done))
            state = next_state
            transition_loss, reward_loss, mask_loss = agent.update(done, episode)

        if episode % args.log_interval == 0 or episode == args.num_episodes - 1:
            if episode % args.save_interval == 0 or episode == args.num_episodes - 1:
                total_reward = evaluate(episode, 1, env, agent, media_path)
            else:
                total_reward = evaluate(episode, 0, env, agent, media_path)
            writer.add_scalar('reward', total_reward, episode)
            writer.add_scalar('transition_loss', transition_loss, episode)
            writer.add_scalar('reward_loss', reward_loss, episode)

        if episode % args.update_freq == 0:
            writer.add_scalar('mask_loss', mask_loss, episode)

        if episode == args.num_episodes - 1:
            agent.save_models(save_path)


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Tetris""")
    parser.add_argument("--env", type=str, default="Maze", choices=["Maze"])
    parser.add_argument("--action_space", type=int, default=10, help="action directions for Maze")
    parser.add_argument("--algo", type=str, default="DQN", choices=["DQN", "MaskDQN"])

    parser.add_argument("--batch_size", type=int, default=256, help="The number of images per batch")
    parser.add_argument('--replay_memory_size', type=int, default=2000)
    parser.add_argument("--policy_lr", type=float, default=1e-3)
    parser.add_argument("--mask_lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.9)

    parser.add_argument("--num_episodes", type=int, default=1500)
    # parser.add_argument("--max_steps", type=int, default=1500)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--reload_freq", type=int, default=50)
    parser.add_argument("--update_freq", type=int, default=50)

    parser.add_argument("--log_path", type=str, default="runs")
    parser.add_argument("--media_path", type=str, default="media")
    parser.add_argument("--save_path", type=str, default="trained_models")
    # parser.add_argument("--load_dir", type=str,
    # default="/Users/elvirawzy/Desktop/wzy/mask_for_RL/trained_models/Maze/MaskDQN_10_1687884638_mask")
    # parser.add_argument("--load_dir", type=str, default="/Users/elvirawzy/Desktop/wzy/mask_for_RL/auto_mask/trained_models/Maze/MaskDQN_12_1688910502_mask")
    parser.add_argument("--load_dir", type=str, default="/Users/elvirawzy/Desktop/wzy/mask_for_RL/auto_mask/trained_models/Maze/MaskDQN_12_1689498963_reward_mask")
    parser.add_argument("--load_flag", type=bool, default=False)

    args = parser.parse_args()
    return args


args = get_args()

train(args)
