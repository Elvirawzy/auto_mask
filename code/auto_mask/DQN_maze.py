from Maze_continuous import Gridworld
import random
import numpy as np
import time

import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[0]))
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
from agents.DQN_mask_agent import DQN, MaskDQN
import  matplotlib.pyplot as plt
from matplotlib.patches import Arrow, Circle
from matplotlib.ticker import NullLocator

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def draw_motion(motion, flag):
    angles = np.linspace(0, 2 * np.pi, 10 + 1)[:-1]
    dirs = list(zip(np.cos(angles), np.sin(angles)))

    plt.figure(figsize=(10, 10))
    if flag:
        plt.title('Maze-Original Action Space')
        flag=0
    else:
        plt.title('Maze-Masked Action Space')

    currentAxis = plt.gca()                        #Turns off the the boundary padding
    currentAxis.xaxis.set_major_locator(NullLocator()) #Turns of ticks of x axis
    currentAxis.yaxis.set_major_locator(NullLocator()) #Turns of ticks of y axis
    plt.ion()

    for i, m in enumerate(motion):
        arrow = Arrow(0.5, 0.5, m[0]*0.5, m[1]*0.5, width=0.05, fill=True, color='darkblue', alpha=0.1)
        currentAxis.add_patch(arrow)
    currentAxis.add_patch(Circle((0.5, 0.5), 0.005, color='black'))
    
    plt.show()

def evaluate(environment, algorithm):
    done = False
    total_reward = 0
    state = environment.reset()

    while not done:
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action, _, mask = algorithm.get_action(state)

        motion = environment.motions
        draw_motion(motion, 1)

        masked_motion = mask.view(-1, 1) * motion
        draw_motion(masked_motion, 0)
        next_obs, reward, done, info = environment.step(action[0])
        total_reward += reward
        state = next_obs

    return total_reward


def train(args):
    print("action directions:", args.action_space)
    FILE_PATH = str(Path(__file__).resolve().parents[0])
    log_time = str(int(time.time()))

    writer = SummaryWriter(os.path.join(FILE_PATH, args.log_path, args.env, args.algo + '_' + str(args.action_space) + '_' + log_time))
    media_path = os.path.join(FILE_PATH, args.media_path, args.env, args.algo + '_' + str(args.action_space) + '_' + log_time)
    save_path = os.path.join(FILE_PATH, args.save_path, args.env, args.algo + '_' + str(args.action_space) + '_' + log_time)

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
            obs_n=env.observation_space.shape[0],
            act_n=env.action_space.n,
            motion=env.motions,
            device=device,
            load_flag=args.load_model,
            policy_lr=args.policy_lr,
            mask_lr=args.mask_lr,
            gamma=args.gamma,
            batch_size=args.batch_size,
            replay_memory_size=args.replay_memory_size,
            reload_freq=args.reload_freq,
            trans_freq=args.trans_freq,
        )
    else:
        agent = None

    if args.load_model:
        agent.load_model('trained_models/Maze/MaskDQN_10_1702345183')

    for episode in range(args.num_episodes):
        epsilon = 0.3

        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        done = False
        total_reward = 0

        while not done:
            sample = random.random()
            if sample > epsilon:
                with torch.no_grad():
                    action, _, mask = agent.get_action(state)
            else:
                if args.algo == "MaskDQN":
                    _, _, mask = agent.get_action(state)
                    indexs = mask.squeeze(0).nonzero()
                    random_index = torch.randint(0, indexs.size(0), (1,)).item()
                    action = indexs[random_index]
                else:
                    action = torch.tensor(np.array([env.action_space.sample()]), device=device, dtype=torch.int64)
            next_obs, reward, done, info = env.step(action[0])
            total_reward += reward

            reward = torch.tensor([reward], device=device, dtype=torch.float32)
            next_state = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)

            agent.replay_memory.push(state, action.unsqueeze(0), next_state, reward, not bool(done))
            state = next_state
            trans_loss, reward_loss, mask_loss = agent.update(args.load_model)

            if trans_loss is not None:
                writer.add_scalar('transition_loss', trans_loss, episode)
            if reward_loss is not None:
                writer.add_scalar('reward_loss', reward_loss, episode)
            if mask_loss is not None:
                writer.add_scalar('mask_loss', mask_loss, episode)

        writer.add_scalar('train_reward', total_reward, episode)
        if episode % args.log_interval == 0 or episode == args.num_episodes - 1:
            eval_reward = evaluate(env, agent)
            writer.add_scalar('eval_reward', eval_reward, episode)
            print(episode, ":", total_reward, 'eval', eval_reward)

        if args.algo == "MaskDQN" and not args.load_model and episode == args.num_episodes - 1:
            os.makedirs(save_path)
            agent.save_model(save_path)


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Tetris""")
    parser.add_argument("--env", type=str, default="Maze", choices=["Maze"])
    parser.add_argument("--action_space", type=int, default=10, help="action directions for Maze")
    parser.add_argument("--algo", type=str, default="MaskDQN", choices=["DQN", "MaskDQN"])

    parser.add_argument("--batch_size", type=int, default=256, help="The number of images per batch")
    parser.add_argument('--replay_memory_size', type=int, default=2000)
    parser.add_argument("--policy_lr", type=float, default=1e-3)
    parser.add_argument("--mask_lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.9)

    parser.add_argument("--num_episodes", type=int, default=1500)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--reload_freq", type=int, default=50)
    parser.add_argument("--trans_freq", type=int, default=50)

    parser.add_argument("--log_path", type=str, default="runs")
    parser.add_argument("--media_path", type=str, default="media")
    parser.add_argument("--save_path", type=str, default="trained_models")
    parser.add_argument("--load_model", type=bool, default=True)

    args = parser.parse_args()
    return args


args = get_args()

train(args)
