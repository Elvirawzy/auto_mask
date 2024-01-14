import random, numpy, argparse
import time
import os
from pathlib import Path
import torch
import pandas as pd

from minatar import Environment
from agents.DQNCNN_mask_agent import BreakoutAgent, AsterixAgent
from collections import deque
from torch.utils.tensorboard import SummaryWriter

log_time = str(int(time.time()))
device = torch.device("cuda:1")


def get_state(s):
    return (torch.tensor(s, device=device).permute(2, 0, 1)).unsqueeze(0).float()


def evaluate(environment, algorithm):
    done = False
    total_reward = 0
    environment.reset()
    state = get_state(environment.state())
    invalid_actions = []

    steps = 0
    while not done and steps < 100:
        steps += 1
        with torch.no_grad():
            action, _, mask = algorithm.get_action(state)
        reward, done = environment.act(action)
        total_reward += reward
        state = get_state(environment.state())
        invalid_actions.append(torch.sum(mask[5:]))

    return total_reward, sum(invalid_actions) / len(invalid_actions)


def train(args):
    env = Environment(args.game)
    num_actions = env.num_actions()
    print("action dimensions:", num_actions)

    FILE_PATH = str(Path(__file__).resolve().parents[0])
    log_path = os.path.join(FILE_PATH, 'data', args.game)

    df1 = pd.DataFrame(columns=['step', 'invalid_action_num', 'label'])  # 列名
    df1.to_csv(log_path + f'/action_nums_{log_time}.csv', index=False)  # 路径

    if not args.load_model:
        save_path = os.path.join(FILE_PATH, 'trained_models', args.game, f'{args.algo}_{num_actions}_{log_time}')
        os.makedirs(save_path)

    writer = SummaryWriter(os.path.join(FILE_PATH, 'runs', f'{args.game}_{int(time.time())}'))

    if args.algo == 'BreakoutMask':
        agent = BreakoutAgent(
            4,
            num_actions,
            device,
            load_model=args.load_model,
            policy_lr=args.policy_lr,
            gamma=args.gamma,
            batch_size=args.batch_size,
            replay_memory_size=args.replay_memory_size,
            reload_freq=args.reload_freq,
            trans_freq=args.trans_freq,
        )
    else:
        agent = AsterixAgent(
            4,
            num_actions,
            device,
            load_model=args.load_model,
            policy_lr=args.policy_lr,
            gamma=args.gamma,
            batch_size=args.batch_size,
            replay_memory_size=args.replay_memory_size,
            reload_freq=args.reload_freq,
            trans_freq=args.trans_freq,
        )

    if args.load_model:
        agent.load_model(os.path.join(FILE_PATH, 'trained_models', args.game, 'AsterixMask_18_1701671799'))

    eps = 0
    total_eps = 0
    eval_rewards = deque([], maxlen=100)
    while eps < args.num_episodes:
        eps += 1
        epsilon = 0.3
        total_reward = 0.0

        env.reset()
        done = False
        state = get_state(env.state())
        steps = 0
        while not done and steps < 100:
            total_eps += 1
            steps += 1
            sample = random.random()
            if sample > epsilon:
                with torch.no_grad():
                    action, _, mask = agent.get_action(state)
            else:
                # action = torch.randint(0, num_actions, (1, )).to(device)

                with torch.no_grad():
                    _, _, mask = agent.get_action(state)
                    indexs = mask.nonzero()
                    random_index = torch.randint(0, indexs.size(0), (1,)).item()
                    action = indexs[random_index]

            reward, done = env.act(action)
            next_state = get_state(env.state())
            total_reward += reward

            # agent.replay_memory.push(state, action.view(1, 1), next_state,
            # torch.tensor([reward], device=device), not bool(done))
            agent.replay_memory.push(state, action.view(1, 1), next_state,
                                     torch.tensor([reward], device=device), not bool(done), mask.unsqueeze(0))

            state = next_state

            # policy_loss = agent.update()
            policy_loss, trans_loss, reward_loss, mask_loss = agent.update(args.load_model)

            if policy_loss is not None:
                writer.add_scalar('loss/policy_loss', policy_loss, total_eps)
            if trans_loss is not None:
                writer.add_scalar('loss/trans_loss', trans_loss, total_eps)
            if reward_loss is not None:
                writer.add_scalar('loss/reward_loss', reward_loss, total_eps)
            if mask_loss is not None:
                writer.add_scalar('loss/mask_loss', mask_loss, total_eps)

        # writer.add_scalar('reward', total_reward, eps)
        if eps % args.log_interval == 0 or eps == args.num_episodes - 1:
            eval_reward, invalid_act_num = evaluate(env, agent)
            eval_rewards.append(eval_reward)
            avg_reward = sum(list(eval_rewards)) / len(list(eval_rewards))

            list1 = [eps, invalid_act_num.item(), 'invalid_action_num']
            data1 = pd.DataFrame([list1])
            data1.to_csv(log_path + f'/action_nums_{log_time}.csv', mode='a', header=False, index=False)

            print(f'eps: {eps}, reward: {eval_reward}, avg: {avg_reward}')

        if not args.load_model and eps == args.num_episodes - 1:
            agent.save_models(save_path)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="breakout", type=str, choices=["breakout", "asterix"])
    parser.add_argument("--num_episodes", default=15000, type=int)
    parser.add_argument("--algo", type=str, default="BreakoutMask", choices=["BreakoutMask", "AsterixMask"])
    parser.add_argument("--log_path", type=str, default="runs")

    parser.add_argument("--batch_size", type=int, default=256, help="The number of images per batch")
    parser.add_argument('--replay_memory_size', type=int, default=1000)
    parser.add_argument("--policy_lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--reload_freq", type=int, default=50)
    parser.add_argument("--trans_freq", type=int, default=200)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--load_model", type=bool, default=False)

    args = parser.parse_args()
    return args


args = get_args()
train(args)