import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import time

from pathlib import Path
import pandas as pd
import random, numpy, argparse, logging, os

from collections import namedtuple
from minatar import Environment
from models.mask import GlobalMask
from torch.utils.tensorboard import SummaryWriter

################################################################################################################
# Constants
#
################################################################################################################
BATCH_SIZE = 32
REPLAY_BUFFER_SIZE = 100000
TARGET_NETWORK_UPDATE_FREQ = 1000
TRAINING_FREQ = 2
NUM_FRAMES = 15000
FIRST_N_FRAMES = 50000
REPLAY_START_SIZE = 5000
END_EPSILON = 0.1
STEP_SIZE = 0.00025
GRAD_MOMENTUM = 0.95
SQUARED_GRAD_MOMENTUM = 0.95
MIN_SQUARED_GRAD = 0.01
GAMMA = 0.99
EPSILON = 1.0

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter(os.path.join(os.getcwd(), 'runs', f'DQN5_asterix_{int(time.time())}'))

class QNetwork(nn.Module):
    def __init__(self, in_channels, num_actions):
        super(QNetwork, self).__init__()
        self.conv = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1)

        def size_linear_unit(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1

        num_linear_units = size_linear_unit(10) * size_linear_unit(10) * 16
        self.fc_hidden = nn.Linear(in_features=num_linear_units, out_features=128)
        self.output = nn.Linear(in_features=128, out_features=num_actions)

    # As per implementation instructions according to pytorch, the forward function should be overwritten by all
    # subclasses
    def forward(self, x):
        # Rectified output from the first conv layer
        x = f.relu(self.conv(x))
        x = f.relu(self.fc_hidden(x.view(x.size(0), -1)))
        return self.output(x)

transition = namedtuple('transition', 'state, next_state, action, mask, reward, is_terminal')

class replay_buffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.location = 0
        self.buffer = []

    def add(self, *args):
        # Append when the buffer is not full but overwrite when the buffer is full
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(transition(*args))
        else:
            self.buffer[self.location] = transition(*args)

        # Increment the buffer location
        self.location = (self.location + 1) % self.buffer_size

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)


def get_state(s):
    return (torch.tensor(s, device=device).permute(2, 0, 1)).unsqueeze(0).float()

def world_dynamics(t, replay_start_size, num_actions, s, env, policy_net, mask_net):
    # A uniform random policy is run before the learning starts
    # mask = mask_net.get_mask().to(device)
    def random_select():
        # indexs = mask.nonzero()
        # random_index = torch.randint(0, indexs.size(0), (1,)).item()
        # act = indexs[random_index]
        # return act.view(1, 1)
        pass
    if t < replay_start_size:
        action = torch.tensor([[random.randrange(num_actions)]], device=device)
        # action = random_select()
    else:
        epsilon = END_EPSILON if t - replay_start_size >= FIRST_N_FRAMES \
            else ((END_EPSILON - EPSILON) / FIRST_N_FRAMES) * (t - replay_start_size) + EPSILON

        if numpy.random.binomial(1, epsilon) == 1:
            action = torch.tensor([[random.randrange(num_actions)]], device=device)
            # action = random_select()
        else:
            with torch.no_grad():
                q = policy_net(s).to(device)
                # q = torch.where(mask.to(torch.bool), q, torch.tensor(-1e+8).to(device))
                action = q.max(1)[1].view(1, 1)
                # action = policy_net(s).max(1)[1].view(1, 1)

    # Act according to the action and observe the transition and reward
    reward, terminated = env.act(action)

    # Obtain s_prime
    s_prime = get_state(env.state())

    # return s_prime, action, mask.unsqueeze(0), torch.tensor([[reward]], device=device).float(), torch.tensor([[terminated]], device=device)
    return s_prime, action, None, torch.tensor([[reward]], device=device).float(), torch.tensor([[terminated]], device=device)

def train(sample, policy_net, target_net, optimizer):
    # Batch is a list of namedtuple's, the following operation returns samples grouped by keys
    batch_samples = transition(*zip(*sample))

    states = torch.cat(batch_samples.state)
    next_states = torch.cat(batch_samples.next_state)
    actions = torch.cat(batch_samples.action)
    # masks = torch.cat(batch_samples.mask)
    rewards = torch.cat(batch_samples.reward)
    is_terminal = torch.cat(batch_samples.is_terminal)

    Q_s_a = policy_net(states)
    # Q_s_a = torch.where(masks.to(torch.bool), Q_s_a, torch.tensor(-1e+8).to(device))
    Q_s_a = Q_s_a.gather(1, actions)

    # Get the indices of next_states that are not terminal
    none_terminal_next_state_index = torch.tensor([i for i, is_term in enumerate(is_terminal) if is_term == 0],
                                                  dtype=torch.int64, device=device)
    # Select the indices of each row
    none_terminal_next_states = next_states.index_select(0, none_terminal_next_state_index)
    # none_terminal_masks = masks.index_select(0, none_terminal_next_state_index)

    Q_s_prime_a_prime = torch.zeros(len(sample), 1, device=device)
    if len(none_terminal_next_states) != 0:
        next_Q_s_a = target_net(none_terminal_next_states).detach()
        # next_Q_s_a = torch.where(none_terminal_masks.to(torch.bool), next_Q_s_a, torch.tensor(-1e+8).to(device))
        Q_s_prime_a_prime[none_terminal_next_state_index] = next_Q_s_a.max(1)[0].unsqueeze(1)

    # Compute the target
    target = rewards + GAMMA * Q_s_prime_a_prime

    # Huber loss
    loss = f.smooth_l1_loss(target, Q_s_a)

    # Zero gradients, backprop, update the weights of policy_net
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def dqn(env, load_model, load_path=None, step_size=STEP_SIZE):
    in_channels = env.state_shape()[2]


    num_actions = env.num_actions()
    print("action dimensions:", num_actions)
    FILE_PATH = str(Path(__file__).resolve().parents[0])
    log_path = os.path.join(FILE_PATH, 'data', 'asterix')
    log_time = str(int(time.time()))
    log_name = f'/DQN_{num_actions}_{log_time}.csv'

    df1 = pd.DataFrame(columns=['step', 'invalid_action_num', 'label'])  # 列名
    df1.to_csv(log_path + log_name, index=False)  # 路径

    # Instantiate networks, optimizer, loss and buffer
    policy_net = QNetwork(in_channels, num_actions).to(device)
    # mask_net = GlobalMask(num_actions, device, load_model)
    # mask_net.load_state_dict(torch.load(os.path.join(load_path, 'mask_model.pth'), map_location=device)['mask'])
    mask_net = None
    replay_start_size = 0

    target_net = QNetwork(in_channels, num_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    r_buffer = replay_buffer(REPLAY_BUFFER_SIZE)
    replay_start_size = REPLAY_START_SIZE

    optimizer = optim.RMSprop(policy_net.parameters(), lr=step_size, alpha=SQUARED_GRAD_MOMENTUM, centered=True,
                              eps=MIN_SQUARED_GRAD)

    # Set initial values
    e_init = 0
    t_init = 0
    policy_net_update_counter_init = 0
    avg_return_init = 0.0
    data_return_init = []
    frame_stamp_init = []

    # Data containers for performance measure and model related data
    data_return = data_return_init
    frame_stamp = frame_stamp_init
    avg_return = avg_return_init
    invalid_act_num = 0
    total_act_num = 0

    # Train for a number of frames
    t = t_init
    e = e_init
    policy_net_update_counter = policy_net_update_counter_init
    t_start = time.time()
    while e < NUM_FRAMES:
        # Initialize the return for every episode (we should see this eventually increase)
        G = 0.0

        # Initialize the environment and start state
        env.reset()
        s = get_state(env.state())
        is_terminated = False
        steps = 0
        while (not is_terminated) and steps < 100:
            steps += 1
            # Generate data
            s_prime, action, mask, reward, is_terminated = world_dynamics(t, replay_start_size, num_actions, s, env,
                                                                    policy_net, mask_net)

            sample = None

            r_buffer.add(s, s_prime, action, mask, reward, is_terminated)
            # Start learning when there's enough data and when we can sample a batch of size BATCH_SIZE
            if t > REPLAY_START_SIZE and len(r_buffer.buffer) >= BATCH_SIZE:
                # Sample a batch
                sample = r_buffer.sample(BATCH_SIZE)

            # Train every n number of frames defined by TRAINING_FREQ
            if t % TRAINING_FREQ == 0 and sample is not None:
                policy_net_update_counter += 1
                train(sample, policy_net, target_net, optimizer)

            # Update the target network only after some number of policy network updates
            if policy_net_update_counter > 0 and policy_net_update_counter % TARGET_NETWORK_UPDATE_FREQ == 0:
                target_net.load_state_dict(policy_net.state_dict())

            G += reward.item()

            t += 1

            # Continue the process
            s = s_prime

            total_act_num += 1
            if action >= 5:
                invalid_act_num += 1

        # Increment the episodes
        e += 1

        # Save the return for each episode
        data_return.append(G)
        frame_stamp.append(t)

        # Logging exponentiated return only when verbose is turned on and only at 1000 episode intervals
        avg_return = 0.99 * avg_return + 0.01 * G
        if e % 100 == 0:
            print(e, ':', avg_return)

        if e % 50 == 0:
            avg_invalid_act_num = invalid_act_num / total_act_num
            list1 = [e, avg_invalid_act_num, 'DQN']
            data1 = pd.DataFrame([list1])
            data1.to_csv(log_path + log_name, mode='a', header=False, index=False)
            invalid_act_num = 0
            total_act_num = 0

    # Print final logging info
    print("Avg return: " + str(numpy.around(avg_return, 2)) + " | Time per frame: " + str((time.time() - t_start) / t))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default='breakout', type=str)
    parser.add_argument("--output", "-o", type=str)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--loadfile", "-l", type=str)
    parser.add_argument("--alpha", "-a", type=float, default=STEP_SIZE)
    parser.add_argument("--load", default=False)
    parser.add_argument("--replayoff", "-r", action="store_true")
    parser.add_argument("--targetoff", "-t", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    if args.output:
        file_name = args.output
    else:
        file_name = os.getcwd() + "/" + args.game

    load_file_path = os.path.join(os.getcwd(), 'trained_models', args.game, 'AsterixMask_18_1701671799')

    env = Environment(args.game)

    print('Cuda available?: ' + str(torch.cuda.is_available()))
    dqn(env, args.load, load_file_path, args.alpha)


if __name__ == '__main__':
    main()

