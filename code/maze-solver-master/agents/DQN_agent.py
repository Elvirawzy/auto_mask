from models.DQN import DQNModel
import torch
import torch.nn as nn
from collections import namedtuple, deque
import random
import numpy as np

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN:
    def __init__(self, obs_space, act_space, args, device):
        super(DQN, self).__init__()
        self.obs_n = obs_space
        self.act_n = act_space
        self.action_space = act_space
        self.model = DQNModel(self.obs_n, self.act_n).to(device)
        self.target_model = DQNModel(self.obs_n, self.act_n).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.criterion = nn.MSELoss()

        self.gamma = args.gamma
        self.batch_size = args.batch_size
        self.device = device

        self.replay_memory = ReplayMemory(args.replay_memory_size)
        self.update_steps = 0
        self.reload_freq = args.reload_freq

    def get_action(self, state, action=None):
        q = self.model(state)
        if action is None:
            max_q = q.max(1)[0]
            action = q.max(1)[1]
        else:
            max_q = q.gather(1, action).squeeze(1)
        return action, max_q

    def update(self, over):
        if len(self.replay_memory) >= self.batch_size:
            transitions = self.replay_memory.sample(self.batch_size)
            batch = Transition(*zip(*transitions))
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                    batch.next_state)), device=self.device, dtype=torch.bool)
            non_final_next_states = torch.cat([s for s in batch.next_state
                                               if s is not None])
            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)
            _, state_action_values = self.get_action(state_batch, action=action_batch)
            next_state_values = torch.zeros(self.batch_size, device=self.device)
            with torch.no_grad():
                # next_q = self.target_model(non_final_next_states).detach()
                # max_next_q = next_q.max(1)[0]
                # next_state_values[non_final_mask] = max_next_q

                next_q = self.model(non_final_next_states).detach()
                max_next_a = next_q.max(1)[1].unsqueeze(1)
                next_target_q = self.target_model(non_final_next_states).detach()
                max_next_q = next_target_q.gather(1, max_next_a).squeeze(1)
                next_state_values[non_final_mask] = max_next_q
            # Compute the expected Q values
            expected_state_action_values = (next_state_values * self.gamma) + reward_batch
            # q = self.model(state_batch)
            # q = torch.softmax(q,dim=1)
            # L1_loss = [sum(p) for p in q]

            loss = self.criterion(state_action_values, expected_state_action_values)
            self.optimizer.zero_grad()
            loss.backward()
            # In-place gradient clipping
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 100)
            self.optimizer.step()
            self.update_steps += 1
            if self.update_steps % self.reload_freq:
                self.target_model.load_state_dict(self.model.state_dict())