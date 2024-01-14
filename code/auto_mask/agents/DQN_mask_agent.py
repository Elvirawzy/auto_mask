from models.DQN import DQNModel
from models.mask import MaskModel
from models.transition import DeterministicTransitionModel
from models.reward import RewardModel
import torch
import torch.nn as nn
from collections import namedtuple, deque
import random
import numpy as np
from numpy import unique
import torch.nn.functional as F
from sklearn.cluster import DBSCAN, AffinityPropagation
import os

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'terminate'))


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
    def __init__(
            self,
            obs_n,
            act_n,
            device,
            policy_lr=1e-3,
            gamma=0.9,
            batch_size=256,
            replay_memory_size=2000,
            reload_freq=50,
            trans_freq=50
    ):
        super(DQN, self).__init__()
        self.obs_n = obs_n
        self.act_n = act_n
        self.policy = DQNModel(self.obs_n, self.act_n).to(device)
        self.policy_target = DQNModel(self.obs_n, self.act_n).to(device)
        self.policy_target.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=policy_lr)
        self.criterion = nn.MSELoss()

        self.gamma = gamma
        self.batch_size = batch_size
        self.device = device

        self.replay_memory = ReplayMemory(replay_memory_size)
        self.update_steps = 0
        self.reload_freq = reload_freq
        self.trans_freq = trans_freq

    def get_action(self, state, action=None, mask=None):
        q = self.policy(state)
        if action is None:
            max_q = q.max(1)[0]
            action = q.max(1)[1]
        else:
            max_q = q.gather(1, action).squeeze(1)
        return action, max_q, mask

    def update(self, load_model):
        trans_loss, reward_loss, mask_loss = None, None, None
        if len(self.replay_memory) < self.batch_size:
            return trans_loss, reward_loss, mask_loss

        def sample_transition():
            transitions = self.replay_memory.sample(self.batch_size)
            batch = Transition(*zip(*transitions))
            non_final_idx = torch.tensor(batch.terminate, device=self.device, dtype=torch.bool).detach()
            next_states_batch = torch.cat(batch.next_state).detach()
            state_batch = torch.cat(batch.state).detach()
            action_batch = torch.cat(batch.action).detach()
            reward_batch = torch.cat(batch.reward).detach()
            return state_batch, action_batch, reward_batch, next_states_batch, non_final_idx

        s, a, r, s_, t = sample_transition()

        self.update_policy(s, a, r, s_, t)

        if not load_model and self.update_steps % self.trans_freq == 0:
            self.update_steps = 0
            trans_loss, reward_loss = self.update_transition_reward(s, a, r.unsqueeze(1), s_)
            mask_loss = self.update_mask(s)

        return trans_loss, reward_loss, mask_loss

    def load_model(self, load_path):
        pass

    def save_model(self, save_path):
        torch.save({'policy': self.policy.state_dict()}, os.path.join(save_path, 'policy_model.pth'))

    def update_policy(self, s, a, r, s_, t):
        _, state_action_values, _ = self.get_action(s, action=a)
        with torch.no_grad():
            next_target_q = self.policy_target(s_)
            max_next_q = next_target_q.max(1)[0]
            next_state_values = max_next_q * t
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + r

        policy_loss = self.criterion(state_action_values, expected_state_action_values.detach())
        self.optimizer.zero_grad()
        policy_loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy.parameters(), 100)
        self.optimizer.step()
        self.update_steps += 1
        if self.update_steps % self.reload_freq == 0:
            self.policy_target.load_state_dict(self.policy.state_dict())

    def update_transition_reward(self, s, a, r, s_):
        return None, None

    def update_mask(self, s):
        return None

class MaskDQN(DQN):
    def __init__(
            self,
            obs_n,
            act_n,
            motion,
            device,
            load_flag,
            policy_lr=1e-3,
            mask_lr=1e-3,
            transition_lr=1e-3,
            gamma=0.9,
            batch_size=256,
            replay_memory_size=2000,
            reload_freq=50,
            trans_freq=100,
    ):
        super(MaskDQN, self).__init__(
            obs_n,
            act_n,
            device,
            policy_lr,
            gamma,
            batch_size,
            replay_memory_size,
            reload_freq,
            trans_freq
        )
        self.obs_n = obs_n
        self.act_n = act_n
        self.motion = motion

        self.mask_model = MaskModel(2, self.act_n, load_flag).to(device)
        self.transition_model = DeterministicTransitionModel(2, 2).to(device)
        self.reward_model = RewardModel(2, 2).to(device)

        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=policy_lr)
        self.mask_optimizer = torch.optim.Adam(self.mask_model.parameters(), lr=mask_lr)
        self.transition_optimizer = torch.optim.Adam(self.transition_model.parameters(), lr=transition_lr)
        self.reward_optimizer = torch.optim.Adam(self.reward_model.parameters(), lr=1e-4)

        self.fit_criterion = nn.MSELoss()
        self.reward_criterion = nn.MSELoss()

        self.gamma = gamma
        self.batch_size = batch_size
        self.device = device

        self.replay_memory = ReplayMemory(replay_memory_size)
        self.update_steps = 0
        self.reload_freq = reload_freq

    def get_action(self, state, action=None, mask=None):
        q = self.policy(state)
        mask, _ = self.mask_model.get_mask(state[:, :2])
        mask_q = mask * q
        if action is None:
            max_q = mask_q.max(1)[0]
            action = mask_q.max(1)[1]
        else:
            max_q = mask_q.gather(1, action).squeeze(1)
        return action, max_q, mask

    def update_transition_reward(self, s, a, r, s_):
        s = s[:, :2]
        s_ = s_[:, :2]
        motion = np.zeros((s.shape[0], self.motion.shape[1]))
        for idx, act in enumerate(a):
            motion[idx] = self.motion[act]
        motion = torch.tensor(motion, dtype=torch.float32, device=self.device)
        pred_s_= self.transition_model(torch.cat([s, motion], dim=1))

        diff = (pred_s_ - s_.detach())
        transition_loss = torch.mean(0.5 * diff.pow(2))

        self.transition_optimizer.zero_grad()
        transition_loss.backward()
        self.transition_optimizer.step()

        pred_reward = self.reward_model(torch.cat([s, motion], dim=1))
        reward_loss = self.reward_criterion(pred_reward, r.detach())

        self.reward_optimizer.zero_grad()
        reward_loss.backward()
        self.reward_optimizer.step()

        return transition_loss, reward_loss

    def update_mask(self, s):
        simple_s = s[:, :2]

        all_motion = torch.tensor(self.motion, dtype=torch.float32, device=self.device)
        labels = []
        for state in simple_s:
            state_batch = state.unsqueeze(0).repeat(self.act_n, 1)
            with torch.no_grad():
                pred_s_ = self.transition_model(torch.cat([state_batch, all_motion], dim=1))
                pred_r = self.reward_model(torch.cat([state_batch, all_motion], dim=1))/1000
                dbscan = DBSCAN(eps=0.05, min_samples=1)
                clusters = dbscan.fit_predict(torch.cat([pred_s_, pred_r], dim=1).cpu().numpy())
            label = []
            tmp = []
            for c in clusters:
                tag = 0 if c in tmp else 1
                tmp.append(c)
                label.append(tag)
            labels.append(label)
        labels = torch.tensor(np.array(labels), dtype=torch.float32).to(self.device)

        mask, probs = self.mask_model.get_mask(simple_s)
        prob_labels = 1 - torch.abs(mask - labels).to(self.device)

        mask_loss = (probs.exp() - prob_labels).pow(2).mean()

        self.mask_optimizer.zero_grad()
        mask_loss.backward()
        self.mask_optimizer.step()

        return mask_loss

    def update_policy(self, s, a, r, s_, t):
        _, state_action_values, _ = self.get_action(s, action=a)
        with torch.no_grad():
            next_mask, _ = self.mask_model.get_mask(s_[:, :2])
            next_target_q = next_mask * self.policy_target(s_)
            max_next_q = next_target_q.max(1)[0]
            next_state_values = max_next_q * t
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + r

        policy_loss = self.fit_criterion(state_action_values, expected_state_action_values.detach())
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy.parameters(), 100)
        self.policy_optimizer.step()
        self.update_steps += 1
        if self.update_steps % self.reload_freq == 0:
            self.policy_target.load_state_dict(self.policy.state_dict())

    def save_model(self, save_path):
        torch.save({'mask': self.mask_model.state_dict()}, os.path.join(save_path, 'mask_model.pth'))

    def load_model(self, load_path):
        mask_state_dict = torch.load(os.path.join(load_path, 'mask_model.pth'), map_location='cpu')
        self.mask_model.load_state_dict(mask_state_dict['mask'])
