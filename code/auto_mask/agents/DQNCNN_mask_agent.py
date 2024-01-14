from models.DQN import DQNCNNModel
from models.mask import GlobalMask
from models.transition import TransCNN
from models.reward import RewardCNN
import torch
import torch.nn as nn
from collections import namedtuple, deque
import random
import numpy as np
import os
from sklearn.cluster import DBSCAN

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'terminate', 'mask'))
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class MaskAgent:
    def __init__(
            self,
            channel,
            act_n,
            device,
            load_model,
            policy_lr=1e-3,
            gamma=0.9,
            batch_size=256,
            replay_memory_size=2000,
            reload_freq=50,
            trans_freq=50,
    ):
        super(MaskAgent).__init__()
        self.channel = channel
        self.act_n = act_n
        self.policy = DQNCNNModel(self.channel, self.act_n).to(device)
        self.policy_target = DQNCNNModel(self.channel, self.act_n).to(device)
        self.policy_target.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=policy_lr)
        self.criterion = nn.MSELoss()
        self.reward_criterion = nn.MSELoss()
        self.classify_criterion = nn.BCELoss()

        self.reward_model = RewardCNN(self.channel, self.act_n).to(device)
        self.mask = GlobalMask(self.act_n, device, load_model)

        self.reward_optimizer = torch.optim.Adam(self.reward_model.parameters(), lr=5e-3)
        self.mask_optimizer = torch.optim.Adam(self.mask.parameters(), lr=1e-2)

        self.gamma = gamma
        self.batch_size = batch_size
        self.device = device

        self.replay_memory = ReplayMemory(replay_memory_size)
        self.update_steps = 0
        self.reload_freq = reload_freq
        self.trans_freq = trans_freq
        self.mask_flag = 0

    def get_action(self, state, action=None, mask=None):
        q = self.policy(state).to(self.device)
        if mask is None:
            mask = self.mask.get_mask().to(self.device)
        # q = mask * q
        q = torch.where(mask.to(torch.bool), q, torch.tensor(-1e+8).to(self.device))
        if action is None:
            max_q = q.max(1)[0]
            action = q.max(1)[1]
        else:
            max_q = q.gather(1, action).squeeze(1)
        return action, max_q, mask

    def update(self, load_model):
        if len(self.replay_memory) < self.batch_size:
            return None, None, None, None

        transition_loss, reward_loss, mask_loss = None, None, None

        def sample_transition():
            transitions = self.replay_memory.sample(self.batch_size)
            batch = Transition(*zip(*transitions))
            non_final_idx = torch.tensor(batch.terminate, device=self.device, dtype=torch.bool).detach()
            next_states_batch = torch.cat(batch.next_state).detach()
            state_batch = torch.cat(batch.state).detach()
            action_batch = torch.cat(batch.action).detach()
            reward_batch = torch.cat(batch.reward).to(torch.float32).detach()
            mask_batch = torch.cat(batch.mask).to(torch.float32).detach()
            return state_batch, action_batch, reward_batch, next_states_batch, non_final_idx, mask_batch

        s, a, r, s_, t, m = sample_transition()

        _, state_action_values, _ = self.get_action(s, action=a, mask=m)
        with torch.no_grad():
            next_target_q = self.policy_target(s_)
            # next_target_q = next_target_q * m
            next_target_q = torch.where(m.to(torch.bool), next_target_q, torch.tensor(-1e+8).to(self.device))
            max_next_q = next_target_q.max(1)[1]
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

        if not load_model and self.update_steps % self.trans_freq == 0:
            self.update_steps = 0
            transition_loss, reward_loss = self.update_transition_reward(s, a, r.unsqueeze(1), s_)
            mask_loss = self.update_mask(s)

        return policy_loss, transition_loss, reward_loss, mask_loss

    def save_models(self, save_path):
        # torch.save({'policy': self.policy.state_dict()}, os.path.join(save_path, 'policy_model.pth'))
        torch.save({'mask': self.mask.state_dict()}, os.path.join(save_path, 'mask_model.pth'))
        torch.save({'transition': self.transition_model.state_dict()},
                   os.path.join(save_path, 'transition_model.pth'))
        torch.save({'reward': self.reward_model.state_dict()}, os.path.join(save_path, 'reward_model.pth'))

    def load_model(self, load_path):
        self.mask.load_state_dict(
            torch.load(os.path.join(load_path, 'mask_model.pth'), map_location=self.device)['mask'])

    def update_transition_reward(self, s, a, r, s_):
        raise NotImplementedError

    def update_mask(self, s):
        raise NotImplementedError


class BreakoutAgent(MaskAgent):
    def __init__(
            self,
            channel,
            act_n,
            device,
            load_model,
            policy_lr=1e-3,
            gamma=0.9,
            batch_size=256,
            replay_memory_size=2000,
            reload_freq=50,
            trans_freq=50,
    ):
        super(BreakoutAgent, self).__init__(
            channel,
            act_n,
            device,
            load_model,
            policy_lr,
            gamma,
            batch_size,
            replay_memory_size,
            reload_freq,
            trans_freq,
        )
        self.transition_model = TransCNN(self.channel, 10, self.act_n).to(device)
        self.trans_optimizer = torch.optim.Adam(self.transition_model.parameters(), lr=5e-3)

    def update_transition_reward(self, s, a, r, s_):
        act_idxs = torch.zeros(a.shape[0], self.act_n).to(torch.float32).to(self.device)
        act_idxs.scatter_(1, a, 1)
        paddle_s = s[:, 0, 9, :]
        pred_next_s, log_probs= self.transition_model.pred_s(torch.cat([paddle_s, act_idxs], dim=1))
        paddle_s_ = s_[:, 0, 9, :].argmax(dim=1)

        diff = (pred_next_s != paddle_s_).float()
        # transition_loss = torch.mean(0.5 * diff.pow(2))
        transition_loss = (torch.exp(log_probs) * diff).mean()

        self.trans_optimizer.zero_grad()
        transition_loss.backward()
        self.trans_optimizer.step()

        pred_reward = self.reward_model(s, act_idxs)
        reward_loss = self.reward_criterion(pred_reward, r.detach())

        self.reward_optimizer.zero_grad()
        reward_loss.backward()
        self.reward_optimizer.step()

        return transition_loss, reward_loss

    def update_mask(self, s):
        all_motion = torch.tensor(np.identity(self.act_n), dtype=torch.float32, device=self.device)
        labels = []
        paddle_s = s[:, 0, 9, :]
        for idx, state in enumerate(paddle_s):
            paddle_batch = state.repeat(self.act_n, 1)
            state_batch = s[idx].repeat(self.act_n, 1, 1, 1)
            with torch.no_grad():
                pred_s_, _ = self.transition_model.pred_s(torch.cat([paddle_batch, all_motion], dim=1))
                s_diff = pred_s_ - paddle_batch.argmax(dim=1)
                pred_r = self.reward_model.pred_r(state_batch, all_motion)
                s_r = torch.concatenate([s_diff.view(self.act_n, 1), pred_r], dim=1)
                dbscan = DBSCAN(eps=0.1, min_samples=1)
                clusters = dbscan.fit_predict(s_r.cpu().numpy())
            label = []
            tmp = []
            for c in clusters:
                tag = 0 if c in tmp else 1
                tmp.append(c)
                label.append(tag)
            labels.append(label)
        labels = torch.tensor(np.array(labels), dtype=torch.float32, device=self.device)
        prob_labels = torch.zeros(labels.shape[0], self.act_n, 2).to(torch.float32).to(self.device)
        prob_labels[:, :, 0] += (1- labels)
        prob_labels[:, :, 1] += labels

        loss = (self.mask.probs - prob_labels).pow(2).mean()
        self.mask_optimizer.zero_grad()
        loss.backward()
        self.mask_optimizer.step()

        return loss.item()


class AsterixAgent(MaskAgent):
    def __init__(
            self,
            channel,
            act_n,
            device,
            load_model,
            policy_lr=1e-3,
            gamma=0.9,
            batch_size=256,
            replay_memory_size=2000,
            reload_freq=50,
            trans_freq=50,
    ):
        super(AsterixAgent, self).__init__(
            channel,
            act_n,
            device,
            load_model,
            policy_lr,
            gamma,
            batch_size,
            replay_memory_size,
            reload_freq,
            trans_freq,
        )
        self.transition_model = TransCNN(self.channel, 100, self.act_n).to(device)
        self.trans_optimizer = torch.optim.Adam(self.transition_model.parameters(), lr=5e-3)

    def update_transition_reward(self, s, a, r, s_):
        act_idxs = torch.zeros(a.shape[0], self.act_n).to(torch.float32).to(self.device)
        act_idxs.scatter_(1, a, 1)
        pos_s = s[:, 0, :, :].reshape(-1, 100)
        pos_s_ = s_[:, 0, :, :].reshape(-1, 100).argmax(dim=1)
        pred_next_s, log_probs = self.transition_model.pred_s(torch.concatenate([pos_s, act_idxs], dim=1))

        diff = (pred_next_s - pos_s_).float().pow(2)
        # diff = pred_next_s - label_s
        # transition_loss = torch.mean(0.5 * diff.pow(2))
        transition_loss = (torch.exp(log_probs) * diff).mean()

        self.trans_optimizer.zero_grad()
        transition_loss.backward()
        self.trans_optimizer.step()

        pred_reward = self.reward_model(s, act_idxs)
        reward_loss = self.reward_criterion(pred_reward, r.detach())

        self.reward_optimizer.zero_grad()
        reward_loss.backward()
        self.reward_optimizer.step()

        return transition_loss, reward_loss

    def update_mask(self, s):
        all_motion = torch.tensor(np.identity(self.act_n), dtype=torch.float32, device=self.device)
        labels = []
        pos_s = s[:, 0, :, :].reshape(-1, 100)
        for idx, state in enumerate(pos_s):
            state_batch = s[idx].repeat(self.act_n, 1, 1, 1)
            pos_batch = state.repeat(self.act_n, 1)
            with torch.no_grad():
                pred_s_, _ = self.transition_model.pred_s(torch.concatenate([pos_batch, all_motion], dim=1))
                pred_r = self.reward_model.pred_r(state_batch, all_motion) / 10
                s_r = torch.concatenate([pred_s_.view(self.act_n, 1), pred_r], dim=1)
                dbscan = DBSCAN(eps=0.1, min_samples=1)
                clusters = dbscan.fit_predict(s_r.cpu().numpy())
            label = []
            tmp = []
            for c in clusters:
                tag = 0 if c in tmp else 1
                tmp.append(c)
                label.append(tag)
            labels.append(label)
        labels = torch.tensor(np.array(labels), dtype=torch.float32, device=self.device)
        prob_labels = torch.zeros(labels.shape[0], self.act_n, 2).to(torch.float32).to(self.device)
        prob_labels[:, :, 0] += (1- labels)
        prob_labels[:, :, 1] += labels

        loss = (self.mask.probs - prob_labels).pow(2).mean()
        self.mask_optimizer.zero_grad()
        loss.backward()
        self.mask_optimizer.step()

        return loss.item()
