import math
import random
from pathlib import Path
from typing import Dict, Any

import torch
from torch import optim, nn

from agents.base_agent import BaseAgent
from buffers.replay_buffer import ReplayBuffer
from common.types import Transition
from networks import QNetwork


class DQNAgent(BaseAgent):
    def __init__(self, state_dim, action_dim, config, device):
        super().__init__(device)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        # 网络
        self.policy_net = QNetwork(
            state_dim, action_dim, hidden_dim=config.hidden_dim
        ).to(device)
        self.target_net = QNetwork(
            state_dim, action_dim, hidden_dim=config.hidden_dim
        ).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        for param in self.target_net.parameters():
            param.requires_grad = False

        # 优化器
        self.optimizer = optim.AdamW(
            self.policy_net.parameters(), lr=config.lr, amsgrad=True
        )
        # 损失
        self.criterion = nn.SmoothL1Loss()
        # Replay Buffer
        self.memory = ReplayBuffer(capacity=config.buffer_size)
        # 超参数
        self.batch_size = config.batch_size
        self.gamma = config.gamma
        self.epsilon_start = config.epsilon_start
        self.epsilon_end = config.epsilon_end
        self.epsilon_decay = config.epsilon_decay
        self.target_update_freq = config.target_update_freq

    @property
    def epsilon(self):
        # 计算当前探索率
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(
            -self.total_steps / self.epsilon_decay
        )

    def select_action(self, state):
        # eval模式直接贪心选择
        if not self.is_training:
            with torch.no_grad():
                return self.policy_net(state).argmax(dim=1).view(1, 1)

        epsilon = self.epsilon
        self.total_steps += 1
        # 以1-ε概率选择最优
        if random.random() > epsilon:
            with torch.no_grad():
                return self.policy_net(state).argmax(dim=1).view(1, 1)
        # ε概率随机选一个
        else:
            return torch.tensor(
                [[random.randrange(self.action_dim)]],
                device=self.device,
                dtype=torch.long,
            )

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def sample_batch(self):
        transitions = self.memory.sample(self.batch_size)
        return Transition(*zip(*transitions))

    def _compute_target_q(self, reward_batch, next_state_batch, done_batch):
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1).values
        target_q_values = reward_batch + self.gamma * next_q_values * (1 - done_batch)
        return target_q_values

    def update(self):
        if len(self.memory) < self.batch_size:
            return {}

        batch = self.sample_batch()

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)
        done_batch = torch.tensor(batch.done, device=self.device, dtype=torch.float32)

        target_q_values = self._compute_target_q(
            reward_batch, next_state_batch, done_batch
        )
        current_q_values = (
            self.policy_net(state_batch).gather(1, action_batch).squeeze(1)
        )

        loss = self.criterion(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        if self.total_steps % self.target_update_freq == 0:
            self.update_target_network()

        return {
            "loss": loss.item(),
            "epsilon": self.epsilon,
            "avg_q": current_q_values.mean().item(),
            "avg_target_q": target_q_values.mean().item(),
        }

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def state_dict(self) -> Dict[str, Any]:
        return {
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
        }

    def save(self, path):
        path = Path(path)
        torch.save(self.state_dict(), path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint)

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.policy_net.load_state_dict(state_dict["policy_net"])
        self.target_net.load_state_dict(state_dict["target_net"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.total_steps = state_dict["total_steps"]
