import math
import random
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

    def select_action(self, state):
        # eval模式直接贪心选择
        if not self.is_training:
            with torch.no_grad():
                return self.policy_net(state).argmax(dim=1).view(1, 1)
        # 计算当前探索率
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(
            -1.0 * self.total_steps / self.epsilon_decay
        )
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
        self.memory.push(state, action, next_state, reward, done)

    def update(self):
        if len(self.memory) < self.batch_size:
            return {}
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)
        done_batch = torch.tensor(batch.done, device=self.device, dtype=torch.float32)

        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1).values
        target_q_values = reward_batch + self.gamma * next_q_values * (1 - done_batch)
        current_q_values = (
            self.policy_net(state_batch).gather(1, action_batch).squeeze(1)
        )

        loss = self.criterion(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        return {"loss": loss.item()}

    def state_dict(self) -> Dict[str, Any]:
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        pass
