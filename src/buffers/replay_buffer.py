import random
from collections import deque

from src.common.types import Transition


class ReplayBuffer:
    def __init__(self, capacity):
        # 用的是双端队列，但只使用他的栈的功能
        self.memory = deque([], maxlen=capacity)
        self.capacity = capacity

    # 将新的数据（这个数据是与环境交互时产生的）入栈
    def push(self, state, action, reward, next_state, done):
        self.memory.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
