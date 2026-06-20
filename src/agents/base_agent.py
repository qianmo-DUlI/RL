from abc import ABC, abstractmethod


class BaseAgent(ABC):
    def __init__(self, device):
        self.device = device
        self.is_training = True
        self.total_steps = 0

    @abstractmethod
    def select_action(self, state):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass

    def train(self):
        self.is_training = True

    def eval(self):
        self.is_training = False

    def reset(self):
        pass
