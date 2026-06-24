from collections import defaultdict

from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, log_dir):
        self.metrics = defaultdict(list)
        self.writer = None
        if log_dir is not None:
            self.writer = SummaryWriter(log_dir)

    def log(self, metrics: dict, step=None):
        for key, value in metrics.items():
            self.metrics[key].append(value)
            if self.writer is not None and step is not None and isinstance(value, (int, float)):
                self.writer.add_scalar(key, value, step)

    def mean(self, key, window=None):
        values = self.metrics[key]
        if len(values) == 0:
            return 0
        if window is not None:
            values = values[-window:]
        return sum(values) / len(values)

    def latest(self, key):
        values = self.metrics[key]
        if not values:
            return None
        return values[-1]

    def close(self):
        if self.writer is not None:
            self.writer.close()
