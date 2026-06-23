from collections import defaultdict


class Logger:

    def __init__(self):
        self.metrics = defaultdict(list)

    def log(self, metrics: dict):

        for key, value in metrics.items():
            self.metrics[key].append(value)

    def mean(self, key, window=None):

        values = self.metrics[key]

        if len(values) == 0:
            return 0

        if window is not None:
            values = values[-window:]

        return sum(values) / len(values)


