from .metric import Metric, MINIMIZE_MODE

MODE = MINIMIZE_MODE


def create(config):
    return LossMetric(**config)


class LossMetric(Metric):
    def __init__(self, loss_key='loss'):
        self.loss_key = loss_key
        self.loss_sum = 0
        self.loss_count = 0

    def update(self, data_dict):
        self.loss_sum += data_dict[self.loss_key].item()
        self.loss_count += 1

    def reset(self):
        self.loss_sum = 0
        self.loss_count = 0

    def value(self):
        return self.loss_sum/self.loss_count
