import torch

from .metric import Metric, MAXIMIZE_MODE

MODE = MAXIMIZE_MODE


def get_metric(config):
    return AccuracyMetric(**config)


class AccuracyMetric(Metric):
    def __init__(self, label_key='labels', logits_key='logits'):
        super().__init__()
        self.label_key = label_key
        self.logits_key = logits_key
        self.correct_preds = 0
        self.total_preds = 0

    def update(self, data_dict):
        labels = data_dict[self.label_key].view(-1)
        logits = data_dict[self.logits_key]
        logits = logits.view(-1, logits.size(1))
        _, predictions = torch.max(logits, dim=1)
        self.correct_preds += (predictions == labels).sum().item()
        self.total_preds += predictions.size(0)

    def reset(self):
        self.correct_preds = 0
        self.total_preds = 0

    def value(self):
        return self.correct_preds/self.total_preds
