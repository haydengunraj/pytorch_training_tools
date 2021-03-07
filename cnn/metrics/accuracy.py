import torch

from .metric import Metric, MAXIMIZE_MODE


def create(config):
    return AccuracyMetric(**config)


class AccuracyMetric(Metric):
    """Running accuracy metric"""
    def __init__(self, name, label_key='labels', output_key='logits', binary_threshold=None):
        super().__init__(name, MAXIMIZE_MODE)
        self.label_key = label_key
        self.output_key = output_key
        self.binary_threshold = binary_threshold
        self.correct_preds = 0
        self.total_preds = 0

    def update(self, data_dict):
        labels = data_dict[self.label_key].view(-1)
        outputs = data_dict[self.output_key]
        if outputs.ndimension() > 1:
            if outputs.size(1) > 1:
                _, predictions = torch.max(outputs, dim=1)
            else:
                predictions = outputs.view(-1)
        else:
            predictions = outputs
            if self.binary_threshold is not None:
                predictions = predictions > self.binary_threshold
                labels = labels > 0

        self.correct_preds += (predictions == labels).sum().item()
        self.total_preds += predictions.size(0)

    def reset(self):
        self.correct_preds = 0
        self.total_preds = 0

    def log(self, writer, step, tag_prefix='val/'):
        writer.add_scalar(tag_prefix + self.name, self.value, step)

    @property
    def value(self):
        return self.correct_preds/self.total_preds
