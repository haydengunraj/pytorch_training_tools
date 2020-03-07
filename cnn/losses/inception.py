import torch
import torch.nn.functional as F


def create(config):
    return InceptionLoss(**config)


class InceptionLoss:
    def __init__(self, loss_type, logit_weight=1.0, aux_weight=1.0, weight=None):
        self.loss_func = getattr(F, loss_type)
        if self.loss_func is None:
            raise ValueError('torch.nn.functional does not define', loss_type)
        self.logit_weight = logit_weight
        self.aux_weight = aux_weight

        self.include_weight = loss_type in ('cross_entropy',)
        self.weight = weight

    def __call__(self, logits, target):
        if isinstance(logits, torch.Tensor):
            return self.loss_func(logits, target)
        elif isinstance(logits, tuple) and len(logits) == 2:
            logits, aux_logits = logits
            return self.logit_weight*self._loss_func(logits, target) \
                + self.aux_weight*self._loss_func(aux_logits, target)
        else:
            raise ValueError('InceptionLoss logits must be a Tensor or a 2-tuple of Tensors')

    def _loss_func(self, logits, target):
        if self.include_weight:
            return self.loss_func(logits, target, weight=self.weight)
        return self.loss_func(logits, target)
