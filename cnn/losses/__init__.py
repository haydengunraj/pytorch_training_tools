import torch.nn as nn

from .wrapper import LossWrapper
from ..utils import import_submodule


def get_loss(config):
    loss_type = config.pop('type')
    input_map = config.pop('input_map')
    output_keys = config.pop('output_keys')
    module = import_submodule(__name__, loss_type)
    if module is not None:
        loss_func = module.get_loss(config)
    else:
        loss_func = getattr(nn, loss_type)
        if loss_func is None:
            raise ValueError('Unrecognized loss type: ' + loss_type)
        loss_func = loss_func(**config)
    return LossWrapper(loss_func, input_map, output_keys)
