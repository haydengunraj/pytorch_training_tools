import torch.nn as nn

from ..utils import get_component, CallableWrapper, LOSS_KEY


def get_loss(config):
    loss_type = config.pop('type')
    input_keys = config.pop('input_keys')
    output_keys = config.pop('output_keys')
    if LOSS_KEY not in output_keys:
        raise ValueError('Loss functions must have a total loss key called "{}"'.format(LOSS_KEY))
    loss_func = get_component(__name__, loss_type, config, (nn,))
    return CallableWrapper(loss_func, input_keys, output_keys)
