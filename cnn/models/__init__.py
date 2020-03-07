import torchvision.models as models
import torch

from ..utils import get_component, CallableWrapper


def get_model(config):
    model_type = config.pop('type')
    input_keys = config.pop('input_keys')
    output_keys = config.pop('output_keys')
    weights_path = config.pop('weights_path', None)
    model = get_component(__name__, model_type, config, (models,))
    if weights_path is not None:
        model.load_state_dict(torch.load(weights_path))
    return CallableWrapper(model, input_keys, output_keys)
