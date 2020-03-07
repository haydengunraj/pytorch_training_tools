import torchvision.models as models
import torch

from .wrapper import ModelWrapper
from ..utils import import_submodule


def get_model(config):
    model_type = config.pop('type')
    input_map = config.pop('input_map')
    output_keys = config.pop('output_keys')
    weights_path = config.pop('weights_path', None)
    module = import_submodule(__name__, model_type)
    if module is not None:
        model = module.get_model(config)
    else:
        model = getattr(models, model_type)
        if model is None:
            raise ValueError('Unrecognized model type: ' + model_type)
        model = model(**config)
    if weights_path is not None:
        model.load_state_dict(torch.load(weights_path))
    return ModelWrapper(model, input_map, output_keys)
