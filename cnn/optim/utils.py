import copy
import torch.nn as nn

from .dws import DepthWiseSeparableConv2d


def conv2d_to_dws_conv2d(model):
    """Replaces nn.Conv2d layers with DepthWiseSeparableConv2d layers"""
    dws_model = copy.deepcopy(model)
    conv2d_layer_dict = get_conv2d_layer_dict(dws_model)
    for name, conv in conv2d_layer_dict.items():
        submodules = name.split('.')
        module = dws_model
        for submod in submodules[:-1]:
            module = module._modules[submod]
        module._modules[submodules[-1]] = DepthWiseSeparableConv2d(
            conv.in_channels, conv.out_channels, conv.kernel_size, stride=conv.stride, padding=conv.padding,
            dilation=conv.dilation, bias=(conv.bias is not None), padding_mode=conv.padding_mode
        )
    return dws_model


def get_conv2d_layer_dict(model):
    """Create a name: module dictionary for Conv2d layers"""
    return _get_layer_dict(model, nn.Conv2d)


def get_dws_conv2d_layer_dict(model):
    """Create a name: module dictionary for DepthWiseSeparableConv2d layers"""
    return _get_layer_dict(model, DepthWiseSeparableConv2d)


def register_hooks(layer_dict):
    """Register forward hooks which place layer outputs in a dictionary"""
    output_dict = {name: None for name in layer_dict.keys()}
    handle_dict = {name: None for name in layer_dict.keys()}
    for name, layer in layer_dict.items():
        def _hook(module, inputs, output):
            output_dict[name] = output
        handle_dict[name] = layer.register_forward_hook(_hook)
    return output_dict, handle_dict


def _get_layer_dict(model, layer_type):
    layer_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, layer_type):
            layer_dict[name] = module
    return layer_dict
