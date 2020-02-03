import torch
import torch.nn as nn

from .utils import get_conv2d_layer_dict, get_dws_conv2d_layer_dict


class DepthWiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, bias=True, padding_mode='zeros'):
        super().__init__()
        self.dws_conv2d = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding,
                      dilation=dilation, bias=bias, padding_mode=padding_mode, groups=in_channels),
            nn.Conv2d(in_channels, out_channels, 1, padding=0, bias=bias)
        )

    def forward(self, x):
        return self.dws_conv2d(x)


class L2DifferenceLoss:
    def __init__(self, base_model, dws_model):
        self.base_model = base_model
        self.base_model.eval()
        self.base_model.requires_grad = False

        self.conv2d_layer_dict = get_conv2d_layer_dict(self.base_model)
        self.dws_conv2d_layer_dict = get_dws_conv2d_layer_dict(dws_model)

    def __call__(self, outputs, inputs):
        _ = self.base_model(inputs)
        loss = 0
        for name in self.conv2d_layer_dict.keys():
            loss += torch.norm(self.conv2d_layer_dict[name] - self.dws_conv2d_layer_dict[name])
        return loss
