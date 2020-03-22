import torch.nn as nn
from torchvision.models import resnet50


def create(config):
    return EmbedNet(**config)


class EmbedNet(nn.Module):
    def __init__(self, embedding_size, pool_size=(1, 1), pretrained=True):
        super().__init__()
        self.resnet = resnet50(pretrained=pretrained)
        self.resnet.avgpool = nn.AdaptiveAvgPool2d(pool_size)
        self.resnet.fc = nn.Linear(2048*pool_size[0]*pool_size[1], embedding_size)

    def forward(self, x):
        return self.resnet(x)
