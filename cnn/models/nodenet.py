import torch
import torch.nn as nn
import torch.nn.functional as F


class NodeNet3D(nn.Module):
    def __init__(self, input_size, input_channels=3, embedding_size=128, classifier=False,
                 n_classes=3, weights_path=None):
        super().__init__()
        self._classifier = classifier

        self.conv = nn.Sequential(
            # Block 1

            nn.Conv3d(input_channels, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2, stride=2),

            # Block 2
            nn.Conv3d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2, stride=2),

            # Block 3
            nn.Conv3d(128, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2, stride=2),
        )
        in_features = 256*(input_size//8)**3
        self.linear = nn.Linear(in_features, embedding_size)
        self.linear_out = nn.Linear(embedding_size, n_classes)

        if weights_path is not None:
            self.load_state_dict(torch.load(weights_path))

    def forward(self, x):
        x = self.conv(x)
        x = x.view((x.size(0), -1))
        x = self.linear(x)
        if self._classifier:
            x = self.linear_out(x)
        else:
            x = F.normalize(x, p=2, dim=1)
        return x

    def inference(self, x):
        x = self.forward(x)
        if self._classifier:
            x = F.softmax(x, dim=1)
        return x

    def classifier(self, classifier=True):
        self._classifier = classifier
