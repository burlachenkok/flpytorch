#!/usr/bin/env python3

"""
CNN model for FEMNIST Dataset.
"""
from torch import nn

# Import PyTorch layers, activations and more
import torch.nn.functional as F


class FEMNIST(nn.Module):
    def __init__(self, channel_1=32, channel_2=64, num_classes=62):
        super(FEMNIST, self).__init__()

        self.conv1 = nn.Conv2d(1, channel_1, (5, 5))
        self.conv2 = nn.Conv2d(channel_1, channel_2, (5, 5))

        # Fully connected layer from 16 * channel_2 to num_classes units
        self.fc = nn.Linear(16 * channel_2, num_classes)

    def forward(self, x):
        out = F.relu(F.max_pool2d(self.conv1(x), 2))
        out = F.relu(F.max_pool2d(self.conv2(out), 2))
        out = self.fc(nn.Flatten(out))
        return out


# TODO: Any way we can actually have an useful pretrained argument here?
def femnist(pretrained=False, num_classes=62):
    return FEMNIST(num_classes=num_classes)


def minifemnist(pretrained=False, num_classes=62):
    return FEMNIST(num_classes=num_classes, channel_1=10, channel_2=20)
