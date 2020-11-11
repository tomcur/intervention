import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np


class SpatialSoftargmax(nn.Module):
    """
    From: https://gist.github.com/jeasinema/1cba9b40451236ba2cfb507687e08834
    """

    def __init__(self, height, width, channel, temperature=None):
        super().__init__()
        self.height = height
        self.width = width
        self.channel = channel

        if temperature:
            self.temperature = torch.ones(1) * temperature
        else:
            self.temperature = Parameter(torch.ones(1))

        pos_y, pos_x = np.meshgrid(
            np.linspace(-1.0, 1.0, self.height),
            np.linspace(-1.0, 1.0, self.width),
            indexing="ij",
        )
        pos_x = torch.from_numpy(pos_x.reshape(self.height * self.width)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self.height * self.width)).float()
        self.register_buffer("pos_x", pos_x)
        self.register_buffer("pos_y", pos_y)

    def forward(self, feature):
        # Output:
        #   (N, C, 2) [[[x_0, y_0], ...]]

        size = feature.size()
        feature = feature.view(-1, self.height * self.width)

        softmax_attention = F.softmax(feature / self.temperature, dim=-1)
        expected_x = torch.sum(self.pos_x * softmax_attention, dim=1, keepdim=True)
        expected_y = torch.sum(self.pos_y * softmax_attention, dim=1, keepdim=True)
        expected_xy = torch.cat([expected_x, expected_y], 1)
        feature_keypoints = expected_xy.view(-1, self.channel, 2)

        return feature_keypoints, softmax_attention.view(*size)

    def to_width_height_range(self, coordinates):
        """
        Transforms coordinates of dimensions [N, C, 2] in value range [-1, 1] to range
        [0, width-1] and [0, height-1]
        """
        return (
            (coordinates + 1.0)
            / 2.0
            * torch.tensor([[[self.width - 1, self.height - 1]]])
        )
