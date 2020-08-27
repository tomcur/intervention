import torch
from torch import nn

from .spatial_softargmax import SpatialSoftargmax

class TaillessResnet34(nn.Module):
    """
    Resnet from torchvision without the average pooling, flattening, and
    fully-connected layers.
    """

    def __init__(self):
        super().__init__()

        self._resnet = torch.hub.load(
            "pytorch/vision:v0.6.0", "resnet34", pretrained=True
        )

    def forward(self, image):
        image = self._resnet.conv1(image)
        image = self._resnet.bn1(image)
        image = self._resnet.relu(image)
        image = self._resnet.maxpool(image)

        image = self._resnet.layer1(image)
        image = self._resnet.layer2(image)
        image = self._resnet.layer3(image)
        image = self._resnet.layer4(image)

        return image


class Image(nn.Module):
    OUTPUTS: int = 4
    HEATMAP_WIDTH: int = 96
    HEATMAP_HEIGHT: int = 40
    COORDINATE_STEPS: int = 5

    def __init__(self):
        super().__init__()

        self.resnet = TaillessResnet34()

        self.higher = nn.Sequential(
            nn.BatchNorm2d(512 + 1),
            nn.ConvTranspose2d(512 + 1, 256, 3, 2, 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),
            nn.ReLU(True),
        )

        self.location_prediction = nn.ModuleList(
            [
                nn.Sequential(
                    nn.BatchNorm2d(64),
                    nn.Conv2d(64, Image.COORDINATE_STEPS, 1, 1, 0),
                    SpatialSoftargmax(
                        Image.HEATMAP_WIDTH,
                        Image.HEATMAP_HEIGHT,
                        Image.COORDINATE_STEPS,
                    ),
                )
                for _ in range(Image.OUTPUTS)
            ]
        )

    def forward(self, image, velocity):
        resnet_out = self.resnet(image)

        velocity = velocity.repeat((1, 1, 5, 12))
        higher_in = torch.cat((resnet_out, velocity), dim=1)

        higher_out = self.higher(higher_in)

        location_predictions = [
            location_prediction(higher_out)
            for location_prediction in self.location_prediction
        ]

        return location_predictions
