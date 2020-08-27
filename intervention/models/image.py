import torch
from torch import nn

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
