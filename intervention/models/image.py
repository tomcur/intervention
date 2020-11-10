from typing import List, Tuple

import numpy as np
import torch
import torchvision
from torch import nn

from .. import coordinates
from ..carla_utils.manager import TickState
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
    """
    This network produces a 2-tuple of outputs.

    The first output is a list (containing `Image.OUTPUTS` members) of the location
    predictions: a number of X and Y coordinate pairs (dimensionality of
    `[N, Image.COORDINATE_STEPS, 2]` with `N` the number of examples in the batch) which
    are the soft argmax of ego (camera perspective) heatmaps of predicted next
    locations.

    The second output is a list (containg `Image.OUTPUTS` members) of the heatmaps
    themselves (dimensionality
    `[N, Image.COORDINATE_STEPS, Image.HEATMAP_HEIGHT, Image.HEATMAP_WIDTH]`).

    The X and Y coordinate pairs are in range of `[-1, 1]`. The resolution is configured
    through `Image.HEATMAP_WIDTH` and `Image.HEATMAP_HEIGHT`.
    """

    OUTPUTS: int = 4
    HEATMAP_WIDTH: int = 384 // 4
    HEATMAP_HEIGHT: int = 160 // 4
    COORDINATE_STEPS: int = 5
    SPEED_FEATURE_MAPS: int = 128

    def __init__(self):
        super().__init__()

        self.resnet = TaillessResnet34()

        self.higher = nn.Sequential(
            nn.BatchNorm2d(512 + Image.SPEED_FEATURE_MAPS),
            nn.ConvTranspose2d(512 + Image.SPEED_FEATURE_MAPS, 256, 3, 2, 1, 1),
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
                        Image.HEATMAP_HEIGHT,
                        Image.HEATMAP_WIDTH,
                        Image.COORDINATE_STEPS,
                    ),
                )
                for _ in range(Image.OUTPUTS)
            ]
        )

    def forward(
        self, image: torch.Tensor, speed: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        resnet_out = self.resnet(image)

        speed = speed[:, None, None, None].repeat((1, Image.SPEED_FEATURE_MAPS, 5, 12))
        higher_in = torch.cat((resnet_out, speed), dim=1)

        higher_out = self.higher(higher_in)

        [location_predictions, location_heatmaps] = zip(
            *[
                location_prediction(higher_out)
                for location_prediction in self.location_prediction
            ]
        )

        return list(location_predictions), list(location_heatmaps)


class Agent:
    def __init__(self, model: nn.Module):
        self._transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self._model = model

        self._img_size = torch.tensor([384, 160])

    def step(self, state: TickState) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Send the state through the underlying model, and return its output
        as predicted ego coordinate waypoints.

        :return: A 3-tuple, where:
            - the first element is the predicted target locations of the commanded
              action in ego top-down coordinates;
            - the second element is the raw model output (dimensions:
              `[Image.OUTPUTS, Image.COORDINATE_STEPS, 2]`); and
            - the third element is the heatmap of the predictions of the commanded
              action.

        """
        image = torch.unsqueeze(self._transforms(state.rgb.copy()), 0)
        speed = torch.tensor([state.speed]).float()
        with torch.no_grad():
            predictions, heatmaps = self._model.forward(image, speed)

        # Get the singular heatmap and prediction based on the commanded action and
        # taking only the first minibatch result (the agent runs with a minibatch of
        # size 1).
        heatmap_out = heatmaps[state.command - 1][0, ...].cpu().detach().numpy()
        command_out = predictions[state.command - 1][0, ...]
        locations = command_out + 1
        locations[..., 0] = locations[..., 0] * 0.5 * self._img_size[0]
        locations[..., 1] = (
            locations[..., 1] * 0.25 * self._img_size[1]
        ) + self._img_size[1] / 2.0

        targets = np.zeros((5, 2))

        for idx, [image_x, image_y] in enumerate(locations.tolist()):
            ego_x, ego_y = coordinates.image_coordinate_to_ego_coordinate(
                image_x, image_y
            )
            targets[idx] = [ego_x, ego_y]

        flattened_predictions = np.array(
            [prediction[0, ...].cpu().detach().numpy() for prediction in predictions]
        )

        return targets, flattened_predictions, heatmap_out
