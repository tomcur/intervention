import torch
from pathlib import Path
from loguru import logger

import numpy as np

from ..models.image import Image
from . import dataset

TRAIN_EPOCHS: int = 5


def select_branch(branches, commands):
    size = branches[0].size()
    result = torch.zeros(size[0], size[1], size[2])
    for (idx, command) in enumerate(commands):
        result[idx, :] += branches[command][idx, :]
    return result
    # shape = branches.size()


def test():
    training_dataset = dataset.off_policy_data(Path("./test-data"))
    training_generator = torch.utils.data.DataLoader(
        training_dataset, batch_size=50, shuffle=True
    )

    model = Image()

    img_size = torch.FloatTensor([384, 160])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(TRAIN_EPOCHS):
        logger.info(f"Performing epoch {epoch+1}/{TRAIN_EPOCHS}.")
        for (batch_number, (rgb_image, datapoint_meta)) in enumerate(
            training_generator
        ):
            all_branch_predictions = model.forward(
                rgb_image.float(), datapoint_meta["speed"].float()
            )

            pred_locations = select_branch(
                all_branch_predictions, datapoint_meta["command"]
            )

            locations = datapoint_meta["next_locations_image_coordinates"]
            locations = locations / (0.5 * img_size) - 1
            loss = torch.mean(torch.abs(pred_locations - locations), dim=(1, 2))
            loss_mean = loss.mean()

            optimizer.zero_grad()
            loss_mean.backward()
            optimizer.step()

            logger.trace(f"Batch {batch_number+1} mean loss: {loss_mean}")
