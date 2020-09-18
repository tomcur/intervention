import torch
from pathlib import Path
from loguru import logger

import numpy as np

from ..models.image import Image
from . import dataset

TRAIN_EPOCHS: int = 5


def select_branch(branches, commands):
    size = branches[0].size()
    result = torch.zeros(*size, device=branches[0].device)
    for (idx, command) in enumerate(commands):
        # commands are 1-based (valid values 1, 2, 3, 4)
        result[idx, :] += branches[command - 1][idx, :]
    return result
    # shape = branches.size()


def test(
    device: torch.device,
    checkpoint_path: Path,
    batch_size=30,
):
    training_dataset = dataset.off_policy_data(Path("./test-data"))
    training_generator = torch.utils.data.DataLoader(
        training_dataset, batch_size=batch_size, shuffle=True
    )

    model = Image().to(device)

    img_size = torch.tensor([384, 160], device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(TRAIN_EPOCHS):
        logger.info(f"Performing epoch {epoch+1}/{TRAIN_EPOCHS}.")
        for (batch_number, (rgb_image, datapoint_meta)) in enumerate(
            training_generator
        ):
            rgb_image = rgb_image.float().to(device)
            speed = datapoint_meta["speed"].float().to(device)

            all_branch_predictions = model.forward(rgb_image, speed)
            del rgb_image, speed

            pred_locations = select_branch(
                all_branch_predictions, datapoint_meta["command"]
            )
            del all_branch_predictions

            locations = datapoint_meta["next_locations_image_coordinates"].to(device)
            locations = locations / (0.5 * img_size) - 1
            loss = torch.mean(torch.abs(pred_locations - locations), dim=(1, 2))
            del pred_locations, locations

            loss_mean = loss.mean()
            del loss

            optimizer.zero_grad()
            loss_mean.backward()
            optimizer.step()

            logger.trace(f"Batch {batch_number+1} mean loss: {loss_mean}")
            del loss_mean

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            checkpoint_path / f"{epoch}.pth",
        )
