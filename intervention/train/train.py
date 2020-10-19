from typing import Optional, List

import torch
from pathlib import Path
from loguru import logger

import numpy as np

from ..models.image import Image
from .. import process
from . import dataset

TRAIN_EPOCHS: int = 5


def select_branch(branches: List[torch.Tensor], commands: List[int]) -> torch.Tensor:
    size = branches[0].size()
    result = torch.zeros(*size, device=branches[0].device)
    for (idx, command) in enumerate(commands):
        # commands are 1-based (valid values 1, 2, 3, 4)
        result[idx, :] += branches[command - 1][idx, :]
    return result


def test(
    dataset_path: Path,
    output_checkpoint_path: Path,
    batch_size: int = 30,
    initial_checkpoint_path: Optional[Path] = None,
) -> None:
    training_dataset = dataset.off_policy_data(dataset_path)
    training_generator = torch.utils.data.DataLoader(
        training_dataset, batch_size=batch_size, shuffle=True
    )

    model = Image().to(process.torch_device)
    model.train()

    img_size = torch.tensor([384, 160], device=process.torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    initial_epoch = 0
    if initial_checkpoint_path is not None:
        logger.info(f"Reading checkpoint from {initial_checkpoint_path}.")
        checkpoint = torch.load(initial_checkpoint_path)

        logger.info(f"Resuming from Epoch {checkpoint['epoch']} checkpoint.")
        initial_epoch = checkpoint["epoch"] + 1

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    for epoch in range(initial_epoch, initial_epoch + TRAIN_EPOCHS):
        out_path = output_checkpoint_path / f"{epoch}.pth"
        if out_path.exists():
            raise Exception(
                f"Output checkpoint for Epoch {epoch} already exists: {out_path}."
            )

        num_batches = len(training_generator)
        logger.info(
            f"Performing Epoch {epoch} ({epoch+1-initial_epoch}/{TRAIN_EPOCHS})."
        )
        for (batch_number, (rgb_image, _, datapoint_meta)) in enumerate(
            training_generator
        ):
            rgb_image = rgb_image.float().to(process.torch_device)
            speed = datapoint_meta["speed"].float().to(process.torch_device)

            all_branch_predictions, *_ = model.forward(rgb_image, speed)
            del rgb_image, speed

            pred_locations = select_branch(
                all_branch_predictions, list(map(int, datapoint_meta["command"]))
            )
            del all_branch_predictions

            locations = datapoint_meta["next_locations_image_coordinates"].to(process.torch_device)
            # Transform X and Y differently; we can never have a waypoint above the
            # horizon (i.e. above the vertical middle of the camera frame).
            locations[..., 0] = locations[..., 0] / (0.5 * img_size[0]) - 1
            locations[..., 1] = (locations[..., 1] - img_size[1] / 2) / (
                0.25 * img_size[1]
            ) - 1

            loss = torch.mean(torch.abs(pred_locations - locations), dim=(1, 2))
            del pred_locations, locations

            loss_mean = loss.mean()
            del loss

            optimizer.zero_grad()
            loss_mean.backward()
            optimizer.step()

            logger.trace(
                f"Finished Batch {batch_number} ({batch_number+1}/{num_batches}). "
                f"Mean loss: {loss_mean}."
            )
            del loss_mean

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            out_path,
        )
        logger.info(f"Saved Epoch {epoch} checkpoint to {out_path}.")
