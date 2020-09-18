from typing import Optional

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


def test(
    device: torch.device,
    output_checkpoint_path: Path,
    batch_size=30,
    initial_checkpoint_path: Optional[Path] = None,
):
    training_dataset = dataset.off_policy_data(Path("./test-data"))
    training_generator = torch.utils.data.DataLoader(
        training_dataset, batch_size=batch_size, shuffle=True
    )

    model = Image().to(device)
    model.train()

    img_size = torch.tensor([384, 160], device=device)
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
            raise Exception(f"Output checkpoint for Epoch {epoch} already exists.")

        num_batches = len(training_generator)
        logger.info(
            f"Performing Epoch {epoch} ({epoch+1-initial_epoch}/{TRAIN_EPOCHS})."
        )
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

            logger.trace(
                f"Finished Batch {batch_number} ({batch_number}/{num_batches}). "
                f"Mean loss: {loss_mean}"
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
        logger.info("Saved Epoch {epoch} checkpoint to {out_path}.")
