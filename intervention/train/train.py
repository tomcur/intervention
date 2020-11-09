from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from loguru import logger

from .. import process
from ..models.image import Image
from . import dataset


def select_branch(branches: List[torch.Tensor], commands: List[int]) -> torch.Tensor:
    size = branches[0].size()
    result = torch.zeros(*size, device=branches[0].device)
    for (idx, command) in enumerate(commands):
        # commands are 1-based (valid values 1, 2, 3, 4)
        result[idx, :] += branches[command - 1][idx, :]
    return result


def imitation(
    dataset_path: Path,
    output_checkpoint_path: Path,
    batch_size: int = 30,
    initial_checkpoint_path: Optional[Path] = None,
    epochs: int = 5,
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

    for epoch in range(initial_epoch, initial_epoch + epochs):
        out_path = output_checkpoint_path / f"{epoch}.pth"
        if out_path.exists():
            raise Exception(
                f"Output checkpoint for Epoch {epoch} already exists: {out_path}."
            )

        num_batches = len(training_generator)
        logger.info(
            f"Performing Epoch {epoch} ({epoch+1-initial_epoch}/{epochs})."
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


def _intervention_data_loaders(
    intervention_dataset_path: Path, imitation_dataset_path: Path, batch_size: int
) -> Tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
]:
    """
    Creates a 3-tuple of `torch.utils.data.DataLoader` respectively:
    (1) generating negative batches, (2) recovery imitation batches and (3) regular
    imitation batches.

    The batch sizes are such that the total batch distribution is the same as the
    natural distribution of the intervention dataset.
    """
    intervention_datasets = dataset.intervention_data(intervention_dataset_path)
    imitation_dataset = dataset.off_policy_data(imitation_dataset_path)

    negative_len = len(intervention_datasets.negative)
    recovery_imitation_len = len(intervention_datasets.imitation)
    supervision_signal_len = len(intervention_datasets.supervision_signal)

    logger.debug(
        "Intervention dataset sizes:\n"
        f"\tNegative: {negative_len}\n"
        f"\tImitation: {recovery_imitation_len}\n"
        f"\tSupervision signal: {supervision_signal_len}\n"
    )

    total_len = negative_len + recovery_imitation_len + supervision_signal_len

    negative_batch_size = round(negative_len / total_len * batch_size)
    recovery_imitation_batch_size = round(
        recovery_imitation_len / total_len * batch_size
    )
    regular_imitation_batch_size = (
        batch_size - negative_batch_size - recovery_imitation_batch_size
    )

    logger.debug(
        "Batch sizes:\n"
        f"\tNegative: {negative_batch_size}\n"
        f"\tRecovery imitation: {recovery_imitation_batch_size}\n"
        f"\tRegular imitation: {regular_imitation_batch_size}\n"
    )

    assert negative_batch_size > 0
    assert recovery_imitation_batch_size > 0
    assert regular_imitation_batch_size > 0

    negative_generator = torch.utils.data.DataLoader(
        intervention_datasets.negative, batch_size=negative_batch_size, shuffle=True
    )
    recovery_imitation_generator = torch.utils.data.DataLoader(
        intervention_datasets.imitation,
        batch_size=recovery_imitation_batch_size,
        shuffle=True,
    )
    regular_imitation_generator = torch.utils.data.DataLoader(
        imitation_dataset, batch_size=regular_imitation_batch_size, shuffle=True,
    )

    return negative_generator, recovery_imitation_generator, regular_imitation_generator


def intervention(
    intervention_dataset_path: Path,
    imitation_dataset_path: Path,
    output_checkpoint_path: Path,
    batch_size: int = 30,
    initial_checkpoint_path: Optional[Path] = None,
    epochs: int = 5,
) -> None:
    (
        negative_generator,
        recovery_imitation_generator,
        regular_imitation_generator,
    ) = _intervention_data_loaders(
        intervention_dataset_path, imitation_dataset_path, batch_size
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
        # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    for epoch in range(initial_epoch, initial_epoch + epochs):
        out_path = output_checkpoint_path / f"{epoch}.pth"
        if out_path.exists():
            raise Exception(
                f"Output checkpoint for Epoch {epoch} already exists: {out_path}."
            )

        num_batches = min(
            len(negative_generator),
            len(recovery_imitation_generator),
            len(regular_imitation_generator),
        )
        logger.info(f"Performing Epoch {epoch} ({epoch+1-initial_epoch}/{epochs}).")
        for (
            batch_number,
            (negative_batch, recovery_imitation_batch, regular_imitation_batch),
        ) in enumerate(
            zip(
                iter(negative_generator),
                iter(recovery_imitation_generator),
                iter(regular_imitation_generator),
            )
        ):
            (
                negative_rgb_images,
                _,
                negative_model_output,
                negative_datapoint,
            ) = negative_batch
            (
                recovery_imitation_rgb_images,
                _,
                recovery_imitation_datapoint,
            ) = recovery_imitation_batch
            (
                regular_imitation_rgb_images,
                _,
                regular_imitation_datapoint,
            ) = regular_imitation_batch

            negative_len = len(negative_rgb_images)
            recovery_imitation_len = len(recovery_imitation_rgb_images)
            regular_imitation_len = len(regular_imitation_rgb_images)

            rgb_images = torch.cat(
                (
                    negative_rgb_images.float(),
                    recovery_imitation_rgb_images.float(),
                    regular_imitation_rgb_images.float(),
                )
            ).to(process.torch_device)
            del negative_rgb_images
            del recovery_imitation_rgb_images
            del regular_imitation_rgb_images

            speeds = torch.cat(
                (
                    negative_datapoint["speed"].float(),
                    recovery_imitation_datapoint["speed"].float(),
                    regular_imitation_datapoint["speed"].float(),
                )
            ).to(process.torch_device)

            all_branch_predictions, *_ = model.forward(rgb_images, speeds)
            del rgb_images, speeds

            commands = torch.cat(
                (
                    negative_datapoint["command"],
                    recovery_imitation_datapoint["command"],
                    regular_imitation_datapoint["command"],
                )
            )

            # Select the predicted locations head based on the planner's command.
            pred_locations = select_branch(
                all_branch_predictions, list(map(int, commands))
            )
            del commands, all_branch_predictions

            # Swap dimensions. Coming from the data loader, the first dimension are the
            # batch samples. We expect the first dimension to be the output heads...
            negative_model_output = torch.transpose(negative_model_output, 0, 1)

            # ... and we actually expect the output heads to be a list.
            converted = [out for out in negative_model_output[:, ...]]

            # Select the original (erroneous) student model output head based on the
            # planner's command.
            original_negative_model_output = select_branch(
                converted, list(map(int, negative_datapoint["command"])),
            )

            locations = torch.cat(
                (
                    original_negative_model_output,
                    recovery_imitation_datapoint[
                        "next_locations_image_coordinates"
                    ].float(),
                    regular_imitation_datapoint[
                        "next_locations_image_coordinates"
                    ].float(),
                )
            ).to(process.torch_device)

            # Transform X and Y differently; we can never have a waypoint above the
            # horizon (i.e. above the vertical middle of the camera frame).
            locations[..., 0] = locations[..., 0] / (0.5 * img_size[0]) - 1
            locations[..., 1] = (locations[..., 1] - img_size[1] / 2) / (
                0.25 * img_size[1]
            ) - 1

            loss = torch.mean(torch.abs(pred_locations - locations), dim=(1, 2))
            del pred_locations, locations

            # The learning rates of the negative samples are negative.
            # TODO: negative sample learning rate curve
            meta_learning_rates = torch.cat(
                (
                    -1 * torch.ones(negative_len),
                    torch.ones(recovery_imitation_len),
                    torch.ones(regular_imitation_len),
                )
            ).to(process.torch_device)

            loss = meta_learning_rates * loss
            loss_mean = loss.mean()
            del loss, meta_learning_rates

            logger.trace(
                f"Finished Batch {batch_number} ({batch_number+1}/{num_batches}). "
                f"Mean loss: {loss_mean}."
            )

            optimizer.zero_grad()
            loss_mean.backward()
            optimizer.step()

            del loss_mean
