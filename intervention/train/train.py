import math
from pathlib import Path, PurePath
from typing import List, Optional, Tuple

import numpy as np
import torch
import torchvision
from loguru import logger
from torch.utils.tensorboard import SummaryWriter

from .. import process
from ..models.image import Image
from . import dataset

EPSILON = 1e-8


def select_branch(branches: List[torch.Tensor], commands: List[int]) -> torch.Tensor:
    size = branches[0].size()
    result = torch.zeros(*size, device=branches[0].device)
    for (idx, command) in enumerate(commands):
        # commands are 1-based (valid values 1, 2, 3, 4)
        result[idx, :] += branches[command - 1][idx, :]
    return result


def cross_entropy_four_hot(x: float, y: float, width: int, height: int) -> torch.Tensor:
    """
    Create a (spatial) probability distribution with four active cells,
    such that the expected value is centered on `(x, y)`.
    :param x: x-component of coordinate, within range -1.0 to 1.0
    :param y: y-component of coordinate, within range -1.0 to 1.0
    """
    t = torch.zeros(height, width)

    from_left = (x + 1.0) / 2.0 * (width - 1)
    from_top = (y + 1.0) / 2.0 * (height - 1)

    width_idx = math.floor(from_left)
    height_idx = math.floor(from_top)

    left_frac = from_left - width_idx
    top_frac = from_top - height_idx

    t[height_idx + 0, width_idx + 0] = (1 - top_frac) * (1 - left_frac)

    if width_idx + 1 < width:
        t[height_idx + 0, width_idx + 1] = (1 - top_frac) * left_frac

    if height_idx + 1 < height:
        t[height_idx + 1, width_idx + 0] = top_frac * (1 - left_frac)

    if width_idx + 1 < width and height_idx + 1 < height:
        t[height_idx + 1, width_idx + 1] = top_frac * left_frac

    return t


def imitation(
    dataset_path: Path,
    output_checkpoint_path: Path,
    batch_size: int = 30,
    initial_checkpoint_path: Optional[Path] = None,
    epochs: int = 5,
) -> None:
    #: Global learning rate multiplier
    LEARNING_RATE = 0.001
    GRADIENT_NORM_CLIPPING = 0.1

    training_dataset = dataset.off_policy_data(dataset_path)
    training_generator = torch.utils.data.DataLoader(
        training_dataset, batch_size=batch_size, shuffle=True
    )

    model = Image().to(process.torch_device)
    model.train()

    img_size = torch.tensor([384, 160], device=process.torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    initial_epoch = 0
    total_batches = 0
    if initial_checkpoint_path is not None:
        logger.info(f"Reading checkpoint from {initial_checkpoint_path}.")
        checkpoint = torch.load(initial_checkpoint_path)

        logger.info(f"Resuming from Epoch {checkpoint['epoch']} checkpoint.")
        initial_epoch = checkpoint["epoch"] + 1
        total_batches = checkpoint["total_batches"]

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    writer = SummaryWriter(
        log_dir=Path("logs") / "imitation" / PurePath(output_checkpoint_path).name,
        purge_step=initial_epoch,
    )

    for epoch in range(initial_epoch, initial_epoch + epochs):
        epoch_total_train_loss = 0.0

        out_path = output_checkpoint_path / f"{epoch}.pth"
        if out_path.exists():
            raise Exception(
                f"Output checkpoint for Epoch {epoch} already exists: {out_path}."
            )

        num_batches = len(training_generator)
        logger.info(f"Performing Epoch {epoch} ({epoch+1-initial_epoch}/{epochs}).")
        for (
            batch_number,
            (rgb_image, untransformed_rgb_image, datapoint_meta),
        ) in enumerate(training_generator):
            this_batch_size = len(rgb_image)

            rgb_image = rgb_image.float().to(process.torch_device)
            speed = datapoint_meta["speed"].float().to(process.torch_device)

            # At start of every epoch, store some data in TensorBoard for sanity
            # checks.
            if batch_number == 0:
                writer.add_text("progress", f"start of epoch {epoch}", total_batches)

                image_grid = torchvision.utils.make_grid(
                    untransformed_rgb_image,
                    nrow=5,
                )
                writer.add_image(
                    "images-rgb-untransformed", image_grid, global_step=total_batches
                )

                image_grid = torchvision.utils.make_grid(
                    rgb_image, nrow=5, normalize=True
                )
                writer.add_image(
                    "images-rgb-transformed", image_grid, global_step=total_batches
                )

                writer.add_graph(model, (rgb_image, speed))
            del untransformed_rgb_image

            _all_branch_predictions, all_branch_heatmaps = model.forward(
                rgb_image, speed
            )
            del _all_branch_predictions, rgb_image, speed

            pred_heatmaps = select_branch(
                all_branch_heatmaps, list(map(int, datapoint_meta["command"]))
            )

            locations = datapoint_meta["next_locations_image_coordinates"].to(
                process.torch_device
            )

            # Transform X and Y differently; we can never have a waypoint above the
            # horizon (i.e. above the vertical middle of the camera frame).
            locations[..., 0] = locations[..., 0] / (0.5 * img_size[0]) - 1
            locations[..., 1] = (locations[..., 1] - img_size[1] / 2) / (
                0.25 * img_size[1]
            ) - 1

            heatmaps_size = pred_heatmaps.size()
            target_four_hot = torch.zeros(*heatmaps_size)

            for batch in range(this_batch_size):
                for step in range(heatmaps_size[1]):
                    target_four_hot[batch, step, ...] = cross_entropy_four_hot(
                        locations[batch, step, 0],
                        locations[batch, step, 1],
                        heatmaps_size[3],
                        heatmaps_size[2],
                    )
            del locations

            target_four_hot = target_four_hot.to(process.torch_device)

            loss = torch.mean(
                -target_four_hot * torch.log(pred_heatmaps + EPSILON), dim=(1, 2, 3)
            )
            del target_four_hot, pred_heatmaps

            writer.add_histogram("loss", loss, global_step=total_batches)

            loss_mean = loss.mean()
            del loss

            writer.add_scalar("loss-mean", loss_mean, global_step=total_batches)

            optimizer.zero_grad()
            loss_mean.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), GRADIENT_NORM_CLIPPING, norm_type=2.0
            )
            optimizer.step()

            logger.trace(
                f"Finished Batch {batch_number} ({batch_number+1}/{num_batches}). "
                f"Mean loss: {loss_mean}."
            )
            epoch_total_train_loss += loss_mean.item()
            del loss_mean

            total_batches += 1

        writer.add_hparams(
            {
                "learning_rate": LEARNING_RATE,
                "gradient_norm_clipping": GRADIENT_NORM_CLIPPING,
                "batch_size": batch_size,
                "epoch": epoch,
            },
            {"hparam/epoch_mean_train_loss": epoch_total_train_loss / num_batches},
        )

        torch.save(
            {
                "epoch": epoch,
                "total_batches": total_batches,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            out_path,
        )
        logger.info(f"Saved Epoch {epoch} checkpoint to {out_path}.")

    writer.close()


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
        imitation_dataset,
        batch_size=regular_imitation_batch_size,
        shuffle=True,
    )

    return negative_generator, recovery_imitation_generator, regular_imitation_generator


def _predicted_locations_to_one_hot_heatmap(predicted_locations, img_size):
    # TODO: unit test this function
    batch_size, coordinate_steps, _ = predicted_locations.shape
    assert Image.COORDINATE_STEPS == coordinate_steps

    one_hot = torch.zeros(
        batch_size, coordinate_steps, Image.HEATMAP_HEIGHT, Image.HEATMAP_WIDTH
    ).to(process.torch_device)

    predicted_locations[..., 0] = (
        (predicted_locations[..., 0] + 1.0) / 2.0 * (Image.HEATMAP_WIDTH - 1)
    )

    predicted_locations[..., 1] = (
        (predicted_locations[..., 1] + 1.0) / 2.0 * (Image.HEATMAP_HEIGHT - 1)
    )

    predicted_locations[..., :] = torch.round(predicted_locations[..., :])
    predicted_locations = predicted_locations.long()

    one_hot[..., predicted_locations[..., 1], predicted_locations[..., 0]] = 1

    return one_hot


def intervention(
    intervention_dataset_path: Path,
    imitation_dataset_path: Path,
    output_checkpoint_path: Path,
    batch_size: int = 30,
    initial_checkpoint_path: Optional[Path] = None,
    epochs: int = 5,
) -> None:
    #: Global learning rate multiplier
    LEARNING_RATE = 0.001
    GRADIENT_NORM_CLIPPING = 0.1

    #: The exponential decay time constant of negative learning, unit: number of frames
    # until intervention
    NEGATIVE_LEARNING_DECAY_TIME = 10.0

    #: Initial learning rate (at the frame right before intervention)
    NEGATIVE_LEARNING_DECAY_INITIAL = 1.0

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
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    initial_epoch = 0
    total_batches = 0
    if initial_checkpoint_path is not None:
        logger.info(f"Reading checkpoint from {initial_checkpoint_path}.")
        checkpoint = torch.load(initial_checkpoint_path)

        logger.info(f"Resuming from Epoch {checkpoint['epoch']} checkpoint.")
        initial_epoch = checkpoint["epoch"] + 1
        total_batches = checkpoint["total_batches"]

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    writer = SummaryWriter(
        log_dir=Path("logs") / "intervention" / PurePath(output_checkpoint_path).name,
        purge_step=initial_epoch,
    )

    for epoch in range(initial_epoch, initial_epoch + epochs):
        epoch_total_train_loss = 0.0

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
                untransformed_negative_rgb_images,
                _negative_image_targets_output,
                negative_image_heatmaps_output,
                negative_datapoint,
            ) = negative_batch
            (
                recovery_imitation_rgb_images,
                untransformed_recovery_imitation_rgb_images,
                recovery_imitation_datapoint,
            ) = recovery_imitation_batch
            (
                regular_imitation_rgb_images,
                untransformed_regular_imitation_rgb_images,
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

            # At start of every epoch, store some data in TensorBoard for sanity
            # checks.
            if batch_number == 0:
                writer.add_text("progress", f"start of epoinfch {epoch}", total_batches)

                untransformed_rgb_images = torch.cat(
                    (
                        untransformed_negative_rgb_images,
                        untransformed_recovery_imitation_rgb_images,
                        untransformed_regular_imitation_rgb_images,
                    )
                )

                image_grid = torchvision.utils.make_grid(
                    untransformed_rgb_images,
                    nrow=5,
                )
                writer.add_image(
                    "images-rgb-untransformed", image_grid, global_step=total_batches
                )

                image_grid = torchvision.utils.make_grid(
                    rgb_images, nrow=5, normalize=True
                )
                writer.add_image(
                    "images-rgb-transformed", image_grid, global_step=total_batches
                )

                writer.add_graph(model, (rgb_images, speeds))

            all_branch_predictions, all_branch_heatmaps = model.forward(
                rgb_images, speeds
            )
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
            pred_locations = pred_locations[negative_len:, ...]

            pred_heatmaps = select_branch(all_branch_heatmaps, list(map(int, commands)))
            del commands, all_branch_predictions, all_branch_heatmaps

            # Swap dimensions. Coming from the data loader, the first dimension are the
            # batch samples. We expect the first dimension to be the output heads...
            negative_image_heatmaps_output = torch.transpose(
                negative_image_heatmaps_output, 0, 1
            )

            # ... and we actually expect the output heads to be a list.
            converted = [out for out in negative_image_heatmaps_output[:, ...]]

            # Select the original (erroneous) student model output head based on the
            # planner's command.
            original_negative_heatmaps_output = select_branch(
                converted,
                list(map(int, negative_datapoint["command"])),
            ).to(process.torch_device)

            meta_learning_rates = torch.ones(negative_len)

            locations = torch.cat(
                (
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

            heatmaps_size = pred_heatmaps.size()
            target_four_hot = torch.zeros(
                (
                    recovery_imitation_len + regular_imitation_len,
                    heatmaps_size[1],
                    heatmaps_size[2],
                    heatmaps_size[3],
                )
            )

            for example_idx in range(recovery_imitation_len + regular_imitation_len):
                for step in range(heatmaps_size[1]):
                    target_four_hot[example_idx, step, ...] = cross_entropy_four_hot(
                        locations[example_idx, step, 0],
                        locations[example_idx, step, 1],
                        heatmaps_size[3],
                        heatmaps_size[2],
                    )
            del locations

            target_four_hot = target_four_hot.to(process.torch_device)

            targets = torch.cat(
                (
                    original_negative_heatmaps_output,
                    target_four_hot,
                )
            ).to(process.torch_device)
            del original_negative_heatmaps_output, target_four_hot

            meta_learning_rates = torch.cat(
                (
                    -(
                        NEGATIVE_LEARNING_DECAY_INITIAL
                        * torch.exp(
                            -negative_datapoint["ticks_to_intervention"].float()
                            / NEGATIVE_LEARNING_DECAY_TIME
                        )
                    ),
                    torch.ones(recovery_imitation_len),
                    torch.ones(regular_imitation_len),
                )
            ).to(process.torch_device)

            loss = torch.mean(
                -targets * torch.log(pred_heatmaps + EPSILON), dim=(1, 2, 3)
            )
            del targets, pred_heatmaps

            writer.add_histogram("loss", loss, global_step=total_batches)

            loss_mean = (meta_learning_rates * loss).mean()
            del loss

            writer.add_scalar("loss-mean", loss_mean, global_step=total_batches)

            optimizer.zero_grad()
            loss_mean.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), GRADIENT_NORM_CLIPPING, norm_type=2.0
            )
            optimizer.step()

            logger.trace(
                f"Finished Batch {batch_number} ({batch_number+1}/{num_batches}). "
                f"Mean loss: {loss_mean}."
            )
            epoch_total_train_loss += loss_mean.item()
            del loss_mean

            total_batches += 1

        writer.add_hparams(
            {
                "learning_rate": LEARNING_RATE,
                "gradient_norm_clipping": GRADIENT_NORM_CLIPPING,
                "negative_learning_decay_initial": NEGATIVE_LEARNING_DECAY_INITIAL,
                "negative_learning_decay_time": NEGATIVE_LEARNING_DECAY_TIME,
                "batch_size": batch_size,
                "epoch": epoch,
            },
            {"hparam/epoch_mean_train_loss": epoch_total_train_loss / num_batches},
        )

        torch.save(
            {
                "epoch": epoch,
                "total_batches": total_batches,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            out_path,
        )

        logger.info(f"Saved Epoch {epoch} checkpoint to {out_path}.")
