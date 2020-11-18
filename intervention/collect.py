import csv
import os
import uuid
import zipfile
from pathlib import Path

import numpy as np
from loguru import logger

from . import data, process, run
from .utils import multiprocessing as utils_process


def collect_example_episode(
    teacher_checkpoint: Path, episode_dir: Path, seed_sequence: np.random.SeedSequence,
) -> data.EpisodeSummary:
    process.rng = np.random.default_rng(seed_sequence)

    with zipfile.ZipFile(episode_dir / "data.zip", mode="w") as zip_archive:
        with open(episode_dir / "episode.csv", mode="w", newline="") as csv_file:
            store = data.ZipStore(zip_archive, csv_file)
            summary = run.run_example_episode(store, teacher_checkpoint)
            store.stop()
            return summary


def collect_example_episodes(
    teacher_checkpoint: Path, data_path: Path, num_episodes: int,
) -> None:
    parent_seed_sequence = np.random.SeedSequence()

    episode_summaries_path = data_path / "episodes.csv"
    file_exists = os.path.isfile(episode_summaries_path)
    with open(episode_summaries_path, mode="a", newline="") as episode_summaries:
        episode_summaries_writer = csv.DictWriter(
            episode_summaries,
            fieldnames=data.EpisodeSummary.__dataclass_fields__.keys(),
        )
        if not file_exists:
            episode_summaries_writer.writeheader()

        for episode in range(num_episodes):
            [seed_sequence] = parent_seed_sequence.spawn(1)

            episode_id = uuid.uuid4()
            logger.info(f"Collecting episode {episode+1}/{num_episodes}: {episode_id}.")
            episode_dir = data_path / str(episode_id)
            episode_dir.mkdir(parents=True, exist_ok=False)

            # Run in process to circumvent Carla bug
            episode_summary = utils_process.process_wrapper(
                collect_example_episode, teacher_checkpoint, episode_dir, seed_sequence,
            )
            episode_summary.uuid = str(episode_id)
            episode_summaries_writer.writerow(episode_summary.as_csv_writeable_dict())

            # try:
            # except (exceptions.EpisodeStuck, exceptions.CollisionInEpisode) as exception:
            #     logger.info(f"Removing episode because of episode exception: {exception}.")
            #     (episode_dir / "data.zip").unlink()
            #     (episode_dir / "episode.csv").unlink()
            #     episode_dir.rmdir()


def collect_on_policy_episode(
    student_checkpoint: Path,
    teacher_checkpoint: Path,
    episode_dir: Path,
    seed_sequence: np.random.SeedSequence,
) -> data.EpisodeSummary:
    process.rng = np.random.default_rng(seed_sequence)

    with zipfile.ZipFile(episode_dir / "data.zip", mode="w") as zip_archive:
        with open(episode_dir / "episode.csv", mode="w", newline="") as csv_file:
            store = data.ZipStore(zip_archive, csv_file)
            summary = run.run_on_policy_episode(
                store, student_checkpoint, teacher_checkpoint
            )
            store.stop()
            return summary


def collect_on_policy_episodes(
    student_checkpoint: Path,
    teacher_checkpoint: Path,
    data_path: Path,
    num_episodes: int,
) -> None:
    parent_seed_sequence = np.random.SeedSequence()

    episode_summaries_path = data_path / "episodes.csv"
    file_exists = os.path.isfile(episode_summaries_path)
    with open(episode_summaries_path, mode="a", newline="") as episode_summaries:
        episode_summaries_writer = csv.DictWriter(
            episode_summaries,
            fieldnames=data.EpisodeSummary.__dataclass_fields__.keys(),
        )
        if not file_exists:
            episode_summaries_writer.writeheader()

        for episode in range(num_episodes):
            [seed_sequence] = parent_seed_sequence.spawn(1)

            episode_id = uuid.uuid4()
            logger.info(f"Collecting episode {episode+1}/{num_episodes}: {episode_id}.")
            episode_dir = data_path / str(episode_id)
            episode_dir.mkdir(parents=True, exist_ok=False)

            # Run in process to circumvent Carla bug
            episode_summary = utils_process.process_wrapper(
                collect_on_policy_episode,
                student_checkpoint,
                teacher_checkpoint,
                episode_dir,
                seed_sequence,
            )
            episode_summary.uuid = str(episode_id)
            episode_summaries_writer.writerow(episode_summary.as_csv_writeable_dict())
