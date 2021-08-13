import csv
import os
import uuid
import zipfile
from pathlib import Path

import numpy as np
from loguru import logger

from . import data, process, run
from .utils import multiprocessing as utils_process


def collect_teacher_episode(
    teacher_checkpoint: Path,
    episode_dir: Path,
    seed_sequence: np.random.SeedSequence,
    metrics_only: bool,
) -> data.EpisodeSummary:
    process.rng = np.random.default_rng(seed_sequence)

    with zipfile.ZipFile(episode_dir / "data.zip", mode="w") as zip_archive:
        with open(episode_dir / "episode.csv", mode="w", newline="") as csv_file:
            store = data.ZipStore(zip_archive, csv_file, metrics_only=metrics_only)
            try:
                summary = run.run_teacher_episode(store, teacher_checkpoint)
            finally:
                store.stop()
            return summary


def collect_teacher_episodes(teacher_checkpoint: Path, metrics_only: bool) -> None:
    parent_seed_sequence = np.random.SeedSequence()

    episode_summaries_path = process.data_path / "episodes.csv"
    file_exists = os.path.isfile(episode_summaries_path)
    with open(episode_summaries_path, mode="a", newline="") as episode_summaries:
        episode_summaries_writer = csv.DictWriter(
            episode_summaries,
            fieldnames=data.EpisodeSummary.__dataclass_fields__.keys(),
        )
        if not file_exists:
            episode_summaries_writer.writeheader()

        for episode in range(process.num_episodes):
            [seed_sequence] = parent_seed_sequence.spawn(1)

            episode_id = uuid.uuid4()
            logger.info(
                f"Collecting episode {episode+1}/{process.num_episodes}: {episode_id}."
            )
            episode_dir = process.data_path / str(episode_id)
            episode_dir.mkdir(parents=True, exist_ok=False)

            # Run in process to circumvent Carla bug
            episode_summary = utils_process.process_wrapper(
                collect_teacher_episode,
                teacher_checkpoint,
                episode_dir,
                seed_sequence,
                metrics_only,
            )
            episode_summary.uuid = str(episode_id)
            episode_summaries_writer.writerow(episode_summary.as_csv_writeable_dict())

            # try:
            # except (exceptions.EpisodeStuck, exceptions.CollisionInEpisode) as exception:
            #     logger.info(f"Removing episode because of episode exception: {exception}.")
            #     (episode_dir / "data.zip").unlink()
            #     (episode_dir / "episode.csv").unlink()
            #     episode_dir.rmdir()


def collect_student_episode(
    student_checkpoint: Path,
    episode_dir: Path,
    seed_sequence: np.random.SeedSequence,
    metrics_only: bool,
    print_to_pdf: bool,
) -> data.EpisodeSummary:
    process.rng = np.random.default_rng(seed_sequence)

    with zipfile.ZipFile(episode_dir / "data.zip", mode="w") as zip_archive:
        with open(episode_dir / "episode.csv", mode="w", newline="") as csv_file:
            store = data.ZipStore(zip_archive, csv_file, metrics_only=metrics_only)
            try:
                summary = run.run_student_episode(
                    store, episode_dir, student_checkpoint, print_to_pdf
                )
            finally:
                store.stop()
            return summary


def collect_student_episodes(
    student_checkpoint: Path, metrics_only: bool, print_to_pdf: bool
) -> None:
    parent_seed_sequence = np.random.SeedSequence()

    episode_summaries_path = process.data_path / "episodes.csv"
    file_exists = os.path.isfile(episode_summaries_path)
    with open(episode_summaries_path, mode="a", newline="") as episode_summaries:
        episode_summaries_writer = csv.DictWriter(
            episode_summaries,
            fieldnames=data.EpisodeSummary.__dataclass_fields__.keys(),
        )
        if not file_exists:
            episode_summaries_writer.writeheader()

        for episode in range(process.num_episodes):
            [seed_sequence] = parent_seed_sequence.spawn(1)

            episode_id = uuid.uuid4()
            logger.info(
                f"Collecting episode {episode+1}/{process.num_episodes}: {episode_id}."
            )
            episode_dir = process.data_path / str(episode_id)
            episode_dir.mkdir(parents=True, exist_ok=False)

            # Run in process to circumvent Carla bug
            episode_summary = utils_process.process_wrapper(
                collect_student_episode,
                student_checkpoint,
                episode_dir,
                seed_sequence,
                metrics_only,
                print_to_pdf,
            )
            episode_summary.uuid = str(episode_id)
            episode_summaries_writer.writerow(episode_summary.as_csv_writeable_dict())


def collect_intervention_episode(
    student_checkpoint: Path,
    teacher_checkpoint: Path,
    episode_dir: Path,
    seed_sequence: np.random.SeedSequence,
    only_student_driving: bool,
    metrics_only: bool,
) -> data.EpisodeSummary:
    process.rng = np.random.default_rng(seed_sequence)

    with zipfile.ZipFile(episode_dir / "data.zip", mode="w") as zip_archive:
        with open(episode_dir / "episode.csv", mode="w", newline="") as csv_file:
            store = data.ZipStore(zip_archive, csv_file, metrics_only=metrics_only)
            try:
                summary = run.run_intervention_episode(
                    store, student_checkpoint, teacher_checkpoint, only_student_driving
                )
            finally:
                store.stop()
            return summary


def collect_intervention_episodes(
    student_checkpoint: Path,
    teacher_checkpoint: Path,
    only_student_driving: bool,
    metrics_only: bool,
) -> None:
    parent_seed_sequence = np.random.SeedSequence()

    episode_summaries_path = process.data_path / "episodes.csv"
    file_exists = os.path.isfile(episode_summaries_path)
    with open(episode_summaries_path, mode="a", newline="") as episode_summaries:
        episode_summaries_writer = csv.DictWriter(
            episode_summaries,
            fieldnames=data.EpisodeSummary.__dataclass_fields__.keys(),
        )
        if not file_exists:
            episode_summaries_writer.writeheader()

        for episode in range(process.num_episodes):
            [seed_sequence] = parent_seed_sequence.spawn(1)

            episode_id = uuid.uuid4()
            logger.info(
                f"Collecting episode {episode+1}/{process.num_episodes}: {episode_id}."
            )
            episode_dir = process.data_path / str(episode_id)
            episode_dir.mkdir(parents=True, exist_ok=False)

            # Run in process to circumvent Carla bug
            episode_summary = utils_process.process_wrapper(
                collect_intervention_episode,
                student_checkpoint,
                teacher_checkpoint,
                episode_dir,
                seed_sequence,
                only_student_driving,
                metrics_only,
            )
            episode_summary.uuid = str(episode_id)
            episode_summaries_writer.writerow(episode_summary.as_csv_writeable_dict())
