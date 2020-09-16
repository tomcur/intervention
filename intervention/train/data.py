import sys

from typing import Tuple, List, Dict

from pathlib import Path
from zipfile import ZipFile
from csv import DictReader
from loguru import logger
from dataclass_csv import DataclassReader

import numpy as np
import torch

from ..run import EpisodeSummary

if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict

LOCATIONS_NUM_STEPS: int = 5


class Location(TypedDict):
    x: float
    y: float
    z: float


class Orientation(TypedDict):
    orientation_x: float
    orientation_y: float
    orientation_z: float


class DatapointMeta(TypedDict):
    current_orientation: Orientation
    current_location: Location
    next_locations: List[Location]


class OffPolicyDataset(torch.utils.data.Dataset):
    def __init__(self, data_directory: Path, episodes: List[str]):
        self._index_map: List[Tuple[str, int]] = []
        self._episodes: Dict[str, List[DatapointMeta]] = {}
        self._zip_files: Dict[str, ZipFile] = {}

        for episode in episodes:
            self._episodes[episode] = []
            self._zip_files[episode] = ZipFile(
                data_directory / episode / "images.zip", mode="r"
            )

            with open(data_directory / episode / "episode.csv") as csv_file:
                csv_reader = DictReader(csv_file)
                rows = list(csv_reader)
                for (idx, row) in enumerate(rows[:-LOCATIONS_NUM_STEPS]):
                    self._index_map.append((episode, idx))
                    meta = DatapointMeta(
                        current_orientation=Orientation(
                            orientation_x=float(row["orientation_x"]),
                            orientation_y=float(row["orientation_y"]),
                            orientation_z=float(row["orientation_z"]),
                        ),
                        current_location=Location(
                            x=float(row["location_x"]),
                            y=float(row["location_y"]),
                            z=float(row["location_z"]),
                        ),
                        next_locations=list(
                            map(
                                lambda row: Location(
                                    x=float(row["location_x"]),
                                    y=float(row["location_y"]),
                                    z=float(row["location_z"]),
                                ),
                                rows[idx + 1 : idx + 1 + LOCATIONS_NUM_STEPS],
                            )
                        ),
                    )
                    self._episodes[episode].append(meta)

    def __len__(self):
        return len(self._index_map)

    def __getitem__(self, idx):
        episode, episode_idx = self._index_map[idx]
        return self._episodes[episode][episode_idx]

    def __del__(self):
        for zip_file in self._zip_files.values():
            zip_file.close()


def off_policy_data(data_directory) -> OffPolicyDataset:
    with open(data_directory / "episodes.csv") as episode_summaries_file:
        episode_summaries_reader = DataclassReader(
            episode_summaries_file, EpisodeSummary
        )
        episode_summaries = list(episode_summaries_reader)

    print(episode_summaries)
    episodes = [ep.uuid for ep in episode_summaries if ep.success]
    logger.info(
        f"Using {len(episodes)} successful episodes "
        f"out of {len(episode_summaries)} total episodes."
    )
    return OffPolicyDataset(data_directory, episodes)
