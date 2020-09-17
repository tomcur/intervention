import sys

from typing import Tuple, List, Dict, Any

from pathlib import Path
from zipfile import ZipFile
from csv import DictReader

from loguru import logger

from dataclass_csv import DataclassReader
import numpy as np
import torch

from ..data import EpisodeSummary
from .. import coordinates

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
    x: float
    y: float
    z: float


class DatapointMeta(TypedDict):
    rgb_filename: str
    command: int
    speed: float
    current_orientation: Orientation
    current_location: Location
    next_locations: List[Location]
    next_locations_image_coordinates: List[Any]


def datapoint_meta_from_dictionaries(dictionaries: List[Any]) -> List[DatapointMeta]:
    datapoints = []
    for (idx, dictionary) in enumerate(dictionaries[:-LOCATIONS_NUM_STEPS]):

        current_orientation = Orientation(
            x=float(dictionary["orientation_x"]),
            y=float(dictionary["orientation_y"]),
            z=float(dictionary["orientation_z"]),
        )

        current_location = Location(
            x=float(dictionary["location_x"]),
            y=float(dictionary["location_y"]),
            z=float(dictionary["location_z"]),
        )

        next_locations = [
            Location(
                x=float(subsequent_dictionary["location_x"]),
                y=float(subsequent_dictionary["location_y"]),
                z=float(subsequent_dictionary["location_z"]),
            )
            for subsequent_dictionary in dictionaries[
                idx + 1 : idx + 1 + LOCATIONS_NUM_STEPS
            ]
        ]

        next_locations_image_coordinates = [
            coordinates.world_coordinate_to_image_coordinate(
                location_x=location["x"],
                location_y=location["y"],
                current_location_x=current_location["x"],
                current_location_y=current_location["y"],
                current_forward_x=current_orientation["x"],
                current_forward_y=current_orientation["y"],
            )
            for location in next_locations
        ]

        meta = DatapointMeta(
            rgb_filename=dictionary["rgb_filename"],
            command=int(dictionary["command"])
            speed=float(dictionary["speed"]),
            current_orientation=current_orientation,
            current_location=current_location,
            next_locations=next_locations,
            next_locations_image_coordinates=np.array(next_locations_image_coordinates),
        )
        datapoints.append(meta)
    return datapoints


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
                self._episodes[episode] = datapoint_meta_from_dictionaries(rows)
                self._index_map.extend(
                    [(episode, idx) for idx in range(len(rows) - LOCATIONS_NUM_STEPS)]
                )

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

    episodes = [ep.uuid for ep in episode_summaries if ep.success]
    logger.info(
        f"Using {len(episodes)} successful episodes "
        f"out of {len(episode_summaries)} total episodes."
    )
    return OffPolicyDataset(data_directory, episodes)
