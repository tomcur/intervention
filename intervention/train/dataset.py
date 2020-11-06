import sys

from typing import Tuple, List, Dict, Mapping, Any, Sequence

from pathlib import Path
from zipfile import ZipFile
from csv import DictReader

from loguru import logger

from dataclass_csv import DataclassReader
import numpy as np
import torch
import torchvision

from ..data import EpisodeSummary, FrameData
from ..utils import image
from .. import coordinates

if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict

LOCATIONS_NUM_STEPS: int = 5
LOCATIONS_STEP_INTERVAL: int = 5


class Location(TypedDict):
    x: float
    y: float
    z: float


class Orientation(TypedDict):
    x: float
    y: float
    z: float


class Datapoint(TypedDict):
    rgb_filename: str
    model_output_filename: str
    command: int
    speed: float
    current_orientation: Orientation
    current_location: Location
    next_locations: List[Location]
    next_locations_image_coordinates: List[Any]


def datapoints_from_dictionaries(dictionaries: List[FrameData],) -> List[Datapoint]:
    datapoints = []
    for (idx, dictionary) in enumerate(
        dictionaries[: -LOCATIONS_NUM_STEPS * LOCATIONS_STEP_INTERVAL]
    ):

        current_orientation = Orientation(
            x=dictionary["orientation_x"],
            y=dictionary["orientation_y"],
            z=dictionary["orientation_z"],
        )

        current_location = Location(
            x=dictionary["location_x"],
            y=dictionary["location_y"],
            z=dictionary["location_z"],
        )

        next_locations = [
            Location(
                x=subsequent_dictionary["location_x"],
                y=subsequent_dictionary["location_y"],
                z=subsequent_dictionary["location_z"],
            )
            for subsequent_dictionary in dictionaries[
                (idx + LOCATIONS_STEP_INTERVAL) : (
                    idx + LOCATIONS_NUM_STEPS * LOCATIONS_STEP_INTERVAL + 1
                ) : LOCATIONS_STEP_INTERVAL
            ]
        ]
        assert len(next_locations) == LOCATIONS_NUM_STEPS

        next_locations_image_coordinates = [
            coordinates.ego_coordinate_to_image_coordinate(
                *coordinates.world_coordinate_to_ego_coordinate(
                    location_x=location["x"],
                    location_y=location["y"],
                    current_location_x=current_location["x"],
                    current_location_y=current_location["y"],
                    current_forward_x=current_orientation["x"],
                    current_forward_y=current_orientation["y"],
                )
            )
            for location in next_locations
        ]

        datapoint = Datapoint(
            rgb_filename=dictionary["rgb_filename"],
            student_output_filename=dictionary["student_output_filename"],
            command=dictionary["command"],
            speed=dictionary["speed"],
            current_orientation=current_orientation,
            current_location=current_location,
            next_locations=next_locations,
            next_locations_image_coordinates=np.array(next_locations_image_coordinates),
        )
        datapoints.append(datapoint)
    return datapoints


def _parse_frame_data(r: Dict[str, str]) -> FrameData:
    return FrameData(
        tick=int(r["tick"]),
        command=int(r["tick"]),
        controller=r["controller"],
        rgb_filename=r["rgb_filename"],
        student_output_filename=r["student_output_filename"]
        if r["student_output_filename"]
        else None,
        ticks_engaged=int(r["ticks_engaged"]) if r["ticks_engaged"] else None,
        ticks_to_intervention=int(r["ticks_to_intervention"])
        if r["ticks_to_intervention"]
        else None,
        ticks_intervened=int(r["ticks_intervened"]) if r["ticks_intervened"] else None,
        ticks_to_engagement=int(r["ticks_to_engagement"])
        if r["ticks_to_engagement"]
        else None,
        ticks_to_end=int(r["ticks_to_end"]) if r["ticks_to_end"] else None,
        lane_invasion=int(r["lane_invasion"]),
        collision=int(r["collision"]),
        location_x=float(r["location_x"]),
        location_y=float(r["location_y"]),
        location_z=float(r["location_z"]),
        velocity_x=float(r["velocity_x"]),
        velocity_y=float(r["velocity_y"]),
        velocity_z=float(r["velocity_z"]),
        speed=float(r["speed"]),
        orientation_x=float(r["orientation_x"]),
        orientation_y=float(r["orientation_y"]),
        orientation_z=float(r["orientation_z"]),
    )


class OffPolicyDataset(torch.utils.data.Dataset):
    def __init__(self, data_directory: Path, episodes: List[str]):
        self._index_map: List[Tuple[str, int]] = []
        self._episodes: Dict[str, List[Datapoint]] = {}
        self._zip_files: Dict[str, ZipFile] = {}

        self._transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        for episode in episodes:
            self._episodes[episode] = []
            self._zip_files[episode] = ZipFile(
                data_directory / episode / "data.zip", mode="r"
            )

            with open(data_directory / episode / "episode.csv") as csv_file:
                csv_reader = DictReader(csv_file)
                rows = [_parse_frame_data(r) for r in csv_reader]
                self._episodes[episode] = datapoints_from_dictionaries(rows)
                self._index_map.extend(
                    [(episode, idx) for idx in range(len(self._episodes[episode]))]
                )

    def __len__(self):
        return len(self._index_map)

    def __getitem__(self, idx):
        episode, episode_idx = self._index_map[idx]
        datapoint = self._episodes[episode][episode_idx]
        episode_img_bytes = self._zip_files[episode].read(datapoint["rgb_filename"])
        episode_img = image.buffer_to_np(episode_img_bytes)
        return self._transforms(episode_img), episode_img, datapoint

    def __del__(self):
        for zip_file in self._zip_files.values():
            zip_file.close()


def off_policy_data(data_directory) -> OffPolicyDataset:
    with open(data_directory / "episodes.csv") as episode_summaries_file:
        episode_summaries_reader = DataclassReader(
            episode_summaries_file, EpisodeSummary
        )
        episode_summaries: List[EpisodeSummary] = list(episode_summaries_reader)

    episodes = [ep.uuid for ep in episode_summaries if ep.end_status == "success"]
    logger.info(
        f"Using {len(episodes)} successful episodes "
        f"out of {len(episode_summaries)} total episodes."
    )
    return OffPolicyDataset(data_directory, episodes)


class OnPolicySupervisionDataset(torch.utils.data.Dataset):
    def __init__(self):
        self._transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, idx):
        raise NotImplementedError()

    def __del__(self):
        for zip_file in self._zip_files.values():
            zip_file.close()


# class InterventionSampler(torch.utils.data.


class _Dataset(torch.utils.data.Dataset):
    def __init__(self, datapoints: Sequence[Tuple[Datapoint, ZipFile]]):
        self._datapoints = datapoints

        self._transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self):
        return len(self._datapoints)

    def __getitem__(self, idx):
        (datapoint, zip_file) = self._datapoints[idx]

        img_bytes = zip_file.read(datapoint["rgb_filename"])
        img = image.buffer_to_np(img_bytes)
        return self._transforms(img), img, datapoint


class _DatasetBuilder:
    def __init__(self):
        self._datapoints: List[Tuple[Datapoint, ZipFile]] = []

    def add_datapoint(self, zip_file: ZipFile, datapoint: Datapoint):
        self._datapoints.append((datapoint, zip_file))

    def build(self) -> _Dataset:
        return _Dataset(self._datapoints)


class _ConcatenatedDataset(torch.utils.data.Dataset):
    def __init__(self, datasets: Sequence[torch.utils.data.Dataset]):
        self._datasets = datasets

        self._len = sum([len(dataset) for dataset in self._datasets])

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        offset = 0
        for dataset in self._datasets:
            if idx < offset + len(dataset):
                return dataset[idx - offset]
            offset += len(dataset)

        raise IndexError()


def intervention_data(
    data_directory,
) -> torch.utils.data.Dataset:  # InterventionDataset:
    with open(data_directory / "episodes.csv") as episode_summaries_file:
        episode_summaries_reader = DataclassReader(
            episode_summaries_file, EpisodeSummary
        )
        episode_summaries: List[EpisodeSummary] = list(episode_summaries_reader)

    logger.info(f"Using {len(episode_summaries)} episodes.")

    negative_dataset_builder = _DatasetBuilder()
    supervision_signal_dataset_builder = _DatasetBuilder()
    imitation_dataset_builder = _DatasetBuilder()

    for episode in episode_summaries:
        # Note we do not explicitly close this zipfile, but as we're opening read-only
        # that's fine. The underlying file will be closed when the reference count drops
        # to 0.
        zip_file = ZipFile(data_directory / episode.uuid / "data.zip", mode="r")

        with open(data_directory / episode.uuid / "episode.csv") as csv_file:
            csv_reader = DictReader(csv_file)
            frames = [_parse_frame_data(r) for r in csv_reader]
            datapoints = datapoints_from_dictionaries(frames)
            for (frame_data, datapoint) in zip(frames, datapoints):
                if frame_data["controller"] == "student":
                    ticks_to_intervention = frame_data["ticks_to_intervention"]
                    if (
                        ticks_to_intervention is None
                        and episode.end_status != "success"
                    ):
                        ticks_to_intervention = frame_data["ticks_to_end"]

                    if ticks_to_intervention is None or ticks_to_intervention > 25:
                        supervision_signal_dataset_builder.add_datapoint(
                            zip_file, datapoint
                        )
                    else:
                        negative_dataset_builder.add_datapoint(zip_file, datapoint)
                else:
                    assert frame_data["controller"] == "teacher"
                    if (
                        frame_data["ticks_to_end"] is None
                        or frame_data["ticks_to_end"] <= 50
                        or episode.end_status == "success"
                    ):
                        imitation_dataset_builder.add_datapoint(zip_file, datapoint)

    return _ConcatenatedDataset(
        [
            negative_dataset_builder.build(),
            supervision_signal_dataset_builder.build(),
            imitation_dataset_builder.build(),
        ]
    )
