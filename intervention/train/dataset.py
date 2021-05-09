import sys
from csv import DictReader
from dataclasses import dataclass
import enum
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple
from zipfile import ZipFile

import numpy as np
import torch
import torchvision
from dataclass_csv import DataclassReader
from loguru import logger

from .. import coordinates
from ..data import EpisodeSummary, FrameData
from ..utils import image

if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict

LOCATIONS_NUM_STEPS: int = 5
LOCATIONS_STEP_INTERVAL: int = 5


class DataType(enum.Enum):
    NEGATIVE = enum.auto()
    SUPERVISION_SIGNAL = enum.auto()
    IMITATION = enum.auto()


class Location(TypedDict):
    x: float
    y: float
    z: float


class Orientation(TypedDict):
    x: float
    y: float
    z: float


class Datapoint(TypedDict):
    tick: int
    controller: str
    rgb_filename: str
    teacher_waypoints_filename: str
    student_waypoints_filename: str
    student_image_targets_filename: str
    student_image_heatmaps_filename: str
    command: int
    speed: float
    current_orientation: Orientation
    current_location: Location
    next_locations: List[Location]
    next_locations_image_coordinates: np.ndarray
    ticks_to_intervention: int


def datapoints_from_dictionaries(dictionaries: List[FrameData]) -> List[Datapoint]:
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
            tick=dictionary["tick"],
            controller=dictionary["controller"],
            rgb_filename=dictionary["rgb_filename"],
            teacher_waypoints_filename=dictionary["teacher_waypoints_filename"] or "",
            student_waypoints_filename=dictionary["student_waypoints_filename"] or "",
            student_image_targets_filename=dictionary["student_image_targets_filename"]
            or "",
            student_image_heatmaps_filename=dictionary[
                "student_image_heatmaps_filename"
            ]
            or "",
            command=dictionary["command"],
            speed=dictionary["speed"],
            current_orientation=current_orientation,
            current_location=current_location,
            next_locations=next_locations,
            next_locations_image_coordinates=np.array(next_locations_image_coordinates),
            ticks_to_intervention=dictionary["ticks_to_intervention"]
            if dictionary["ticks_to_intervention"] is not None
            else -1,
        )
        datapoints.append(datapoint)
    return datapoints


def _parse_frame_data(r: Dict[str, str]) -> FrameData:
    return FrameData(
        tick=int(r["tick"]),
        command=int(r["command"]),
        controller=r["controller"],
        rgb_filename=r["rgb_filename"],
        teacher_waypoints_filename=r["teacher_waypoints_filename"]
        if "teacher_waypoints_filename" in r and r["teacher_waypoints_filename"]
        else None,
        student_waypoints_filename=r["student_waypoints_filename"]
        if "student_waypoints_filename" in r and r["student_waypoints_filename"]
        else None,
        student_image_targets_filename=r["student_image_targets_filename"]
        if "student_image_targets_filename" in r and r["student_image_targets_filename"]
        else None,
        student_image_heatmaps_filename=r["student_image_heatmaps_filename"]
        if "student_image_heatmaps_filename" in r
        and r["student_image_heatmaps_filename"]
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


class _Dataset(torch.utils.data.Dataset):
    """
    This data loader holds handles to open zip files. It loads images and
    other binary data from the zip-file as needed.
    """

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

    def __getitem__(self, idx) -> Any:
        """
        Images (both transformed and non-transformed) are of dimensionality
        `[N, C, H, W]`.
        """
        (datapoint, zip_file) = self._datapoints[idx]

        rgb_filename = Path(datapoint["rgb_filename"])
        if rgb_filename.suffix == ".png":
            with zip_file.open(str(rgb_filename), mode="r") as file:
                img = image.image_file_to_np(file)
        else:
            # Read the image as raw bytes
            img_bytes = zip_file.read(datapoint["rgb_filename"])
            img = image.buffer_to_np(img_bytes)

        with zip_file.open(datapoint["teacher_waypoints_filename"]) as f:
            teacher_waypoints = np.load(f)

        if datapoint["student_image_targets_filename"] != "":
            assert datapoint["student_image_heatmaps_filename"] != ""

            with zip_file.open(datapoint["student_image_targets_filename"]) as f:
                student_image_targets = np.load(f)

            with zip_file.open(datapoint["student_image_heatmaps_filename"]) as f:
                student_image_heatmaps = np.load(f)

            return {
                "rgb_image": self._transforms(img),
                "untransformed_rgb_image": np.moveaxis(img, [2], [0]),
                "teacher_waypoint": teacher_waypoints,
                "datapoint": datapoint,
                "student_image_targets": student_image_targets,
                "student_image_heatmaps": student_image_heatmaps,
            }
        else:
            return {
                "rgb_image": self._transforms(img),
                "untransformed_rgb_image": np.moveaxis(img, [2], [0]),
                "teacher_waypoint": teacher_waypoints,
                "datapoint": datapoint,
            }


class _DatasetBuilder:
    """
    A dataset builder. It takes handles to open zip files and datapoints.
    It can be consumed to create a dataset.
    """

    def __init__(self):
        self._datapoints: List[Tuple[Datapoint, ZipFile]] = []

    def add_datapoint(self, zip_file: ZipFile, datapoint: Datapoint) -> int:
        """
        Add a datapoint to the set.

        :return: the index the datapoint is assigned.
        """
        idx = len(self._datapoints)
        self._datapoints.append((datapoint, zip_file))
        return idx

    def build(self) -> _Dataset:
        """
        Consumes the dataset builder, returning a dataset.
        """
        d = _Dataset(self._datapoints)
        self._datapoints = []
        return d


class _CombinedDataset(torch.utils.data.Dataset):
    """
    Combines negative, supervision and imitation datasets into one.
    """

    def __init__(
        self,
        datapoints: Sequence[Tuple[DataType, int]],
        negative: torch.utils.data.Dataset,
        supervision_signal: torch.utils.data.Dataset,
        imitation: torch.utils.data.Dataset,
    ):
        self._datapoints = datapoints
        self._negative = negative
        self._supervision_signal = supervision_signal
        self._imitation = imitation

    def __len__(self):
        return len(self._datapoints)

    def __getitem__(self, idx) -> Any:
        (data_type, idx_in_dataset) = self._datapoints[idx]
        if data_type is DataType.NEGATIVE:
            return (data_type, self._negative[idx_in_dataset])
        elif data_type is DataType.SUPERVISION_SIGNAL:
            return (data_type, self._supervision_signal[idx_in_dataset])
        elif data_type is DataType.IMITATION:
            return (data_type, self._imitation[idx_in_dataset])


class _CombinedDatasetBuilder:
    """
    A combined dataset builder. It takes the indices of datapoints in intervention datasets.
    It can be consumed, given references to intervention datasets, to create a combined dataset.
    """

    def __init__(self):
        self._datapoints: List[Tuple[DataType, int]] = []

    def add_datapoint(self, data_type: DataType, index_in_dataset: int) -> int:
        """
        Add a datapoint to the set.

        :return: the index the datapoint is assigned.
        """
        idx = len(self._datapoints)
        self._datapoints.append((data_type, index_in_dataset))
        return idx

    def build(
        self,
        negative: torch.utils.data.Dataset,
        supervision_signal: torch.utils.data.Dataset,
        imitation: torch.utils.data.Dataset,
    ) -> _Dataset:
        """
        Consumes the dataset builder, returning a dataset.
        """
        d = _CombinedDataset(
            self._datapoints,
            negative=negative,
            supervision_signal=supervision_signal,
            imitation=imitation,
        )
        self._datapoints = []
        return d


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


@dataclass
class InterventionDatasets:
    """
    An (on-policy) intervention dataset. This consists of three separate datasets:
        - `negative` consists of examples of student driving leading up to an
          intervention;
        - `supervision_signal` consists of examples of student driving that take
          place more than 2.5 seconds before an intervention;
        - `imitation` consists of examples of teacher driving; and
        - `combined` all examples in original order.
    """

    negative: torch.utils.data.Dataset
    supervision_signal: torch.utils.data.Dataset
    imitation: torch.utils.data.Dataset
    combined: torch.utils.data.Dataset


def intervention_data(data_directory) -> InterventionDatasets:
    """
    Load an (on-policy) intervention dataset. This consists of three separate datasets.
    """
    with open(data_directory / "episodes.csv") as episode_summaries_file:
        episode_summaries_reader = DataclassReader(
            episode_summaries_file, EpisodeSummary
        )
        episode_summaries: List[EpisodeSummary] = list(episode_summaries_reader)

    logger.info(f"Using {len(episode_summaries)} episodes.")

    negative_dataset_builder = _DatasetBuilder()
    supervision_signal_dataset_builder = _DatasetBuilder()
    imitation_dataset_builder = _DatasetBuilder()
    combined_dataset_builder = _CombinedDatasetBuilder()

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
                        idx = supervision_signal_dataset_builder.add_datapoint(
                            zip_file, datapoint
                        )
                        combined_dataset_builder.add_datapoint(
                            DataType.SUPERVISION_SIGNAL, idx
                        )
                    else:
                        idx = negative_dataset_builder.add_datapoint(
                            zip_file, datapoint
                        )
                        combined_dataset_builder.add_datapoint(DataType.NEGATIVE, idx)
                else:
                    assert frame_data["controller"] == "teacher"
                    if (
                        frame_data["ticks_to_end"] is None
                        or frame_data["ticks_to_end"] <= 50
                        or episode.end_status == "success"
                    ):
                        idx = imitation_dataset_builder.add_datapoint(
                            zip_file, datapoint
                        )
                        combined_dataset_builder.add_datapoint(DataType.IMITATION, idx)

    negative = negative_dataset_builder.build()
    supervision_signal = supervision_signal_dataset_builder.build()
    imitation = imitation_dataset_builder.build()

    return InterventionDatasets(
        negative=negative,
        supervision_signal=supervision_signal,
        imitation=imitation,
        combined=combined_dataset_builder.build(
            negative=negative,
            supervision_signal=supervision_signal,
            imitation=imitation,
        ),
    )


def off_policy_data(data_directory) -> torch.utils.data.Dataset:
    """
    Load an off-policy dataset.
    """
    # Off-policy data can be seen as a special case of intervention data with only
    # imitation frames. Load it as intervention data, and return only the imitation
    # dataset.
    datasets = intervention_data(data_directory)
    assert (
        len(datasets.negative) == 0
    ), "Off-policy data should not have student driving"
    assert (
        len(datasets.supervision_signal) == 0
    ), "Off-policy data should not have student driving"
    return datasets.imitation
