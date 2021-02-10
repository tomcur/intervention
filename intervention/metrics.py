import csv
from pathlib import Path
from typing import BinaryIO, Tuple

from dataclass_csv import DataclassReader

from .data import EpisodeSummary, FrameData
from .train.dataset import _parse_frame_data


def episode_metrics(episode_directory: Path) -> Tuple[float, float, int]:
    with open(episode_directory / "episode.csv") as csv_file:
        csv_reader = csv.DictReader(csv_file)
        frames = [_parse_frame_data(r) for r in csv_reader]

    total_distance = 0.0
    engaged_distance = 0.0
    interventions = 0
    controller = "teacher"
    for frame in frames:
        total_distance += frame["speed"] * 0.1

        if controller == "student":
            engaged_distance += frame["speed"] * 0.1
            if frame["controller"] == "teacher":
                interventions += 1

        controller = frame["controller"]

    return total_distance, engaged_distance, interventions


def on_policy_metrics(out: BinaryIO, data_directory: Path) -> None:
    with open(data_directory / "episodes.csv") as episode_summaries_file:
        episode_summaries_reader = DataclassReader(
            episode_summaries_file, EpisodeSummary
        )
        episode_summaries: List[EpisodeSummary] = list(episode_summaries_reader)

    total_distance = 0.0
    engaged_distance = 0.0
    interventions = 0
    for summary in episode_summaries:
        total_distance_, engaged_distance_, interventions_ = episode_metrics(
            data_directory / summary.uuid
        )
        total_distance += total_distance_
        engaged_distance += engaged_distance_
        interventions += interventions_

    writer = csv.DictWriter(out, fieldnames=["metric", "value"])
    writer.writeheader()

    # total_distance_meters = sum(
    #     map(lambda summary: summary.distance_travelled, episode_summaries)
    # )
    # total_interventions = sum(
    #     map(lambda summary: summary.interventions, episode_summaries)
    # )

    writer.writerow(
        {
            "metric": "total_distance",
            "value": total_distance,
        }
    )
    writer.writerow(
        {
            "metric": "engaged_distance",
            "value": engaged_distance,
        }
    )
    writer.writerow(
        {
            "metric": "interventions",
            "value": interventions,
        }
    )
    writer.writerow(
        {
            "metric": "total_distance_per_intervention",
            "value": total_distance / interventions,
        }
    )
    writer.writerow(
        {
            "metric": "engaged_distance_per_intervention",
            "value": engaged_distance / interventions,
        }
    )

    # print("test")
    # print(episode_summaries)
    # for episode in episode_summaries:

    pass
