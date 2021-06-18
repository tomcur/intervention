import csv
from pathlib import Path
from typing import BinaryIO, List, SupportsWrite, Tuple

import pandas as pd
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


def intervention_metrics(out: BinaryIO, data_directory: Path) -> None:
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


def summarize(out: SupportsWrite[str], data_directory: Path) -> None:
    episodes = pd.read_csv(data_directory / "episodes.csv")

    episodes[["time"]] = episodes[["ticks"]] / 10.0

    print("dataset\t\t\t\t\t\t", data_directory.name, file=out)
    print("total episodes\t\t\t\t\t", len(episodes), file=out)
    print(
        "total distance travelled (km)\t\t\t",
        episodes["distance_travelled"].sum() / 1000.0,
        file=out,
    )
    print(
        "total teacher distance travelled (km)\t\t",
        episodes["teacher_distance_travelled"].sum() / 1000.0,
        file=out,
    )
    print(
        "total student distance travelled (km)\t\t",
        episodes["student_distance_travelled"].sum() / 1000.0,
        file=out,
    )
    print(
        "total simulated time (h)\t\t\t",
        episodes["time"].sum() / (60.0 * 60.0),
        file=out,
    )
    print(
        "success rate\t\t\t\t\t",
        (episodes["end_status"] == "success").mean(),
        file=out,
    )
    print(
        "interventions per episode\t\t\t",
        episodes["interventions"].mean(),
        file=out,
    )
    print(
        "interventions per student distance travelled\t",
        (episodes["interventions"] / episodes["student_distance_travelled"]).mean(),
        file=out,
    )

    print(file=out)
    print("end status distribution", file=out)
    print("==============", file=out)
    print(episodes[["end_status"]].value_counts(normalize=True).to_string())

    print(file=out)
    print("end status by town and weather", file=out)
    print("==============", file=out)
    print(
        episodes.groupby(["town", "weather", "end_status"])
        .size()
        .unstack(fill_value=0)
        .to_string(),
        file=out,
    )

    print(file=out)
    print("weather and town distribution", file=out)
    print("==============", file=out)
    print(episodes.groupby(["town", "weather"]).size().unstack(fill_value=0), file=out)

    print(file=out)
    print("distance travelled per weather and town (km)", file=out)
    print("==============", file=out)
    print(
        (
            episodes.groupby(["town", "weather"])[["distance_travelled"]].sum() / 1000.0
        ).to_string(),
        file=out,
    )

    print(file=out)
    print("simulated time per weather and town (h)", file=out)
    print("==============", file=out)
    print(
        (
            episodes.groupby(["town", "weather"])[["time"]].sum() / (60.0 * 60.0)
        ).to_string(),
        file=out,
    )
