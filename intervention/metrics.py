import csv
from pathlib import Path
from typing import List, TextIO, Tuple

import pandas as pd
from dataclass_csv import DataclassReader

from .data import EpisodeSummary, FrameData
from .train.dataset import _parse_frame_data


def intervention_metrics(out: TextIO, data_directory: Path) -> None:
    episodes = pd.read_csv(data_directory / "episodes.csv")

    def _sample_statistics(r):
        episode = pd.read_csv(data_directory / r["uuid"] / "episode.csv")
        return pd.Series(
            {
                "uuid": r["uuid"],
                "town": r["town"],
                "weather": r["weather"],
                "student_ticks": (episode["controller"] == "student").sum(),
                "teacher_ticks": (episode["controller"] == "teacher").sum(),
                "total_ticks": len(episode.index),
                "interventions": (episode["ticks_to_intervention"] == 0).sum(),
            }
        )

    sample_statistics = episodes.apply(_sample_statistics, axis="columns")

    print("dataset\t\t\t\t\t\t", data_directory.name, file=out)
    print("total episodes\t\t\t\t\t", len(episodes), file=out)

    print(file=out)
    print("data distribution by town and weather", file=out)
    print("==============", file=out)
    print(
        sample_statistics.groupby(["town", "weather"])[
            ["student_ticks", "teacher_ticks", "total_ticks", "interventions"]
        ].sum(),
        file=out,
    )


def summarize(out: TextIO, data_directory: Path) -> None:
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
