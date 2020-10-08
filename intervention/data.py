import dataclasses
from datetime import datetime, timezone

import dataclass_csv


@dataclasses.dataclass
@dataclass_csv.dateformat("%Y-%m-%dT%H:%M:%S.%f%z")
class EpisodeSummary:
    uuid: str = ""
    collection_start_datetime: datetime = dataclasses.field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    collection_end_datetime: datetime = dataclasses.field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    town: str = ""
    terminated: bool = False
    success: bool = False
    collisions: int = 0
    distance_travelled: float = 0.0
    interventions: int = 0
    ticks: int = 0

    def set_end_datetime(self):
        self.collection_end_datetime = datetime.now(timezone.utc)

    def as_csv_writeable_dict(self):
        values = self.__dict__
        for (key, value) in values.items():
            if isinstance(value, bool):
                values[key] = int(value)
            elif isinstance(value, datetime):
                values[key] = value.isoformat()
        return values
