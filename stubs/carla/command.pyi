from typing import Optional, Union

from . import Actor, ActorBlueprint, Transform


class Command:
    ...


class Response:
    actor_id: int
    error: str

    def has_error(self) -> bool:
        ...


class DestroyActor(Command):
    def __init__(self, actor: Actor):
        ...


class SpawnActor(Command):
    def __init__(
        self,
        blueprint: ActorBlueprint,
        transform: Transform,
        parent: Optional[Union[Actor, int]] = None,
    ):
        ...
