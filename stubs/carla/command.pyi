from . import ActorBlueprint, Actor, Transform


class Command:
    ...


class DestroyActor(Command):
    def __init__(self, actor: Actor):
        ...


class SpawnActor(Command):
    def __init__(
        self,
        blueprint: ActorBlueprint,
        transform: Transform,
        parent: Optional[Actor] = None,
    ):
        ...
