import multiprocessing
import traceback
from typing import Callable, Tuple, TypeVar, Union

from loguru import logger

T = TypeVar("T")


def process_wrapper(target: Callable[..., T], *args, **kwargs) -> T:
    """
    Runs `target` (with `*args` and `**kwargs`) in a subprocess. Blocks until `target`
    is done. Returns the return value of `target`.
    """
    queue: multiprocessing.SimpleQueue[
        Tuple[bool, Union[T, Exception]]
    ] = multiprocessing.SimpleQueue()

    def _wrapper(
        target: Callable[..., T], queue, *args, **kwargs,
    ):
        """
        Runs `target` with `*args` and `**kwargs`. Puts a 2-tuple in `queue`. The first
        member of the tuple is a boolean indicating whether`target` succeeded or raised
        an exception. The second member is the return value of `target` or the raised
        exception.
        """
        try:
            value = target(*args, **kwargs)
            queue.put((True, value))
        except Exception:
            queue.put((False, traceback.format_exc()))

    process = multiprocessing.Process(
        target=_wrapper, args=(target, queue) + args, kwargs=kwargs
    )
    process.start()
    success, value = queue.get()
    if success:
        return value  # type: ignore
    else:
        logger.error(f"Child process raised the following exception: {value}")
        raise Exception("Child process raised an exception")
