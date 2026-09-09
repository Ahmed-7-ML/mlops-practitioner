# Timed Decorator

import time
import functools
from typing import Callable, Any
from prodml.logging_conf import setup_logging

logger = setup_logging("prodml.utils")


def timed(func: Callable) -> Callable:
    """
    Decorator to measure the execution time of a function.

    Args:
        func (Callable): The function to be decorated.

    Returns:
        Callable: The wrapped function with timing functionality.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = round(end_time - start_time, 4)
        logger.info(
            f"⏱️ Function '{func.__name__}' executed",
            extra={
                "extra": {
                    "function_name": func.__name__,
                    "duration_sec": round(elapsed_time, 4),
                }
            },
        )
        return result

    return wrapper
