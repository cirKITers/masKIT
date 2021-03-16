from typing import List, Dict, TypeVar, Union, Any, Callable
import time
import json
import pathlib
from functools import wraps

# JSON compatibility type
T = TypeVar("T")
_JSON_Element = Union[str, int, float, bool, None, Dict[str, T], List[T]]
JSON = _JSON_Element[_JSON_Element[_JSON_Element[Any]]]
CJ = TypeVar("CJ", bound=Callable[..., JSON])


log_path = pathlib.Path(__file__).parent / "logs" / f"{time.time()}.json"
log_path.parent.mkdir(exist_ok=True)


def serialize(o):
    return json.dumps(repr(o))
    # raise TypeError(f"Cannot serialize {o} to JSON")


def log_results(executor: CJ) -> CJ:
    @wraps(executor)
    def wrapper(*args, **kwargs):
        wallclock, cpuclock = time.perf_counter, time.process_time
        start_time = wallclock(), cpuclock()
        result = executor(*args, **kwargs)
        end_time = wallclock(), cpuclock()
        with open(log_path, "a") as out_file:
            json.dump(
                {
                    "event": "log_results",
                    "call": [
                        f"{executor.__module__}.{executor.__qualname__}",
                        args,
                        kwargs,
                    ],
                    "result": result
                    if type(result) is not dict
                    else {
                        key: value
                        for key, value in result.items()
                        if type(key) is str and not key.startswith("__")
                    },
                    "walltime": end_time[0] - start_time[0],
                    "cputime": end_time[1] - start_time[1],
                },
                out_file,
                default=serialize,
            )
        return result

    return wrapper
