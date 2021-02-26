from typing import List, Dict, TypeVar, Union, Any, Callable
import time
import json
import pathlib
from functools import wraps
# TODO: Es sollte möglich sein, dass aus der Funktion heraus alle x Schritte Daten geloggt werden
# TODO: Random Seed sollte geloggt werden

# JSON compatibility type
T = TypeVar('T')
_JSON_Element = Union[str, int, float, bool, None, Dict[str, T], List[T]]
JSON = _JSON_Element[_JSON_Element[_JSON_Element[Any]]]
CJ = TypeVar('CJ', bound=Callable[..., JSON])

# Analysen, die wir machen wollen (immer mit Ensembles/ohne Ensembles)
# * Erzeugung von unterschiedlichen circuits via setzen des Random Seeds
# * Parameterstudie unterschiedliche # wires, unterschiedliche # layers
# * Wenn möglich Parameterstudie dazu, wie lange Kosten beobachtet werden sollen, bevor das Ensemble einsetzt


log_path = pathlib.Path(__file__).parent / "logs" / f"{time.time()}.json"
log_path.parent.mkdir(exist_ok=True)


def serialize(o):
    return repr(o)
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
                        kwargs
                    ],
                    "result": result,
                    "walltime": end_time[0] - start_time[0],
                    "cputime": end_time[1] - start_time[1],
                },
                out_file,
                default=serialize,
            )
        return result
    return wrapper
