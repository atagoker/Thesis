"""Read-only progress observation of the frozen training function.

Python tracing observes completed optimizer steps and appended epoch rows. It
does not modify the frozen implementation, model, objective or random generators.
"""

from contextlib import contextmanager
import sys


class TrainingCancelled(Exception):
    pass


@contextmanager
def observe_training(function, emit, cancelled=lambda: False):
    previous = sys.gettrace()
    state = {"epoch": 0, "seen": 0, "rows": 0, "totals": None}

    def local_trace(frame, event, arg):
        if event != "line":
            return local_trace
        if cancelled():
            raise TrainingCancelled("Training stopped by the user.")
        values = frame.f_locals
        epoch = values.get("epoch", 0)
        totals = values.get("totals", {})
        seen = totals.get("source_seen", 0)
        if epoch != state["epoch"]:
            state["epoch"] = epoch
        # The loop variable advances before the previous totals dictionary is
        # replaced. Reset only when the new epoch's actual accumulator exists.
        if totals is not state["totals"]:
            state["totals"], state["seen"] = totals, 0
        if seen > state["seen"]:
            state["seen"] = seen
            batch_size = values["cfg"]["batch_size"]
            step = (seen + batch_size - 1) // batch_size
            steps = len(values["loader"])
            if step == 1 or step % 5 == 0 or step == steps:
                emit("batch", epoch=epoch, total_epochs=values["cfg"]["adaptation_epochs"],
                     step=step, steps=steps, source_loss=float(values["source_loss"].detach()),
                     target_loss=float(values["target_loss"].detach()))
        history = values.get("history", [])
        if len(history) > state["rows"]:
            state["rows"] = len(history)
            emit("epoch", **history[-1])
        return local_trace

    def global_trace(frame, event, arg):
        if event == "call" and frame.f_code is function.__code__:
            return local_trace
        return None

    sys.settrace(global_trace)
    try:
        yield
    finally:
        sys.settrace(previous)
