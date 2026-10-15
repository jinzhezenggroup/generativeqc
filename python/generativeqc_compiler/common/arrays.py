"""Owned immutable finite FP64 arrays shared by method and grid consumers."""

import typing

import numpy as np

from .array_execution import current_execution


def immutable(value: typing.Any, *, shape: typing.Any = None) -> typing.Any:
    """Detach finite FP64 values; host publication is irreversibly read-only.

    Explicit device consumers own their detached private storage through the
    scoped executor. They must publish host results before promising the public
    immutable-array contract; writable device panels never escape that owner.
    """
    execution = current_execution()
    if execution is not None:
        return execution.freeze(value, shape=shape)
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError("complex references are unsupported")
    array = np.asarray(raw, dtype=np.float64, order="C")
    if shape is not None and array.shape != shape:
        raise ValueError(f"expected shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("reference arrays must be finite")
    return np.frombuffer(array.tobytes(), dtype=np.float64).reshape(array.shape)
