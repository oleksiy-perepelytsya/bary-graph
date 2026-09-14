"""Compact float32 binary vector encode/decode for MongoDB storage.

Vectors are stored as ``binData`` (``bytes`` of raw float32, little-endian)
instead of Python ``list[float]``. BSON has no float32 type, so a ``.tolist()``
list of float32 values round-trips through Mongo as 64-bit doubles — 2× the
bytes. Packing as float32 binary halves every stored vector (4096-d -> 16 KB)
with no precision loss, and Mongo ``$vectorSearch`` accepts float32 ``binData``
for the ``vector`` path, so the existing vector index is unaffected.

Every pipeline reader that currently does ``np.asarray(doc["vector"], float32)``
must instead call :func:`unpack_vec`, since ``np.asarray`` cannot decode a bytes
blob into a 1-D vector.
"""

from __future__ import annotations

from typing import Any

import numpy as np

_DTYPE = np.float32


def pack_vec(v: np.ndarray | list[float] | None) -> bytes | None:
    """Pack a vector (array or list) into a float32 ``binData`` byte blob.

    ``None`` stays ``None`` (placeholder vectors on word nodes before s05).
    """
    if v is None:
        return None
    return np.asarray(v, dtype=_DTYPE).tobytes()


def unpack_vec(blob: Any, dim: int | None = None) -> np.ndarray:
    """Decode a float32 ``binData`` blob back into a 1-D float32 array.

    ``dim`` defaults to the blob length (all vectors in this build are 4096-d).
    """
    if isinstance(blob, np.ndarray):
        return np.asarray(blob, dtype=_DTYPE)
    if isinstance(blob, (list, tuple)):
        return np.asarray(blob, dtype=_DTYPE)
    arr = np.frombuffer(blob, dtype=_DTYPE)
    if dim is not None:
        arr = arr.reshape(dim)
    return arr
