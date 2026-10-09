"""Binary vector codec for MongoDB storage.

Stored vectors use the BSON **binary vector** subtype 9 (``BinaryVector``, dtype
``FLOAT32``): a 2-byte metadata header (``[dtype][padding]``) followed by the
packed little-endian float32 values.

Why subtype 9 and not a plain ``binData`` blob: mongot silently ignores generic
binData (subtype 0) on the ``vector`` path — the search index reports ``READY``
but indexes **zero** documents, so ``$vectorSearch`` always returns nothing. It
does ingest subtype 9. The header costs 2 bytes, so switching from the previous
raw float32 blob is lossless and size-neutral (4098 B vs 4096 B for 1024-d).

Legacy subtype-0 blobs remain readable via :func:`unpack_vec`, so documents
written before the migration keep working.
"""

from __future__ import annotations

import base64
from typing import Any

import numpy as np
from bson.binary import Binary, BinaryVectorDtype

_DTYPE = np.float32

VECTOR_SUBTYPE = 9
# BinaryVectorDtype values are single bytes; pull the numeric tag out.
_DTYPE_F32 = BinaryVectorDtype.FLOAT32.value[0]  # 0x27
_DTYPE_I8 = BinaryVectorDtype.INT8.value[0]  # 0x03


def pack_vec(v: np.ndarray | list[float] | None) -> Binary | None:
    """Encode a vector as a BSON binary vector (subtype 9, dtype FLOAT32).

    ``None`` stays ``None`` (placeholder vectors on word nodes before s05).
    """
    if v is None:
        return None
    return Binary.from_vector(np.asarray(v, dtype=_DTYPE), dtype=BinaryVectorDtype.FLOAT32)


def _decode_subtype9(blob: bytes) -> np.ndarray:
    """Decode a subtype-9 binary vector payload (2-byte metadata + data)."""
    tag = blob[0]
    data = blob[2:]
    if tag == _DTYPE_F32:
        return np.frombuffer(data, dtype=np.float32)
    if tag == _DTYPE_I8:
        # Pre-quantized int8: cast back to float. Direction is preserved; the
        # per-vector scale is not stored, so magnitudes are approximate.
        return np.frombuffer(data, dtype=np.int8).astype(np.float32)
    raise ValueError(f"unsupported BSON vector dtype 0x{tag:02x}")


def unpack_vec(blob: Any, dim: int | None = None) -> np.ndarray:
    """Decode a stored vector back into a 1-D float32 array.

    Accepts subtype-9 binary vectors (header stripped), legacy subtype-0 raw
    float32 blobs, and plain array-likes (list/ndarray).
    """
    if isinstance(blob, np.ndarray):
        arr = np.asarray(blob, dtype=_DTYPE)
    elif isinstance(blob, (list, tuple)):
        arr = np.asarray(blob, dtype=_DTYPE)
    elif isinstance(blob, Binary) and blob.subtype == VECTOR_SUBTYPE:
        arr = _decode_subtype9(bytes(blob))
    else:
        # bytes / Binary subtype 0 / memoryview — legacy raw float32.
        arr = np.frombuffer(blob, dtype=_DTYPE)
    if dim is not None:
        arr = arr.reshape(dim)
    return arr


def encode_vec(v: np.ndarray | list[float]) -> str:
    """JSON-safe vector: base64 of the **legacy** raw float32 blob.

    Deliberately not subtype 9: base64 loses the BSON subtype, so a subtype-9
    payload decoded back to ``bytes`` is indistinguishable from legacy data.
    The JSONL intermediates (``senses_embedded.jsonl``) and their consumers
    (s03) use the flat float32 layout, which keeps existing files readable and
    keeps ``v.tolist()``'s ~15 KB/row off disk (~5.5 KB here).
    """
    return base64.b64encode(np.asarray(v, dtype=_DTYPE).tobytes()).decode("ascii")


def decode_vec(s: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(s), dtype=_DTYPE)
