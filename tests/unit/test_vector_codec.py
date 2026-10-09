from __future__ import annotations

import base64

import numpy as np
from bson.binary import Binary, BinaryVectorDtype

from lib.vector import (
    VECTOR_SUBTYPE,
    decode_vec,
    encode_vec,
    pack_vec,
    unpack_vec,
)


def _v(n=8, seed=0):
    return np.random.default_rng(seed).standard_normal(n).astype(np.float32)


def test_pack_vec_emits_subtype9_float32():
    v = _v()
    blob = pack_vec(v)
    assert isinstance(blob, Binary)
    assert blob.subtype == VECTOR_SUBTYPE
    assert blob[0] == BinaryVectorDtype.FLOAT32.value[0]  # dtype tag
    assert blob[1] == 0  # padding
    assert len(blob) == 2 + v.size * 4


def test_pack_vec_none_stays_none():
    assert pack_vec(None) is None


def test_round_trip_subtype9():
    v = _v(seed=1)
    np.testing.assert_array_equal(unpack_vec(pack_vec(v)), v)


def test_unpack_legacy_raw_float32_bytes():
    # Pre-migration subtype-0 blob: raw float32, no header.
    v = _v(seed=2)
    legacy = v.tobytes()
    np.testing.assert_array_equal(unpack_vec(legacy), v)


def test_unpack_accepts_array_likes():
    v = _v(seed=3)
    np.testing.assert_allclose(unpack_vec(v.tolist()), v)
    np.testing.assert_allclose(unpack_vec(v), v)


def test_unpack_reshapes_to_dim():
    v = _v(8, seed=4)
    assert unpack_vec(pack_vec(v), dim=8).shape == (8,)


def test_unpack_decodes_int8_subtype9():
    codes = np.array([1, -2, 3, -4], dtype=np.int8)
    blob = Binary.from_vector(codes, dtype=BinaryVectorDtype.INT8)
    np.testing.assert_array_equal(unpack_vec(blob), codes.astype(np.float32))


def test_encode_decode_vec_is_legacy_float32():
    v = _v(seed=5)
    # encode_vec stays on the flat float32 layout (s03-compatible).
    raw = base64.b64decode(encode_vec(v))
    assert len(raw) == v.size * 4
    np.testing.assert_array_equal(decode_vec(encode_vec(v)), v)
