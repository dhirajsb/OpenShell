# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib

import pytest

from openshell.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    Checkpoint,
    CheckpointError,
    CheckpointIntegrityError,
    CheckpointMetadata,
    pack_checkpoint,
    unpack_checkpoint,
)

_KEY = b"0123456789abcdef0123456789abcdef"
_PAYLOAD = b"the opaque driver-produced state blob \x00\x01\x02" * 32


def _meta(**overrides: object) -> CheckpointMetadata:
    base = {
        "sandbox_id": "sbx-abc123",
        "engine": "criu",
        "engine_version": "3.19",
        "arch": "x86_64",
        "driver": "docker",
        "includes_process_memory": True,
        "labels": {"reason": "golden"},
    }
    base.update(overrides)
    return CheckpointMetadata(**base)  # type: ignore[arg-type]


def test_pack_unpack_round_trip() -> None:
    blob = pack_checkpoint(_PAYLOAD, metadata=_meta(), key=_KEY)
    restored = unpack_checkpoint(blob, key=_KEY)
    assert restored.payload == _PAYLOAD
    assert restored.metadata.sandbox_id == "sbx-abc123"
    assert restored.metadata.engine == "criu"
    assert restored.metadata.arch == "x86_64"
    assert restored.metadata.driver == "docker"
    assert restored.metadata.includes_process_memory is True
    assert restored.metadata.labels == {"reason": "golden"}


def test_pack_stamps_digest_size_and_version() -> None:
    blob = pack_checkpoint(_PAYLOAD, metadata=_meta(digest="", size_bytes=0), key=_KEY)
    meta = unpack_checkpoint(blob, key=_KEY).metadata
    assert meta.digest == hashlib.sha256(_PAYLOAD).hexdigest()
    assert meta.size_bytes == len(_PAYLOAD)
    assert meta.format_version == CHECKPOINT_FORMAT_VERSION


def test_pack_defaults_created_at_ms() -> None:
    blob = pack_checkpoint(_PAYLOAD, metadata=_meta(), key=_KEY)
    assert unpack_checkpoint(blob, key=_KEY).metadata.created_at_ms > 0


def test_pack_preserves_explicit_created_at_ms() -> None:
    blob = pack_checkpoint(_PAYLOAD, metadata=_meta(created_at_ms=42), key=_KEY)
    assert unpack_checkpoint(blob, key=_KEY).metadata.created_at_ms == 42


def test_checkpoint_to_from_bytes_helpers() -> None:
    ckpt = Checkpoint(metadata=_meta(), payload=_PAYLOAD)
    blob = ckpt.to_bytes(key=_KEY)
    restored = Checkpoint.from_bytes(blob, key=_KEY)
    assert restored.payload == _PAYLOAD
    assert restored.metadata.sandbox_id == ckpt.metadata.sandbox_id


def test_empty_payload_round_trips() -> None:
    blob = pack_checkpoint(b"", metadata=_meta(), key=_KEY)
    restored = unpack_checkpoint(blob, key=_KEY)
    assert restored.payload == b""
    assert restored.metadata.size_bytes == 0


def test_tampered_payload_fails_hmac() -> None:
    blob = bytearray(pack_checkpoint(_PAYLOAD, metadata=_meta(), key=_KEY))
    # Flip a byte inside the payload region (well past the header).
    blob[-40] ^= 0xFF
    with pytest.raises(CheckpointIntegrityError):
        unpack_checkpoint(bytes(blob), key=_KEY)


def test_tampered_tag_fails_hmac() -> None:
    blob = bytearray(pack_checkpoint(_PAYLOAD, metadata=_meta(), key=_KEY))
    blob[-1] ^= 0xFF
    with pytest.raises(CheckpointIntegrityError):
        unpack_checkpoint(bytes(blob), key=_KEY)


def test_wrong_key_fails_hmac() -> None:
    blob = pack_checkpoint(_PAYLOAD, metadata=_meta(), key=_KEY)
    with pytest.raises(CheckpointIntegrityError):
        unpack_checkpoint(blob, key=b"x" * 32)


def test_bad_magic_rejected() -> None:
    blob = bytearray(pack_checkpoint(_PAYLOAD, metadata=_meta(), key=_KEY))
    blob[0] ^= 0xFF
    with pytest.raises(CheckpointIntegrityError):
        unpack_checkpoint(bytes(blob), key=_KEY)


def test_unknown_version_rejected() -> None:
    blob = bytearray(pack_checkpoint(_PAYLOAD, metadata=_meta(), key=_KEY))
    # The version byte immediately follows the 6-byte magic.
    blob[6] = 0xFF
    with pytest.raises(CheckpointIntegrityError):
        unpack_checkpoint(bytes(blob), key=_KEY)


def test_truncated_blob_rejected() -> None:
    blob = pack_checkpoint(_PAYLOAD, metadata=_meta(), key=_KEY)
    with pytest.raises(CheckpointIntegrityError):
        unpack_checkpoint(blob[:10], key=_KEY)


def test_verify_false_skips_crypto_but_parses() -> None:
    blob = bytearray(pack_checkpoint(_PAYLOAD, metadata=_meta(), key=_KEY))
    blob[-1] ^= 0xFF  # break the tag
    # verify=False must still return the parsed content without raising.
    restored = unpack_checkpoint(bytes(blob), key=_KEY, verify=False)
    assert restored.payload == _PAYLOAD


def test_pack_requires_key() -> None:
    with pytest.raises(CheckpointError):
        pack_checkpoint(_PAYLOAD, metadata=_meta(), key=b"")


def test_unpack_requires_key_when_verifying() -> None:
    blob = pack_checkpoint(_PAYLOAD, metadata=_meta(), key=_KEY)
    with pytest.raises(CheckpointError):
        unpack_checkpoint(blob, key=b"")


def test_metadata_json_is_canonical() -> None:
    # Canonical JSON: sorted keys, compact separators. Stable across runs so the
    # HMAC is reproducible.
    meta = _meta()
    assert meta.to_json() == meta.to_json()
    assert ", " not in meta.to_json()
    assert '"arch":"x86_64"' in meta.to_json()
