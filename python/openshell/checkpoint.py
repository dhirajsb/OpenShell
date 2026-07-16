# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Portable, integrity-protected sandbox checkpoint format (``.osckpt``).

This module is the client-side half of the checkpoint/pause/resume proposal
tracked in ``rfc/0011-checkpoint-pause-resume``. It defines the on-disk /
on-the-wire envelope the SDK uses to hand a sandbox's captured state back to a
caller and to feed it back into a restore, independently of any gateway or
compute-driver support.

The envelope authenticates and integrity-protects an opaque payload (the state
blob produced by a compute driver) so a checkpoint can safely cross a trust
boundary — be written to disk, moved between gateways, and later restored:

    +-----------+---------+-------------+----------------+--------------+---------+----------+
    | magic (6) | ver (1) | hlen (4 BE) | header (hlen)  | plen (8 BE)  | payload | tag (32) |
    +-----------+---------+-------------+----------------+--------------+---------+----------+
      b"OSCKPT"    0x01      uint32        canonical JSON    uint64        bytes    HMAC-SHA256

Two independent checks are enforced on unpack:

- ``digest`` (SHA-256 of the payload, recorded in the header) catches accidental
  corruption or truncation even without the key.
- ``tag`` (HMAC-SHA256 over the whole envelope, compared in constant time)
  catches tampering and authenticates the writer.

The payload itself is opaque to this module: it is whatever a driver's
checkpoint engine produced (a CRIU image, a tar of the filesystem, a VM memory
snapshot, ...). This module does not interpret it.

Nothing here talks to the gateway; the round-trip RPCs (``CheckpointSandbox`` /
``RestoreSandbox``) are described in the RFC and are not yet implemented.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import struct
import time
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "Checkpoint",
    "CheckpointError",
    "CheckpointIntegrityError",
    "CheckpointMetadata",
    "pack_checkpoint",
    "unpack_checkpoint",
]

# Magic prefix identifying an OpenShell checkpoint envelope.
_MAGIC = b"OSCKPT"
# Wire/format version. Bump when the envelope layout changes incompatibly;
# unknown versions fail closed on unpack. An encrypted (AEAD) variant would be
# introduced as a new version rather than an in-place change (see RFC 0011).
CHECKPOINT_FORMAT_VERSION = 1
# HMAC-SHA256 output size.
_TAG_LEN = 32
# struct formats: big-endian u32 header length, big-endian u64 payload length.
_HLEN = struct.Struct(">I")
_PLEN = struct.Struct(">Q")


class CheckpointError(Exception):
    """Base class for checkpoint packing/unpacking failures."""


class CheckpointIntegrityError(CheckpointError):
    """Raised when a checkpoint fails an integrity or authenticity check.

    This covers a bad magic/version, a header/payload length that does not
    match the envelope, a payload digest mismatch, or an HMAC that does not
    verify under the supplied key. A restore path must treat this as fatal and
    never forward the payload to the gateway.
    """


@dataclass(frozen=True)
class CheckpointMetadata:
    """Self-describing header for a checkpoint.

    Everything needed to decide whether a checkpoint can be restored, and to
    detect a mismatched or corrupted artifact, without interpreting the opaque
    payload. Serialized as canonical (sorted-key, compact) JSON inside the
    envelope header.
    """

    sandbox_id: str
    format_version: int = CHECKPOINT_FORMAT_VERSION
    created_at_ms: int = 0
    # Checkpoint engine that produced the payload, e.g. "criu", "runc-checkpoint",
    # "fs-tar", "vm-snapshot". Provenance used to reject mismatched restores.
    engine: str = ""
    engine_version: str = ""
    # CPU architecture the checkpoint was captured on, e.g. "x86_64", "aarch64".
    arch: str = ""
    # Originating compute driver name, e.g. "docker", "kubernetes", "vm".
    driver: str = ""
    # SHA-256 (hex) of the payload. Filled in by pack_checkpoint.
    digest: str = ""
    size_bytes: int = 0
    # Whether the payload captures process/memory state (vs filesystem+spec only).
    includes_process_memory: bool = False
    # Opaque, caller-supplied labels stamped at capture time.
    labels: Mapping[str, str] = field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize to canonical JSON (sorted keys, no insignificant spaces)."""
        return json.dumps(
            {
                "sandbox_id": self.sandbox_id,
                "format_version": self.format_version,
                "created_at_ms": self.created_at_ms,
                "engine": self.engine,
                "engine_version": self.engine_version,
                "arch": self.arch,
                "driver": self.driver,
                "digest": self.digest,
                "size_bytes": self.size_bytes,
                "includes_process_memory": self.includes_process_memory,
                "labels": dict(self.labels),
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, raw: bytes) -> CheckpointMetadata:
        try:
            obj = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CheckpointIntegrityError(
                f"checkpoint header is not valid JSON: {exc}"
            ) from exc
        if not isinstance(obj, dict):
            raise CheckpointIntegrityError("checkpoint header must be a JSON object")
        labels = obj.get("labels") or {}
        if not isinstance(labels, dict):
            raise CheckpointIntegrityError("checkpoint header labels must be a map")
        return cls(
            sandbox_id=str(obj.get("sandbox_id", "")),
            format_version=int(obj.get("format_version", 0)),
            created_at_ms=int(obj.get("created_at_ms", 0)),
            engine=str(obj.get("engine", "")),
            engine_version=str(obj.get("engine_version", "")),
            arch=str(obj.get("arch", "")),
            driver=str(obj.get("driver", "")),
            digest=str(obj.get("digest", "")),
            size_bytes=int(obj.get("size_bytes", 0)),
            includes_process_memory=bool(obj.get("includes_process_memory", False)),
            labels={str(k): str(v) for k, v in labels.items()},
        )


@dataclass(frozen=True)
class Checkpoint:
    """A captured sandbox checkpoint: metadata plus an opaque state payload.

    Use :meth:`to_bytes` to produce a signed ``.osckpt`` envelope and
    :meth:`from_bytes` to parse and verify one. The payload is never
    interpreted by the SDK.
    """

    metadata: CheckpointMetadata
    payload: bytes

    def to_bytes(self, *, key: bytes) -> bytes:
        """Serialize to a signed ``.osckpt`` envelope authenticated with ``key``."""
        return pack_checkpoint(self.payload, metadata=self.metadata, key=key)

    @classmethod
    def from_bytes(cls, blob: bytes, *, key: bytes, verify: bool = True) -> Checkpoint:
        """Parse and (by default) verify a ``.osckpt`` envelope.

        Raises :class:`CheckpointIntegrityError` if ``verify`` is true and the
        digest or HMAC does not check out under ``key``.
        """
        return unpack_checkpoint(blob, key=key, verify=verify)


def pack_checkpoint(
    payload: bytes,
    *,
    metadata: CheckpointMetadata,
    key: bytes,
) -> bytes:
    """Build a signed ``.osckpt`` envelope around ``payload``.

    The payload SHA-256 digest and size are (re)computed and stamped into the
    header so the returned envelope is internally consistent regardless of what
    the caller passed in ``metadata``. The whole envelope is then authenticated
    with an HMAC-SHA256 tag over ``key``.
    """
    if not key:
        raise CheckpointError("a non-empty HMAC key is required to pack a checkpoint")

    digest = hashlib.sha256(payload).hexdigest()
    header_meta = replace(
        metadata,
        format_version=CHECKPOINT_FORMAT_VERSION,
        digest=digest,
        size_bytes=len(payload),
        created_at_ms=metadata.created_at_ms or int(time.time() * 1000),
    )
    header = header_meta.to_json().encode("utf-8")

    prefix = (
        _MAGIC
        + bytes([CHECKPOINT_FORMAT_VERSION])
        + _HLEN.pack(len(header))
        + header
        + _PLEN.pack(len(payload))
        + payload
    )
    tag = hmac.new(key, prefix, hashlib.sha256).digest()
    return prefix + tag


def unpack_checkpoint(
    blob: bytes,
    *,
    key: bytes,
    verify: bool = True,
) -> Checkpoint:
    """Parse a ``.osckpt`` envelope, verifying integrity and authenticity.

    When ``verify`` is true (the default), both the payload SHA-256 digest and
    the envelope HMAC are checked; either failure raises
    :class:`CheckpointIntegrityError`. ``verify=False`` still validates the
    structural framing (magic, version, lengths) but skips the cryptographic
    checks — use it only for inspection of an untrusted blob, never before a
    restore.
    """
    if verify and not key:
        raise CheckpointError("a non-empty HMAC key is required to verify a checkpoint")

    # magic(6) + ver(1) + hlen(4) + plen(8) + tag(32) is the fixed overhead.
    min_len = len(_MAGIC) + 1 + _HLEN.size + _PLEN.size + _TAG_LEN
    if len(blob) < min_len:
        raise CheckpointIntegrityError("checkpoint is too short to be valid")

    off = 0
    if blob[: len(_MAGIC)] != _MAGIC:
        raise CheckpointIntegrityError("not an OpenShell checkpoint (bad magic)")
    off += len(_MAGIC)

    version = blob[off]
    off += 1
    if version != CHECKPOINT_FORMAT_VERSION:
        raise CheckpointIntegrityError(
            f"unsupported checkpoint format version {version}; "
            f"this SDK supports version {CHECKPOINT_FORMAT_VERSION}"
        )

    (header_len,) = _HLEN.unpack_from(blob, off)
    off += _HLEN.size
    header_end = off + header_len
    if header_end + _PLEN.size + _TAG_LEN > len(blob):
        raise CheckpointIntegrityError("checkpoint header length is out of bounds")
    header = blob[off:header_end]
    off = header_end

    (payload_len,) = _PLEN.unpack_from(blob, off)
    off += _PLEN.size
    payload_end = off + payload_len
    if payload_end + _TAG_LEN != len(blob):
        raise CheckpointIntegrityError("checkpoint payload length does not match frame")
    payload = blob[off:payload_end]
    tag = blob[payload_end:]

    if verify:
        expected = hmac.new(key, blob[:payload_end], hashlib.sha256).digest()
        if not hmac.compare_digest(expected, tag):
            raise CheckpointIntegrityError("checkpoint HMAC verification failed")

    metadata = CheckpointMetadata.from_json(header)

    if verify:
        actual_digest = hashlib.sha256(payload).hexdigest()
        if metadata.digest and not hmac.compare_digest(actual_digest, metadata.digest):
            raise CheckpointIntegrityError("checkpoint payload digest mismatch")

    return Checkpoint(metadata=metadata, payload=payload)
