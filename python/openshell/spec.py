# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed, hand-authorable models for constructing a sandbox spec.

These stdlib `dataclasses` are a friendlier surface over the generated
protobuf messages in `openshell._proto`. Callers build a `SandboxSpec`
(and the specs it nests) with normal Python types — `str`, `int`, `dict`,
`list` — instead of mutating raw protobuf objects, then hand it to
`SandboxClient.create` / `Sandbox(spec=...)`, which accept either form.

Every model provides:

- `to_proto()` — build the corresponding `openshell_pb2` message.
- `from_proto(msg)` — read one back into a typed model.

The conversions are round-trip faithful: `Model.from_proto(m).to_proto()`
reproduces `m`, and `Model.to_proto()` followed by `from_proto` reproduces
the model. Message-typed fields use proto3 field presence (`HasField`) so an
unset submessage maps to `None` rather than a zero-valued instance.

Two boundaries are deliberate, to keep this a medium-weight, dependency-free
addition:

- `google.protobuf.Struct` fields (`SandboxTemplate.resources` and
  `driver_config`) are exposed as plain `dict`s. Struct stores all numbers as
  doubles, so a value like `2` reads back as `2.0`; the underlying protobuf
  round-trips exactly.
- `SandboxSpec.policy` is passed through as the raw
  `sandbox_pb2.SandboxPolicy` message. The policy tree (L7 network rules and
  friends) is large and is normally discovered by the sandbox container rather
  than hand-built by SDK callers; a fully typed policy model is left as a
  follow-up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from google.protobuf.json_format import MessageToDict

from ._proto import openshell_pb2, sandbox_pb2

if TYPE_CHECKING:
    from collections.abc import Mapping


def _fill_struct(target: Any, value: Mapping[str, Any]) -> None:
    # `SetInParent` marks the singular message field present even when
    # `value` is empty, so an explicitly empty dict round-trips as a set
    # (rather than absent) Struct. `target` is a `google.protobuf.Struct`.
    target.SetInParent()
    target.update(dict(value))


def _struct_to_dict(struct: Any) -> dict[str, Any]:
    return MessageToDict(struct)


@dataclass
class GpuRequirements:
    """GPU portion of a sandbox's resource requirements.

    `count` is optional: leave it `None` to request the driver's default
    assignment (typically one GPU), matching the proto's `optional uint32`.
    """

    count: int | None = None

    def __post_init__(self) -> None:
        if self.count is not None and self.count < 0:
            raise ValueError("GpuRequirements: count must be non-negative")

    def to_proto(self) -> openshell_pb2.GpuResourceRequirements:
        proto = openshell_pb2.GpuResourceRequirements()
        if self.count is not None:
            proto.count = self.count
        return proto

    @classmethod
    def from_proto(
        cls, proto: openshell_pb2.GpuResourceRequirements
    ) -> GpuRequirements:
        return cls(count=proto.count if proto.HasField("count") else None)


@dataclass
class ResourceRequirements:
    """Portable resource requirements used for driver selection.

    Presence of `gpu` indicates a GPU request; leave it `None` for a
    CPU-only sandbox.
    """

    gpu: GpuRequirements | None = None

    def to_proto(self) -> openshell_pb2.ResourceRequirements:
        proto = openshell_pb2.ResourceRequirements()
        if self.gpu is not None:
            proto.gpu.CopyFrom(self.gpu.to_proto())
        return proto

    @classmethod
    def from_proto(
        cls, proto: openshell_pb2.ResourceRequirements
    ) -> ResourceRequirements:
        return cls(
            gpu=GpuRequirements.from_proto(proto.gpu)
            if proto.HasField("gpu")
            else None,
        )


@dataclass
class SandboxTemplate:
    """Container or VM template used to provision the sandbox.

    `resources` and `driver_config` are opaque `google.protobuf.Struct`
    envelopes surfaced as `dict`s. `user_namespaces` is a tri-state
    (`None` defers to the cluster-wide default), matching the proto's
    `optional bool`.
    """

    image: str = ""
    runtime_class_name: str = ""
    agent_socket: str = ""
    labels: dict[str, str] = field(default_factory=dict)
    annotations: dict[str, str] = field(default_factory=dict)
    environment: dict[str, str] = field(default_factory=dict)
    resources: dict[str, Any] | None = None
    user_namespaces: bool | None = None
    driver_config: dict[str, Any] | None = None

    def to_proto(self) -> openshell_pb2.SandboxTemplate:
        proto = openshell_pb2.SandboxTemplate(
            image=self.image,
            runtime_class_name=self.runtime_class_name,
            agent_socket=self.agent_socket,
        )
        proto.labels.update(self.labels)
        proto.annotations.update(self.annotations)
        proto.environment.update(self.environment)
        if self.resources is not None:
            _fill_struct(proto.resources, self.resources)
        if self.user_namespaces is not None:
            proto.user_namespaces = self.user_namespaces
        if self.driver_config is not None:
            _fill_struct(proto.driver_config, self.driver_config)
        return proto

    @classmethod
    def from_proto(cls, proto: openshell_pb2.SandboxTemplate) -> SandboxTemplate:
        return cls(
            image=proto.image,
            runtime_class_name=proto.runtime_class_name,
            agent_socket=proto.agent_socket,
            labels=dict(proto.labels),
            annotations=dict(proto.annotations),
            environment=dict(proto.environment),
            resources=_struct_to_dict(proto.resources)
            if proto.HasField("resources")
            else None,
            user_namespaces=proto.user_namespaces
            if proto.HasField("user_namespaces")
            else None,
            driver_config=_struct_to_dict(proto.driver_config)
            if proto.HasField("driver_config")
            else None,
        )


@dataclass
class SandboxSpec:
    """Desired sandbox configuration submitted through the API.

    Pass an instance directly to `SandboxClient.create`, `create_session`,
    or `Sandbox(spec=...)`; the client converts it via `to_proto()`. An
    all-default `SandboxSpec()` is equivalent to the server-side default —
    the sandbox container discovers its policy from the image.

    `policy` is the raw `sandbox_pb2.SandboxPolicy` message (see the module
    docstring); leave it `None` to let the container supply its baked-in
    policy.
    """

    log_level: str = ""
    environment: dict[str, str] = field(default_factory=dict)
    template: SandboxTemplate | None = None
    policy: sandbox_pb2.SandboxPolicy | None = None
    providers: list[str] = field(default_factory=list)
    resource_requirements: ResourceRequirements | None = None

    def to_proto(self) -> openshell_pb2.SandboxSpec:
        proto = openshell_pb2.SandboxSpec(log_level=self.log_level)
        proto.environment.update(self.environment)
        if self.template is not None:
            proto.template.CopyFrom(self.template.to_proto())
        if self.policy is not None:
            proto.policy.CopyFrom(self.policy)
        proto.providers.extend(self.providers)
        if self.resource_requirements is not None:
            proto.resource_requirements.CopyFrom(self.resource_requirements.to_proto())
        return proto

    @classmethod
    def from_proto(cls, proto: openshell_pb2.SandboxSpec) -> SandboxSpec:
        policy: sandbox_pb2.SandboxPolicy | None = None
        if proto.HasField("policy"):
            policy = sandbox_pb2.SandboxPolicy()
            policy.CopyFrom(proto.policy)
        return cls(
            log_level=proto.log_level,
            environment=dict(proto.environment),
            template=SandboxTemplate.from_proto(proto.template)
            if proto.HasField("template")
            else None,
            policy=policy,
            providers=list(proto.providers),
            resource_requirements=ResourceRequirements.from_proto(
                proto.resource_requirements
            )
            if proto.HasField("resource_requirements")
            else None,
        )
