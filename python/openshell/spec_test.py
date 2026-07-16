# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from openshell._proto import openshell_pb2, sandbox_pb2
from openshell.spec import (
    GpuRequirements,
    ResourceRequirements,
    SandboxSpec,
    SandboxTemplate,
)

# ---------------------------------------------------------------------------
# GpuRequirements
# ---------------------------------------------------------------------------


def test_gpu_requirements_default_omits_count() -> None:
    proto = GpuRequirements().to_proto()
    # `optional uint32` — an unset count must not set field presence.
    assert not proto.HasField("count")


def test_gpu_requirements_sets_count() -> None:
    proto = GpuRequirements(count=4).to_proto()
    assert proto.HasField("count")
    assert proto.count == 4


def test_gpu_requirements_round_trip() -> None:
    for count in (None, 0, 1, 8):
        model = GpuRequirements(count=count)
        assert GpuRequirements.from_proto(model.to_proto()) == model


def test_gpu_requirements_rejects_negative_count() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        GpuRequirements(count=-1)


# ---------------------------------------------------------------------------
# ResourceRequirements
# ---------------------------------------------------------------------------


def test_resource_requirements_default_has_no_gpu() -> None:
    proto = ResourceRequirements().to_proto()
    assert not proto.HasField("gpu")


def test_resource_requirements_round_trip_with_gpu() -> None:
    model = ResourceRequirements(gpu=GpuRequirements(count=2))
    restored = ResourceRequirements.from_proto(model.to_proto())
    assert restored == model
    assert restored.gpu is not None
    assert restored.gpu.count == 2


def test_resource_requirements_round_trip_without_gpu() -> None:
    model = ResourceRequirements()
    assert ResourceRequirements.from_proto(model.to_proto()) == model


# ---------------------------------------------------------------------------
# SandboxTemplate
# ---------------------------------------------------------------------------


def test_sandbox_template_defaults_round_trip() -> None:
    model = SandboxTemplate()
    assert SandboxTemplate.from_proto(model.to_proto()) == model


def test_sandbox_template_full_round_trip() -> None:
    model = SandboxTemplate(
        image="ghcr.io/example/sandbox:latest",
        runtime_class_name="gvisor",
        agent_socket="/run/agent.sock",
        labels={"team": "aiq"},
        annotations={"note": "demo"},
        environment={"LOG": "debug"},
        # String/bool values survive Struct round-trip exactly.
        resources={"tier": "gold", "spot": True},
        user_namespaces=True,
        driver_config={"driver": "docker"},
    )
    restored = SandboxTemplate.from_proto(model.to_proto())
    assert restored == model


def test_sandbox_template_user_namespaces_tristate() -> None:
    assert not SandboxTemplate().to_proto().HasField("user_namespaces")
    for value in (True, False):
        proto = SandboxTemplate(user_namespaces=value).to_proto()
        assert proto.HasField("user_namespaces")
        assert proto.user_namespaces is value
        assert SandboxTemplate.from_proto(proto).user_namespaces is value


def test_sandbox_template_empty_struct_keeps_presence() -> None:
    # An explicitly empty dict is distinct from an unset Struct field.
    proto = SandboxTemplate(resources={}).to_proto()
    assert proto.HasField("resources")
    assert SandboxTemplate.from_proto(proto).resources == {}


def test_sandbox_template_struct_numbers_become_floats() -> None:
    # Documented behavior: google.protobuf.Struct stores numbers as doubles,
    # so integers read back as floats. The protobuf itself round-trips.
    model = SandboxTemplate(resources={"cpu": 2})
    restored = SandboxTemplate.from_proto(model.to_proto())
    assert restored.resources == {"cpu": 2.0}


# ---------------------------------------------------------------------------
# SandboxSpec
# ---------------------------------------------------------------------------


def test_sandbox_spec_default_matches_empty_proto() -> None:
    assert SandboxSpec().to_proto() == openshell_pb2.SandboxSpec()


def test_sandbox_spec_defaults_round_trip() -> None:
    model = SandboxSpec()
    assert SandboxSpec.from_proto(model.to_proto()) == model


def test_sandbox_spec_full_round_trip() -> None:
    model = SandboxSpec(
        log_level="debug",
        environment={"FOO": "bar"},
        template=SandboxTemplate(image="img:1", environment={"NESTED": "1"}),
        providers=["claude", "gitlab"],
        resource_requirements=ResourceRequirements(gpu=GpuRequirements(count=1)),
    )
    restored = SandboxSpec.from_proto(model.to_proto())
    assert restored == model


def test_sandbox_spec_populates_expected_proto_fields() -> None:
    proto = SandboxSpec(
        log_level="info",
        environment={"K": "V"},
        template=SandboxTemplate(image="img:2"),
        providers=["openai"],
        resource_requirements=ResourceRequirements(gpu=GpuRequirements(count=3)),
    ).to_proto()

    assert proto.log_level == "info"
    assert dict(proto.environment) == {"K": "V"}
    assert proto.template.image == "img:2"
    assert list(proto.providers) == ["openai"]
    assert proto.resource_requirements.gpu.count == 3


def test_sandbox_spec_policy_passthrough_round_trip() -> None:
    policy = sandbox_pb2.SandboxPolicy(version=7)
    policy.filesystem.include_workdir = True
    policy.filesystem.read_only.append("/etc")

    model = SandboxSpec(policy=policy)
    proto = model.to_proto()
    assert proto.HasField("policy")
    assert proto.policy.version == 7

    restored = SandboxSpec.from_proto(proto)
    assert restored.policy == policy
    # Passthrough copies rather than aliasing the caller's message.
    assert restored.policy is not policy


def test_sandbox_spec_omits_unset_optional_submessages() -> None:
    proto = SandboxSpec().to_proto()
    assert not proto.HasField("template")
    assert not proto.HasField("policy")
    assert not proto.HasField("resource_requirements")


def test_proto_to_model_to_proto_round_trip() -> None:
    proto = openshell_pb2.SandboxSpec(log_level="warn")
    proto.environment["A"] = "1"
    proto.template.image = "img:3"
    proto.template.resources.update({"tier": "silver"})
    proto.providers.extend(["p1", "p2"])
    proto.resource_requirements.gpu.count = 5
    proto.policy.version = 2

    assert SandboxSpec.from_proto(proto).to_proto() == proto


# ---------------------------------------------------------------------------
# SandboxClient.create accepts the typed model
# ---------------------------------------------------------------------------


def _make_sandbox_proto(name: str) -> openshell_pb2.Sandbox:
    sandbox = openshell_pb2.Sandbox()
    sandbox.metadata.id = "sandbox-1"
    sandbox.metadata.name = name
    sandbox.status.phase = openshell_pb2.SANDBOX_PHASE_READY
    return sandbox


class _FakeSandboxStub:
    def __init__(self) -> None:
        self.create_request: openshell_pb2.CreateSandboxRequest | None = None

    def CreateSandbox(
        self,
        request: openshell_pb2.CreateSandboxRequest,
        timeout: float | None = None,
    ) -> Any:
        self.create_request = request
        _ = timeout
        return SimpleNamespace(sandbox=_make_sandbox_proto(request.name or "generated"))


def _client_with_fake_stub(stub: object) -> Any:
    from openshell.sandbox import SandboxClient

    client = cast("SandboxClient", object.__new__(SandboxClient))
    client._timeout = 30.0
    client._stub = cast("Any", stub)
    return client


def test_create_accepts_typed_spec_model() -> None:
    stub = _FakeSandboxStub()
    client = _client_with_fake_stub(stub)

    spec = SandboxSpec(log_level="debug", template=SandboxTemplate(image="img:1"))
    client.create(spec=spec)

    assert stub.create_request is not None
    # The typed model is converted to its protobuf form before sending.
    assert stub.create_request.spec == spec.to_proto()


def test_create_still_accepts_raw_proto_spec() -> None:
    stub = _FakeSandboxStub()
    client = _client_with_fake_stub(stub)

    raw = openshell_pb2.SandboxSpec(log_level="info")
    client.create(spec=raw)

    assert stub.create_request is not None
    assert stub.create_request.spec == raw
