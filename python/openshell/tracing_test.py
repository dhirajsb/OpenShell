# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from openshell import _tracing
from openshell._proto import openshell_pb2
from openshell.sandbox import SandboxClient


class _FakeExecStub:
    def __init__(self) -> None:
        self.request: openshell_pb2.ExecSandboxRequest | None = None

    def ExecSandbox(
        self,
        request: openshell_pb2.ExecSandboxRequest,
        timeout: float | None = None,
    ):
        self.request = request
        _ = timeout
        yield openshell_pb2.ExecSandboxEvent(
            exit=openshell_pb2.ExecSandboxExit(exit_code=7)
        )


def _make_sandbox_proto(
    id_: str,
    name: str,
    labels: dict[str, str] | None = None,
) -> openshell_pb2.Sandbox:
    sandbox = openshell_pb2.Sandbox()
    sandbox.metadata.id = id_
    sandbox.metadata.name = name
    for key, value in (labels or {}).items():
        sandbox.metadata.labels[key] = value
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
        return SimpleNamespace(
            sandbox=_make_sandbox_proto(
                "sandbox-1", request.name or "generated", dict(request.labels)
            )
        )


def _client_with_fake_stub(stub: object) -> SandboxClient:
    client = cast("SandboxClient", object.__new__(SandboxClient))
    client._timeout = 30.0
    client._stub = cast("Any", stub)
    return client


@pytest.fixture(autouse=True)
def _clear_tracer_cache() -> Any:
    _tracing._reset_tracer_cache()
    yield
    _tracing._reset_tracer_cache()


# ---------------------------------------------------------------------------
# No-op path: OpenTelemetry unavailable / not resolvable.
# ---------------------------------------------------------------------------


def test_span_yields_none_when_tracer_unavailable(monkeypatch: Any) -> None:
    monkeypatch.setattr(_tracing, "_load_tracer", lambda: None)

    with _tracing.span("openshell.test", attributes={"k": "v"}) as active:
        assert active is None


def test_load_tracer_returns_none_when_opentelemetry_missing(
    monkeypatch: Any,
) -> None:
    import builtins

    real_import = builtins.__import__

    def _blocked_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "opentelemetry" or name.startswith("opentelemetry."):
            raise ImportError("opentelemetry is not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)

    assert _tracing._load_tracer() is None


def test_client_create_works_without_tracer(monkeypatch: Any) -> None:
    monkeypatch.setattr(_tracing, "_load_tracer", lambda: None)
    stub = _FakeSandboxStub()
    client = _client_with_fake_stub(stub)

    ref = client.create(name="job-1", labels={"team": "aiq"})

    assert ref.id == "sandbox-1"
    assert stub.create_request is not None
    assert stub.create_request.name == "job-1"


def test_client_exec_works_without_tracer(monkeypatch: Any) -> None:
    monkeypatch.setattr(_tracing, "_load_tracer", lambda: None)
    stub = _FakeExecStub()
    client = _client_with_fake_stub(stub)

    result = client.exec("sandbox-1", ["echo", "ok"])

    assert result.exit_code == 7


# ---------------------------------------------------------------------------
# Span-emitting path: an in-memory exporter captures the emitted spans.
# ---------------------------------------------------------------------------


def _in_memory_tracer() -> tuple[Any, Any]:
    """Build a tracer backed by an in-memory span exporter."""
    pytest.importorskip("opentelemetry.sdk")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer("openshell-test"), exporter


def test_create_emits_span_with_attributes(monkeypatch: Any) -> None:
    tracer, exporter = _in_memory_tracer()
    monkeypatch.setattr(_tracing, "_load_tracer", lambda: tracer)

    stub = _FakeSandboxStub()
    client = _client_with_fake_stub(stub)

    client.create(name="job-1", labels={"team": "aiq"})

    spans = exporter.get_finished_spans()
    assert [s.name for s in spans] == ["openshell.sandbox.create"]
    assert spans[0].attributes["openshell.sandbox.name"] == "job-1"
    assert spans[0].attributes["openshell.sandbox.id"] == "sandbox-1"


def test_exec_emits_span_with_exit_code(monkeypatch: Any) -> None:
    tracer, exporter = _in_memory_tracer()
    monkeypatch.setattr(_tracing, "_load_tracer", lambda: tracer)

    stub = _FakeExecStub()
    client = _client_with_fake_stub(stub)

    client.exec("sandbox-1", ["echo", "ok"])

    spans = exporter.get_finished_spans()
    assert [s.name for s in spans] == ["openshell.sandbox.exec"]
    assert spans[0].attributes["openshell.sandbox.id"] == "sandbox-1"
    assert spans[0].attributes["openshell.exec.argc"] == 2
    assert spans[0].attributes["openshell.exec.exit_code"] == 7


def test_span_drops_none_valued_attributes(monkeypatch: Any) -> None:
    tracer, exporter = _in_memory_tracer()
    monkeypatch.setattr(_tracing, "_load_tracer", lambda: tracer)

    with _tracing.span("openshell.test", attributes={"kept": "v", "dropped": None}):
        pass

    span = exporter.get_finished_spans()[0]
    assert span.attributes["kept"] == "v"
    assert "dropped" not in span.attributes


def test_span_records_exception_and_reraises(monkeypatch: Any) -> None:
    tracer, exporter = _in_memory_tracer()
    monkeypatch.setattr(_tracing, "_load_tracer", lambda: tracer)

    with pytest.raises(ValueError, match="boom"), _tracing.span("openshell.test"):
        raise ValueError("boom")

    span = exporter.get_finished_spans()[0]
    assert any(event.name == "exception" for event in span.events)
