# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional OpenTelemetry tracing hooks for sandbox operations.

Tracing is entirely opt-in and degrades to a no-op when OpenTelemetry is
not installed or not configured:

- **OpenTelemetry absent**: `opentelemetry` is imported lazily the first
  time a span is requested. If the import fails (the `openshell[otel]`
  extra was not installed) the tracer resolves to `None` and every
  `span(...)` block becomes a no-op that never touches the OTEL API.
- **OpenTelemetry present but unconfigured**: `opentelemetry.trace`
  returns its built-in no-op tracer until an application installs an SDK
  `TracerProvider`. Spans created against that tracer do nothing and are
  never exported, so instrumentation stays silent until the host process
  opts in by configuring a provider.

The tracer is resolved once and cached; the OTEL API's proxy tracer
resolves the active provider lazily per span, so a provider configured
after import is still honored. Nothing here raises if OpenTelemetry is
missing or misbehaves — sandbox operations must work identically whether
or not tracing is active.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

# Instrumentation scope name reported on every emitted span.
_INSTRUMENTATION_NAME = "openshell"

# Sentinel distinguishing "tracer not yet resolved" from "resolved to
# None" (OpenTelemetry unavailable) so we attempt the lazy import at most
# once per process.
_UNSET: Any = object()
_tracer: Any = _UNSET


def _instrumentation_version() -> str | None:
    """Best-effort package version for the instrumentation scope."""
    try:
        from importlib.metadata import version

        return version("openshell")
    except Exception:
        return None


def _load_tracer() -> Any:
    """Resolve an OpenTelemetry tracer, or `None` when unavailable.

    Returns `None` if `opentelemetry` is not installed or the API cannot
    hand back a tracer for any reason. The caller treats `None` as "do
    nothing".
    """
    try:
        from opentelemetry import trace
    except ImportError:
        return None
    try:
        return trace.get_tracer(_INSTRUMENTATION_NAME, _instrumentation_version())
    except Exception:  # pragma: no cover - defensive; OTEL API should not raise
        return None


def _get_tracer() -> Any:
    """Return the cached tracer, resolving it lazily on first use."""
    global _tracer
    if _tracer is _UNSET:
        _tracer = _load_tracer()
    return _tracer


def _reset_tracer_cache() -> None:
    """Clear the cached tracer so the next `span(...)` re-resolves it.

    Intended for tests that swap the tracer via `_load_tracer`; production
    callers never need to invoke this.
    """
    global _tracer
    _tracer = _UNSET


@contextlib.contextmanager
def span(
    name: str,
    *,
    attributes: Mapping[str, Any] | None = None,
) -> Iterator[Any]:
    """Start a span named `name`, yielding it (or `None` when tracing is off).

    When OpenTelemetry is unavailable the block runs as a plain no-op and
    yields `None`, so callers can guard optional attribute enrichment with
    `if active is not None:`. Attributes whose value is `None` are dropped
    so callers can pass optional fields without branching. Exceptions
    raised inside the block are recorded on the span and re-raised,
    matching OpenTelemetry's default `start_as_current_span` behavior.
    """
    tracer = _get_tracer()
    if tracer is None:
        yield None
        return
    span_attributes = (
        {key: value for key, value in attributes.items() if value is not None}
        if attributes
        else None
    )
    with tracer.start_as_current_span(name, attributes=span_attributes) as active_span:
        yield active_span
