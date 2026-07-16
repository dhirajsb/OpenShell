# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LangChain tool factories for OpenShell sandboxes.

Expose an OpenShell sandbox's execution surface (Python code execution,
shell commands, and file read/write) as `langchain_core` tools so a
sandbox can be dropped into an agent framework as a first-class tool
provider.

LangChain is an *optional* dependency. This module imports cleanly
without it — the `langchain-core` import is deferred until a tool
factory is actually called, at which point a clear, actionable error is
raised if the extra is not installed::

    pip install "openshell[langchain]"

Typical usage::

    from openshell import Sandbox
    from openshell.langchain import create_sandbox_tools

    with Sandbox() as sandbox:
        tools = create_sandbox_tools(sandbox, timeout_seconds=30)
        # hand `tools` to any LangChain agent / executor

Every factory accepts anything exposing the sandbox `exec` contract
(`Sandbox`, `SandboxSession`), so the tools work with the high-level
context manager or a lower-level session.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from langchain_core.tools import BaseTool

    from .sandbox import ExecResult


# Safety defaults. Callers can override per factory. A non-positive
# `max_output_chars` disables truncation; a `None` timeout defers to the
# sandbox / gateway default.
_DEFAULT_TIMEOUT_SECONDS = 60
_DEFAULT_MAX_OUTPUT_CHARS = 10_000
_DEFAULT_PYTHON_BIN = "python"
# `sh -c <script>` is the most portable way to run an arbitrary shell
# command; the sandbox image is not guaranteed to ship bash.
_DEFAULT_SHELL: tuple[str, ...] = ("sh", "-c")


class SandboxExecutor(Protocol):
    """Structural type for objects that can run a command in a sandbox.

    Both `openshell.Sandbox` and `openshell.SandboxSession` satisfy this
    protocol, so the tool factories accept either without importing a
    concrete class.
    """

    def exec(
        self,
        command: Sequence[str],
        *,
        stream_output: bool = ...,
        workdir: str | None = ...,
        env: Mapping[str, str] | None = ...,
        stdin: bytes | None = ...,
        timeout_seconds: int | None = ...,
    ) -> ExecResult: ...


def _require_langchain() -> type[BaseTool]:
    """Import and return `langchain_core.tools.StructuredTool`.

    Deferred so this module imports without `langchain-core` present.
    Raises `ImportError` with an install hint the first time a factory
    is invoked without the optional extra.
    """
    try:
        tools = importlib.import_module("langchain_core.tools")
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise ImportError(
            "LangChain support requires the optional 'langchain' extra. "
            "Install it with: pip install 'openshell[langchain]'"
        ) from exc
    return tools.StructuredTool


def _truncate(text: str, max_output_chars: int) -> str:
    if max_output_chars > 0 and len(text) > max_output_chars:
        omitted = len(text) - max_output_chars
        return f"{text[:max_output_chars]}\n... [truncated {omitted} characters]"
    return text


def _format_exec_result(result: ExecResult, *, max_output_chars: int) -> str:
    """Render an `ExecResult` as an LLM-friendly, capped text block."""
    sections = [f"exit_code: {result.exit_code}"]
    if result.stdout:
        sections.append("stdout:\n" + _truncate(result.stdout, max_output_chars))
    if result.stderr:
        sections.append("stderr:\n" + _truncate(result.stderr, max_output_chars))
    return "\n".join(sections)


def create_python_tool(
    sandbox: SandboxExecutor,
    *,
    timeout_seconds: int | None = _DEFAULT_TIMEOUT_SECONDS,
    max_output_chars: int = _DEFAULT_MAX_OUTPUT_CHARS,
    python_bin: str = _DEFAULT_PYTHON_BIN,
    name: str = "openshell_run_python",
    description: str | None = None,
) -> BaseTool:
    """Tool that runs a Python source snippet inside the sandbox.

    The snippet is executed with `python -c <code>`; capture output by
    printing to stdout. Use for computation the agent should perform in
    the isolated environment rather than in the host process.

    Args:
        sandbox: a `Sandbox`/`SandboxSession` (anything satisfying
            `SandboxExecutor`).
        timeout_seconds: per-call execution timeout forwarded to the
            sandbox. `None` uses the sandbox default.
        max_output_chars: cap on each of stdout/stderr in the returned
            text; non-positive disables truncation.
        python_bin: interpreter to invoke inside the sandbox.
        name: tool name surfaced to the agent.
        description: tool description; a sensible default is used when
            omitted.
    """
    structured_tool = _require_langchain()

    def run_python(code: str) -> str:
        result = sandbox.exec(
            [python_bin, "-c", code],
            timeout_seconds=timeout_seconds,
        )
        return _format_exec_result(result, max_output_chars=max_output_chars)

    return structured_tool.from_function(
        func=run_python,
        name=name,
        description=(
            description
            or "Execute a Python 3 code snippet inside the OpenShell sandbox "
            "and return its exit code, stdout, and stderr. Print results to "
            "stdout to capture them."
        ),
    )


def create_shell_tool(
    sandbox: SandboxExecutor,
    *,
    timeout_seconds: int | None = _DEFAULT_TIMEOUT_SECONDS,
    max_output_chars: int = _DEFAULT_MAX_OUTPUT_CHARS,
    shell: Sequence[str] = _DEFAULT_SHELL,
    name: str = "openshell_run_shell",
    description: str | None = None,
) -> BaseTool:
    """Tool that runs a shell command inside the sandbox.

    The command string is passed to `sh -c` (configurable via `shell`).

    Args:
        sandbox: a `Sandbox`/`SandboxSession`.
        timeout_seconds: per-call execution timeout forwarded to the
            sandbox. `None` uses the sandbox default.
        max_output_chars: cap on each of stdout/stderr in the returned
            text; non-positive disables truncation.
        shell: argv prefix used to run the command string, e.g.
            `("sh", "-c")` or `("bash", "-lc")`.
        name: tool name surfaced to the agent.
        description: tool description; a sensible default is used when
            omitted.
    """
    structured_tool = _require_langchain()
    shell_prefix = list(shell)

    def run_shell(command: str) -> str:
        result = sandbox.exec(
            [*shell_prefix, command],
            timeout_seconds=timeout_seconds,
        )
        return _format_exec_result(result, max_output_chars=max_output_chars)

    return structured_tool.from_function(
        func=run_shell,
        name=name,
        description=(
            description
            or "Run a shell command inside the OpenShell sandbox and return "
            "its exit code, stdout, and stderr."
        ),
    )


def create_read_file_tool(
    sandbox: SandboxExecutor,
    *,
    timeout_seconds: int | None = _DEFAULT_TIMEOUT_SECONDS,
    max_output_chars: int = _DEFAULT_MAX_OUTPUT_CHARS,
    name: str = "openshell_read_file",
    description: str | None = None,
) -> BaseTool:
    """Tool that reads a file from the sandbox filesystem.

    Returns the file contents on success, or an `error:`-prefixed
    message carrying the exit code and stderr on failure (e.g. missing
    file or permission denied).

    Args:
        sandbox: a `Sandbox`/`SandboxSession`.
        timeout_seconds: per-call execution timeout forwarded to the
            sandbox. `None` uses the sandbox default.
        max_output_chars: cap on the returned contents; non-positive
            disables truncation.
        name: tool name surfaced to the agent.
        description: tool description; a sensible default is used when
            omitted.
    """
    structured_tool = _require_langchain()

    def read_file(path: str) -> str:
        result = sandbox.exec(
            ["cat", "--", path],
            timeout_seconds=timeout_seconds,
        )
        if result.exit_code != 0:
            return _format_exec_result(result, max_output_chars=max_output_chars)
        return _truncate(result.stdout, max_output_chars)

    return structured_tool.from_function(
        func=read_file,
        name=name,
        description=(
            description
            or "Read a file from the OpenShell sandbox filesystem and return "
            "its contents."
        ),
    )


def create_write_file_tool(
    sandbox: SandboxExecutor,
    *,
    timeout_seconds: int | None = _DEFAULT_TIMEOUT_SECONDS,
    max_output_chars: int = _DEFAULT_MAX_OUTPUT_CHARS,
    encoding: str = "utf-8",
    name: str = "openshell_write_file",
    description: str | None = None,
) -> BaseTool:
    """Tool that writes text content to a file in the sandbox.

    Content is streamed via stdin to `sh -c 'cat > "$1"' sh <path>`, so
    the destination path is passed as a positional argument rather than
    interpolated into the shell script — avoiding quoting/injection
    pitfalls. Overwrites the file if it exists.

    Args:
        sandbox: a `Sandbox`/`SandboxSession`.
        timeout_seconds: per-call execution timeout forwarded to the
            sandbox. `None` uses the sandbox default.
        max_output_chars: cap on stderr echoed back on failure;
            non-positive disables truncation.
        encoding: encoding used to serialize `content` before writing.
        name: tool name surfaced to the agent.
        description: tool description; a sensible default is used when
            omitted.
    """
    structured_tool = _require_langchain()

    def write_file(path: str, content: str) -> str:
        result = sandbox.exec(
            ["sh", "-c", 'cat > "$1"', "sh", path],
            stdin=content.encode(encoding),
            timeout_seconds=timeout_seconds,
        )
        if result.exit_code != 0:
            return _format_exec_result(result, max_output_chars=max_output_chars)
        return f"Wrote {len(content)} characters to {path}"

    return structured_tool.from_function(
        func=write_file,
        name=name,
        description=(
            description
            or "Write text content to a file in the OpenShell sandbox, "
            "overwriting it if it already exists."
        ),
    )


def create_sandbox_tools(
    sandbox: SandboxExecutor,
    *,
    timeout_seconds: int | None = _DEFAULT_TIMEOUT_SECONDS,
    max_output_chars: int = _DEFAULT_MAX_OUTPUT_CHARS,
) -> list[BaseTool]:
    """Create the full set of OpenShell sandbox tools.

    Returns the Python-execution, shell-command, file-read, and
    file-write tools, each wired to `sandbox` and sharing the given
    safety knobs. Convenience wrapper over the individual factories for
    the common "give the agent everything" case.

    Args:
        sandbox: a `Sandbox`/`SandboxSession`.
        timeout_seconds: per-call execution timeout applied to every
            tool. `None` uses the sandbox default.
        max_output_chars: output cap applied to every tool; non-positive
            disables truncation.
    """
    return [
        create_python_tool(
            sandbox,
            timeout_seconds=timeout_seconds,
            max_output_chars=max_output_chars,
        ),
        create_shell_tool(
            sandbox,
            timeout_seconds=timeout_seconds,
            max_output_chars=max_output_chars,
        ),
        create_read_file_tool(
            sandbox,
            timeout_seconds=timeout_seconds,
            max_output_chars=max_output_chars,
        ),
        create_write_file_tool(
            sandbox,
            timeout_seconds=timeout_seconds,
            max_output_chars=max_output_chars,
        ),
    ]


__all__ = [
    "SandboxExecutor",
    "create_python_tool",
    "create_read_file_tool",
    "create_sandbox_tools",
    "create_shell_tool",
    "create_write_file_tool",
]
