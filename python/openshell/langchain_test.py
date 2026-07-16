# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any

import pytest

from openshell import langchain as lc
from openshell.sandbox import ExecResult


class _FakeSandbox:
    """Records `exec` calls and returns a scripted `ExecResult`."""

    def __init__(self, result: ExecResult | None = None) -> None:
        self.calls: list[SimpleNamespace] = []
        self._result = result or ExecResult(exit_code=0, stdout="ok\n", stderr="")

    def exec(
        self,
        command: Any,
        *,
        stream_output: bool = False,
        workdir: str | None = None,
        env: Any = None,
        stdin: bytes | None = None,
        timeout_seconds: int | None = None,
    ) -> ExecResult:
        _ = (stream_output, workdir, env)
        self.calls.append(
            SimpleNamespace(
                command=list(command),
                stdin=stdin,
                timeout_seconds=timeout_seconds,
            )
        )
        return self._result


# ---------------------------------------------------------------------------
# Lazy-import contract: the module imports without langchain, and the error
# only surfaces when a factory is actually called.
# ---------------------------------------------------------------------------


def test_module_exposes_factories_without_importing_langchain() -> None:
    # Importing openshell.langchain (done at file top) must not require
    # langchain to be installed; the public factories are present.
    for factory in (
        "create_python_tool",
        "create_shell_tool",
        "create_read_file_tool",
        "create_write_file_tool",
        "create_sandbox_tools",
    ):
        assert hasattr(lc, factory)


def test_factory_raises_clear_error_when_langchain_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When langchain-core cannot be imported, calling a factory raises a
    clear ImportError pointing at the optional extra."""
    real_import = importlib.import_module

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.startswith("langchain"):
            raise ImportError("No module named 'langchain_core'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", fake_import)

    with pytest.raises(ImportError, match=r"openshell\[langchain\]"):
        lc.create_python_tool(_FakeSandbox())


# ---------------------------------------------------------------------------
# Tool behavior. These require langchain-core; skip cleanly when absent.
# ---------------------------------------------------------------------------


def test_python_tool_executes_code_and_formats_output() -> None:
    pytest.importorskip("langchain_core")
    sandbox = _FakeSandbox(ExecResult(exit_code=0, stdout="42\n", stderr=""))
    tool = lc.create_python_tool(sandbox, timeout_seconds=15)

    out = tool.invoke({"code": "print(6 * 7)"})

    assert sandbox.calls[0].command == ["python", "-c", "print(6 * 7)"]
    assert sandbox.calls[0].timeout_seconds == 15
    assert "exit_code: 0" in out
    assert "42" in out


def test_python_tool_honors_custom_interpreter() -> None:
    pytest.importorskip("langchain_core")
    sandbox = _FakeSandbox()
    tool = lc.create_python_tool(sandbox, python_bin="python3.12")

    tool.invoke({"code": "pass"})

    assert sandbox.calls[0].command[0] == "python3.12"


def test_shell_tool_runs_command() -> None:
    pytest.importorskip("langchain_core")
    sandbox = _FakeSandbox(ExecResult(exit_code=0, stdout="hello\n", stderr=""))
    tool = lc.create_shell_tool(sandbox, timeout_seconds=30)

    out = tool.invoke({"command": "echo hello"})

    assert sandbox.calls[0].command == ["sh", "-c", "echo hello"]
    assert sandbox.calls[0].timeout_seconds == 30
    assert "hello" in out


def test_shell_tool_honors_custom_shell() -> None:
    pytest.importorskip("langchain_core")
    sandbox = _FakeSandbox()
    tool = lc.create_shell_tool(sandbox, shell=("bash", "-lc"))

    tool.invoke({"command": "true"})

    assert sandbox.calls[0].command == ["bash", "-lc", "true"]


def test_read_file_tool_returns_contents() -> None:
    pytest.importorskip("langchain_core")
    sandbox = _FakeSandbox(ExecResult(exit_code=0, stdout="file body", stderr=""))
    tool = lc.create_read_file_tool(sandbox)

    out = tool.invoke({"path": "/etc/hostname"})

    assert sandbox.calls[0].command == ["cat", "--", "/etc/hostname"]
    assert out == "file body"


def test_read_file_tool_reports_failure() -> None:
    pytest.importorskip("langchain_core")
    sandbox = _FakeSandbox(
        ExecResult(exit_code=1, stdout="", stderr="cat: /nope: No such file")
    )
    tool = lc.create_read_file_tool(sandbox)

    out = tool.invoke({"path": "/nope"})

    assert "exit_code: 1" in out
    assert "No such file" in out


def test_write_file_tool_streams_content_via_stdin() -> None:
    pytest.importorskip("langchain_core")
    sandbox = _FakeSandbox(ExecResult(exit_code=0, stdout="", stderr=""))
    tool = lc.create_write_file_tool(sandbox)

    out = tool.invoke({"path": "/tmp/out.txt", "content": "hi there"})

    assert sandbox.calls[0].command == ["sh", "-c", 'cat > "$1"', "sh", "/tmp/out.txt"]
    assert sandbox.calls[0].stdin == b"hi there"
    assert "8 characters" in out
    assert "/tmp/out.txt" in out


def test_write_file_tool_reports_failure() -> None:
    pytest.importorskip("langchain_core")
    sandbox = _FakeSandbox(
        ExecResult(exit_code=2, stdout="", stderr="cannot create: Permission denied")
    )
    tool = lc.create_write_file_tool(sandbox)

    out = tool.invoke({"path": "/root/x", "content": "data"})

    assert "exit_code: 2" in out
    assert "Permission denied" in out


def test_output_is_truncated_to_cap() -> None:
    pytest.importorskip("langchain_core")
    big = "x" * 5000
    sandbox = _FakeSandbox(ExecResult(exit_code=0, stdout=big, stderr=""))
    tool = lc.create_shell_tool(sandbox, max_output_chars=100)

    out = tool.invoke({"command": "cat big"})

    assert "truncated 4900 characters" in out
    # The capped stdout section carries only the first 100 chars of payload.
    assert "x" * 100 in out
    assert "x" * 101 not in out


def test_create_sandbox_tools_returns_all_factories() -> None:
    pytest.importorskip("langchain_core")
    sandbox = _FakeSandbox()

    tools = lc.create_sandbox_tools(sandbox, timeout_seconds=42)

    names = [tool.name for tool in tools]
    assert names == [
        "openshell_run_python",
        "openshell_run_shell",
        "openshell_read_file",
        "openshell_write_file",
    ]


def test_tools_carry_descriptions() -> None:
    pytest.importorskip("langchain_core")
    sandbox = _FakeSandbox()

    for tool in lc.create_sandbox_tools(sandbox):
        assert tool.description
