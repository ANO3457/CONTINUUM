"""The filesystem sandbox adapter (issue #1684).

``FilesystemSandboxAdapter`` is the default adapter for docs and CI examples: no
docker, no browser, no cluster, just a sandbox directory and every shell step
recorded through the ``ActionLedger``.

Two things are pinned here beyond the happy path. The argv form exists because
a string command runs through the platform shell, so redirections, quoting, and
builtins written for one shell family fail on the other (#842); an argv list
runs without a shell and is portable. And a command that exits non-zero is
captured as a failed ``AdapterResult`` rather than raised, because the ledger
attempt has to stay recorded either way.

Commands that must fail use ``sys.executable`` so the tests behave the same on
Windows, where ``false`` and ``/bin/false`` do not exist.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from continuum.actions.ledger import ActionLedger
from continuum.adapters.filesystem import FilesystemSandboxAdapter
from continuum.models import Run
from continuum.storage import SQLiteStorage


def make_storage() -> SQLiteStorage:
    storage = SQLiteStorage(":memory:")
    storage.create_run(Run(run_id="run_1", goal="sandbox"))
    return storage


def make_adapter(tmp_path: Path) -> FilesystemSandboxAdapter:
    return FilesystemSandboxAdapter(make_storage(), str(tmp_path / "sandbox"))


def records(adapter: FilesystemSandboxAdapter) -> list[object]:
    return ActionLedger(adapter.storage, "run_1").all()


def shell_python(code: str) -> str:
    """A string command that runs ``code`` through the platform shell.

    The interpreter path is quoted because a checkout can live under a path
    with spaces, which is the same class of quoting problem #842 is about.
    """
    return f'"{sys.executable}" -c "{code}"'


# --- sandbox directory ------------------------------------------------------ #


def test_the_sandbox_directory_is_created_when_absent(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "sandbox"
    assert not target.exists()

    adapter = FilesystemSandboxAdapter(make_storage(), str(target))

    assert target.is_dir()
    assert adapter.sandbox_dir == target


def test_an_existing_sandbox_directory_is_reused(tmp_path: Path) -> None:
    target = tmp_path / "sandbox"
    target.mkdir()
    marker = target / "keep-me.txt"
    marker.write_text("already here", encoding="utf-8")

    FilesystemSandboxAdapter(make_storage(), str(target))

    assert marker.read_text(encoding="utf-8") == "already here"


def test_the_command_runs_in_the_sandbox_directory(tmp_path: Path) -> None:
    adapter = make_adapter(tmp_path)

    adapter.run_shell("run_1", [sys.executable, "-c", "import os; print(os.getcwd())"])

    recorded = records(adapter)[0].result["__return_value__"]
    assert str(adapter.sandbox_dir) in recorded


# --- the string form -------------------------------------------------------- #


def test_a_string_command_runs_through_the_shell(tmp_path: Path) -> None:
    adapter = make_adapter(tmp_path)

    result = adapter.run_shell("run_1", shell_python("print('hi')"))

    assert result.status == "completed"
    assert result.error is None
    assert "hi" in result.output


def test_a_string_command_is_recorded_as_a_shell_action(tmp_path: Path) -> None:
    adapter = make_adapter(tmp_path)

    adapter.run_shell("run_1", shell_python("pass"))

    record = records(adapter)[0]
    assert record.action_type == "shell"


# --- the argv form ---------------------------------------------------------- #


def test_an_argv_list_runs_without_a_shell(tmp_path: Path) -> None:
    adapter = make_adapter(tmp_path)

    result = adapter.run_shell("run_1", [sys.executable, "-c", "print('argv form')"])

    assert result.status == "completed"
    assert "argv form" in result.output


def test_an_argv_list_does_not_expand_a_metacharacter(tmp_path: Path) -> None:
    """The point of the argv form: no shell means no glob, no split, no subshell."""
    adapter = make_adapter(tmp_path)

    result = adapter.run_shell(
        "run_1", [sys.executable, "-c", "import sys; print(sys.argv[1])", "*"]
    )

    assert result.output.strip() == "*"


def test_an_argv_argument_with_spaces_stays_one_argument(tmp_path: Path) -> None:
    adapter = make_adapter(tmp_path)

    result = adapter.run_shell(
        "run_1", [sys.executable, "-c", "import sys; print(sys.argv[1])", "two words"]
    )

    assert result.output.strip() == "two words"


def test_stderr_does_not_leak_into_stdout(tmp_path: Path) -> None:
    adapter = make_adapter(tmp_path)

    result = adapter.run_shell(
        "run_1",
        [sys.executable, "-c", "import sys; sys.stderr.write('noise')"],
    )

    assert "noise" not in result.output


# --- the failure path ------------------------------------------------------- #


def test_a_non_zero_exit_is_captured_not_raised(tmp_path: Path) -> None:
    adapter = make_adapter(tmp_path)

    result = adapter.run_shell("run_1", [sys.executable, "-c", "import sys; sys.exit(1)"])

    assert result.status == "failed"
    assert result.error


def test_a_failing_string_command_is_captured(tmp_path: Path) -> None:
    adapter = make_adapter(tmp_path)

    result = adapter.run_shell("run_1", shell_python("import sys; sys.exit(2)"))

    assert result.status == "failed"
    assert result.error


def test_a_failing_command_is_still_recorded_in_the_ledger(tmp_path: Path) -> None:
    adapter = make_adapter(tmp_path)

    adapter.run_shell("run_1", [sys.executable, "-c", "import sys; sys.exit(1)"])

    assert len(records(adapter)) == 1


def test_a_failure_does_not_block_the_next_command(tmp_path: Path) -> None:
    adapter = make_adapter(tmp_path)

    adapter.run_shell("run_1", [sys.executable, "-c", "import sys; sys.exit(1)"])
    result = adapter.run_shell("run_1", [sys.executable, "-c", "print('after')"])

    assert result.status == "completed"
    assert "after" in result.output


def test_a_missing_executable_is_captured(tmp_path: Path) -> None:
    adapter = make_adapter(tmp_path)

    result = adapter.run_shell("run_1", ["definitely-not-a-real-binary-anywhere"])

    assert result.status == "failed"
    assert result.error


# --- dep_scope -------------------------------------------------------------- #


def test_dep_scope_is_preserved_on_the_recorded_action(tmp_path: Path) -> None:
    adapter = make_adapter(tmp_path)

    adapter.run_shell("run_1", [sys.executable, "-c", "pass"], dep_scope="pandas")

    assert records(adapter)[0].dep_scope == "pandas"


def test_dep_scope_defaults_to_unset(tmp_path: Path) -> None:
    adapter = make_adapter(tmp_path)

    adapter.run_shell("run_1", [sys.executable, "-c", "pass"])

    assert records(adapter)[0].dep_scope is None


# --- idempotency ------------------------------------------------------------ #


def test_the_same_argv_list_is_a_replay_rather_than_a_second_run(tmp_path: Path) -> None:
    adapter = make_adapter(tmp_path)
    command = [sys.executable, "-c", "print('once')"]

    first = adapter.run_shell("run_1", command)
    second = adapter.run_shell("run_1", command)

    assert first.output == second.output
    assert len(records(adapter)) == 1


@pytest.mark.skipif(os.name == "nt", reason="POSIX shell builtin")
def test_different_string_commands_are_different_actions(tmp_path: Path) -> None:
    adapter = make_adapter(tmp_path)

    adapter.run_shell("run_1", shell_python("print('a')"))
    adapter.run_shell("run_1", shell_python("print('b')"))

    assert len(records(adapter)) == 2
