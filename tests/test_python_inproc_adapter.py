"""The in-process Python environment adapter (issue #1685).

``PythonInProcAdapter`` runs a snippet with the interpreter that is already
running CONTINUUM, so it is the environment adapter that works in CI with no
external service (#116). Every run is recorded through the ``ActionLedger``, so
it has the same recovery parity as the shell adapter.

The failure paths matter most here. A snippet that raises, or exits non-zero,
makes ``subprocess.run(check=True)`` raise, and ``run_action`` catches it so
the ledger attempt stays recorded and auditable. The caller sees a failed
``AdapterResult``, not an exception, and a failed attempt is exactly what a
resumed run needs to see in order to refuse to repeat the effect.
"""

from __future__ import annotations

from pathlib import Path

from continuum.actions.ledger import ActionLedger
from continuum.adapters.python_inproc import PythonInProcAdapter
from continuum.models import Run
from continuum.storage import SQLiteStorage


def make_storage() -> SQLiteStorage:
    storage = SQLiteStorage(":memory:")
    storage.create_run(Run(run_id="run_1", goal="inproc"))
    return storage


def make_adapter(tmp_path: Path) -> PythonInProcAdapter:
    return PythonInProcAdapter(make_storage(), str(tmp_path / "work"))


def records(adapter: PythonInProcAdapter) -> list[object]:
    return ActionLedger(adapter.storage, "run_1").all()


# --- workdir ---------------------------------------------------------------- #


def test_the_workdir_is_created_when_absent(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "work"
    assert not target.exists()

    adapter = PythonInProcAdapter(make_storage(), str(target))

    assert target.is_dir()
    assert adapter.workdir == target


def test_an_existing_workdir_is_reused(tmp_path: Path) -> None:
    target = tmp_path / "work"
    target.mkdir()
    (target / "keep-me.txt").write_text("already here", encoding="utf-8")

    PythonInProcAdapter(make_storage(), str(target))

    assert (target / "keep-me.txt").read_text(encoding="utf-8") == "already here"


def test_the_snippet_runs_in_the_workdir(tmp_path: Path) -> None:
    adapter = make_adapter(tmp_path)

    result = adapter.run_python("run_1", "import os; print(os.getcwd())")

    assert str(adapter.workdir) in result.output


def test_a_file_written_by_the_snippet_lands_in_the_workdir(tmp_path: Path) -> None:
    adapter = make_adapter(tmp_path)

    adapter.run_python("run_1", "open('written.txt', 'w').write('data')")

    assert (adapter.workdir / "written.txt").read_text(encoding="utf-8") == "data"


# --- the happy path --------------------------------------------------------- #


def test_stdout_comes_back_in_the_result(tmp_path: Path) -> None:
    adapter = make_adapter(tmp_path)

    result = adapter.run_python("run_1", "print('hello from inproc')")

    assert result.status == "completed"
    assert result.error is None
    assert "hello from inproc" in result.output


def test_a_silent_snippet_still_completes(tmp_path: Path) -> None:
    adapter = make_adapter(tmp_path)

    result = adapter.run_python("run_1", "pass")

    assert result.status == "completed"
    assert result.output == ""


def test_a_completed_run_is_recorded(tmp_path: Path) -> None:
    adapter = make_adapter(tmp_path)

    adapter.run_python("run_1", "pass")

    record = records(adapter)[0]
    assert record.action_type == "python"


# --- failure: an unhandled exception --------------------------------------- #


def test_an_unhandled_exception_is_captured_not_raised(tmp_path: Path) -> None:
    adapter = make_adapter(tmp_path)

    result = adapter.run_python("run_1", "raise ValueError('boom')")

    assert result.status == "failed"
    assert result.error


def test_a_failing_snippet_stays_recorded_in_the_ledger(tmp_path: Path) -> None:
    adapter = make_adapter(tmp_path)

    adapter.run_python("run_1", "raise ValueError('boom')")

    assert len(records(adapter)) == 1


def test_a_failure_does_not_block_the_next_snippet(tmp_path: Path) -> None:
    adapter = make_adapter(tmp_path)

    adapter.run_python("run_1", "raise ValueError('boom')")
    result = adapter.run_python("run_1", "print('after')")

    assert result.status == "completed"
    assert "after" in result.output


def test_a_snippet_that_writes_then_raises_is_recorded_as_failed(tmp_path: Path) -> None:
    """A partial effect before the raise must not read as a completed one."""
    adapter = make_adapter(tmp_path)

    result = adapter.run_python(
        "run_1", "open('partial.txt', 'w').write('half'); raise ValueError('boom')"
    )

    assert result.status == "failed"
    record = records(adapter)[0]
    assert record.status != "completed"


# --- failure: a non-zero exit ---------------------------------------------- #


def test_a_non_zero_exit_is_captured_not_raised(tmp_path: Path) -> None:
    adapter = make_adapter(tmp_path)

    result = adapter.run_python("run_1", "import sys; sys.exit(1)")

    assert result.status == "failed"
    assert result.error


def test_an_explicit_exit_code_is_captured(tmp_path: Path) -> None:
    adapter = make_adapter(tmp_path)

    result = adapter.run_python("run_1", "import sys; sys.exit(2)")

    assert result.status == "failed"


def test_a_zero_exit_after_a_warning_still_completes(tmp_path: Path) -> None:
    adapter = make_adapter(tmp_path)

    result = adapter.run_python("run_1", "import sys; print('warned'); sys.exit(0)")

    assert result.status == "completed"
    assert "warned" in result.output


# --- dep_scope -------------------------------------------------------------- #


def test_dep_scope_is_preserved_on_the_recorded_action(tmp_path: Path) -> None:
    adapter = make_adapter(tmp_path)

    adapter.run_python("run_1", "pass", dep_scope="numpy")

    assert records(adapter)[0].dep_scope == "numpy"


def test_dep_scope_defaults_to_unset(tmp_path: Path) -> None:
    adapter = make_adapter(tmp_path)

    adapter.run_python("run_1", "pass")

    assert records(adapter)[0].dep_scope is None


def test_a_scoped_failure_still_records_its_scope(tmp_path: Path) -> None:
    adapter = make_adapter(tmp_path)

    adapter.run_python("run_1", "raise ValueError('boom')", dep_scope="pandas")

    assert records(adapter)[0].dep_scope == "pandas"


# --- idempotency ------------------------------------------------------------ #


def test_the_same_snippet_is_a_replay_rather_than_a_second_run(tmp_path: Path) -> None:
    adapter = make_adapter(tmp_path)
    code = "open('once.txt', 'w').write('data')"

    adapter.run_python("run_1", code)
    adapter.run_python("run_1", code)

    assert len(records(adapter)) == 1


def test_different_snippets_are_different_actions(tmp_path: Path) -> None:
    adapter = make_adapter(tmp_path)

    adapter.run_python("run_1", "print('a')")
    adapter.run_python("run_1", "print('b')")

    assert len(records(adapter)) == 2
