import sys
from unittest.mock import MagicMock, patch

import pytest

import continuum.adapters.browser as browser_mod
from continuum.adapters.browser import BrowserAdapter


@pytest.fixture
def storage():
    mock_store = MagicMock()
    mock_store.record_action_start.return_value = "action_1"
    return mock_store


@pytest.fixture
def run_id():
    return "run_1"


def test_browser_adapter_available_flag(storage):
    adapter = BrowserAdapter(storage)

    with patch.dict(sys.modules, {"playwright": None, "playwright.sync_api": None}):
        assert adapter.available() is False

    mock_pw = MagicMock()
    with patch.dict(sys.modules, {"playwright": mock_pw, "playwright.sync_api": mock_pw}):
        assert adapter.available() is True


def test_browser_adapter_navigate_without_playwright(storage, run_id):
    adapter = BrowserAdapter(storage)
    with patch.object(adapter, "available", return_value=False), pytest.raises(RuntimeError):
        adapter.navigate(run_id, "https://example.com")


def test_browser_adapter_navigate_success(storage, run_id):
    adapter = BrowserAdapter(storage)

    fake_result = MagicMock()
    fake_result.status = "completed"
    fake_result.output = "<html><body>Hello World</body></html>"

    with (
        patch.object(adapter, "available", return_value=True),
        patch.object(browser_mod, "run_action", return_value=fake_result) as mock_run_action,
    ):
        result = adapter.navigate(run_id, "https://example.com")

        assert result.status == "completed"
        assert "Hello World" in result.output
        mock_run_action.assert_called_once()


def test_browser_adapter_navigate_failure_captured(storage, run_id):
    adapter = BrowserAdapter(storage)

    fake_result = MagicMock()
    fake_result.status = "failed"
    fake_result.error = "Connection timed out"

    with (
        patch.object(adapter, "available", return_value=True),
        patch.object(browser_mod, "run_action", return_value=fake_result),
    ):
        result = adapter.navigate(run_id, "https://invalid-url.fake")

        assert result.status == "failed"
        assert "Connection timed out" in str(result.error)


def test_browser_adapter_dep_scope_forwarding(storage, run_id):
    adapter = BrowserAdapter(storage)

    with (
        patch.object(adapter, "available", return_value=True),
        patch.object(browser_mod, "run_action") as mock_run_action,
    ):
        adapter.navigate(run_id, "https://example.com", dep_scope="custom_scope")

        mock_run_action.assert_called_once()
        action_arg = mock_run_action.call_args[0][2]
        assert action_arg.dep_scope == "custom_scope"
