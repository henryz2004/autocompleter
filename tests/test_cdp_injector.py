"""Tests for the CDP injector module."""

import json
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from autocompleter.cdp_injector import (
    CDPConnection,
    find_debug_port,
    is_chromium_app,
    probe_editable_dom_state,
    _port_from_process_args,
    _probe_debug_port,
)


# ---------------------------------------------------------------------------
# is_chromium_app
# ---------------------------------------------------------------------------

class TestIsChromiumApp:
    """Test detection of Chromium-based applications."""

    @pytest.mark.parametrize("app_name", [
        "Google Chrome",
        "Google Chrome Canary",
        "Chromium",
        "Microsoft Edge",
        "Arc",
        "Brave Browser",
        "Vivaldi",
        "Opera",
        "Google Gemini",
        "ChatGPT",
        "Claude",
        "Slack",
        "Discord",
        "Visual Studio Code",
        "Cursor",
        "Notion",
    ])
    def test_known_chromium_apps(self, app_name):
        assert is_chromium_app(app_name) is True

    @pytest.mark.parametrize("app_name", [
        "Safari",
        "TextEdit",
        "Notes",
        "Pages",
        "Terminal",
        "Finder",
        "Preview",
        "Firefox",
        "Mail",
        "Calendar",
    ])
    def test_non_chromium_apps(self, app_name):
        assert is_chromium_app(app_name) is False

    def test_empty_string(self):
        assert is_chromium_app("") is False

    def test_pattern_match_chrome_variant(self):
        """Apps with 'chrome' in the name should match via pattern."""
        assert is_chromium_app("My Chrome Browser") is True

    def test_pattern_match_electron_variant(self):
        """Apps with 'electron' in the name should match via pattern."""
        assert is_chromium_app("My Electron App") is True


# ---------------------------------------------------------------------------
# find_debug_port
# ---------------------------------------------------------------------------

class TestFindDebugPort:

    @patch("autocompleter.cdp_injector._port_from_process_args")
    def test_finds_port_from_process_args(self, mock_from_args):
        mock_from_args.return_value = 9222
        assert find_debug_port(12345) == 9222
        mock_from_args.assert_called_once_with(12345)

    @patch("autocompleter.cdp_injector._probe_debug_port")
    @patch("autocompleter.cdp_injector._port_from_process_args")
    def test_probes_common_ports_when_args_fail(self, mock_from_args, mock_probe):
        mock_from_args.return_value = None
        # First port probe fails, second succeeds
        mock_probe.side_effect = [False, True]
        result = find_debug_port(12345)
        assert result == 9229  # Second common port

    @patch("autocompleter.cdp_injector._probe_debug_port")
    @patch("autocompleter.cdp_injector._port_from_process_args")
    def test_returns_none_when_no_port_found(self, mock_from_args, mock_probe):
        mock_from_args.return_value = None
        mock_probe.return_value = False
        assert find_debug_port(12345) is None

    def test_invalid_pid_returns_none(self):
        assert find_debug_port(0) is None
        assert find_debug_port(-1) is None


class TestPortFromProcessArgs:

    @patch("subprocess.run")
    def test_extracts_port_from_args(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome "
                   "--remote-debugging-port=9222 --flag",
        )
        assert _port_from_process_args(12345) == 9222

    @patch("subprocess.run")
    def test_returns_none_when_no_port_flag(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome --no-sandbox",
        )
        assert _port_from_process_args(12345) is None

    @patch("subprocess.run")
    def test_returns_none_on_process_error(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        assert _port_from_process_args(12345) is None

    @patch("subprocess.run")
    def test_handles_timeout(self, mock_run):
        import subprocess
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="ps", timeout=2)
        assert _port_from_process_args(12345) is None


# ---------------------------------------------------------------------------
# CDPConnection — discover_targets
# ---------------------------------------------------------------------------

class TestDiscoverTargets:

    @patch("urllib.request.urlopen")
    def test_returns_targets_list(self, mock_urlopen):
        targets = [
            {
                "id": "abc123",
                "type": "page",
                "title": "Google",
                "url": "https://google.com",
                "webSocketDebuggerUrl": "ws://localhost:9222/devtools/page/abc123",
            },
            {
                "id": "def456",
                "type": "page",
                "title": "GitHub",
                "url": "https://github.com",
                "webSocketDebuggerUrl": "ws://localhost:9222/devtools/page/def456",
            },
        ]
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(targets).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        cdp = CDPConnection(port=9222)
        result = cdp.discover_targets()

        assert len(result) == 2
        assert result[0]["title"] == "Google"
        assert result[1]["url"] == "https://github.com"

    @patch("urllib.request.urlopen")
    def test_returns_empty_on_connection_error(self, mock_urlopen):
        mock_urlopen.side_effect = Exception("Connection refused")
        cdp = CDPConnection(port=9222)
        result = cdp.discover_targets()
        assert result == []


# ---------------------------------------------------------------------------
# CDPConnection — find_active_target
# ---------------------------------------------------------------------------

class TestFindActiveTarget:

    def test_finds_first_page_target(self):
        targets = [
            {
                "id": "sw1",
                "type": "service_worker",
                "url": "https://example.com/sw.js",
            },
            {
                "id": "page1",
                "type": "page",
                "url": "https://example.com",
                "webSocketDebuggerUrl": "ws://localhost:9222/devtools/page/page1",
            },
            {
                "id": "page2",
                "type": "page",
                "url": "https://other.com",
                "webSocketDebuggerUrl": "ws://localhost:9222/devtools/page/page2",
            },
        ]
        cdp = CDPConnection(port=9222)
        result = cdp.find_active_target(targets)
        assert result is not None
        assert result["id"] == "page1"

    def test_skips_extension_pages(self):
        targets = [
            {
                "id": "ext1",
                "type": "page",
                "url": "chrome-extension://abc123/popup.html",
                "webSocketDebuggerUrl": "ws://localhost:9222/devtools/page/ext1",
            },
            {
                "id": "page1",
                "type": "page",
                "url": "https://example.com",
                "webSocketDebuggerUrl": "ws://localhost:9222/devtools/page/page1",
            },
        ]
        cdp = CDPConnection(port=9222)
        result = cdp.find_active_target(targets)
        assert result is not None
        assert result["id"] == "page1"

    def test_skips_devtools_pages(self):
        """Devtools pages should be excluded from both page and fallback filters."""
        targets = [
            {
                "id": "dt1",
                "type": "page",
                "url": "devtools://devtools/bundled/inspector.html",
                "webSocketDebuggerUrl": "ws://localhost:9222/devtools/page/dt1",
            },
        ]
        cdp = CDPConnection(port=9222)
        result = cdp.find_active_target(targets)
        assert result is None

    def test_falls_back_to_target_with_ws_url(self):
        """When no page-type targets exist, fall back to any with a ws URL."""
        targets = [
            {
                "id": "other1",
                "type": "background_page",
                "url": "https://example.com",
                "webSocketDebuggerUrl": "ws://localhost:9222/devtools/page/other1",
            },
        ]
        cdp = CDPConnection(port=9222)
        result = cdp.find_active_target(targets)
        assert result is not None
        assert result["id"] == "other1"

    def test_returns_none_for_empty_targets(self):
        cdp = CDPConnection(port=9222)
        assert cdp.find_active_target([]) is None

    def test_returns_none_for_none_targets(self):
        """When discover_targets fails, find_active_target should return None."""
        cdp = CDPConnection(port=9222)
        with patch.object(cdp, "discover_targets", return_value=[]):
            assert cdp.find_active_target() is None


class TestProbeEditableDomState:
    @patch("autocompleter.cdp_injector.find_debug_port", return_value=None)
    def test_returns_no_debug_port_when_port_missing(self, mock_find_port):
        result = probe_editable_dom_state("Google Chrome", 1234)
        assert result["status"] == "no_debug_port"
        mock_find_port.assert_called_once_with(1234)

    @patch("autocompleter.cdp_injector.find_debug_port", return_value=9222)
    @patch("autocompleter.cdp_injector.CDPConnection")
    def test_returns_no_target_when_no_active_target(self, mock_cdp_cls, mock_find_port):
        mock_cdp = MagicMock()
        mock_cdp.discover_targets.return_value = []
        mock_cdp.find_active_target.return_value = None
        mock_cdp_cls.return_value = mock_cdp

        result = probe_editable_dom_state("Google Chrome", 1234)

        assert result["status"] == "no_target"
        mock_cdp.close.assert_called_once()

    @patch("autocompleter.cdp_injector.find_debug_port", return_value=9222)
    @patch("autocompleter.cdp_injector.CDPConnection")
    def test_returns_connect_failed_when_target_connection_fails(self, mock_cdp_cls, mock_find_port):
        target = {
            "id": "page-1",
            "title": "ChatGPT",
            "url": "https://chatgpt.com",
            "webSocketDebuggerUrl": "ws://localhost:9222/devtools/page/page-1",
        }
        mock_cdp = MagicMock()
        mock_cdp.discover_targets.return_value = [target]
        mock_cdp.find_active_target.return_value = target
        mock_cdp.connect_to_target.return_value = False
        mock_cdp_cls.return_value = mock_cdp

        result = probe_editable_dom_state("Google Chrome", 1234)

        assert result["status"] == "connect_failed"
        assert result["target_title"] == "ChatGPT"
        mock_cdp.close.assert_called_once()

    @patch("autocompleter.cdp_injector.find_debug_port", return_value=9222)
    @patch("autocompleter.cdp_injector.CDPConnection")
    def test_returns_js_failed_when_runtime_evaluate_errors(self, mock_cdp_cls, mock_find_port):
        target = {
            "id": "page-1",
            "title": "ChatGPT",
            "url": "https://chatgpt.com",
            "webSocketDebuggerUrl": "ws://localhost:9222/devtools/page/page-1",
        }
        mock_cdp = MagicMock()
        mock_cdp.discover_targets.return_value = [target]
        mock_cdp.find_active_target.return_value = target
        mock_cdp.connect_to_target.return_value = True
        mock_cdp.send_command.return_value = {"error": {"message": "boom"}}
        mock_cdp_cls.return_value = mock_cdp

        result = probe_editable_dom_state("Google Chrome", 1234)

        assert result["status"] == "js_failed"
        assert result["error"] == "boom"
        mock_cdp.close.assert_called_once()

    @patch("autocompleter.cdp_injector.find_debug_port", return_value=9222)
    @patch("autocompleter.cdp_injector.CDPConnection")
    def test_returns_success_with_active_element_and_candidates(self, mock_cdp_cls, mock_find_port):
        target = {
            "id": "page-1",
            "title": "ChatGPT",
            "url": "https://chatgpt.com",
            "webSocketDebuggerUrl": "ws://localhost:9222/devtools/page/page-1",
        }
        mock_cdp = MagicMock()
        mock_cdp.discover_targets.return_value = [target]
        mock_cdp.find_active_target.return_value = target
        mock_cdp.connect_to_target.return_value = True
        mock_cdp.send_command.return_value = {
            "result": {
                "result": {
                    "value": {
                        "active_element": {"tag": "textarea", "value_length": 12},
                        "editable_candidates": [{"tag": "textarea", "value_length": 12}],
                    }
                }
            }
        }
        mock_cdp_cls.return_value = mock_cdp

        result = probe_editable_dom_state("Google Chrome", 1234)

        assert result["status"] == "success"
        assert result["target_url"] == "https://chatgpt.com"
        assert result["active_element"]["tag"] == "textarea"
        assert result["editable_candidates"][0]["tag"] == "textarea"
        mock_cdp.close.assert_called_once()


# ---------------------------------------------------------------------------
# CDPConnection — send_command
# ---------------------------------------------------------------------------

class TestSendCommand:

    def test_sends_json_and_returns_response(self):
        mock_ws = MagicMock()
        mock_ws.recv.return_value = json.dumps({
            "id": 1,
            "result": {"success": True},
        })

        cdp = CDPConnection(port=9222)
        cdp._ws = mock_ws

        result = cdp.send_command("Input.insertText", {"text": "hello"})

        # Verify the sent message
        sent = json.loads(mock_ws.send.call_args[0][0])
        assert sent["method"] == "Input.insertText"
        assert sent["params"] == {"text": "hello"}
        assert sent["id"] == 1

        # Verify the response
        assert result["result"]["success"] is True

    def test_skips_event_messages(self):
        """Events (no id field) should be skipped until the matching response arrives."""
        mock_ws = MagicMock()
        mock_ws.recv.side_effect = [
            json.dumps({"method": "Page.loadEventFired", "params": {}}),
            json.dumps({"id": 1, "result": {}}),
        ]

        cdp = CDPConnection(port=9222)
        cdp._ws = mock_ws

        result = cdp.send_command("Page.enable")
        assert result["id"] == 1

    def test_returns_error_when_not_connected(self):
        cdp = CDPConnection(port=9222)
        cdp._ws = None
        result = cdp.send_command("Input.insertText", {"text": "test"})
        assert "error" in result
        assert "Not connected" in result["error"]["message"]

    def test_returns_error_on_ws_exception(self):
        mock_ws = MagicMock()
        mock_ws.send.side_effect = Exception("WebSocket broken")

        cdp = CDPConnection(port=9222)
        cdp._ws = mock_ws

        result = cdp.send_command("Input.insertText", {"text": "test"})
        assert "error" in result


# ---------------------------------------------------------------------------
# CDPConnection — insert_text
# ---------------------------------------------------------------------------

class TestInsertText:

    def test_successful_insertion(self):
        mock_ws = MagicMock()
        mock_ws.recv.return_value = json.dumps({
            "id": 1,
            "result": {},
        })

        cdp = CDPConnection(port=9222)
        cdp._ws = mock_ws

        assert cdp.insert_text("Hello world") is True

        sent = json.loads(mock_ws.send.call_args[0][0])
        assert sent["method"] == "Input.insertText"
        assert sent["params"]["text"] == "Hello world"

    def test_failed_insertion(self):
        mock_ws = MagicMock()
        mock_ws.recv.return_value = json.dumps({
            "id": 1,
            "error": {"code": -32000, "message": "Not focused"},
        })

        cdp = CDPConnection(port=9222)
        cdp._ws = mock_ws

        assert cdp.insert_text("test") is False

    def test_insertion_when_not_connected(self):
        cdp = CDPConnection(port=9222)
        cdp._ws = None
        assert cdp.insert_text("test") is False


# ---------------------------------------------------------------------------
# CDPConnection — execute_js
# ---------------------------------------------------------------------------

class TestExecuteJs:

    def test_successful_execution(self):
        mock_ws = MagicMock()
        mock_ws.recv.return_value = json.dumps({
            "id": 1,
            "result": {
                "result": {"type": "boolean", "value": True},
            },
        })

        cdp = CDPConnection(port=9222)
        cdp._ws = mock_ws

        result = cdp.execute_js("document.execCommand('insertText', false, 'hi')")
        assert result is True

        sent = json.loads(mock_ws.send.call_args[0][0])
        assert sent["method"] == "Runtime.evaluate"
        assert "execCommand" in sent["params"]["expression"]

    def test_returns_none_on_error(self):
        mock_ws = MagicMock()
        mock_ws.recv.return_value = json.dumps({
            "id": 1,
            "error": {"code": -32000, "message": "Execution context was destroyed"},
        })

        cdp = CDPConnection(port=9222)
        cdp._ws = mock_ws

        assert cdp.execute_js("1 + 1") is None

    def test_returns_none_for_undefined_result(self):
        mock_ws = MagicMock()
        mock_ws.recv.return_value = json.dumps({
            "id": 1,
            "result": {
                "result": {"type": "undefined"},
            },
        })

        cdp = CDPConnection(port=9222)
        cdp._ws = mock_ws

        assert cdp.execute_js("void 0") is None


# ---------------------------------------------------------------------------
# CDPConnection — insert_text_via_js
# ---------------------------------------------------------------------------

class TestInsertTextViaJs:

    def test_successful_js_insertion(self):
        mock_ws = MagicMock()
        mock_ws.recv.return_value = json.dumps({
            "id": 1,
            "result": {
                "result": {"type": "boolean", "value": True},
            },
        })

        cdp = CDPConnection(port=9222)
        cdp._ws = mock_ws

        assert cdp.insert_text_via_js("Hello world") is True

    def test_failed_js_insertion(self):
        mock_ws = MagicMock()
        mock_ws.recv.return_value = json.dumps({
            "id": 1,
            "result": {
                "result": {"type": "boolean", "value": False},
            },
        })

        cdp = CDPConnection(port=9222)
        cdp._ws = mock_ws

        assert cdp.insert_text_via_js("test") is False


# ---------------------------------------------------------------------------
# CDPConnection — connect / close
# ---------------------------------------------------------------------------

class TestConnection:

    @patch("autocompleter.cdp_injector.HAS_WEBSOCKET", True)
    @patch("autocompleter.cdp_injector.ws_module")
    def test_successful_connect(self, mock_ws_module):
        mock_ws_obj = MagicMock()
        mock_ws_module.create_connection.return_value = mock_ws_obj

        cdp = CDPConnection(port=9222)
        assert cdp.connect("ws://localhost:9222/devtools/page/abc") is True
        assert cdp._ws is mock_ws_obj

    @patch("autocompleter.cdp_injector.HAS_WEBSOCKET", True)
    @patch("autocompleter.cdp_injector.ws_module")
    def test_failed_connect(self, mock_ws_module):
        mock_ws_module.create_connection.side_effect = Exception("Connection refused")

        cdp = CDPConnection(port=9222)
        assert cdp.connect("ws://localhost:9222/devtools/page/abc") is False
        assert cdp._ws is None

    @patch("autocompleter.cdp_injector.HAS_WEBSOCKET", False)
    def test_connect_without_websocket_library(self):
        cdp = CDPConnection(port=9222)
        assert cdp.connect("ws://localhost:9222/devtools/page/abc") is False

    def test_close_when_connected(self):
        mock_ws = MagicMock()
        cdp = CDPConnection(port=9222)
        cdp._ws = mock_ws

        cdp.close()
        mock_ws.close.assert_called_once()
        assert cdp._ws is None

    def test_close_when_not_connected(self):
        cdp = CDPConnection(port=9222)
        cdp._ws = None
        cdp.close()  # Should not raise

    def test_close_handles_ws_exception(self):
        mock_ws = MagicMock()
        mock_ws.close.side_effect = Exception("Already closed")
        cdp = CDPConnection(port=9222)
        cdp._ws = mock_ws

        cdp.close()  # Should not raise
        assert cdp._ws is None


# ---------------------------------------------------------------------------
# CDPConnection — connect_to_target
# ---------------------------------------------------------------------------

class TestConnectToTarget:

    @patch.object(CDPConnection, "connect")
    @patch.object(CDPConnection, "find_active_target")
    def test_connects_to_discovered_target(self, mock_find, mock_connect):
        mock_find.return_value = {
            "id": "page1",
            "type": "page",
            "webSocketDebuggerUrl": "ws://localhost:9222/devtools/page/page1",
        }
        mock_connect.return_value = True

        cdp = CDPConnection(port=9222)
        assert cdp.connect_to_target() is True
        mock_connect.assert_called_once_with("ws://localhost:9222/devtools/page/page1")

    @patch.object(CDPConnection, "find_active_target")
    def test_returns_false_when_no_target(self, mock_find):
        mock_find.return_value = None
        cdp = CDPConnection(port=9222)
        assert cdp.connect_to_target() is False

    @patch.object(CDPConnection, "connect")
    def test_connects_to_provided_target(self, mock_connect):
        mock_connect.return_value = True
        target = {
            "id": "page1",
            "webSocketDebuggerUrl": "ws://localhost:9222/devtools/page/page1",
        }

        cdp = CDPConnection(port=9222)
        assert cdp.connect_to_target(target) is True

    @patch.object(CDPConnection, "connect")
    def test_returns_false_when_target_has_no_ws_url(self, mock_connect):
        target = {"id": "page1", "type": "page"}
        cdp = CDPConnection(port=9222)
        assert cdp.connect_to_target(target) is False
        mock_connect.assert_not_called()


# ---------------------------------------------------------------------------
# TextInjector — CDP integration in the strategy chain
# ---------------------------------------------------------------------------

class TestInjectorCDPIntegration:
    """Test that CDP is correctly integrated into TextInjector's strategy chain."""

    @patch("autocompleter.text_injector.TextInjector._inject_via_clipboard")
    @patch("autocompleter.text_injector.TextInjector._inject_via_cdp")
    @patch("autocompleter.text_injector.TextInjector._inject_via_ax")
    def test_cdp_tried_after_ax_fails(self, mock_ax, mock_cdp, mock_clipboard):
        """When AX fails, CDP should be attempted before clipboard."""
        from autocompleter.text_injector import TextInjector

        mock_ax.return_value = False
        mock_cdp.return_value = True
        mock_clipboard.return_value = True

        injector = TextInjector()
        result = injector.inject("hello", app_name="Google Chrome", app_pid=1234)

        assert result is True
        mock_ax.assert_called_once_with("hello", insertion_point=None)
        mock_cdp.assert_called_once_with("hello", app_name="Google Chrome", app_pid=1234)
        mock_clipboard.assert_not_called()

    @patch("autocompleter.text_injector.TextInjector._inject_via_clipboard")
    @patch("autocompleter.text_injector.TextInjector._inject_via_cdp")
    @patch("autocompleter.text_injector.TextInjector._inject_via_ax")
    def test_falls_through_to_clipboard_when_cdp_fails(self, mock_ax, mock_cdp, mock_clipboard):
        """When both AX and CDP fail, clipboard should be used."""
        from autocompleter.text_injector import TextInjector

        mock_ax.return_value = False
        mock_cdp.return_value = False
        mock_clipboard.return_value = True

        injector = TextInjector()
        result = injector.inject("hello", app_name="Google Chrome", app_pid=1234)

        assert result is True
        mock_cdp.assert_called_once()
        mock_clipboard.assert_called_once_with("hello")

    @patch("autocompleter.text_injector.TextInjector._inject_via_clipboard")
    @patch("autocompleter.text_injector.TextInjector._inject_via_cdp")
    @patch("autocompleter.text_injector.TextInjector._inject_via_ax")
    def test_non_chromium_app_skips_cdp(self, mock_ax, mock_cdp, mock_clipboard):
        """Non-Chromium apps should skip CDP entirely."""
        from autocompleter.text_injector import TextInjector

        mock_ax.return_value = False
        mock_cdp.return_value = False  # Will return False because is_chromium_app fails
        mock_clipboard.return_value = True

        injector = TextInjector()
        result = injector.inject("hello", app_name="Safari", app_pid=1234)

        assert result is True
        mock_cdp.assert_called_once_with("hello", app_name="Safari", app_pid=1234)
        mock_clipboard.assert_called_once()

    @patch("autocompleter.text_injector.TextInjector._inject_via_clipboard")
    @patch("autocompleter.text_injector.TextInjector._inject_via_cdp")
    @patch("autocompleter.text_injector.TextInjector._inject_via_ax")
    def test_replace_mode_skips_ax_tries_cdp(self, mock_ax, mock_cdp, mock_clipboard):
        """In replace mode, AX is skipped but CDP is still tried."""
        from autocompleter.text_injector import TextInjector

        mock_cdp.return_value = True

        injector = TextInjector()
        result = injector.inject(
            "hello", replace=True, app_name="Google Chrome", app_pid=1234,
        )

        assert result is True
        mock_ax.assert_not_called()
        mock_cdp.assert_called_once()

    @patch("autocompleter.text_injector.TextInjector._inject_via_keystrokes")
    @patch("autocompleter.text_injector.TextInjector._inject_via_clipboard")
    @patch("autocompleter.text_injector.TextInjector._inject_via_cdp")
    @patch("autocompleter.text_injector.TextInjector._inject_via_ax")
    def test_codex_prefers_keystrokes_before_clipboard(
        self, mock_ax, mock_cdp, mock_clipboard, mock_keys,
    ):
        """Codex should avoid AX rewrites and clipboard before keystrokes."""
        from autocompleter.text_injector import TextInjector

        mock_cdp.return_value = False
        mock_keys.return_value = True
        mock_clipboard.return_value = True

        injector = TextInjector()
        result = injector.inject("hello", app_name="Codex", app_pid=1234)

        assert result is True
        mock_ax.assert_not_called()
        mock_cdp.assert_called_once_with("hello", app_name="Codex", app_pid=1234)
        mock_keys.assert_called_once_with("hello")
        mock_clipboard.assert_not_called()

    @patch("autocompleter.text_injector.TextInjector._inject_via_clipboard")
    @patch("autocompleter.text_injector.TextInjector._inject_via_cdp")
    @patch("autocompleter.text_injector.TextInjector._inject_via_ax")
    def test_backward_compat_no_app_info(self, mock_ax, mock_cdp, mock_clipboard):
        """Calling inject() without app_name/app_pid should still work."""
        from autocompleter.text_injector import TextInjector

        mock_ax.return_value = False
        mock_cdp.return_value = False
        mock_clipboard.return_value = True

        injector = TextInjector()
        result = injector.inject("hello")

        assert result is True
        # CDP called with defaults (will return False since app_name is empty)
        mock_cdp.assert_called_once_with("hello", app_name="", app_pid=0)


# ---------------------------------------------------------------------------
# TextInjector._inject_via_cdp (unit-level)
# ---------------------------------------------------------------------------

class TestInjectViaCdpMethod:
    """Test the _inject_via_cdp method directly with mocks."""

    @patch("autocompleter.text_injector.find_debug_port")
    @patch("autocompleter.text_injector.is_chromium_app")
    def test_skips_non_chromium(self, mock_is_chromium, mock_find_port):
        from autocompleter.text_injector import TextInjector

        mock_is_chromium.return_value = False
        injector = TextInjector()
        assert injector._inject_via_cdp("hello", app_name="Safari", app_pid=100) is False
        mock_find_port.assert_not_called()

    @patch("autocompleter.text_injector.find_debug_port")
    @patch("autocompleter.text_injector.is_chromium_app")
    def test_skips_when_no_debug_port(self, mock_is_chromium, mock_find_port):
        from autocompleter.text_injector import TextInjector

        mock_is_chromium.return_value = True
        mock_find_port.return_value = None
        injector = TextInjector()
        assert injector._inject_via_cdp("hello", app_name="Chrome", app_pid=100) is False

    @patch("autocompleter.text_injector.CDPConnection")
    @patch("autocompleter.text_injector.find_debug_port")
    @patch("autocompleter.text_injector.is_chromium_app")
    def test_success_via_insert_text(self, mock_is_chromium, mock_find_port, mock_cdp_cls):
        from autocompleter.text_injector import TextInjector

        mock_is_chromium.return_value = True
        mock_find_port.return_value = 9222
        mock_cdp = MagicMock()
        mock_cdp.connect_to_target.return_value = True
        mock_cdp.insert_text.return_value = True
        mock_cdp_cls.return_value = mock_cdp

        injector = TextInjector()
        assert injector._inject_via_cdp("hello", app_name="Chrome", app_pid=100) is True
        mock_cdp.insert_text.assert_called_once_with("hello")
        mock_cdp.close.assert_called_once()

    @patch("autocompleter.text_injector.CDPConnection")
    @patch("autocompleter.text_injector.find_debug_port")
    @patch("autocompleter.text_injector.is_chromium_app")
    def test_falls_back_to_js_when_insert_text_fails(
        self, mock_is_chromium, mock_find_port, mock_cdp_cls,
    ):
        from autocompleter.text_injector import TextInjector

        mock_is_chromium.return_value = True
        mock_find_port.return_value = 9222
        mock_cdp = MagicMock()
        mock_cdp.connect_to_target.return_value = True
        mock_cdp.insert_text.return_value = False
        mock_cdp.insert_text_via_js.return_value = True
        mock_cdp_cls.return_value = mock_cdp

        injector = TextInjector()
        assert injector._inject_via_cdp("hello", app_name="Chrome", app_pid=100) is True
        mock_cdp.insert_text.assert_called_once()
        mock_cdp.insert_text_via_js.assert_called_once_with("hello")

    @patch("autocompleter.text_injector.CDPConnection")
    @patch("autocompleter.text_injector.find_debug_port")
    @patch("autocompleter.text_injector.is_chromium_app")
    def test_returns_false_when_both_cdp_methods_fail(
        self, mock_is_chromium, mock_find_port, mock_cdp_cls,
    ):
        from autocompleter.text_injector import TextInjector

        mock_is_chromium.return_value = True
        mock_find_port.return_value = 9222
        mock_cdp = MagicMock()
        mock_cdp.connect_to_target.return_value = True
        mock_cdp.insert_text.return_value = False
        mock_cdp.insert_text_via_js.return_value = False
        mock_cdp_cls.return_value = mock_cdp

        injector = TextInjector()
        assert injector._inject_via_cdp("hello", app_name="Chrome", app_pid=100) is False
        mock_cdp.close.assert_called_once()

    @patch("autocompleter.text_injector.CDPConnection")
    @patch("autocompleter.text_injector.find_debug_port")
    @patch("autocompleter.text_injector.is_chromium_app")
    def test_returns_false_when_connect_fails(
        self, mock_is_chromium, mock_find_port, mock_cdp_cls,
    ):
        from autocompleter.text_injector import TextInjector

        mock_is_chromium.return_value = True
        mock_find_port.return_value = 9222
        mock_cdp = MagicMock()
        mock_cdp.connect_to_target.return_value = False
        mock_cdp_cls.return_value = mock_cdp

        injector = TextInjector()
        assert injector._inject_via_cdp("hello", app_name="Chrome", app_pid=100) is False
        mock_cdp.close.assert_called_once()

    @patch("autocompleter.text_injector.CDPConnection")
    @patch("autocompleter.text_injector.find_debug_port")
    @patch("autocompleter.text_injector.is_chromium_app")
    def test_handles_exception_gracefully(
        self, mock_is_chromium, mock_find_port, mock_cdp_cls,
    ):
        from autocompleter.text_injector import TextInjector

        mock_is_chromium.return_value = True
        mock_find_port.return_value = 9222
        mock_cdp = MagicMock()
        mock_cdp.connect_to_target.side_effect = Exception("Unexpected error")
        mock_cdp_cls.return_value = mock_cdp

        injector = TextInjector()
        assert injector._inject_via_cdp("hello", app_name="Chrome", app_pid=100) is False
        mock_cdp.close.assert_called_once()

    def test_skips_when_empty_app_name(self):
        from autocompleter.text_injector import TextInjector

        injector = TextInjector()
        assert injector._inject_via_cdp("hello", app_name="", app_pid=100) is False
