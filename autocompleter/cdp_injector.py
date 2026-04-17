"""Chrome DevTools Protocol (CDP) injection for Chromium-based apps.

Uses the CDP to dispatch Input.insertText or Runtime.evaluate directly into
the JavaScript runtime of Chromium-based apps (Chrome, Edge, Arc, Brave,
Electron PWAs).  This avoids clipboard side effects and works reliably with
contenteditable elements and custom web components.

Requires the target app to be running with --remote-debugging-port enabled.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import threading
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import urllib.request
    import urllib.error

    HAS_URLLIB = True
except ImportError:
    HAS_URLLIB = False

try:
    import websocket as ws_module

    HAS_WEBSOCKET = True
except ImportError:
    ws_module = None  # type: ignore[assignment]
    HAS_WEBSOCKET = False


# Known Chromium-based application names (as reported by NSWorkspace).
# This covers browsers, Electron apps, and PWAs.
_CHROMIUM_APP_NAMES: frozenset[str] = frozenset({
    # Browsers
    "Google Chrome",
    "Google Chrome Canary",
    "Chromium",
    "Microsoft Edge",
    "Arc",
    "Brave Browser",
    "Vivaldi",
    "Opera",
    # Common Electron / PWA apps
    "Electron",
    "Visual Studio Code",
    "VS Code",
    "Code",
    "Slack",
    "Discord",
    "Notion",
    "Figma",
    "Spotify",
    "1Password",
    "Obsidian",
    "Cursor",
    # Google PWAs
    "Google Gemini",
    "ChatGPT",
    "Claude",
})

# Patterns that identify Chromium-based apps by partial name match
_CHROMIUM_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"chrome", re.IGNORECASE),
    re.compile(r"chromium", re.IGNORECASE),
    re.compile(r"electron", re.IGNORECASE),
    re.compile(r"edge", re.IGNORECASE),
]

# Common debug ports used by Chrome/Chromium and Node.js
_COMMON_DEBUG_PORTS: list[int] = [9222, 9229, 9223, 9224]


def _build_dom_probe_expression(
    *,
    candidate_limit: int = 8,
    preview_chars: int = 240,
) -> str:
    return f"""
(() => {{
  const CANDIDATE_LIMIT = {candidate_limit};
  const PREVIEW_CHARS = {preview_chars};

  function textValue(el) {{
    if (!el) return "";
    if (typeof el.value === "string") return el.value;
    if (el.isContentEditable) {{
      if (typeof el.innerText === "string" && el.innerText) return el.innerText;
      if (typeof el.textContent === "string") return el.textContent;
    }}
    return "";
  }}

  function summarize(el) {{
    if (!el) return null;
    const getAttr = el.getAttribute ? (name => el.getAttribute(name) || "") : (() => "");
    const value = textValue(el);
    return {{
      tag: (el.tagName || "").toLowerCase(),
      role: getAttr("role"),
      type: getAttr("type"),
      contenteditable: !!el.isContentEditable,
      contenteditable_attr: getAttr("contenteditable"),
      placeholder: getAttr("placeholder"),
      aria_label: getAttr("aria-label"),
      id: typeof el.id === "string" ? el.id : "",
      class_name: typeof el.className === "string" ? el.className.slice(0, 200) : "",
      value_length: value.length,
      value_preview: value.slice(0, PREVIEW_CHARS),
    }};
  }}

  const selectors = [
    "textarea",
    "input:not([type])",
    'input[type="text"]',
    'input[type="search"]',
    'input[type="email"]',
    'input[type="url"]',
    'input[type="tel"]',
    'input[type="number"]',
    "[contenteditable]",
    'div[role="textbox"]'
  ];

  const seen = new Set();
  const editableCandidates = [];
  for (const selector of selectors) {{
    for (const el of document.querySelectorAll(selector)) {{
      if (!el || seen.has(el)) continue;
      seen.add(el);
      const summary = summarize(el);
      if (summary) editableCandidates.push(summary);
      if (editableCandidates.length >= CANDIDATE_LIMIT) break;
    }}
    if (editableCandidates.length >= CANDIDATE_LIMIT) break;
  }}

  return {{
    active_element: summarize(document.activeElement),
    editable_candidates: editableCandidates,
  }};
}})()
""".strip()


def is_chromium_app(app_name: str) -> bool:
    """Check if the given application name belongs to a Chromium-based app.

    Returns True for known Chrome browsers, Electron apps, and PWAs.
    """
    if not app_name:
        return False

    # Exact match against known names
    if app_name in _CHROMIUM_APP_NAMES:
        return True

    # Pattern-based match for variants
    for pattern in _CHROMIUM_PATTERNS:
        if pattern.search(app_name):
            return True

    return False


def find_debug_port(pid: int) -> Optional[int]:
    """Try to discover the Chrome DevTools debug port for a given process.

    Checks the process command-line arguments for --remote-debugging-port,
    then probes common ports.

    Returns the port number or None if not found.
    """
    if pid <= 0:
        return None

    # Strategy 1: Read process args for --remote-debugging-port=NNNN
    port = _port_from_process_args(pid)
    if port is not None:
        return port

    # Strategy 2: Probe common ports to see if any respond
    for port in _COMMON_DEBUG_PORTS:
        if _probe_debug_port(port):
            return port

    return None


def _port_from_process_args(pid: int) -> Optional[int]:
    """Extract --remote-debugging-port from the process command line."""
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "args="],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0 and result.stdout.strip():
            args_str = result.stdout.strip()
            match = re.search(r"--remote-debugging-port=(\d+)", args_str)
            if match:
                port = int(match.group(1))
                logger.debug(f"Found debug port {port} in process args for PID {pid}")
                return port
    except (subprocess.TimeoutExpired, OSError, ValueError):
        logger.debug(f"Could not read process args for PID {pid}", exc_info=True)
    return None


def _probe_debug_port(port: int) -> bool:
    """Check if a CDP endpoint is responding on the given port."""
    if not HAS_URLLIB:
        return False
    try:
        req = urllib.request.Request(
            f"http://localhost:{port}/json/version",
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=1) as resp:
            data = json.loads(resp.read())
            return "Browser" in data or "webSocketDebuggerUrl" in data
    except Exception:
        return False


def probe_editable_dom_state(
    app_name: str,
    app_pid: int,
    *,
    candidate_limit: int = 8,
    preview_chars: int = 240,
) -> Dict[str, Any]:
    """Probe the active Chromium/Electron DOM for editable state."""
    result: Dict[str, Any] = {
        "app_name": app_name,
        "app_pid": app_pid,
    }
    if not app_name or not is_chromium_app(app_name):
        result["status"] = "no_debug_port"
        result["reason"] = "non_chromium_app"
        return result

    port = find_debug_port(app_pid)
    if port is None:
        result["status"] = "no_debug_port"
        return result

    result["port"] = port
    cdp = CDPConnection(port=port)
    try:
        targets = cdp.discover_targets()
        result["discovered_target_count"] = len(targets)
        target = cdp.find_active_target(targets)
        if target is None:
            result["status"] = "no_target"
            return result

        result["target_title"] = target.get("title", "")
        result["target_url"] = target.get("url", "")
        result["target_id"] = target.get("id", "")

        if not cdp.connect_to_target(target):
            result["status"] = "connect_failed"
            return result

        response = cdp.send_command(
            "Runtime.evaluate",
            {
                "expression": _build_dom_probe_expression(
                    candidate_limit=candidate_limit,
                    preview_chars=preview_chars,
                ),
                "returnByValue": True,
            },
        )
        if "error" in response:
            result["status"] = "js_failed"
            result["error"] = response["error"].get("message", "unknown")
            return result

        runtime_result = response.get("result", {})
        if runtime_result.get("exceptionDetails"):
            result["status"] = "js_failed"
            result["error"] = runtime_result["exceptionDetails"].get("text", "exception")
            return result

        value = runtime_result.get("result", {}).get("value")
        if not isinstance(value, dict):
            result["status"] = "js_failed"
            result["error"] = "missing_value"
            return result

        result["status"] = "success"
        result["active_element"] = value.get("active_element")
        result["editable_candidates"] = value.get("editable_candidates") or []
        return result
    finally:
        cdp.close()


class CDPConnection:
    """A connection to a Chromium process via the Chrome DevTools Protocol.

    Connects over WebSocket to send CDP commands and receive responses.
    """

    def __init__(self, port: int = 9222, timeout: float = 5.0):
        self.port = port
        self.timeout = timeout
        self._ws = None
        self._msg_id = 0
        self._lock = threading.Lock()

    def connect(self, ws_url: str) -> bool:
        """Connect to a specific WebSocket debugger URL.

        Returns True on success, False on failure.
        """
        if not HAS_WEBSOCKET:
            logger.warning("websocket-client not installed, CDP injection unavailable")
            return False

        try:
            self._ws = ws_module.create_connection(
                ws_url, timeout=self.timeout
            )
            logger.debug(f"CDP connected to {ws_url}")
            return True
        except Exception:
            logger.debug(f"CDP WebSocket connection failed: {ws_url}", exc_info=True)
            self._ws = None
            return False

    def close(self) -> None:
        """Close the WebSocket connection."""
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None

    def discover_targets(self) -> List[Dict[str, Any]]:
        """List available CDP targets (tabs/pages) via HTTP.

        Returns a list of target dicts from http://localhost:{port}/json.
        """
        if not HAS_URLLIB:
            return []

        try:
            req = urllib.request.Request(
                f"http://localhost:{self.port}/json",
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                targets = json.loads(resp.read())
                logger.debug(f"CDP discovered {len(targets)} targets on port {self.port}")
                return targets
        except Exception:
            logger.debug(
                f"CDP target discovery failed on port {self.port}",
                exc_info=True,
            )
            return []

    def find_active_target(
        self, targets: Optional[List[Dict[str, Any]]] = None
    ) -> Optional[Dict[str, Any]]:
        """Find the most likely active target (focused tab/page).

        Prefers targets of type 'page' that are not extension pages.
        If no targets are provided, calls discover_targets() first.

        Returns the target dict or None.
        """
        if targets is None:
            targets = self.discover_targets()

        if not targets:
            return None

        # Filter to page-type targets (not service workers, extensions, etc.)
        pages = [
            t for t in targets
            if t.get("type") == "page"
            and not t.get("url", "").startswith("chrome-extension://")
            and not t.get("url", "").startswith("devtools://")
        ]

        if not pages:
            # Fall back to any target with a webSocketDebuggerUrl,
            # still excluding extension and devtools URLs.
            pages = [
                t for t in targets
                if "webSocketDebuggerUrl" in t
                and not t.get("url", "").startswith("chrome-extension://")
                and not t.get("url", "").startswith("devtools://")
            ]

        if not pages:
            return None

        # Prefer the first page (Chrome lists the active tab first)
        return pages[0]

    def connect_to_target(self, target: Optional[Dict[str, Any]] = None) -> bool:
        """Discover targets and connect to the active one.

        If target is provided, connects directly to it.
        Returns True on success.
        """
        if target is None:
            target = self.find_active_target()

        if target is None:
            logger.debug("No active CDP target found")
            return False

        ws_url = target.get("webSocketDebuggerUrl")
        if not ws_url:
            logger.debug("Target has no webSocketDebuggerUrl")
            return False

        return self.connect(ws_url)

    def send_command(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Send a CDP command and wait for the response.

        Returns the response dict. On error, returns a dict with an 'error' key.
        """
        if self._ws is None:
            return {"error": {"message": "Not connected"}}

        with self._lock:
            self._msg_id += 1
            msg_id = self._msg_id

        message = {
            "id": msg_id,
            "method": method,
            "params": params or {},
        }

        try:
            self._ws.send(json.dumps(message))
            # Read responses until we get the one matching our ID
            while True:
                raw = self._ws.recv()
                response = json.loads(raw)
                if response.get("id") == msg_id:
                    return response
                # Skip events (no id field) — they are asynchronous notifications
        except Exception as exc:
            logger.debug(f"CDP command {method} failed: {exc}", exc_info=True)
            return {"error": {"message": str(exc)}}

    def insert_text(self, text: str) -> bool:
        """Insert text at the current cursor position using CDP Input.insertText.

        This dispatches the text through the browser's input handling pipeline,
        which triggers all JavaScript event listeners (input, change, etc.)
        without touching the system clipboard.

        Returns True if the command succeeded.
        """
        response = self.send_command("Input.insertText", {"text": text})
        if "error" in response:
            logger.debug(f"CDP Input.insertText failed: {response['error']}")
            return False
        return True

    def execute_js(self, expression: str) -> Any:
        """Execute a JavaScript expression via Runtime.evaluate.

        Returns the result value, or None on error.
        """
        response = self.send_command(
            "Runtime.evaluate",
            {
                "expression": expression,
                "returnByValue": True,
            },
        )

        if "error" in response:
            logger.debug(f"CDP Runtime.evaluate failed: {response['error']}")
            return None

        result = response.get("result", {}).get("result", {})
        if result.get("type") == "undefined":
            return None
        return result.get("value")

    def insert_text_via_js(self, text: str) -> bool:
        """Insert text using document.execCommand as a fallback.

        This is an alternative to Input.insertText that works in some
        cases where the CDP input method doesn't trigger the right events.

        Returns True if execCommand reported success.
        """
        # Escape the text for embedding in a JS string
        escaped = text.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n")
        result = self.execute_js(
            f"document.execCommand('insertText', false, '{escaped}')"
        )
        return result is True
