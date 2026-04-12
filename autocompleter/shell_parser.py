"""Terminal buffer parser for shell-aware autocomplete.

Parses raw terminal buffer text (from AXTextArea) to extract the current
command being typed, the prompt string, and recent command history.

Also handles TUI detection (e.g. Claude Code) so the autocompleter can
extract the user's actual input from inside a TUI running in the terminal.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Regex patterns for common shell prompts.
# All anchored to start-of-line (^) to avoid matching inside program output.
# Order matters: more specific patterns first to avoid false matches.
_PROMPT_PATTERNS = [
    # zsh: user@host ~ %   or  user@host dir %
    re.compile(r"^(\S+@\S+\s+\S+\s+% )"),
    # zsh default: %
    re.compile(r"^(% )"),
    # bash: user@host:~/dir$   or  user@host dir $
    re.compile(r"^(\S+@\S+[:\s]\S*\$ )"),
    # bash default: $
    re.compile(r"^(\$ )"),
    # Custom prompts: ❯, →, #
    re.compile(r"^(.*?❯ )"),
    re.compile(r"^(.*?→ )"),
    re.compile(r"^(# )"),
]

# Fallback separators — ONLY for the current line (last line in the buffer).
# Too aggressive for history scanning since they match inside program output.
_FALLBACK_SEPARATORS = ("% ", "$ ", "> ", "# ")

# History limits
_MAX_RECENT_COMMANDS = 15
_MAX_OUTPUT_LINES_PER_CMD = 10
_MAX_HISTORY_CHARS = 1500

# Lines that are purely decorative UI chrome (Warp separators, etc.)
_CHROME_RE = re.compile(
    r"^[\s─━═┄┅┈┉╌╍▬▭⎯—–\-_=~·•·]{10,}$"  # rows of separator chars
    r"|^✻ "                                     # Warp "baked" status
    r"|^\s*⏵⏵ "                                 # Warp running-task bar
    r"|^\s*↓ to manage\s*$"                     # Warp manage hint
)

# Pattern for numbered suggestion lines (e.g. "  1. some text", "  2. it seems")
_NUMBERED_ITEM_RE = re.compile(r"^\s+\d+\.\s")

# Known command prefixes for the _looks_like_command heuristic.
_COMMAND_PREFIXES = {
    "git", "cd", "ls", "npm", "npx", "yarn", "pnpm", "docker", "python",
    "python3", "pip", "pip3", "node", "cargo", "go", "make", "cmake",
    "brew", "apt", "yum", "dnf", "pacman", "curl", "wget", "ssh", "scp",
    "rsync", "tar", "zip", "unzip", "cat", "head", "tail", "grep", "rg",
    "find", "sed", "awk", "sort", "wc", "mv", "cp", "rm", "mkdir", "rmdir",
    "chmod", "chown", "ln", "touch", "echo", "export", "source", "which",
    "man", "less", "more", "vim", "vi", "nano", "emacs", "code", "open",
    "pbcopy", "pbpaste", "xargs", "tee", "diff", "patch", "kill", "ps",
    "top", "htop", "df", "du", "env", "set", "unset", "alias", "fg", "bg",
    "jobs", "history", "sudo", "su", "systemctl", "journalctl", "kubectl",
    "helm", "terraform", "aws", "gcloud", "az", "ruby", "gem", "bundle",
    "rails", "java", "javac", "mvn", "gradle", "rustc", "ghc", "stack",
    "claude", "bun", "deno",
}

# Shell metacharacters that suggest a line is a command, not prose.
_SHELL_META_RE = re.compile(r"[|&]|--|\.\/|^/|[<>]|\$\(")

# Sentence-ending punctuation — prose lines tend to end with these.
_SENTENCE_END_RE = re.compile(r"[.?!]\s*$")


def _is_chrome_line(line: str) -> bool:
    """Return True if a line is terminal UI chrome rather than real content."""
    return bool(_CHROME_RE.match(line.strip()))


def _is_indented_or_numbered(line: str) -> bool:
    """Return True if line is indented or looks like a numbered suggestion.

    Real shell prompt lines start at column 0. TUI output and numbered
    suggestions (e.g. Claude Code's ``  2. it seems``) are indented.
    """
    if line and line[0] in (" ", "\t"):
        return True
    if _NUMBERED_ITEM_RE.match(line):
        return True
    return False


def _looks_like_command(text: str) -> bool:
    """Heuristic: does *text* look like a shell command (vs prose/code fragment)?

    A line is likely a command if it:
    - Starts with a known command prefix, OR
    - Contains shell metacharacters (|, &&, --, ./, /), OR
    - Is short (< 80 chars) and has no sentence-ending punctuation
    """
    text = text.strip()
    if not text:
        return False

    # Check for known command prefix
    first_word = text.split()[0] if text.split() else ""
    if first_word in _COMMAND_PREFIXES:
        return True

    # Check for shell metacharacters
    if _SHELL_META_RE.search(text):
        return True

    # Short line without sentence-ending punctuation
    if len(text) < 80 and not _SENTENCE_END_RE.search(text):
        return True

    return False


def _detect_prompt_strict(line: str) -> tuple[str, str]:
    """Try to detect a prompt using only regex patterns (no fallback).

    Used for scanning history lines — strict matching avoids false
    positives from ``$`` or ``>`` inside program output.

    Returns (prompt_string, command) if found, else ("", "").
    """
    for pattern in _PROMPT_PATTERNS:
        m = pattern.match(line)
        if m:
            prompt = m.group(1)
            return prompt, line[len(prompt):]
    return "", ""


def _detect_prompt(line: str) -> tuple[str, str]:
    """Try to detect a prompt, with fallback separators.

    Only used for the current (last) line where we want maximum recall.

    Returns (prompt_string, command) if found, else ("", "").
    """
    prompt, cmd = _detect_prompt_strict(line)
    if prompt:
        return prompt, cmd
    for sep in _FALLBACK_SEPARATORS:
        idx = line.rfind(sep)
        if idx >= 0:
            return line[: idx + len(sep)], line[idx + len(sep):]
    return "", ""


@dataclass
class ParsedTerminalBuffer:
    """Structured components extracted from a terminal buffer."""
    current_command: str       # Text after the last prompt marker (e.g. "git sta")
    prompt_string: str         # Detected prompt prefix (e.g. "user@host ~ % ")
    recent_commands: list[str] = field(default_factory=list)  # Previous commands extracted from buffer
    recent_output: str = ""    # Output from the most recent command(s)
    raw_current_line: str = "" # The full last line including prompt
    is_likely_shell_context: bool = False  # True when user appears to be at a shell prompt


def strip_tmux_split_panes(before_cursor: str) -> str:
    """Strip tmux vertical-split pane dividers from the terminal buffer.

    When tmux has a vertical split, macOS AX reports the full terminal
    buffer with both panes interleaved on every line, separated by
    ``│`` (U+2502 BOX DRAWINGS LIGHT VERTICAL).  For example::

        left pane text              │right pane text
        > user input                │$ shell prompt

    This function detects the pattern and extracts only the pane the
    cursor is most likely in (the one containing the last prompt or
    command text).  If no consistent column divider is found the text
    is returned unchanged.
    """
    if "│" not in before_cursor:
        return before_cursor

    lines = before_cursor.split("\n")

    # Detect consistent │ column position across recent lines
    _SAMPLE = 40  # check last N lines
    col_counts: dict[int, int] = {}
    sample_lines = lines[-_SAMPLE:] if len(lines) > _SAMPLE else lines
    for line in sample_lines:
        idx = line.find("│")
        if idx > 0:
            col_counts[idx] = col_counts.get(idx, 0) + 1

    if not col_counts:
        return before_cursor

    # The column divider should appear at the same position in most lines
    best_col, best_count = max(col_counts.items(), key=lambda kv: kv[1])
    if best_count < len(sample_lines) * 0.4:
        # Not enough consistency — probably not a tmux split
        return before_cursor

    # Split each line into left and right panes
    left_lines: list[str] = []
    right_lines: list[str] = []
    for line in lines:
        idx = line.find("│")
        if idx >= 0 and abs(idx - best_col) <= 2:
            left_lines.append(line[:idx].rstrip())
            right_lines.append(line[idx + 1:])
        else:
            # Lines without divider (e.g. tmux status bar) — keep in both
            left_lines.append(line)
            right_lines.append(line)

    # Decide which pane the cursor is in.
    # Heuristic: the pane where the last few lines have a shell prompt
    # or TUI markers is most likely the active pane.
    def _has_prompt_tail(pane_lines: list[str]) -> bool:
        for line in reversed(pane_lines[-10:]):
            stripped = line.strip()
            if not stripped:
                continue
            if _detect_prompt(stripped)[0]:
                return True
            break
        return False

    def _has_tui_markers(pane_lines: list[str]) -> bool:
        """Check whether the pane contains TUI markers (e.g. Claude Code)."""
        for line in pane_lines[-30:]:
            stripped = line.strip()
            if stripped.startswith(_CLAUDE_OUTPUT_MARKER):
                return True
            if _CLAUDE_HINT_RE.match(stripped):
                return True
        return False

    left_has = _has_prompt_tail(left_lines)
    right_has = _has_prompt_tail(right_lines)

    if right_has and not left_has:
        return "\n".join(right_lines)
    if left_has and not right_has:
        return "\n".join(left_lines)

    if not left_has and not right_has:
        # Neither has a shell prompt — check for TUI markers as a fallback
        left_tui = _has_tui_markers(left_lines)
        right_tui = _has_tui_markers(right_lines)
        if left_tui and not right_tui:
            return "\n".join(left_lines)
        if right_tui and not left_tui:
            return "\n".join(right_lines)

    # Both have prompts or no distinguishing signal — default to right pane
    # (where cursor position typically falls in the AX buffer ordering).
    return "\n".join(right_lines)


def parse_terminal_buffer(before_cursor: str) -> ParsedTerminalBuffer:
    """Parse a terminal's before_cursor text into structured components.

    Scans the full buffer for lines that look like command prompts to
    extract actual commands, rather than blindly taking the tail (which
    may be program output like a TUI).

    Args:
        before_cursor: The full AXTextArea value up to the cursor position.

    Returns:
        ParsedTerminalBuffer with extracted command, prompt, and history.
    """
    # Strip tmux split-pane dividers to isolate the active pane
    before_cursor = strip_tmux_split_panes(before_cursor)

    if not before_cursor:
        return ParsedTerminalBuffer(
            current_command="",
            prompt_string="",
        )

    # Split into lines; the last non-empty line contains the current prompt + command
    lines = before_cursor.split("\n")

    # Find the last non-empty/non-chrome line for current command detection.
    # Warp appends status/chrome lines after the actual prompt line.
    # Also skip indented lines and numbered suggestions — real prompts
    # start at column 0, while TUI output and Claude Code numbered
    # suggestions are indented.
    raw_current_line = ""
    history_end = len(lines) - 1
    for k in range(len(lines) - 1, -1, -1):
        stripped = lines[k].strip()
        if not stripped or _is_chrome_line(lines[k]):
            continue
        if _is_indented_or_numbered(lines[k]):
            continue
        raw_current_line = lines[k]
        history_end = k
        break

    history_lines = lines[:history_end] if history_end > 0 else []

    # --- Detect prompt on the current (last meaningful) line ---
    # Use lenient detection (with fallbacks) for the current line only
    prompt_string, current_command = _detect_prompt(raw_current_line)
    if not prompt_string:
        current_command = raw_current_line

    # --- Scan history for command lines ---
    # Walk backwards through history, finding lines with strict prompt matches.
    # For each command found, also grab a few lines of output after it.
    command_blocks: list[str] = []
    recent_commands: list[str] = []

    i = len(history_lines) - 1
    while i >= 0 and len(recent_commands) < _MAX_RECENT_COMMANDS:
        line = history_lines[i]

        # Skip chrome and empty lines
        if not line.strip() or _is_chrome_line(line):
            i -= 1
            continue

        # Strict matching only — no fallback separators for history
        prompt, cmd = _detect_prompt_strict(line)
        if prompt and cmd.strip():
            recent_commands.append(cmd.strip())
            # Collect output lines after this command
            output_lines = []
            for j in range(i + 1, min(i + 1 + _MAX_OUTPUT_LINES_PER_CMD, len(history_lines))):
                out_line = history_lines[j]
                if _detect_prompt_strict(out_line)[0] or _is_chrome_line(out_line):
                    break
                if out_line.strip():
                    output_lines.append(out_line)

            block = f"$ {cmd.strip()}"
            if output_lines:
                block += "\n" + "\n".join(output_lines)
            command_blocks.append(block)

        i -= 1

    # Reverse so they're in chronological order
    recent_commands.reverse()
    command_blocks.reverse()

    # Build recent_output: the command blocks, capped at _MAX_HISTORY_CHARS
    recent_output = ""
    if command_blocks:
        combined = "\n\n".join(command_blocks)
        if len(combined) > _MAX_HISTORY_CHARS:
            # Keep the most recent blocks that fit
            kept: list[str] = []
            total = 0
            for block in reversed(command_blocks):
                if total + len(block) + 2 > _MAX_HISTORY_CHARS:
                    break
                kept.append(block)
                total += len(block) + 2
            kept.reverse()
            combined = "\n\n".join(kept)
        recent_output = combined

    # Filter recent_commands to only include lines that look like commands
    filtered_commands = [c for c in recent_commands if _looks_like_command(c)]

    # Determine if this looks like a real shell context:
    # - A shell prompt was detected on the current line, AND
    # - At least 1 of the recent commands passes the command heuristic
    #   (or there are no history commands but a prompt was detected)
    is_likely_shell = bool(prompt_string) and (
        len(filtered_commands) > 0 or len(recent_commands) == 0
    )

    return ParsedTerminalBuffer(
        current_command=current_command,
        prompt_string=prompt_string,
        recent_commands=filtered_commands,
        recent_output=recent_output,
        raw_current_line=raw_current_line,
        is_likely_shell_context=is_likely_shell,
    )


# ---------------------------------------------------------------------------
# TUI detection (Claude Code, etc.)
# ---------------------------------------------------------------------------

# Patterns used to detect TUI apps running inside a terminal.
# Detection is intentionally loose — we just need to distinguish "inside a
# TUI" from "at a shell prompt" so the right code path fires.  The raw
# buffer text (with its visual markers) is passed through to the LLM as-is
# rather than being parsed into structured conversation turns.

# Separator lines used by Claude Code between turns
_TUI_SEPARATOR_RE = re.compile(r"^[─━═]{20,}$")

# Claude Code output markers
_CLAUDE_OUTPUT_MARKER = "⏺"

# Claude Code hint bar — multiple versions across Claude Code releases
_CLAUDE_HINT_RE = re.compile(
    r"^\s*⏵⏵\s"             # v2.1.63-style: "⏵⏵ accept edits on ..."
    r"|^\s*\?\s+for\s"       # v2.1.89-style: "? for shortcuts"
)

# Claude Code welcome banner (stable across versions)
_CLAUDE_BANNER_RE = re.compile(r"Claude Code v\d")

# Claude Code box-drawing welcome frame
_CLAUDE_BOX_RE = re.compile(r"^[╭╰][─━═].*Claude Code")


def detect_tui(before_cursor: str, after_cursor: str = "") -> bool:
    """Detect whether a TUI app (e.g. Claude Code) is running in the terminal.

    This is a lightweight check — it only answers "is this a TUI?" so the
    caller can avoid misinterpreting the buffer as a shell session.  No
    attempt is made to parse conversation structure; the raw buffer text
    is passed to the LLM which can interpret the visual markers itself.

    Detection uses multiple signals, any sufficient combination triggers:
    - ``⏺`` output markers (present once Claude has responded)
    - ``─────`` separator lines (20+ box-drawing dashes)
    - Hint bar (``⏵⏵ ...`` or ``? for shortcuts``)
    - Welcome banner (``Claude Code v...``)

    Args:
        before_cursor: Terminal buffer text up to the cursor position.
        after_cursor: Terminal buffer text after the cursor position.

    Returns:
        True if a TUI app was detected.
    """
    if not before_cursor:
        return False

    # Strip tmux split-pane dividers so TUI markers aren't interleaved
    # with content from an adjacent pane.
    before_cursor = strip_tmux_split_panes(before_cursor)

    lines = before_cursor.split("\n")

    has_output_marker = False
    separator_count = 0
    has_hint_bar = False
    has_banner = False
    for line in lines[-200:]:  # scan recent portion
        stripped = line.strip()
        if stripped.startswith(_CLAUDE_OUTPUT_MARKER):
            has_output_marker = True
        if _TUI_SEPARATOR_RE.match(stripped):
            separator_count += 1
        if _CLAUDE_HINT_RE.match(stripped):
            has_hint_bar = True
        if _CLAUDE_BANNER_RE.search(line) or _CLAUDE_BOX_RE.match(stripped):
            has_banner = True
        if has_output_marker and separator_count >= 2:
            break

    # Also check after_cursor for hint bar and separators
    if after_cursor:
        for line in after_cursor.split("\n")[:20]:
            stripped = line.strip()
            if _CLAUDE_HINT_RE.match(stripped):
                has_hint_bar = True
            if _TUI_SEPARATOR_RE.match(stripped):
                separator_count += 1

    # Detection rules (any match triggers):
    # 1. Output marker + ≥2 separators (active conversation)
    # 2. Hint bar + at least 1 output marker or separator
    # 3. Welcome banner + at least 1 separator (fresh session, no response yet)
    return (
        (has_output_marker and separator_count >= 2)
        or (has_hint_bar and (has_output_marker or separator_count >= 1))
        or (has_banner and separator_count >= 1)
    )


# Keep ParsedTUIBuffer and parse_tui_buffer as deprecated aliases so that
# existing tests and call sites still compile.  They will be removed once
# all callers are migrated to detect_tui().

@dataclass
class ParsedTUIBuffer:
    """Structured components extracted from a TUI buffer (e.g. Claude Code).

    .. deprecated::
        Use :func:`detect_tui` instead.  The raw terminal buffer is now
        passed to the LLM without structured turn extraction.
    """
    is_tui: bool = False
    tui_name: str = ""
    user_input: str = ""
    conversation_context: str = ""
    conversation_turns: list[dict[str, str]] = field(default_factory=list)


def parse_tui_buffer(before_cursor: str, after_cursor: str = "") -> ParsedTUIBuffer:
    """Detect a TUI app and return a minimal result.

    .. deprecated::
        Use :func:`detect_tui` instead.
    """
    if detect_tui(before_cursor, after_cursor):
        return ParsedTUIBuffer(is_tui=True, tui_name="claude_code")
    return ParsedTUIBuffer()
