# Refinement Roadmap

## Phase 1: Reliability & Polish — DONE

1. ~~**Crash resilience** — 10s API timeout, try/except around LLM calls, clipboard try/finally restoration~~
2. ~~**Overlay bounds checking** — `_clamp_to_screen()` handles multi-monitor, flips above caret, shifts from edges~~
3. ~~**Loading indicator** — "Generating..." overlay shown immediately on Ctrl+Space~~
4. ~~**Shared AX utility module** — `ax_utils.py` extracted with get/set attribute, position, size, PID helpers~~

## Phase 2: Context Quality — PARTIALLY DONE

5. ~~**Context deduplication** — 5-second window dedup in `add_entry`; live `visible_text` bypass avoids stale DB~~
6. ~~**Filter UI chrome noise** — `_SKIP_ROLES` / `_CONTENT_ROLES` filtering in `_collect_text`~~
7. **Timestamp/recency signal** — Include relative timestamps in context so LLM can weight recent vs old
8. **Conversation boundaries** — Detect app switches and insert boundaries so LLM doesn't conflate sessions

## Phase 3: Smarter Suggestions

9. **App-specific prompts** — Tailor system prompt by target app (code editor vs chat vs email vs terminal)
10. **Structured LLM output** — Switch from delimiter parsing to JSON for reliability + confidence scores
11. **Suggestion caching** — Cache recent suggestions keyed by (input_prefix, app) to avoid redundant API calls
12. **Adaptive debounce** — Shorter debounce on manual hotkey trigger vs auto-trigger on typing pause

## Phase 4: UX Enhancements

13. **Click-away dismissal** — Hide overlay on clicks outside it (global mouse event monitor)
14. **Overlay animation** — Fade-in/out transitions
15. **Suggestion regeneration** — Hotkey to dismiss current batch and generate new suggestions
16. **Undo support** — Track last injection, support Cmd+Z to revert
17. **Menu bar agent** — System tray icon for status, settings, context history, pause/resume

## Phase 5: Architecture

18. **Event-driven context collection** — Replace 2s polling with AX notifications for content changes (lower CPU/battery)
19. **Semantic context ranking** — Use embeddings to surface relevant historical context, not just recency
20. **Plugin system** — App-specific adapters for better text extraction (VS Code, Slack, etc.)
21. **Integration tests** — Mock AX API and test end-to-end flow including thread interactions

---

## Next 10: High-Impact Improvement Directions

### N1. Cursor-aware injection
`_inject_via_ax` always appends to end of field (`current_value + text`), ignoring `insertion_point`. Mid-field completions land at the wrong position. Fix: splice text at `insertion_point` for continuation mode. Clipboard/keystroke fallbacks already inject at cursor naturally.

### N2. Auto-trigger on typing pause
`has_typing_paused()` exists in `InputObserver` but is never called. Wire it up: after a configurable pause (e.g. 800ms), automatically trigger suggestion generation. Use a longer debounce than manual hotkey to avoid interrupting fast typing. Show suggestions with a softer visual style (dimmer, smaller) to distinguish auto-triggered from manually requested.

### N3. Streaming suggestions
Currently waits for the full LLM response before showing anything. Stream tokens and display the first suggestion as soon as it's complete (hit the `---SUGGESTION---` delimiter). Perceived latency drops from 2-3s to <1s for the first suggestion. Anthropic's streaming API is already supported by the SDK.

### N4. App-specific conversation extractors
The generic `_extract_conversation_turns` heuristic (short child = speaker, long child = body) misses many chat UIs. Build lightweight extractors for known apps (Gemini, ChatGPT, Claude Desktop, Slack, Discord, iMessage) that understand their specific AX tree structure. Register them in a dict keyed by app name, falling back to the generic heuristic.

### N5. Local/on-device model support
Add an `ollama` provider for privacy-sensitive use or offline scenarios. The API shape is similar to OpenAI's. Would need: provider routing in `suggestion_engine.py`, configurable endpoint URL, and potentially a longer timeout since local models are slower. Good for code completion where sending proprietary code to a cloud API is a concern.

### N6. Suggestion quality feedback loop
Track accept/dismiss/regenerate events per suggestion. Use this signal to:
- Adjust temperature dynamically (lower if user keeps dismissing creative suggestions)
- Build a per-app preference profile (e.g. user prefers shorter suggestions in Slack, longer in email)
- Filter out patterns similar to recently dismissed suggestions
Store events in the existing SQLite DB with a new `suggestion_feedback` table.

### N7. Multi-line and block suggestions
Current overlay shows single-line/sentence suggestions. For email drafts, code blocks, and long-form writing, support multi-paragraph suggestions with:
- A taller, scrollable overlay panel
- Preview mode (show first line, expand on hover/arrow)
- A "partial accept" mechanism (accept first sentence only via Shift+Tab)

### N8. Smart context windowing with embeddings
Replace pure recency-based context selection with lightweight semantic search. Embed recent context entries (could use a small local model or API call) and retrieve entries most relevant to the current input. A user writing about "React hooks" should see their React docs from 30min ago, not their most recent unrelated Slack message. Cache embeddings in the SQLite DB.

### N9. Rich injection for web apps
Current injection either sets AXValue (bypasses JS) or pastes (works but clobbers clipboard). A third strategy: use the Chrome DevTools Protocol (CDP) to dispatch `Input.insertText` events directly into the web page's JS runtime. This would work perfectly for all Chromium-based apps (PWAs, Electron, browsers) without clipboard side effects. Requires connecting to Chrome's debug port.

### N10. Cross-app context linking
When the user switches between apps (e.g. reading docs in Chrome, then writing code in VS Code), link the context across apps so the LLM knows what the user was just reading. Currently each trigger captures only the current app's visible content. Store a "context trail" of the last N app switches with timestamps, and include the most recent cross-app context when it's fresh (<30s old).
