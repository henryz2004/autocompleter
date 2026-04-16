# Fixture Workflow

This doc is the working guide for capturing, analyzing, and benchmarking Accessibility fixtures while building or revising app-specific extractors.

There are two related but different artifact types:

- raw AX tree fixtures: best for extractor design
- trigger/replay dumps: best for end-to-end replay and benchmarking

## Quick Start

From the repo root:

```bash
cd /Users/henryz2004/code/orchestrator/autocompleter
source venv/bin/activate
```

To capture raw AX fixtures:

```bash
python dump_ax_tree_json.py -d tests/fixtures/ax_trees --depth 35 --notes "what was on screen"
```

To analyze one fixture:

```bash
python analyze_fixture.py tests/fixtures/ax_trees/claude-current-chat.json
```

To replay one fixture through the current offline pipeline:

```bash
python replay_fixture.py tests/fixtures/ax_trees/claude-current-chat.json
```

To capture full trigger artifacts from live app usage:

```bash
python -m autocompleter --dump-dir dumps
```

## When To Use Which Capture Path

Use `dump_ax_tree_json.py` when you want:

- the raw Accessibility tree
- the focused element snapshot
- a clean input for extractor design
- stable offline analysis of app structure

Use `python -m autocompleter --dump-dir dumps` when you want:

- the full trigger artifact
- extracted conversation turns
- assembled prompt context
- model request inputs and suggestions
- end-to-end replay/backtesting

Recommended rule:

- redesign extractor logic from raw AX fixtures
- validate end-to-end behavior from trigger dumps

## Raw AX Fixture Capture

Run:

```bash
python dump_ax_tree_json.py -d tests/fixtures/ax_trees --depth 35 --notes "what was on screen"
```

Then, for each target app state:

1. Focus the target app and the relevant window.
2. Put the text cursor in the composing input if one exists.
3. Make sure the on-screen state is the one you want to preserve.
4. Press `ctrl+space`.
5. Repeat for several meaningful states.

The capture is saved into `tests/fixtures/ax_trees/`.

New captures include:

- `artifactType: ax_tree_fixture_v1`
- capture metadata like app, window title, URL, and notes
- focused element metadata when available
- the serialized AX tree with focus annotations when possible

## Capture Checklist

For each app, try to gather fixtures covering:

- new or empty chat
- active conversation with a short history
- active conversation with a long history
- draft already typed into the input
- cursor in the input but no draft
- thread or sub-conversation views
- group chats vs 1:1 chats where relevant
- sidebars open
- search panels open
- attachment or action menus visible
- loading or partially rendered states
- any known weird layouts for that app

Good notes matter. Include the exact state in `--notes`, for example:

- `claude current chat, sidebar open, 2 visible turns`
- `discord DM, long thread, input focused, draft typed`
- `chatgpt new chat, no messages yet, toolbar visible`

## Analyze Captured Fixtures

Use the analyzer on a single file:

```bash
python analyze_fixture.py tests/fixtures/ax_trees/discord.json
```

Or on a whole directory:

```bash
python analyze_fixture.py tests/fixtures/ax_trees --limit 20
```

JSON output is also available:

```bash
python analyze_fixture.py tests/fixtures/ax_trees/discord.json --json
```

The analyzer reports:

- selected extractor for that app/window
- extracted conversation turns from the current extractor
- subtree overview and bottom-up context
- text-rich container candidates for extractor design
- sample text-bearing nodes
- tree stats like total nodes, depth, and role distribution

This is the main tool for deciding:

- where the useful transcript content lives
- which UI regions are mostly chrome
- what structural anchors an app-specific extractor should use
- whether subtree heuristics are surfacing the right region

## Replay Fixtures Offline

To replay a raw fixture:

```bash
python replay_fixture.py tests/fixtures/ax_trees/claude-current-chat.json
```

To replay both continuation and reply context:

```bash
python replay_fixture.py tests/fixtures/ax_trees/claude-current-chat.json --both-modes
```

To replay a saved trigger dump:

```bash
python replay_fixture.py dumps/<file>.json
```

The replay tool now uses the current context assembly flow:

- normalized fixture loading
- current extractor selection
- current subtree bundle logic
- current `ContextStore` continuation/reply formatting

That makes it suitable for checking whether:

- extracted turns look right
- reply context is built correctly
- subtree context fills gaps when no turns are found
- focused draft state is reconstructed correctly from saved dumps

## Build Extractor Regression Fixtures

The extractor regression suite pairs:

- `tests/fixtures/ax_trees/<name>.json`
- `tests/fixtures/expected/<name>.json`

If you want to generate a starter expected file from the current extractor:

```bash
./venv/bin/python -m tests.generate_expected tests/fixtures/ax_trees/claude-current-chat.json
```

Or regenerate them all:

```bash
./venv/bin/python -m tests.generate_expected --all
```

Then review the output carefully and edit it into the intended golden behavior.

Run the extractor regression suite with:

```bash
./venv/bin/python -m pytest tests/test_extractor_regression.py -v
```

## Context And Replay Benchmarks

For context-processing benchmarks, use:

- raw AX fixtures under `tests/fixtures/ax_trees/`
- context expectations under `tests/fixtures/context_expected/`

These expectations assert that the current pipeline surfaces the right content in assembled context for representative fixtures.

Run:

```bash
./venv/bin/python -m pytest tests/test_fixture_workflow.py -v
```

This covers:

- normalized fixture loading
- focused-state reconstruction from replay dumps
- offline replay context assembly
- analyzer sanity checks
- fixture-backed context expectations

## Full Trigger Dumps

To capture end-to-end artifacts from live usage:

```bash
python -m autocompleter --dump-dir dumps
```

Then use the app normally and trigger suggestions in the target apps.

These dump files can contain:

- focused input state
- detection flags
- extracted conversation turns
- assembled context
- request payload metadata
- suggestions
- serialized AX tree

They are useful for:

- reproducing bugs from real sessions
- checking how extractor output flowed into prompt assembly
- replaying a concrete bad suggestion path

They are not as clean as raw AX fixtures for extractor design, but they are much better for end-to-end debugging.

## Practical Workflow

For a new app-specific extractor project:

1. Capture 5-15 raw AX fixtures with `dump_ax_tree_json.py`.
2. Run `analyze_fixture.py` across them and identify repeated structural anchors.
3. Implement or revise the app-specific extractor.
4. Save or update `tests/fixtures/expected/*.json` golden outputs.
5. Run `tests/test_extractor_regression.py`.
6. Add or update `tests/fixtures/context_expected/*.json` if context assembly behavior should be benchmarked too.
7. Run `tests/test_fixture_workflow.py`.
8. Capture a few real `--dump-dir` artifacts and replay them if you want end-to-end validation.

## Privacy Note

Fixtures and trigger dumps may contain sensitive user data, message history, URLs, and drafts.

Before sharing or committing them:

- review the raw JSON
- remove secrets or personal content when needed
- prefer synthetic or redacted captures when possible
