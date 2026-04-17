# AutocompleterBootstrap

Minimal signed macOS `.app` whose only job is to prove TCC + Accessibility API + CGEvent tap work on the user's machine. No Python, no LLM, no injection — just diagnostics.

## Why this exists

The Python autocompleter fails on fresh Macs (particularly macOS 15+) with "no focused text elements found" because unsigned venv Python binaries don't get stable TCC grants. This `.app` is the signed, stable TCC-trusted binary that the rest of the product will live inside eventually. v0.1 ships only the diagnostic panel.

## Build (local dev)

```bash
./scripts/build.sh
open .build/AutocompleterBootstrap.app
```

Produces an ad-hoc signed `.app` suitable for testing on your own machine. You'll need to grant it accessibility permission the first time.

### Gotcha: TCC grant revoked on rebuild

Ad-hoc signing uses a per-build signature, which macOS treats as a different app identity each time. That means **every `./scripts/build.sh` invalidates the Accessibility grant** — after rebuilding you'll see ✗ on the TCC check until you toggle the app off and back on in System Settings → Privacy & Security → Accessibility.

This does not affect friends: a Developer-ID-signed, notarized build (`scripts/sign.sh` → `scripts/notarize.sh`) has a stable signature, so the grant sticks across updates. The pain is dev-only.

## Ship (signed + notarized)

```bash
DEVELOPER_ID="Developer ID Application: Your Name (TEAMID)" ./scripts/sign.sh
APPLE_ID="..." TEAM_ID="..." APP_PASSWORD="..." ./scripts/notarize.sh
```

Output: `dist/AutocompleterBootstrap.dmg` ready to hand to a friend.

## What's in v0.1

Four checks in the diagnostic panel:

1. **Accessibility permission** (`AXIsProcessTrustedWithOptions`)
2. **Focused element read** (`AXFocusedUIElement` returns non-nil on frontmost app)
3. **CGEvent tap creation** (hotkey infrastructure)
4. **End-to-end hotkey capture** (user presses hotkey in TextEdit; panel shows captured `AXValue` + `insertion_point`)

A "Copy diagnostic report" button dumps all four results + OS version as JSON for pasting back when a friend reports breakage.
