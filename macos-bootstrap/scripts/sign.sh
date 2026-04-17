#!/usr/bin/env bash
# Produce a Developer-ID-signed .app ready for notarization.
#
# Requires the env var DEVELOPER_ID, e.g.:
#   DEVELOPER_ID="Developer ID Application: Henry Zhang (XXXXXXXXXX)" ./scripts/sign.sh
#
# The resulting .app lives at dist/AutocompleterBootstrap.app.

set -euo pipefail
cd "$(dirname "$0")/.."

if [ -z "${DEVELOPER_ID:-}" ]; then
    echo "Set DEVELOPER_ID to your 'Developer ID Application: ...' cert name." >&2
    exit 1
fi

APP_NAME="AutocompleterBootstrap"
DIST_DIR="dist"
APP_DIR="$DIST_DIR/$APP_NAME.app"
ENTITLEMENTS="Resources/AutocompleterBootstrap.entitlements"

echo "==> swift build -c release"
swift build -c release

BIN_PATH=".build/release/$APP_NAME"

echo "==> Assembling $APP_DIR"
rm -rf "$APP_DIR"
mkdir -p "$APP_DIR/Contents/MacOS"
mkdir -p "$APP_DIR/Contents/Resources"
cp "$BIN_PATH" "$APP_DIR/Contents/MacOS/$APP_NAME"
cp Resources/Info.plist "$APP_DIR/Contents/Info.plist"
printf 'APPL????' > "$APP_DIR/Contents/PkgInfo"

echo "==> codesign (hardened runtime, timestamped) with $DEVELOPER_ID"
codesign --force \
    --sign "$DEVELOPER_ID" \
    --entitlements "$ENTITLEMENTS" \
    --options runtime \
    --timestamp \
    "$APP_DIR"

echo "==> Verify signature"
codesign --verify --deep --strict --verbose=2 "$APP_DIR"
spctl --assess --type execute --verbose=4 "$APP_DIR" || \
    echo "(spctl will reject until notarization has been stapled — run scripts/notarize.sh next.)"

echo ""
echo "Signed: $APP_DIR"
echo "Next:   APPLE_ID=... TEAM_ID=... APP_PASSWORD=... ./scripts/notarize.sh"
