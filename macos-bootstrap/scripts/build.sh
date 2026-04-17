#!/usr/bin/env bash
# Build a local .app bundle with ad-hoc signing. For dev / testing only.
# Use scripts/sign.sh for a proper Developer-ID-signed build.

set -euo pipefail
cd "$(dirname "$0")/.."

APP_NAME="AutocompleterBootstrap"
BUILD_DIR=".build"
APP_DIR="$BUILD_DIR/$APP_NAME.app"

echo "==> swift build -c release"
swift build -c release

BIN_PATH="$BUILD_DIR/release/$APP_NAME"
if [ ! -f "$BIN_PATH" ]; then
    echo "Build did not produce $BIN_PATH" >&2
    exit 1
fi

echo "==> Assembling $APP_DIR"
rm -rf "$APP_DIR"
mkdir -p "$APP_DIR/Contents/MacOS"
mkdir -p "$APP_DIR/Contents/Resources"
cp "$BIN_PATH" "$APP_DIR/Contents/MacOS/$APP_NAME"
cp Resources/Info.plist "$APP_DIR/Contents/Info.plist"
printf 'APPL????' > "$APP_DIR/Contents/PkgInfo"

echo "==> Ad-hoc signing"
codesign --force \
    --sign - \
    --entitlements Resources/AutocompleterBootstrap.entitlements \
    --options runtime \
    "$APP_DIR"

echo ""
echo "Built: $APP_DIR"
echo "Launch with: open $APP_DIR"
