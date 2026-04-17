#!/usr/bin/env bash
# Zip, submit to Apple notarization, staple, and produce a .dmg.
#
# Requires:
#   APPLE_ID       your Apple ID email
#   TEAM_ID        your Developer Team ID (10 chars)
#   APP_PASSWORD   an app-specific password generated at appleid.apple.com
#
# Assumes scripts/sign.sh has already produced dist/AutocompleterBootstrap.app.

set -euo pipefail
cd "$(dirname "$0")/.."

: "${APPLE_ID:?set APPLE_ID}"
: "${TEAM_ID:?set TEAM_ID}"
: "${APP_PASSWORD:?set APP_PASSWORD}"

APP_NAME="AutocompleterBootstrap"
DIST_DIR="dist"
APP_DIR="$DIST_DIR/$APP_NAME.app"
ZIP_PATH="$DIST_DIR/$APP_NAME.zip"
DMG_PATH="$DIST_DIR/$APP_NAME.dmg"

if [ ! -d "$APP_DIR" ]; then
    echo "$APP_DIR not found — run scripts/sign.sh first." >&2
    exit 1
fi

echo "==> Zipping for notarization submission"
rm -f "$ZIP_PATH"
ditto -c -k --sequesterRsrc --keepParent "$APP_DIR" "$ZIP_PATH"

echo "==> Submitting to Apple (notarytool)"
xcrun notarytool submit "$ZIP_PATH" \
    --apple-id "$APPLE_ID" \
    --team-id "$TEAM_ID" \
    --password "$APP_PASSWORD" \
    --wait

echo "==> Stapling ticket to $APP_DIR"
xcrun stapler staple "$APP_DIR"
xcrun stapler validate "$APP_DIR"

echo "==> Creating $DMG_PATH"
rm -f "$DMG_PATH"
hdiutil create \
    -volname "$APP_NAME" \
    -srcfolder "$APP_DIR" \
    -ov -format UDZO \
    "$DMG_PATH"

echo ""
echo "Signed, notarized, stapled: $APP_DIR"
echo "Distributable DMG:          $DMG_PATH"
