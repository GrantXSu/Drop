#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SERVER="$ROOT/.venv/bin/drop-card-grader"
LABEL="com.cardlens.server"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
LOG_DIR="$HOME/Library/Logs/CardLens"
APP_DIR="$HOME/Applications"

if [[ ! -x "$SERVER" ]]; then
  echo "CardLens is not installed in $ROOT/.venv."
  echo "Run: cd \"$ROOT\" && python3 -m venv .venv && .venv/bin/pip install -e '.[scanner]'"
  exit 1
fi

mkdir -p "$(dirname "$PLIST")" "$LOG_DIR" "$APP_DIR"
cat >"$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>$SERVER</string>
    <string>--host</string>
    <string>127.0.0.1</string>
    <string>--port</string>
    <string>8000</string>
    <string>--reload</string>
  </array>
  <key>WorkingDirectory</key>
  <string>$ROOT</string>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>StandardOutPath</key>
  <string>$LOG_DIR/server.log</string>
  <key>StandardErrorPath</key>
  <string>$LOG_DIR/server-error.log</string>
</dict>
</plist>
EOF

DOMAIN="gui/$(id -u)"
launchctl bootout "$DOMAIN" "$PLIST" 2>/dev/null || true
launchctl bootstrap "$DOMAIN" "$PLIST"
launchctl enable "$DOMAIN/$LABEL"
launchctl kickstart -k "$DOMAIN/$LABEL"

if [[ ! -d "$APP_DIR/CardLens.app" ]] && command -v osacompile >/dev/null; then
  osacompile -o "$APP_DIR/CardLens.app" \
    -e 'on run' \
    -e 'open location "http://127.0.0.1:8000"' \
    -e 'end run'
fi

open "http://127.0.0.1:8000"
echo "CardLens now starts automatically at login."
echo "Open $APP_DIR/CardLens.app or visit http://127.0.0.1:8000."
