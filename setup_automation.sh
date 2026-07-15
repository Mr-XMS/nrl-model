#!/bin/bash
# One-time setup: installs a launchd agent that runs nrl_auto.sh on schedule.
#   Mon 10:00 | Tue 20:30 | Thu 18:00 | Fri 11:00 | Sat 11:00 | Sun 11:00
# Run once:  bash setup_automation.sh      Remove:  bash setup_automation.sh remove

MODEL_DIR="$(cd "$(dirname "$0")" && pwd)"
PLIST="$HOME/Library/LaunchAgents/com.nrlmodel.auto.plist"

if [ "$1" = "remove" ]; then
  launchctl unload "$PLIST" 2>/dev/null
  rm -f "$PLIST"
  echo "Automation removed."
  exit 0
fi

chmod +x "$MODEL_DIR/nrl_auto.sh"
mkdir -p "$HOME/Library/LaunchAgents"

cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
 "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.nrlmodel.auto</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>$MODEL_DIR/nrl_auto.sh</string>
  </array>
  <key>StartCalendarInterval</key>
  <array>
    <dict><key>Weekday</key><integer>1</integer><key>Hour</key><integer>10</integer><key>Minute</key><integer>0</integer></dict>
    <dict><key>Weekday</key><integer>2</integer><key>Hour</key><integer>20</integer><key>Minute</key><integer>30</integer></dict>
    <dict><key>Weekday</key><integer>4</integer><key>Hour</key><integer>18</integer><key>Minute</key><integer>0</integer></dict>
    <dict><key>Weekday</key><integer>5</integer><key>Hour</key><integer>11</integer><key>Minute</key><integer>0</integer></dict>
    <dict><key>Weekday</key><integer>6</integer><key>Hour</key><integer>11</integer><key>Minute</key><integer>0</integer></dict>
    <dict><key>Weekday</key><integer>0</integer><key>Hour</key><integer>11</integer><key>Minute</key><integer>0</integer></dict>
  </array>
  <key>StandardOutPath</key><string>$MODEL_DIR/automation.log</string>
  <key>StandardErrorPath</key><string>$MODEL_DIR/automation.log</string>
</dict>
</plist>
EOF

launchctl unload "$PLIST" 2>/dev/null
launchctl load "$PLIST"
echo "Automation installed. Schedule:"
echo "  Mon 10:00  results + grading"
echo "  Tue 20:30  full forecast + SGM ledger + odds scan"
echo "  Thu 18:00  odds scan"
echo "  Fri-Sun 11:00  game-day odds scan + lineup check"
echo "Logs: $MODEL_DIR/automation.log"
echo "Test it now with:  bash $MODEL_DIR/nrl_auto.sh"
