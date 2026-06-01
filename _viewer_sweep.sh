#!/bin/bash
# Continuous log-viewer cleanup — kills all but newest every 3 min.
cd /c/JARVIS
LOG=data/instance_hygiene.log
while true; do
  viewers=$(powershell.exe -NoProfile -Command "
    Get-CimInstance Win32_Process | Where-Object {
      \$_.Name -eq 'powershell.exe' -and (\$_.CommandLine -like '*Get-Content -Wait*' -or \$_.CommandLine -like '*JARVIS LIVE LOG*')
    } | Sort-Object CreationDate | Select-Object -ExpandProperty ProcessId
  " 2>/dev/null | tr -d '\r' | tr '\n' ' ')
  count=$(echo $viewers | wc -w)
  if [ "$count" -gt 1 ]; then
    keep=$(echo $viewers | awk '{print $NF}')
    printf '[%s] viewer-sweep: %d viewers, keeping PID %s\n' "$(date +%FT%T)" "$count" "$keep" >> "$LOG"
    for v in $viewers; do
      if [ "$v" != "$keep" ]; then
        powershell.exe -NoProfile -Command "Stop-Process -Id $v -Force -ErrorAction SilentlyContinue" 2>/dev/null
      fi
    done
  fi
  sleep 180
done
