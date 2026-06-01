#!/bin/bash
# Periodic check: keep only ONE JARVIS instance + ONE pipeline driver alive.
# Kill duplicates, log cleanup actions. Runs forever in 5-min loop.
cd /c/JARVIS

mkdir -p data
HYG_LOG=data/instance_hygiene.log

log() { printf '[%s] %s\n' "$(date +%FT%T)" "$*" >> "$HYG_LOG"; }

# Kill a PID, verify it actually died, escalate to taskkill /F /T if it didn't.
# Adds ~2s per stale PID so Stop-Process has time to take effect before re-check.
kill_pid() {
  local p=$1
  powershell.exe -NoProfile -Command "Stop-Process -Id $p -Force -ErrorAction SilentlyContinue" 2>/dev/null
  sleep 2
  local alive
  alive=$(powershell.exe -NoProfile -Command "if (Get-Process -Id $p -ErrorAction SilentlyContinue) { 'YES' } else { 'NO' }" 2>/dev/null | tr -d '\r\n ')
  if [ "$alive" = "YES" ]; then
    log "  PID $p survived Stop-Process — escalating to taskkill /F /T"
    local tk_out
    tk_out=$(taskkill //PID $p //F //T 2>&1)
    local tk_exit=$?
    if [ "$tk_exit" -ne 0 ]; then
      log "[hygiene] CANNOT KILL PID $p — manual intervention needed (taskkill exit=$tk_exit: $(printf '%s' "$tk_out" | tr '\n' ' '))"
    fi
  fi
}

log "=== instance-hygiene loop started ==="

while true; do
  # === Count JARVIS processes (bobert_companion) ===
  jarvis_pids=$(powershell.exe -NoProfile -Command "
    Get-CimInstance Win32_Process | Where-Object {
      (\$_.Name -eq 'python.exe' -or \$_.Name -eq 'pythonw.exe') -and
      \$_.CommandLine -like '*bobert_companion*'
    } | Sort-Object CreationDate | Select-Object -ExpandProperty ProcessId
  " 2>/dev/null | tr -d '\r' | tr '\n' ' ')
  jarvis_count=$(echo $jarvis_pids | wc -w)

  if [ "$jarvis_count" -gt 1 ]; then
    log "JARVIS DUPES detected: $jarvis_pids (count=$jarvis_count) — keeping newest"
    # Keep last (newest), kill rest
    keep=$(echo $jarvis_pids | awk '{print $NF}')
    for p in $jarvis_pids; do
      if [ "$p" != "$keep" ]; then
        log "  killing stale JARVIS PID $p (keeping $keep)"
        kill_pid $p
      fi
    done
  fi

  # === Count upgrade-pipeline drivers ===
  pipe_pids=$(powershell.exe -NoProfile -Command "
    Get-CimInstance Win32_Process | Where-Object {
      \$_.Name -eq 'python.exe' -and \$_.CommandLine -like '*pipeline_loop*'
    } | Sort-Object CreationDate | Select-Object -ExpandProperty ProcessId
  " 2>/dev/null | tr -d '\r' | tr '\n' ' ')
  pipe_count=$(echo $pipe_pids | wc -w)

  if [ "$pipe_count" -gt 1 ]; then
    log "PIPELINE DUPES detected: $pipe_pids — keeping newest"
    keep=$(echo $pipe_pids | awk '{print $NF}')
    for p in $pipe_pids; do
      if [ "$p" != "$keep" ]; then
        log "  killing stale pipeline driver $p"
        kill_pid $p
      fi
    done
  fi

  # === Orphan HUD check — HUDs without a parent JARVIS ===
  hud_pids=$(powershell.exe -NoProfile -Command "
    Get-CimInstance Win32_Process | Where-Object {
      (\$_.Name -eq 'python.exe' -or \$_.Name -eq 'pythonw.exe') -and
      (\$_.CommandLine -like '*hud\*' -or \$_.CommandLine -like '*tray.py*')
    } | Select-Object -ExpandProperty ProcessId
  " 2>/dev/null | tr -d '\r' | tr '\n' ' ')

  for hud in $hud_pids; do
    # Get parent of this HUD
    parent=$(powershell.exe -NoProfile -Command "
      (Get-CimInstance Win32_Process -Filter \"ProcessId = $hud\").ParentProcessId
    " 2>/dev/null | tr -d '\r\n ')
    # Is parent alive AND a JARVIS?
    if [ -n "$parent" ]; then
      parent_alive=$(powershell.exe -NoProfile -Command "
        if (Get-Process -Id $parent -ErrorAction SilentlyContinue) { 'YES' } else { 'NO' }
      " 2>/dev/null | tr -d '\r\n ')
      if [ "$parent_alive" = "NO" ]; then
        log "ORPHAN HUD PID $hud (dead parent $parent) — killing"
        kill_pid $hud
      fi
    fi
  done

  # === Periodic summary every 30 min (i.e. every 6 cycles of 5 min) ===
  cycle=$(( ${cycle:-0} + 1 ))
  if [ "$((cycle % 6))" -eq "0" ]; then
    log "STATUS: $jarvis_count JARVIS, $pipe_count pipeline driver(s) alive"
  fi

  sleep 300   # 5 min
done
