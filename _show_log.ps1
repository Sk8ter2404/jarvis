# JARVIS LIVE LOG VIEWER — standalone tail window
#
# Opens (or reopens) a window that streams the latest JARVIS session log in
# real time. The window is READ-ONLY and DISPOSABLE — closing it does NOT
# affect JARVIS in any way. Run this any time you want to see what JARVIS
# is doing, especially after closing the launcher window.
#
# Usage:
#   - Double-click _show_log.ps1
#   - Or from any PowerShell: powershell -File C:\JARVIS\_show_log.ps1

$logFile = Get-ChildItem 'C:\JARVIS\logs\session_*.log' -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending | Select-Object -First 1

if (-not $logFile) {
    Write-Host 'No JARVIS session log found at C:\JARVIS\logs\session_*.log' -ForegroundColor Red
    Write-Host 'Is JARVIS running? Boot it with C:\JARVIS\_boot_jarvis.ps1 first.'
    Start-Sleep -Seconds 4
    exit 1
}

$Host.UI.RawUI.WindowTitle = "JARVIS LIVE LOG ($(Split-Path -Leaf $logFile.FullName))"

Write-Host '=== JARVIS LIVE LOG ===' -ForegroundColor Cyan
Write-Host "File: $($logFile.FullName)" -ForegroundColor DarkGray
Write-Host "Started: $($logFile.CreationTime)" -ForegroundColor DarkGray
Write-Host ''
Write-Host 'Closing this window will NOT affect JARVIS.' -ForegroundColor DarkGray
Write-Host 'JARVIS is detached and survives independently.' -ForegroundColor DarkGray
Write-Host ''

# Tail the file, showing the last 30 lines first then live-following.
# Color-code key event types so it's scan-able.
Get-Content -Path $logFile.FullName -Wait -Tail 30 | ForEach-Object {
    switch -Regex ($_) {
        '\[FATAL\]|Traceback|Exception|APPCRASH' { Write-Host $_ -ForegroundColor Red ;     continue }
        'WARNING|\[FAIL\]|failed|API request rejected' { Write-Host $_ -ForegroundColor Yellow ; continue }
        'You:\s' { Write-Host $_ -ForegroundColor White ; continue }
        'JARVIS:|\[action\]' { Write-Host $_ -ForegroundColor Cyan ; continue }
        '\[inject\]|\[local-llm\]|\[local\]' { Write-Host $_ -ForegroundColor Magenta ; continue }
        '🔔|\[reminder\]' { Write-Host $_ -ForegroundColor Green ; continue }
        default { Write-Host $_ }
    }
}
