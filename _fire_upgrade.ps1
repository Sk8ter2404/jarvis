# One-shot pipeline launcher — kills stale processes, clears lock,
# spawns upgrade_jarvis.py --relaunch in a new visible window.

Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -like '*bobert_companion*' -or
    $_.CommandLine -like '*upgrade_jarvis*'   -or
    $_.CommandLine -like '*\.local\bin\claude*'
} | ForEach-Object {
    Write-Host ("pre-kill: PID " + $_.ProcessId)
    Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
}
Start-Sleep -Seconds 2

if (Test-Path 'jarvis.lock') {
    Remove-Item 'jarvis.lock' -Force
    Write-Host 'cleared stale jarvis.lock'
}

$env:ANTHROPIC_API_KEY = ''
$inner = "`$env:ANTHROPIC_API_KEY=''; cd 'C:\JARVIS'; Write-Host '=== JARVIS UPGRADE PIPELINE ===' -ForegroundColor Cyan; python upgrade_jarvis.py --relaunch"
Start-Process -FilePath 'powershell.exe' -ArgumentList '-NoExit','-Command',$inner
Write-Host ''
Write-Host 'Upgrade pipeline spawned in a new window.' -ForegroundColor Green
Write-Host 'It will drain all 25 pending tasks (audit fixes first, wish-list last),'
Write-Host 'then auto-launch JARVIS when done.'
