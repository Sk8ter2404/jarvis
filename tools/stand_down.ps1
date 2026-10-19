# JARVIS peer handoff, the side that is asked (see core/peer_handoff.py).
#
# Run over SSH by the OTHER machine's JARVIS as it starts: writes the
# stand-down request (data\stand_down.json) that the JARVIS running here
# polls, then waits for it to exit. Runs in an SSH session, which has no
# desktop, so it never STARTS anything.
#
#   -From <name>   who is taking over (logged by the JARVIS standing down)
#   -Query         only print "running <n>" (no request written); the
#                  desktop's resume waiter asks this every 30 s
param(
    [string] $From = 'the other machine',
    [switch] $Query
)

$root = Split-Path -Parent $PSScriptRoot

function Get-Jarvis {
    @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
        ($_.Name -eq 'python.exe' -or $_.Name -eq 'pythonw.exe') -and
        $_.CommandLine -like '*bobert_companion*' -and
        $_.CommandLine -notlike '*--staging*'
    })
}

if ($Query) { "running $(@(Get-Jarvis).Count)"; exit 0 }

$req = Join-Path $root 'data\stand_down.json'
$at = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds() / 1000.0
$safe = ($From -replace '[^A-Za-z0-9 _-]', '')
(@{ from = $safe; at = $at } | ConvertTo-Json -Compress) | Set-Content -Path $req -Encoding ascii

if (@(Get-Jarvis).Count -eq 0) { 'not running'; exit 0 }

$deadline = (Get-Date).AddSeconds(90)   # memory hand-over + graceful teardown
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 2
    if (@(Get-Jarvis).Count -eq 0) { 'stood down'; exit 0 }
}

# A JARVIS older than the handoff never reads the request: stop it the way
# the boot scripts stop a stale instance.
Get-Jarvis | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
$lock = Join-Path $root 'jarvis.lock'
if (Test-Path $lock) { Remove-Item $lock -Force -ErrorAction SilentlyContinue }
'stopped (forced)'
