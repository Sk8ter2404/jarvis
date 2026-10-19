# JARVIS peer handoff, the comeback (see core/peer_handoff.py).
#
# Launched by this machine's JARVIS, through WMI, as it stands down for the
# other machine (HANDOFF_RESUME_WHEN_PEER_STOPS). Boots JARVIS here again
# once the other machine's copy has stopped:
#   * the peer answers "running 0" twice in a row (JARVIS quit there), or
#   * the peer cannot be reached for 6 checks running, about 3 minutes
#     (laptop shut, asleep, off Tailscale) - long enough that a Wi-Fi blip
#     never puts two copies on the air.
# Exits without booting anything as soon as JARVIS runs here again (the
# owner started it himself). Not a standing service: it lives only while the
# other machine has JARVIS.
param(
    [Parameter(Mandatory = $true)] [string] $Peer,
    [int] $StoodDownPid = 0,
    [int] $EverySeconds = 30
)

$root = Split-Path -Parent $PSScriptRoot
$log = Join-Path $root 'logs\handoff_resume.log'
function Log([string] $m) {
    try { "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $m" | Add-Content -Path $log -Encoding utf8 } catch {}
}

function Get-LocalJarvis {
    @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
        ($_.Name -eq 'python.exe' -or $_.Name -eq 'pythonw.exe') -and
        $_.CommandLine -like '*bobert_companion*' -and
        $_.CommandLine -notlike '*--staging*' -and
        $_.ProcessId -ne $StoodDownPid      # the copy still shutting down
    }).Count
}

# -1 = unreachable, else the peer's JARVIS process count.
function Get-PeerJarvis {
    $remote = 'powershell -NoProfile -ExecutionPolicy Bypass -File "' +
              (Join-Path $root 'tools\stand_down.ps1') + '" -Query'
    $out = & ssh.exe -n -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new $Peer $remote 2>$null
    if ($LASTEXITCODE -ne 0) { return -1 }
    $m = [regex]::Match(($out -join "`n"), 'running (\d+)')
    if ($m.Success) { return [int]$m.Groups[1].Value }
    return -1
}

Log "waiting: JARVIS comes back here when $Peer's copy stops"
$gone = 0
$unreachable = 0
while ($true) {
    Start-Sleep -Seconds $EverySeconds
    if ((Get-LocalJarvis) -gt 0) { Log 'JARVIS is running here again - nothing to do'; exit 0 }
    $n = Get-PeerJarvis
    if ($n -gt 0) { $gone = 0; $unreachable = 0; continue }
    if ($n -eq 0) { $gone++; $unreachable = 0 } else { $unreachable++; $gone = 0 }
    if ($gone -ge 2 -or $unreachable -ge 6) { break }
}

$why = if ($gone -ge 2) { 'JARVIS stopped there' } else { 'it has been unreachable for about 3 minutes' }
$boot = Join-Path $root '_boot_laptop.ps1'
if (-not (Test-Path $boot)) { $boot = Join-Path $root '_boot_jarvis.ps1' }
Log "booting JARVIS here ($why): $boot"
Start-Process powershell -WindowStyle Hidden -ArgumentList @(
    '-NoProfile', '-ExecutionPolicy', 'Bypass', '-WindowStyle', 'Hidden', '-File', $boot)
