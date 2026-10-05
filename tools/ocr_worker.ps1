# JARVIS OCR worker - Windows.Media.Ocr, built into Windows (no installs).
#
# Started lazily by core/screen_ocr.py (hidden, idle priority) and kept
# alive between requests; it exits when stdin closes or on "quit" (the
# Python side retires it after 120 s idle).
#
# stdin : one JSON object per line {"w":int,"h":int,"fmt":"gray8"|"bgra8","b64":"..."}
#         - the raw pixels of an image JARVIS is allowed to read (the privacy
#         gate ran before the pixels were taken; private windows never get
#         here).
# stdout: one JSON object per line
#         {"ms":engine_ms,"lines":[{"t":text,"words":[[text,x,y,w,h],...]},...]}
#         or {"error":"..."}; the first line is {"ready":true}.
#
# Nothing is written to disk and nothing leaves the machine.
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding $false
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$null = [Windows.Media.Ocr.OcrEngine, Windows.Foundation, ContentType = WindowsRuntime]
$null = [Windows.Graphics.Imaging.SoftwareBitmap, Windows.Graphics, ContentType = WindowsRuntime]
$asTaskGeneric = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
    $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and
    $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' })[0]
$asTaskOcr = $asTaskGeneric.MakeGenericMethod([Windows.Media.Ocr.OcrResult])
$engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromUserProfileLanguages()
if ($null -eq $engine) {
    [Console]::Out.WriteLine('{"ready":false,"error":"no OCR language installed"}')
    [Console]::Out.Flush()
    exit 2
}
[Console]::Out.WriteLine('{"ready":true}')
[Console]::Out.Flush()
$stdin = [Console]::In
while ($true) {
    $line = $stdin.ReadLine()
    if ($null -eq $line -or $line -eq 'quit') { break }
    try {
        $req = $line | ConvertFrom-Json
        $bytes = [Convert]::FromBase64String($req.b64)
        $buf = [System.Runtime.InteropServices.WindowsRuntime.WindowsRuntimeBufferExtensions]::AsBuffer($bytes)
        if ($req.fmt -eq 'bgra8') { $pf = [Windows.Graphics.Imaging.BitmapPixelFormat]::Bgra8 }
        else { $pf = [Windows.Graphics.Imaging.BitmapPixelFormat]::Gray8 }
        $sw = [Diagnostics.Stopwatch]::StartNew()
        $bmp = [Windows.Graphics.Imaging.SoftwareBitmap]::CreateCopyFromBuffer($buf, $pf, [int]$req.w, [int]$req.h)
        $task = $asTaskOcr.Invoke($null, @($engine.RecognizeAsync($bmp)))
        $null = $task.Wait(-1)
        $res = $task.Result
        $ms = $sw.Elapsed.TotalMilliseconds
        $sb = New-Object System.Text.StringBuilder
        $null = $sb.Append('{"ms":' + [math]::Round($ms, 1) + ',"lines":[')
        $first = $true
        foreach ($l in $res.Lines) {
            if (-not $first) { $null = $sb.Append(',') }; $first = $false
            $words = @()
            foreach ($wd in $l.Words) {
                $r = $wd.BoundingRect
                $words += ('[' + (ConvertTo-Json $wd.Text -Compress) + ',' + [int]$r.X + ',' + [int]$r.Y + ',' + [int]$r.Width + ',' + [int]$r.Height + ']')
            }
            $null = $sb.Append('{"t":' + (ConvertTo-Json $l.Text -Compress) + ',"words":[' + ($words -join ',') + ']}')
        }
        $null = $sb.Append(']}')
        $bmp.Dispose()
        [Console]::Out.WriteLine($sb.ToString())
    } catch {
        [Console]::Out.WriteLine('{"error":' + (ConvertTo-Json ($_.Exception.Message) -Compress) + '}')
    }
    [Console]::Out.Flush()
}
