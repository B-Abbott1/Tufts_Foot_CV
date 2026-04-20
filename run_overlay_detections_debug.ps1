param(
    [string]$GameId = "test",
    [string]$PlayId = "test_frame1",
    [string]$Tag = "new",
    [string]$FramePath = "data/frames/test/test_frame1/frame_1.png",
    [int]$FrameId = 1,
    [string]$Out = "",
    [ValidateSet("foot", "center", "both")]
    [string]$Point = "foot",
    [switch]$NoConfidence
)
$ErrorActionPreference = "Stop"
$detections = "data/detections/${GameId}_${PlayId}_${Tag}.parquet"
$overlayOut = if ($Out) { $Out } else { "overlay_detections_debug_${Tag}.jpg" }
if (-not (Test-Path -LiteralPath $detections)) {
    Write-Error "Detections file not found: $detections (run detect first, e.g. .\run_yolo_tuning.ps1)"
}
$args = @(
    "tools/overlay_detections_debug.py",
    "--detections", $detections,
    "--frame", $FramePath,
    "--frame-id", "$FrameId",
    "--out", $overlayOut,
    "--point", $Point
)
if ($NoConfidence) {
    $args += "--no-show-confidence"
}
& python @args
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "Wrote $overlayOut"