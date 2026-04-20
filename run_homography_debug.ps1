param(
    [string]$GameId = "test",
    [string]$PlayId = "test_frame1",
    [string]$Weights = "yolo11s.pt",
    [string]$FramePath = "data/frames/test/test_frame1/frame_1.png",
    [string]$DetectionsOut = "",
    [string]$HomographyDebugOut = "",
    [switch]$NoWhiteMask
)

$ErrorActionPreference = "Stop"

$detections = if ($DetectionsOut) {
    $DetectionsOut
} else {
    "data/detections/${GameId}_${PlayId}_homography_debug.parquet"
}

$homographyOut = if ($HomographyDebugOut) {
    $HomographyDebugOut
} else {
    "homography_debug_${GameId}_${PlayId}.jpg"
}

Write-Host "Running pipeline.detect (writes homography debug image + detections parquet)..."
$args = @(
    "-m", "pipeline.detect",
    "--game-id", $GameId,
    "--play-id", $PlayId,
    "--weights", $Weights,
    "--homography-ref-image", $FramePath,
    "--homography-debug-image", $homographyOut,
    "--out", $detections
)
if (-not $NoWhiteMask) {
    $args += "--homography-debug-show-white-mask"
}
& python @args
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "Homography debug: $homographyOut"
Write-Host "Detections:       $detections"
