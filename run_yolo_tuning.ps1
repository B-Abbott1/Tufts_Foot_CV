param(
    [string]$GameId = "test",
    [string]$PlayId = "test_frame1",
    [string]$Weights = "yolo11s.pt",
    [string]$FramePath = "data/frames/test/test_frame1/frame_1.png",
    [int]$FrameId = 1,
    [string]$Tag = "new"
)

$ErrorActionPreference = "Stop"

function Invoke-YoloRun {
    param(
        [string]$RunTag
    )

    $detectionsOut = "data/detections/${GameId}_${PlayId}_${RunTag}.parquet"
    $trackingOut = "data/tracking/${GameId}_${PlayId}_${RunTag}.parquet"
    $overlayOut = "overlay_preview_${RunTag}.jpg"

    Write-Host ""
    Write-Host "=== Run tag: $RunTag ==="
    Write-Host "Running detect..."
    & python -m pipeline.detect `
        --game-id $GameId `
        --play-id $PlayId `
        --weights $Weights `
        --homography-ref-image $FramePath `
        --out $detectionsOut
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    Write-Host "Running track..."
    & python -m pipeline.track `
        --detections $detectionsOut `
        --out $trackingOut
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    Write-Host "Running overlay preview..."
    & python tools/overlay_preview.py `
        --tracking $trackingOut `
        --frame $FramePath `
        --frame-id $FrameId `
        --out $overlayOut
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }


    Write-Host "Detections: $detectionsOut"
    Write-Host "Tracking:   $trackingOut"
    Write-Host "Overlay:    $overlayOut"
}

Invoke-YoloRun -RunTag $Tag

Write-Host ""
Write-Host "Done."
