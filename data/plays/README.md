# Play manifest and game splits

## `manifest.csv` (one row per play clip)

| Column | Required | Description |
|--------|----------|-------------|
| `game_id` | yes | Game folder id, e.g. `g01` |
| `play_id` | yes | Play folder id, e.g. `p0042` |
| `source_video` | yes | Path to the per-play MP4 |
| `notes` | no | Free text |

Batch extraction writes frames to `data/frames/{game_id}/{play_id}/frame_{NNNNNN}.jpg`.

See `manifest.example.csv`.

## `splits.csv` (one row per game)

| Column | Required | Description |
|--------|----------|-------------|
| `game_id` | yes | Must match manifest |
| `split` | yes | `train`, `val`, or `test` |

Used by dataset sync and crop export so frames from the same game stay in one split (no leakage).

See `splits.example.csv`.

## Python helpers

[`pipeline/play_manifest.py`](../pipeline/play_manifest.py) — `load_manifest`, `load_game_splits`.

## Optional: player home/away for crop export

For [`tools/export_team_ref_crops.py`](../tools/export_team_ref_crops.py), use a CSV with columns `game_id`, `play_id`, `frame_stem`, `line_index`, `team` where `team` is `home`, `away`, or `unknown`. `line_index` is the 0-based line in the YOLO label file for that frame. See `team_box.example.csv`.

## Typical commands

```text
python -m pipeline.batch_extract_frames --manifest data/plays/manifest.csv
python tools/sync_frames_to_yolo_dataset.py --frames-root data/frames --labels-root data/labels --splits data/plays/splits.csv
python -m training.train_detection --epochs 50
python tools/export_team_ref_crops.py --team-box-csv data/plays/team_labels.csv
python -m training.train_team_classifier --data-root datasets/team_ref
```
