# NESCAC Football Computer Vision Project — Cursor Context

## Project Overview
A computer vision and data science pipeline to analyze NESCAC football game film.
Built by Tufts University football players / data science students.
Primary goal: extract player tracking data from game tape and generate actionable
analytics for coaches (formations, heat maps, route tendencies).

---

## Tech Stack
- **CV / Detection:** YOLOv8 (ultralytics), OpenCV
- **Tracking:** ByteTrack (via supervision library)
- **Deep Learning:** PyTorch
- **Data:** Pandas, NumPy, Parquet/CSV
- **Clustering / ML:** scikit-learn, tslearn (DTW)
- **Visualization:** Matplotlib, Plotly, Seaborn
- **Labeling:** Roboflow
- **IDE:** Cursor

---

## Folder Structure
```
project/
│
├── context.md                  # ← you are here
│
├── data/
│   ├── raw/                    # original exported Hudl film (.mp4)
│   ├── frames/                 # extracted frames organized by game/play
│   │   └── {game_id}/{play_id}/frame_{n}.jpg
│   ├── labels/                 # YOLO-format annotation files
│   ├── tracking/               # output tracking CSVs per game
│   └── plays/                  # per-play metadata CSVs
│
├── models/
│   ├── weights/                # YOLOv8 fine-tuned .pt files
│   └── configs/                # training config YAMLs
│
├── pipeline/
│   ├── extract_frames.py       # video → frames using OpenCV
│   ├── detect.py               # YOLOv8 inference on frames
│   ├── track.py                # ByteTrack multi-object tracking
│   ├── team_separation.py      # HSV jersey color segmentation
│   └── homography.py           # pixel → field coordinate transform
│
├── analytics/
│   ├── formation_classifier.py # pre-snap formation clustering + classification
│   ├── heat_maps.py            # run/pass location heat maps
│   ├── route_classifier.py     # receiver trajectory clustering (DTW)
│   ├── coverage_detection.py   # man vs zone shell detection
│   └── tendency_report.py      # opponent scouting tendency summaries
│
├── utils/
│   ├── field.py                # field coordinate helpers, yard line references
│   ├── video.py                # shared video I/O helpers
│   └── visualization.py        # shared plotting utilities (field overlay, etc.)
│
├── notebooks/
│   └── exploration.ipynb       # EDA, prototyping, one-off analysis
│
└── requirements.txt
```

---

## Core Data Schemas

### Per-Frame Tracking Record
```python
{
  "frame_id":    int,      # absolute frame number in the video
  "play_id":     str,      # unique play identifier e.g. "g01_p04"
  "game_id":     str,      # unique game identifier e.g. "g01"
  "player_id":   int,      # tracker-assigned ID, consistent within a play
  "team":        str,      # "offense" | "defense" | "official"
  "x":           float,    # field coord — yards from line of scrimmage
  "y":           float,    # field coord — yards from left sideline
  "bbox":        list,     # [x1, y1, x2, y2] pixel coords (raw)
  "confidence":  float,    # detection confidence score
  "timestamp":   float     # seconds relative to snap frame (negative = pre-snap)
}
```

### Per-Play Metadata Record
```python
{
  "play_id":     str,      # unique play identifier
  "game_id":     str,
  "snap_frame":  int,      # frame number of the snap
  "down":        int,      # 1-4
  "distance":    int,      # yards to go
  "hash":        str,      # "left" | "middle" | "right"
  "play_type":   str,      # "run" | "pass"
  "result":      float,    # yards gained (negative = loss)
  "complete":    bool,     # pass completion (None for run plays)
  "formation":   str,      # filled in by formation_classifier.py
  "camera_angle": str      # "endzone" | "sideline"
}
```

---

## Pipeline Stages (in order)

1. **Frame Extraction** (`pipeline/extract_frames.py`)
   - Input: raw .mp4 files in `data/raw/`
   - Output: frames at 10fps saved to `data/frames/{game_id}/{play_id}/`

2. **Player Detection** (`pipeline/detect.py`)
   - Input: frames
   - Model: YOLOv8 fine-tuned weights from `models/weights/`
   - Output: YOLO-format detection results per frame

3. **Team Separation** (`pipeline/team_separation.py`)
   - Input: detection bounding boxes + original frames
   - Method: HSV color masking on cropped player bounding boxes
   - Output: team label added to each detection

4. **Player Tracking** (`pipeline/track.py`)
   - Input: per-frame detections with team labels
   - Model: ByteTrack (via supervision)
   - Output: tracking CSV saved to `data/tracking/{game_id}.csv`

5. **Homography Transform** (`pipeline/homography.py`)
   - Input: pixel (x, y) centroids from tracker
   - Method: OpenCV findHomography using yard lines as reference points
   - Output: field (x, y) coordinates appended to tracking CSV

6. **Snap Frame Tagging**
   - Currently: manual tagging in per-play metadata CSV
   - Future: automatic detection via offensive line movement signal

---

## Analytics Modules (depend on completed pipeline)

- **Formation Classifier** — K-means on pre-snap coordinates → classifier
- **Heat Maps** — play result points plotted on field grid, weighted by yards gained
- **Route Classifier** — DTW clustering on receiver post-snap trajectories
- **Coverage Detection** — DB vs WR relative distance post-snap (man/zone signal)
- **Tendency Report** — formation + down/distance → play type/direction tendencies

---

## Key Conventions
- All field coordinates: origin at **bottom-left of end zone**, x = length (0-100 yds), y = width (0-53.3 yds)
- Play IDs follow format: `g{game_number}_p{play_number}` e.g. `g01_p004`
- Snap frame timestamp = 0.0, pre-snap = negative, post-snap = positive
- Team labels: always lowercase strings `"offense"`, `"defense"`, `"official"`
- Tracking CSVs saved as Parquet for performance, metadata as CSV

---

## Current Status
- [ ] Film acquired
- [ ] Labeling complete
- [ ] YOLOv8 fine-tuned
- [ ] Team separation working
- [ ] ByteTrack integrated
- [ ] Homography calibrated
- [ ] Snap frames tagged
- [ ] Formation classifier built
- [ ] Heat maps built
- [ ] Route classifier built

---

## Notes & Known Issues
- Pose estimation was ruled out — film quality (Hudl exports) is too low resolution
- Ball detection is out of scope for now
- Homography needs to be recalibrated per unique camera angle/position
- Player re-ID across plays is not yet implemented — player_id resets each play
