# Tennis Action Recognition Pipeline

This repository is a complete tennis-video analysis pipeline for:

- player selection and tracking
- 2D human pose extraction
- supervised tennis action recognition
- motion-proxy analysis from pose
- explained visualization videos for qualitative review

The current project is focused on tennis actions such as:

- `forehand`
- `backhand`
- `serve`
- `neutral` (no annotated stroke window)

This repo combines two earlier codebases into one cleaner workflow:

1. a pose and motion-analysis pipeline
2. a supervised action-classification pipeline

The result is a single repo that can be used both for:

- training and evaluation on labeled tennis videos
- inference on a new unseen tennis video

## What This Project Does

The pipeline takes a tennis video and does the following:

1. detects the people in the frame
2. selects one player to track
3. runs pose estimation frame by frame on that player
4. saves the tracked keypoints to CSV
5. converts the pose sequence into fixed windows
6. runs a pretrained supervised action-recognition model
7. predicts action labels frame by frame
8. groups frame predictions into stroke segments
9. computes motion-analysis features from the predicted segments
10. exports side-by-side visualization videos and structured result files

There are two main modes:

- **Mode 1: supervised training/evaluation**
- **Mode 2: inference on a new raw video**

## Repository Structure

```text
Final_pipeline/
├── checkpoints/
│   └── action_recognition/
│       └── current/
├── notebooks/
│   └── final_pipeline_demo.ipynb
├── reports/
│   └── current_model_report.md
├── scripts/
│   ├── analyze_motion_features.py
│   ├── build_dataset.py
│   ├── evaluate_and_render.py
│   ├── export_action_windows.py
│   ├── inspect_players.py
│   ├── make_explained_motion_video.py
│   ├── parse_gt_excel.py
│   ├── predict_actions_on_video.py
│   ├── prepare_pose_dataset.py
│   ├── render_action_predictions.py
│   ├── run_complete_pipeline.py
│   ├── track_player_pose.py
│   ├── track_racket.py
│   ├── train_action_pipeline.py
│   └── train_models.py
├── src/
│   ├── tennis_action/
│   │   ├── common.py
│   │   ├── dataset_utils.py
│   │   ├── lstm_model.py
│   │   └── preprocess.py
│   └── tennis_pipeline/
│       └── pose_utils.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Core Workflow

### A. Training / Evaluation Workflow

This path is for building a supervised action classifier from labeled videos.

1. parse the Excel annotations
2. prepare tracked pose CSV files for the selected videos
3. assign frame labels using the annotation contact frame and a fixed window
4. build pose-sequence training samples
5. normalize the pose sequences
6. train classical ML models and an LSTM baseline
7. evaluate on held-out videos
8. render GT-vs-prediction videos

### B. Inference Workflow

This path is for analyzing a new tennis video using a pretrained model.

1. inspect player candidates
2. select and track one player
3. extract pose keypoints
4. run the pretrained action model
5. generate frame predictions and predicted segments
6. compute motion-analysis features
7. render:
   - action-recognition video
   - explained motion video

## Models Used

### 1. Pose Estimation Backbone

This pipeline uses **Ultralytics YOLO11 pose** checkpoints, specifically a checkpoint such as:

- `yolo11x-pose.pt`

In this project, pose estimation is used to extract 17 human body keypoints per frame:

- nose
- eyes
- ears
- shoulders
- elbows
- wrists
- hips
- knees
- ankles

These keypoints are then used to compute:

- pose windows for action classification
- wrist motion proxies
- torso rotation proxies
- follow-through and path-length proxies

### 2. Action Recognition Models

This repo contains a supervised pose-based classification stage with:

- Linear SVM
- Logistic Regression
- Random Forest
- LSTM

The current best saved checkpoint in this repo is the **Random Forest** baseline.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Bikudebug/tennis_action_recognisation.git
cd tennis_action_recognisation
```

### 2. Create an environment

Example with conda:

```bash
conda create -n tennis_action python=3.10 -y
conda activate tennis_action
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install PyTorch

Install the PyTorch build that matches your CUDA version from the official PyTorch site if needed.

### 5. Install Ultralytics

If `ultralytics` is not already installed through your environment setup:

```bash
pip install ultralytics
```

## YOLO11 Pose Model Setup

This repo expects a pose checkpoint path such as:

```text
yolo11x-pose.pt
```

You have two common options:

### Option A: use a local checkpoint you already downloaded

Example:

```bash
--pose-model /path/to/yolo11x-pose.pt
```

### Option B: let Ultralytics download it automatically once

```bash
python -c "from ultralytics import YOLO; YOLO('yolo11x-pose.pt')"
```

After that, you can pass the downloaded checkpoint path to the pipeline.

## Input Data Expected by This Repo

### Raw videos

The pipeline expects standard tennis videos, for example:

```text
videos/tennis_video_1.MOV
```

### Excel ground-truth file

For supervised training, the pipeline expects an Excel file with annotated action frames, for example:

```text
Tennise_dataset_GT.xlsx
```

In the original workflow for this project:

- videos `1, 4, 6` were used for training
- videos `2, 5` were used for testing
- uncertain labels such as `D`, `DS`, and doubtful notes were excluded from clean training labels

## How Frame Labels Are Built

For supervised action recognition, each annotated contact/proxy frame is expanded into a fixed window.

Current default:

- window radius = `15`
- window size = `31` frames total

Example:

- if a `forehand` is annotated at frame `100`
- the labeled action window becomes `85` to `115`

Frames outside any action window are labeled:

- `neutral`

This is how the full video sequence becomes a frame-wise labeled dataset.

## Pose Preprocessing Used for Classification

Before classification, the pose sequence is normalized using body-centered transformations so the model depends less on camera position and subject scale.

The preprocessing stage includes:

- pose-window extraction
- root/body-centered normalization
- body-scale normalization
- missing-keypoint handling with `NaN` support
- fixed-size window formatting

This makes the classifier focus more on relative motion and posture than on absolute pixel coordinates.

## How To Run the Full Pipeline on a New Video

This is the main end-to-end command for inference:

```bash
python scripts/run_complete_pipeline.py \
  --video /path/to/tennis_video_1.MOV \
  --pose-model /path/to/yolo11x-pose.pt \
  --checkpoint-dir checkpoints/action_recognition/current \
  --output-root outputs/inference \
  --target-person-id largest \
  --pose-device 0 \
  --handedness right \
  --h264
```

### What this command does

It runs:

1. `inspect_players.py`
2. `track_player_pose.py`
3. `predict_actions_on_video.py`
4. `render_action_predictions.py`
5. `analyze_motion_features.py`
6. `make_explained_motion_video.py`

Optional:

7. `track_racket.py`

## Optional Racket Tracking

If you also want the racket stage:

```bash
python scripts/run_complete_pipeline.py \
  --video /path/to/tennis_video_1.MOV \
  --pose-model /path/to/yolo11x-pose.pt \
  --checkpoint-dir checkpoints/action_recognition/current \
  --output-root outputs/inference \
  --target-person-id largest \
  --pose-device 0 \
  --handedness right \
  --detector-model /path/to/racket_detector.pt \
  --h264
```

Racket tracking in this repo is optional and depends on visibility, scale, motion blur, and video quality.

## How To Train the Supervised Action Model

If you want to rebuild the pose classifier from labeled videos:

```bash
python scripts/train_action_pipeline.py \
  --excel /path/to/Tennise_dataset_GT.xlsx \
  --video-dir /path/to/videos \
  --pose-root /path/to/outputs/pose_dataset \
  --output-root /path/to/outputs/train_eval \
  --prepare-pose \
  --pose-model /path/to/yolo11x-pose.pt \
  --device cuda:0
```

### What this training command does

1. prepares tracked pose CSVs for the selected train/test videos
2. parses the Excel labels
3. builds frame-level and sequence-level datasets
4. trains the saved models
5. evaluates the models
6. renders GT-vs-prediction videos

If your pose CSVs already exist, you can skip `--prepare-pose`.

## Important Scripts and What They Do

### `inspect_players.py`

- runs YOLO pose on one selected frame
- shows candidate detections
- helps choose which player to track

### `track_player_pose.py`

- tracks one player through the whole video
- writes tracked pose CSV
- writes tracked pose video

### `prepare_pose_dataset.py`

- batch-prepares tracked pose CSV files for the labeled dataset videos

### `build_dataset.py`

- converts labeled videos and pose CSVs into train/test datasets

### `train_models.py`

- trains the supervised classifiers

### `evaluate_and_render.py`

- runs evaluation on held-out videos
- writes metrics and GT-vs-prediction renderings

### `predict_actions_on_video.py`

- uses a pretrained model to predict action labels frame by frame on a new video

### `render_action_predictions.py`

- creates the action-recognition side-by-side video

### `analyze_motion_features.py`

- computes pose-derived motion proxies from predicted or labeled stroke windows

### `make_explained_motion_video.py`

- creates the explained motion-analysis video with the left/right layout

## Output Structure

For each inference video, the pipeline creates a folder like:

```text
outputs/inference/<video_name>/
```

Inside that folder you will typically get:

- `inspection/`
- `pose/`
- `action_predictions/`
- `action_video/`
- `motion_analysis/`
- `explained_video/`
- optional `racket_tracking/`

### Important output files

#### Pose stage

- tracked pose CSV
- tracked pose visualization video

#### Action stage

- `frame_predictions.csv`
- `predicted_action_segments.csv`
- `prediction_summary.json`
- action-recognition video

#### Motion-analysis stage

- `frame_motion_features.csv`
- `action_window_features.csv`
- `motion_analysis_summary.json`
- explained motion video

## Current Saved Checkpoints

Saved checkpoints are included in:

```text
checkpoints/action_recognition/current
```

Included files:

- `random_forest.joblib`
- `linear_svm.joblib`
- `logistic.joblib`
- `lstm.pt`
- `scaler.joblib`
- `metrics_summary.json`
- `training_summary.json`
- `best_model_summary.json`

## Current Model Status

Please read:

- [reports/current_model_report.md](reports/current_model_report.md)

Short honest summary:

- this repo already works end to end
- it is useful as a proof-of-concept tennis analysis pipeline
- the current classifier is strongest on `serve` and `neutral`
- `forehand` / `backhand` generalization still needs improvement
- the current system should be treated as a strong baseline, not a final production model

## Known Limitations

- small labeled dataset
- class imbalance, especially for backhand
- uncertain labels in the source Excel file
- no camera calibration
- no true ball-contact ground truth
- racket tracking is harder when the racket is blurred or very small
- action recognition can be sensitive to viewpoint, player handedness, and pose noise

## Notebook

A small demo notebook is included here:

- [notebooks/final_pipeline_demo.ipynb](notebooks/final_pipeline_demo.ipynb)

## Citation and References

This repo uses **Ultralytics YOLO11 pose** as the human pose estimation backbone.

Primary sources:

- Ultralytics YOLO11 model documentation: https://docs.ultralytics.com/models/yolo11
- Ultralytics pose task documentation: https://docs.ultralytics.com/tasks/pose
- Ultralytics GitHub repository: https://github.com/ultralytics/ultralytics

Useful details from the official Ultralytics docs:

- YOLO11 supports pose estimation as one of its official tasks.
- YOLO11 pose checkpoints use the `-pose` suffix, such as `yolo11x-pose.pt`.
- Ultralytics pose models are trained on COCO keypoints and use the standard 17-keypoint human layout.

If you want to cite the Ultralytics YOLO11 software in academic or technical work, the Ultralytics docs provide a software citation format. A practical citation block is:

```bibtex
@software{yolo11_ultralytics,
  author = {Glenn Jocher and Jing Qiu},
  title = {Ultralytics YOLO11},
  version = {11.0.0},
  year = {2024},
  url = {https://github.com/ultralytics/ultralytics},
  license = {AGPL-3.0}
}
```

Please also cite this repository if you reuse the tennis-specific pipeline design, dataset preparation logic, or visualization workflow.

## Summary

If you are a new user, the shortest practical path is:

1. install dependencies
2. prepare or download `yolo11x-pose.pt`
3. run `run_complete_pipeline.py` on one tennis video
4. inspect the outputs:
   - tracked pose CSV
   - action-recognition video
   - explained motion video
5. if needed, rebuild the supervised model with `train_action_pipeline.py`

That is the intended workflow of this repository.
