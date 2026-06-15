# Final Tennis Pipeline

This repository merges the older motion-analysis pipeline and the newer action-classification pipeline into one organized project that can be pushed to GitHub and reused by other users.

## What this pipeline does

The final merged pipeline supports two major modes:

1. **Supervised training / evaluation**
2. **Inference on a new raw tennis video using pretrained checkpoints**

## Important clarification

The current action-recognition model is **supervised**.

It is trained from labeled pose windows extracted from your manually annotated Excel ground truth. It is **not** an unsupervised model.

## Repository structure

```text
Final_pipeline/
  src/
    tennis_pipeline/
      pose_utils.py
    tennis_action/
      common.py
      dataset_utils.py
      preprocess.py
      lstm_model.py
  scripts/
    inspect_players.py
    track_player_pose.py
    prepare_pose_dataset.py
    parse_gt_excel.py
    export_action_windows.py
    analyze_motion_features.py
    track_racket.py
    make_explained_motion_video.py
    build_dataset.py
    train_models.py
    evaluate_and_render.py
    train_action_pipeline.py
    predict_actions_on_video.py
    render_action_predictions.py
    run_complete_pipeline.py
  checkpoints/
    action_recognition/
      current/
  reports/
    current_model_report.md
  notebooks/
    final_pipeline_demo.ipynb
  requirements.txt
```

## The two old pipelines and how they are merged

### Old `Project_pipeline`

Main purpose:
- player inspection
- pose tracking
- motion analysis
- racket tracking
- explained motion videos

### Old `codex`

Main purpose:
- build skeleton datasets
- train action-recognition models
- evaluate frame-level predictions
- render action prediction videos

### Merged design in `Final_pipeline`

The merged pipeline now works like this:

1. inspect/select player
2. track player pose from raw video
3. run pretrained action recognition on every frame
4. convert frame predictions into predicted action segments
5. compute motion-analysis features from those predicted segments
6. optionally track racket
7. export:
   - action recognition video
   - motion analysis video
   - prediction CSV/JSON
   - motion features CSV/JSON

## Installation

Create an environment and install:

```bash
pip install -r requirements.txt
```

## Pretrained checkpoints

Current pretrained checkpoints are stored here:

- `checkpoints/action_recognition/current`

Included files:
- `random_forest.joblib`
- `linear_svm.joblib`
- `logistic.joblib`
- `lstm.pt`
- `scaler.joblib`
- training summary and metrics summary

## How to train the action-recognition model

Use this when you want to rebuild the supervised classifier from the existing labeled dataset.

```bash
python scripts/train_action_pipeline.py \
  --excel /media/cv/HDD/Throwing_Dataset/Tennis_Internship/Tennise_dataset_GT.xlsx \
  --video-dir /media/cv/HDD/Throwing_Dataset/Tennis_Internship/videos \
  --pose-root /media/cv/HDD/Throwing_Dataset/Tennis_Internship/Final_pipeline/outputs/pose_dataset \
  --output-root /media/cv/HDD/Throwing_Dataset/Tennis_Internship/Final_pipeline/outputs/train_eval \
  --prepare-pose \
  --pose-model /media/cv/HDD/Throwing_Dataset/Tennis_Internship/yolo11x-pose.pt \
  --device cuda:0
```

This produces:
- dataset files
- trained models
- evaluation metrics
- GT-vs-prediction evaluation videos

If you already have tracked pose CSV files, you can skip `--prepare-pose` and point `--pose-root` to that existing pose directory tree.

## How to run inference on a new raw tennis video

This is the main user-facing command.

```bash
python scripts/run_complete_pipeline.py \
  --video /media/cv/HDD/Throwing_Dataset/Tennis_Internship/videos/tennis_video_2.mov \
  --pose-model /media/cv/HDD/Throwing_Dataset/Tennis_Internship/yolo11x-pose.pt \
  --checkpoint-dir /media/cv/HDD/Throwing_Dataset/Tennis_Internship/Final_pipeline/checkpoints/action_recognition/current \
  --output-root /media/cv/HDD/Throwing_Dataset/Tennis_Internship/Final_pipeline/outputs/inference \
  --target-person-id largest \
  --pose-device 0 \
  --handedness right \
  --h264
```

Optional racket tracking:

```bash
python scripts/run_complete_pipeline.py \
  --video /media/cv/HDD/Throwing_Dataset/Tennis_Internship/videos/tennis_video_2.mov \
  --pose-model /media/cv/HDD/Throwing_Dataset/Tennis_Internship/yolo11x-pose.pt \
  --checkpoint-dir /media/cv/HDD/Throwing_Dataset/Tennis_Internship/Final_pipeline/checkpoints/action_recognition/current \
  --output-root /media/cv/HDD/Throwing_Dataset/Tennis_Internship/Final_pipeline/outputs/inference \
  --target-person-id largest \
  --pose-device 0 \
  --handedness right \
  --detector-model /media/cv/HDD/Throwing_Dataset/TU_D/ultralytics/yolov8s.pt \
  --h264
```

## Outputs from the complete pipeline

For each video, the complete pipeline creates:

- `inspection/`
- `pose/`
- `action_predictions/`
- `action_video/`
- `motion_analysis/`
- optional `racket_tracking/`
- `motion_video/`

### Video outputs

1. **Action recognition video**
   - side-by-side RGB and 2D pose prediction view
   - uses the pretrained classifier output

2. **Motion analysis video**
   - uses the predicted action segments
   - visualizes wrist motion, stroke window, and motion proxies
   - can optionally include racket tracking

## Current model quality

Please read:
- [reports/current_model_report.md](reports/current_model_report.md)

Short honest summary:
- the current model works as a baseline
- it is good enough to integrate into a full pipeline
- it is **not yet strong enough** to claim robust final forehand/backhand recognition
- serve and neutral work much better than forehand/backhand

## Can these two old pipelines be merged?

Yes. That is exactly what this folder is doing.

The correct final design is:

- **training mode** for supervised learning
- **inference mode** for any unseen tennis video

That is the cleanest way for a GitHub-style repo and for other users.
