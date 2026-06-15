# Current Action Recognition Report

## Learning setup

The current action-recognition model is **supervised**, not unsupervised.

Why:
- it uses manually annotated labels from the Excel ground truth
- training windows are assigned labels such as `forehand`, `backhand`, `serve`, and `neutral`
- the classifier is fitted directly on these labeled examples

So this is a **supervised skeleton-based action-classification pipeline**.

## Current pretrained model status

The current best saved model is:
- `random_forest`

Current held-out test metrics from the existing split:

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| Linear SVM | 0.8734 | 0.4202 |
| Logistic Regression | 0.6436 | 0.2365 |
| Random Forest | 0.9718 | 0.4896 |
| LSTM | 0.6463 | 0.2328 |

## Honest interpretation

The model **partially works**, but it is **not yet strong enough** to claim robust tennis stroke recognition.

What works:
- `serve` is recognized well on the current test split
- `neutral` is recognized very well

What does not work well enough:
- `forehand` generalization is weak on the current held-out evaluation
- `backhand` is not truly validated in the held-out split because the test videos do not contain confirmed backhand samples

## Main limitation

The current split is imbalanced:
- train videos: `tennis_video_1`, `tennis_video_4`, `tennis_video_6`
- test videos: `tennis_video_2`, `tennis_video_5`

This means:
- test mostly contains `serve`, `forehand`, and lots of `neutral`
- no confirmed held-out `backhand` exists in the test set

## Conclusion

So the correct conclusion is:

> The current action-recognition model is supervised and operational, but not yet fully reliable for final tennis stroke classification. It is good enough as a working baseline and inference engine for pipeline integration, but it still needs better class balance and stronger held-out validation for forehand/backhand performance.

