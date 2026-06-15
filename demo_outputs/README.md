# Demo Outputs

This folder contains a compact example output package from one inference run:

- `tennis_video_1`

It is intentionally limited to smaller, shareable artifacts such as:

- compressed H.264 demo videos
- pose CSV
- prediction CSV/JSON
- motion-analysis CSV/JSON
- player inspection preview files

The full raw output tree is **not** stored in Git because:

- it is much larger
- some files exceed normal GitHub file-size limits
- the uncompressed intermediate videos are not necessary for most users

If you want to reproduce these outputs yourself, run the pipeline on the same input video using the commands documented in the main [README.md](../README.md).
