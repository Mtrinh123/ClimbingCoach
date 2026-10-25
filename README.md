# 🧗 Climbing Coach

Upload a climbing video and get technique feedback. Free, local pose detection tracks your body on the wall and flags habits like resting on bent arms.

> **Status:** early prototype. The pose tracking works; the coaching rules are being refined.

## How it works

1. **Sample frames** from the video (2 per second by default) with OpenCV.
2. **Find the body** in each frame with [MediaPipe Pose](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker): 33 points such as shoulders, elbows, hips and ankles.
3. **Measure technique** from those points:
   - **Elbow angle:** under 120° counts as a bent arm, which tires you out fast
   - **Hip height vs. feet:** a first step toward body-position feedback
4. **Save the results:** annotated frames with the skeleton drawn on, plus a `report.json`.

## First results

On a 35-second bouldering video filmed from behind:

- The body was found in **65 of 70 frames (92%)**, and the skeleton lined up with the climber throughout the climb.

## Setup

Needs **Python 3.10+**.

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install mediapipe opencv-python
```

Download the pose model (about 9 MB) into this folder:

```bash
curl -L -o pose_landmarker_full.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
```

## Usage

```bash
python analyze.py path/to/climb.mov        # 2 frames per second
python analyze.py path/to/climb.mov 4      # 4 frames per second
```

Results go to `out/`:

| File | What it is |
|---|---|
| `frame_000.jpg` … | Sampled frames with the skeleton drawn on. Elbow angles in red are bent arms |
| `report.json` | Per-frame measurements and the times bent arms were found |

## Roadmap

- [x] Pose tracking on climbing video
- [x] Ignore frames that aren't climbing (uses which way the climber faces)
- [ ] Flag bent arms only while holding still
- [ ] More tips: feet not moving while reaching, hips far from the wall
- [ ] Web app: React + TypeScript frontend with pose detection in the browser, ASP.NET Core (C#) API
- [ ] Tests, CI and a live demo
