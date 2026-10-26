# 🧗 Climbing Coach

[![CI](https://github.com/Mtrinh123/ClimbingCoach/actions/workflows/ci.yml/badge.svg)](https://github.com/Mtrinh123/ClimbingCoach/actions/workflows/ci.yml)

Upload a climbing video and get technique feedback. Free, local pose detection tracks your body on the wall and flags habits like resting on bent arms.

> **Status:** early prototype. The pose tracking works; the coaching rules are being refined.

## How it works

1. **Sample frames** from the video (2 per second by default) with OpenCV.
2. **Find the body** in each frame with [MediaPipe Pose](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker): 33 points such as shoulders, elbows, hips and ankles.
3. **Skip frames that aren't climbing.** The video is filmed from behind, so a climber facing the camera has stepped off the wall.
4. **Measure body shape in torso lengths** from the hips, so how close the camera is and where it points don't matter.
5. **Give coaching tips:**

   | Tip | When it triggers |
   |---|---|
   | **Resting on bent arms** | Holding still for 1+ second with an elbow under 120°. Bent arms mid-move are fine |
   | **Feet low on a big reach** | A hand well above the shoulders while the legs are straight |

6. **Save the results:** annotated frames with the skeleton drawn on, plus a `report.json` with the tips.

## First results

On a 35-second bouldering video filmed from behind:

- The body was found in **65 of 70 frames (92%)**, and the skeleton lined up with the climber throughout the climb.
- **57 frames** were climbing; the walk away from the wall afterwards was skipped.
- **2 tips**, both confirmed by watching the video: resting on a bent arm at 16–17s, and a big reach with straight legs at 24s.

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
| `report.json` | Per-frame measurements and the coaching tips with their times |

## Project layout

| File | What it does |
|---|---|
| `analyze.py` | Reads the video, runs MediaPipe, draws the skeleton, writes the results |
| `coach.py` | Measurements and coaching rules. Plain Python, no video libraries |
| `test_coach.py` | Tests for `coach.py` using made-up body positions |

## Tests

```bash
python test_coach.py
```

## Roadmap

- [x] Pose tracking on climbing video
- [x] Ignore frames that aren't climbing (uses which way the climber faces)
- [x] Flag bent arms only while holding still
- [x] Tip: feet low on a big reach
- [ ] More tips: hips far from the wall, foot placement
- [ ] Web app: React + TypeScript frontend with pose detection in the browser, ASP.NET Core (C#) API
- [x] Tests and CI
- [ ] Live demo
