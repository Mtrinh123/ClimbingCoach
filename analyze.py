import json
import sys
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

from coach import BENT_ARM, L_ELBOW, R_ELBOW, bent_arm, find_tips, measure

HERE = Path(__file__).parent
MODEL = HERE / "pose_landmarker_full.task"
OUT = HERE / "out"

BONES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
         (23, 25), (25, 27), (24, 26), (26, 28)]
MIN_VISIBILITY = 0.5


def main(video, fps=2.0):
    if not MODEL.exists():
        sys.exit(f"Missing {MODEL.name}. Download it next to this script first.")
    OUT.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    step = max(1, round(cap.get(cv2.CAP_PROP_FPS) / fps))
    options = PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(MODEL)),
                                    running_mode=RunningMode.VIDEO)
    frames, index = [], 0
    with PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index % step == 0:
                ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ms)
                frames.append(annotate(frame, result, ms, len(frames)))
            index += 1
    cap.release()

    found = [f for f in frames if f["body_found"]]
    # Climbing = back to the camera (filmed from behind). Facing it, or shoulders not seen, means off the wall
    climbing = [f for f in found if f["metrics"].get("facing_camera") is False]
    for f in frames:
        f["climbing"] = f in climbing
    bent = [f for f in climbing if bent_arm(f["metrics"])]
    tips = find_tips(frames)
    report = {
        "frames_checked": len(frames),
        "body_found_in": len(found),
        "climbing_frames": len(climbing),
        "bent_arm_frames": [f["time_s"] for f in bent],
        "tips": tips,
        "frames": frames,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print(f"Checked {len(frames)} frames, body found in {len(found)} ({100 * len(found) // max(1, len(frames))}%), climbing in {len(climbing)}.")
    print(f"Bent-arm moments (elbow < {BENT_ARM} deg): {len(bent)} frames at {report['bent_arm_frames']} s")
    print(f"Tips ({len(tips)}):")
    for t in tips:
        print(f"  {t['start_s']}-{t['end_s']}s  {t['message']}")
    print(f"Annotated frames and report.json are in {OUT}")


def annotate(frame, result, ms, n):
    h, w = frame.shape[:2]
    entry = {"time_s": round(ms / 1000, 1), "body_found": bool(result.pose_landmarks), "metrics": {}}
    if result.pose_landmarks:
        lms = result.pose_landmarks[0]
        points = [(lm.x * w, lm.y * h) for lm in lms]
        seen = [(lm.visibility or 0) >= MIN_VISIBILITY for lm in lms]
        entry["metrics"] = measure(points, seen)
        for a, b in BONES:
            if seen[a] and seen[b]:
                cv2.line(frame, tuple(map(int, points[a])), tuple(map(int, points[b])), (80, 220, 80), 6)
        for i in (L_ELBOW, R_ELBOW):
            if seen[i] and f"{'left' if i == L_ELBOW else 'right'}_elbow" in entry["metrics"]:
                deg = entry["metrics"][f"{'left' if i == L_ELBOW else 'right'}_elbow"]
                color = (60, 60, 255) if deg < BENT_ARM else (255, 255, 255)
                cv2.putText(frame, f"{deg}", tuple(map(int, points[i])), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 5)
    cv2.putText(frame, f"{entry['time_s']}s", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 6)
    small = cv2.resize(frame, (w // 3, h // 3))
    cv2.imwrite(str(OUT / f"frame_{n:03d}.jpg"), small)
    return entry


DEFAULT_VIDEO = Path("dji_export_20261022_111050_0.f_compose_0.MOV")

if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VIDEO
    if not Path(video).exists():
        sys.exit(f"Video not found: {video}\nUsage: python analyze.py <video> [frames_per_second]")
    main(video, float(sys.argv[2]) if len(sys.argv) > 2 else 2.0)
