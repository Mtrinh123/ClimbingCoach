import math

L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 11, 12, 13, 14, 15, 16
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = 23, 24, 25, 26, 27, 28
TRACKED = {"l_wrist": L_WRIST, "r_wrist": R_WRIST, "l_elbow": L_ELBOW, "r_elbow": R_ELBOW,
           "l_ankle": L_ANKLE, "r_ankle": R_ANKLE}

BENT_ARM = 120       # elbow angle (degrees) below this is a bent arm
STILL = 0.15         # body moved less than this (in torso lengths) between frames = holding still
MIN_REST_S = 1.0     # holding still on bent arms this long is worth a tip
HIGH_REACH = 0.4     # a hand this far (torso lengths) above the shoulders is a high reach
STRAIGHT_LEGS = 1.2  # hips-to-feet over this many torso lengths = legs straight, feet low
LONG_PAUSE_S = 3.0   # holding still anywhere this long = hesitating on the wall
OVERHEAD_S = 3.0     # both hands above the shoulders this long drains blood from the forearms
# Feet off the wall = hanging straight down. Tuned on a real video: catches the dangle, nothing else
HANGING_LEGS = 1.1   # hips-to-feet over this many torso lengths
FEET_UNDER = 0.4     # midpoint of the feet within this far sideways of the hips
FEET_TOGETHER = 0.8  # ankles closer than this sideways


def angle(a, b, c):
    """Angle at b, in degrees, between points a-b-c."""
    ab = (a[0] - b[0], a[1] - b[1])
    cb = (c[0] - b[0], c[1] - b[1])
    dot = ab[0] * cb[0] + ab[1] * cb[1]
    norm = math.hypot(*ab) * math.hypot(*cb)
    return math.degrees(math.acos(max(-1, min(1, dot / norm)))) if norm else 0.0


def measure(points, seen):
    """Technique signals for one frame. points are pixel (x, y), y grows downward; seen says which are visible."""
    m = {}
    for side, (s, e, w) in {"left": (L_SHOULDER, L_ELBOW, L_WRIST), "right": (R_SHOULDER, R_ELBOW, R_WRIST)}.items():
        if seen[s] and seen[e] and seen[w]:
            m[f"{side}_elbow"] = round(angle(points[s], points[e], points[w]))
    if seen[L_SHOULDER] and seen[R_SHOULDER]:
        m["facing_camera"] = points[L_SHOULDER][0] > points[R_SHOULDER][0]
    if not all(seen[i] for i in (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)):
        return m
    hip = ((points[L_HIP][0] + points[R_HIP][0]) / 2, (points[L_HIP][1] + points[R_HIP][1]) / 2)
    shoulder_y = (points[L_SHOULDER][1] + points[R_SHOULDER][1]) / 2
    torso = hip[1] - shoulder_y
    if torso <= 0:
        return m
    m["pose"] = {name: [round((points[i][0] - hip[0]) / torso, 3), round((points[i][1] - hip[1]) / torso, 3)]
                 for name, i in TRACKED.items() if seen[i]}
    if seen[L_ANKLE] and seen[R_ANKLE]:
        feet_y = (points[L_ANKLE][1] + points[R_ANKLE][1]) / 2
        m["legs_vs_torso"] = round((feet_y - hip[1]) / torso, 2)
    return m


def movement(before, after):
    """How much the body shape changed between two frames: the farthest any tracked point moved.

    The largest, not the average: moving one hand is a move, even if everything else stays put."""
    shared = before.keys() & after.keys()
    if not shared:
        return None
    return max(math.dist(before[k], after[k]) for k in shared)


def bent_arm(metrics):
    return min(metrics.get("left_elbow", 180), metrics.get("right_elbow", 180)) < BENT_ARM


def hands_overhead(metrics):
    pose = metrics.get("pose", {})
    return all(k in pose and pose[k][1] < -1 for k in ("l_wrist", "r_wrist"))


def feet_off(metrics):
    pose = metrics.get("pose", {})
    if not (hands_overhead(metrics) and "l_ankle" in pose and "r_ankle" in pose):
        return False
    left, right = pose["l_ankle"][0], pose["r_ankle"][0]
    return (metrics.get("legs_vs_torso", 0) > HANGING_LEGS
            and abs((left + right) / 2) < FEET_UNDER and abs(left - right) < FEET_TOGETHER)


def high_reach_feet_low(metrics):
    """A hand well above the shoulders while the legs are straight: the feet are low for this reach."""
    pose = metrics.get("pose", {})
    hand_y = min((pose[k][1] for k in ("l_wrist", "r_wrist") if k in pose), default=None)
    return (hand_y is not None and hand_y < -1 - HIGH_REACH
            and metrics.get("legs_vs_torso", 0) > STRAIGHT_LEGS)


def runs(frames, test):
    """Groups consecutive frames that pass test into (first, last) pairs."""
    groups, start = [], None
    for prev, f in zip([None] + frames, frames + [None]):
        if f is not None and test(f, prev):
            start = start or f
            last = f
        elif start:
            groups.append((start, last))
            start = None
    return groups


def still(f, prev):
    if prev is None:
        return False
    moved = movement(prev["metrics"].get("pose", {}), f["metrics"].get("pose", {}))
    return moved is not None and moved < STILL


def find_tips(frames):
    """Coaching tips for a climb. frames: dicts with time_s, climbing and metrics, in time order."""
    climbing = [f for f in frames if f.get("climbing")]
    tips = []

    def tip(start, end, name, message):
        tips.append({"start_s": start, "end_s": end, "tip": name, "message": message})

    def still_spans(test, min_s):
        """Stretches of holding still that also pass test, lasting at least min_s seconds."""
        for first, last in runs(climbing, lambda f, prev: still(f, prev) and test(f["metrics"])):
            # first is the end of the first still step, so the stretch began one frame earlier
            start = climbing[climbing.index(first) - 1]["time_s"]
            if last["time_s"] - start >= min_s:
                yield start, last["time_s"]

    for start, end in still_spans(bent_arm, MIN_REST_S):
        tip(start, end, "rest_on_bent_arms",
            "You're holding still with bent arms. Straighten them to rest on your skeleton, not your muscles.")

    for start, end in still_spans(lambda m: True, LONG_PAUSE_S):
        tip(start, end, "long_pause",
            "Long pause on the wall. Plan the sequence from the ground so you can climb through without stopping.")

    for first, last in runs(climbing, lambda f, _: hands_overhead(f["metrics"])):
        if last["time_s"] - first["time_s"] >= OVERHEAD_S:
            tip(first["time_s"], last["time_s"], "hands_overhead",
                "Both hands above your head for a while. Drop one arm and shake it out when you can.")

    for first, last in runs(climbing, lambda f, _: feet_off(f["metrics"])):
        tip(first["time_s"], last["time_s"], "feet_off",
            "Your feet came off the wall. Keep weight on your feet to save your arms.")

    for first, last in runs(climbing, lambda f, _: high_reach_feet_low(f["metrics"]) and not feet_off(f["metrics"])):
        tip(first["time_s"], last["time_s"], "feet_low_on_reach",
            "Big reach with straight legs. Step your feet up first so the hold comes to you.")

    return sorted(tips, key=lambda t: t["start_s"])
