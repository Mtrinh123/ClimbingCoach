"""Run with: python test_coach.py (no video or MediaPipe needed)."""
import coach
from coach import angle, find_tips, measure, movement


def body(scale=1.0, facing=False, elbow=(0, 0.5), wrist=(0, 1.0), ankle_y=2.0):
    """33 fake landmarks for a climber seen from behind, in torso lengths around the hips (y grows down).

    elbow and wrist are the left arm's offsets from the left shoulder; the right arm hangs straight.
    """
    points = [(0.0, 0.0)] * 33
    flip = -1 if facing else 1  # facing the camera swaps left and right in the picture
    def at(x, y):
        return (500 + x * 100 * scale, 500 + y * 100 * scale)
    points[coach.L_SHOULDER] = at(-0.5 * flip, -1)
    points[coach.R_SHOULDER] = at(0.5 * flip, -1)
    points[coach.L_ELBOW] = at(-0.5 * flip + elbow[0], -1 + elbow[1])
    points[coach.L_WRIST] = at(-0.5 * flip + wrist[0], -1 + wrist[1])
    points[coach.R_ELBOW] = at(0.5 * flip, 0)
    points[coach.R_WRIST] = at(0.5 * flip, 1)
    points[coach.L_HIP] = at(-0.3 * flip, 0)
    points[coach.R_HIP] = at(0.3 * flip, 0)
    points[coach.L_ANKLE] = at(-0.4 * flip, ankle_y)
    points[coach.R_ANKLE] = at(0.4 * flip, ankle_y)
    return points


SEEN = [True] * 33
BENT = {"elbow": (0.5, 0.5), "wrist": (0, 0)}   # left arm folded: about 90 degrees at the elbow
REACH = {"elbow": (0, -0.6), "wrist": (0, -1.2)}  # left hand well above the shoulders


def frames(*poses, climbing=True):
    """Frames half a second apart, like the default 2 frames per second."""
    return [{"time_s": i * 0.5, "climbing": climbing, "metrics": measure(p, SEEN)} for i, p in enumerate(poses)]


# Angles
assert round(angle((1, 0), (0, 0), (0, 1))) == 90
assert round(angle((-1, 0), (0, 0), (1, 0))) == 180
assert angle((0, 0), (0, 0), (1, 1)) == 0.0  # zero-length arm doesn't crash

# Which way the climber faces
assert measure(body(), SEEN)["facing_camera"] is False
assert measure(body(facing=True), SEEN)["facing_camera"] is True

# Body shape is measured in torso lengths, so camera distance doesn't change it
near, far = measure(body(scale=2), SEEN), measure(body(scale=0.5), SEEN)
assert near["pose"] == far["pose"] and near["legs_vs_torso"] == far["legs_vs_torso"] == 2.0

# Missing shoulders: no body-shape measurements, and no crash
no_shoulders = [s and i not in (coach.L_SHOULDER, coach.R_SHOULDER) for i, s in enumerate(SEEN)]
assert "pose" not in measure(body(), no_shoulders)

# Movement between frames
assert movement({"a": [0, 0]}, {"a": [3, 4]}) == 5
assert movement({"a": [0, 0]}, {"b": [1, 1]}) is None

# Tip: holding still on bent arms for 1 second or more
bent_still = body(ankle_y=0.8, **BENT)
assert [t["tip"] for t in find_tips(frames(*[bent_still] * 3))] == ["rest_on_bent_arms"]
assert find_tips(frames(*[bent_still] * 2)) == []                        # only half a second
assert find_tips(frames(*[bent_still] * 4, climbing=False)) == []        # not on the wall
moving = [body(ankle_y=0.8, elbow=(0.5, 0.5), wrist=(x, 0)) for x in (0, 0.6, 0, 0.6)]
assert find_tips(frames(*moving)) == []                                  # bent arms mid-move are fine
assert find_tips(frames(*[body(ankle_y=0.8)] * 4)) == []                 # resting on straight arms is good

# Tip: big reach with straight legs
reach = find_tips(frames(body(ankle_y=0.8), body(ankle_y=2.0, **REACH)))
assert [(t["tip"], t["start_s"]) for t in reach] == [("feet_low_on_reach", 0.5)]
assert find_tips(frames(body(ankle_y=0.8, **REACH))) == []               # same reach with feet up is fine

print("angle, facing, body shape, movement and both coaching tips: all checks passed")
