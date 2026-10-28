"""Run with: python test_coach.py (no video or MediaPipe needed)."""
import coach
from coach import angle, find_tips, measure, movement


def body(scale=1.0, facing=False, elbow=(0, 0.5), wrist=(0, 1.0), ankle_y=2.0, right_up=False, ankle_x=0.4):
    points = [(0.0, 0.0)] * 33
    flip = -1 if facing else 1  # facing the camera swaps left and right in the picture
    def at(x, y):
        return (500 + x * 100 * scale, 500 + y * 100 * scale)
    points[coach.L_SHOULDER] = at(-0.5 * flip, -1)
    points[coach.R_SHOULDER] = at(0.5 * flip, -1)
    points[coach.L_ELBOW] = at(-0.5 * flip + elbow[0], -1 + elbow[1])
    points[coach.L_WRIST] = at(-0.5 * flip + wrist[0], -1 + wrist[1])
    points[coach.R_ELBOW] = at(0.5 * flip, -1.6 if right_up else 0)
    points[coach.R_WRIST] = at(0.5 * flip, -2.2 if right_up else 1)
    points[coach.L_HIP] = at(-0.3 * flip, 0)
    points[coach.R_HIP] = at(0.3 * flip, 0)
    points[coach.L_ANKLE] = at(-ankle_x * flip, ankle_y)
    points[coach.R_ANKLE] = at(ankle_x * flip, ankle_y)
    return points


SEEN = [True] * 33
BENT = {"elbow": (0.5, 0.5), "wrist": (0, 0)}   # left arm folded: about 90 degrees at the elbow
REACH = {"elbow": (0, -0.6), "wrist": (0, -1.2)}  # left hand well above the shoulders


def frames(*poses, climbing=True):
    """Frames half a second apart, like the default 2 frames per second."""
    return [{"time_s": i * 0.5, "climbing": climbing, "metrics": measure(p, SEEN)} for i, p in enumerate(poses)]



assert round(angle((1, 0), (0, 0), (0, 1))) == 90
assert round(angle((-1, 0), (0, 0), (1, 0))) == 180
assert angle((0, 0), (0, 0), (1, 1)) == 0.0  # zero-length arm doesn't crash


assert measure(body(), SEEN)["facing_camera"] is False
assert measure(body(facing=True), SEEN)["facing_camera"] is True


near, far = measure(body(scale=2), SEEN), measure(body(scale=0.5), SEEN)
assert near["pose"] == far["pose"] and near["legs_vs_torso"] == far["legs_vs_torso"] == 2.0

no_shoulders = [s and i not in (coach.L_SHOULDER, coach.R_SHOULDER) for i, s in enumerate(SEEN)]
assert "pose" not in measure(body(), no_shoulders)

assert movement({"a": [0, 0]}, {"a": [3, 4]}) == 5
assert movement({"a": [0, 0]}, {"b": [1, 1]}) is None

bent_still = body(ankle_y=0.8, **BENT)
assert [t["tip"] for t in find_tips(frames(*[bent_still] * 3))] == ["rest_on_bent_arms"]
assert find_tips(frames(*[bent_still] * 2)) == []                        # only half a second
assert find_tips(frames(*[bent_still] * 4, climbing=False)) == []        # not on the wall
moving = [body(ankle_y=0.8, elbow=(0.5, 0.5), wrist=(x, 0)) for x in (0, 0.6, 0, 0.6)]
assert find_tips(frames(*moving)) == []                                  # bent arms mid-move are fine
assert find_tips(frames(*[body(ankle_y=0.8)] * 4)) == []                 # resting on straight arms is good

reach = find_tips(frames(body(ankle_y=0.8), body(ankle_y=2.0, **REACH)))
assert [(t["tip"], t["start_s"]) for t in reach] == [("feet_low_on_reach", 0.5)]
assert find_tips(frames(body(ankle_y=0.8, **REACH))) == []               # same reach with feet up is fine

def tip_names(*poses):
    return [t["tip"] for t in find_tips(frames(*poses))]


# Tip: long pause anywhere (3+ seconds of holding still), even on straight arms
assert tip_names(*[body(ankle_y=0.8)] * 7) == ["long_pause"]             # 0 to 3.0 s
assert tip_names(*[body(ankle_y=0.8)] * 6) == []                          # 2.5 s is fine

# Tip: both hands overhead for 3+ seconds, even while moving
both_up = [body(ankle_y=0.8, right_up=True, ankle_x=x, **REACH) for x in (0.4, 0.9) * 4][:7]
assert tip_names(*both_up) == ["hands_overhead"]
assert tip_names(*both_up[:6]) == []
assert tip_names(*[body(ankle_y=0.8, **REACH)] * 7) == ["long_pause"]    # one hand up doesn't count

# Tip: feet off the wall (hanging straight down from both hands)
hanging = body(ankle_y=1.5, ankle_x=0.2, right_up=True, **REACH)
assert tip_names(body(ankle_y=0.8), hanging) == ["feet_off"]             # not also "feet low on reach"
assert tip_names(body(ankle_y=0.8), body(ankle_y=1.5, ankle_x=0.8, right_up=True, **REACH)) == [
    "feet_low_on_reach"]                                                   # feet wide apart: still on holds

print("angle, facing, body shape, movement and all five coaching tips: all checks passed")
