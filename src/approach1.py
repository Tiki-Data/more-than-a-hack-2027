"""Approach 1: the rule-based baseline, but with football's possession rules built in.

    python src/approach1.py      # -> out/scores/approach1.csv  (per-frame evidence, for the glue)
                                 #    out/approach1.csv         (a full submission on its own)

Two steps, kept separate so the glue / P2's HMM can use the first without the second:

1. EVIDENCE, per frame (out/scores/approach1.csv)
   - The ball is in a player's control when the nearest player is within CONTROL_RADIUS_M, the
     ball is slower than MAX_CONTROL_SPEED_MS, and no opponent is also within the radius.
     That frame says "this team", with probability EVIDENCE_CONFIDENCE.
   - Ball out of the pitch: says "D", same confidence.
   - Anything else (no ball, ball in flight, loose, contested): 0,0,0, no idea.

2. SWITCHING, the possession rules (out/approach1.csv)
   - A TOUCH is a stretch of control frames by one track slot, allowing short breaks
     (TOUCH_BREAK_FRAMES) where the ball drifts out of reach during a dribble.
   - Possession goes to a new team only when it is CONFIRMED: a touch of at least
     CONFIRM_FRAMES control frames (the player keeps it), or a completed PASS (a touch followed by
     a touch of a different teammate, at least MIN_PASS_DISTANCE_M away, within MAX_PASS_FRAMES,
     with no opponent touching it in between). A single unconfirmed touch -- a deflection, a block,
     a header -- changes nothing. That is the ground truth's sticky-possession rule.
   - The switch is BACKDATED to where the confirming touch began, so the boundary is placed where
     control started, not where we became sure of it.
     Two exceptions:
       * After a dead ball, the switch goes to the first frame the ball actually moves
         (RESTART_SPEED_MS): the taker standing over the ball is still D.
       * If the ball was missing for most of the time since the previous team's last evidence,
         the change happened somewhere we could not see, and the switch goes to the middle of that
         blind stretch. The midpoint minimises the worst-case boundary error when we know nothing.
   - Ball out of the pitch: D from the first frame it is out, until a team is confirmed.
   - Each half starts as D (the kick-off is a restart) and is decoded on its own. X outside.

Every constant below was set from football reasoning or from the tracking's own statistics on
this match (no labels), never from the public 5 minutes. The measurements quoted are reproduced by
`python src/approach1.py --stats`.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES  # noqa: E402
from src.load import load_frames  # noqa: E402
from src.submission import OUT_DIR, SCORES_DIR, validate, write_scores, write_submission  # noqa: E402

# Params
FPS = GAMES[GAME]["fps"]

# Control radius. With the ball slower than 3 m/s, the distance to the nearest player peaks at
# 0.25-0.75 m and has mostly tailed off by 1.5 m (59% of slow-ball frames are within it). Player
# tracking jitter is only ~0.6 cm/frame, so the radius is set by reach, not by noise: a stride plus
# a touch pushed slightly ahead.
CONTROL_RADIUS_M = 1.5

# A player cannot keep a ball under control while it moves faster than they can run (top sprint
# ~10 m/s), plus a margin for a ball pushed ahead. Measured: above 15 m/s the ball is within the
# control radius of anyone only 6% of the time, between 8 and 15 m/s 13%: those are passes and
# shots in flight passing players by.
MAX_CONTROL_SPEED_MS = 12.0

# A touch survives this many frames without control (0.2 s): the ball leaves the radius for a
# moment during a dribble or the tracker drops a frame.
TOUCH_BREAK_FRAMES = 5

# Frames of control that confirm a change of possession on their own: 0.4 s. A clean first touch
# takes 0.3-0.5 s; a deflection or a block is over in 1-3 frames.
CONFIRM_FRAMES = 10

# A completed pass confirms possession even if neither touch is long (one-touch play). The two
# players must be at least 5 m apart -- otherwise it is more likely the tracker swapping slots on
# one player -- and the ball must arrive within 3 s (a 40 m pass at ~13 m/s).
MIN_PASS_DISTANCE_M = 5.0
MAX_PASS_FRAMES = 3 * FPS

# The ball "moves" for a restart when it goes faster than walking pace with it: 3 m/s.
RESTART_SPEED_MS = 3.0
# How long after a dead-ball touch starts we look for the restart kick: 10 s covers setting up a
# free kick or a goal kick.
RESTART_SEARCH_FRAMES = 10 * FPS

# A blind stretch: if more than half the frames between the old team's last evidence and the new
# team's confirming touch had no ball, and the stretch lasts at least 1 s, place the switch in the
# middle of it.
BLIND_MIN_FRAMES = FPS
BLIND_MISSING_SHARE = 0.5

# Confidence of a single frame of evidence. Modest on purpose: one frame can be a deflection, and
# over-confident evidence overpowers the HMM's transition penalties and brings back flicker.
EVIDENCE_CONFIDENCE = 0.8

H, A, D, NONE = 0, 1, 2, -1
LETTERS = np.array(["H", "A", "D"])


# ---------------------------------------------------------------------------------------------
# Step 1: per-frame evidence
# ---------------------------------------------------------------------------------------------

def control_frames(F):
    """(team, slot) of the player in control on each frame, (-1, -1) when nobody is."""
    d = F["d_near"]
    dh = np.where(np.isfinite(d[:, 0]), d[:, 0], np.inf)
    da = np.where(np.isfinite(d[:, 1]), d[:, 1], np.inf)
    speed = F["ball_speed"]
    slow = ~(speed > MAX_CONTROL_SPEED_MS)  # NaN speed (edge of a gap) does not rule control out
    ok = slow & ~F["ball_out"]

    home = ok & (dh <= CONTROL_RADIUS_M) & (da > CONTROL_RADIUS_M)
    away = ok & (da <= CONTROL_RADIUS_M) & (dh > CONTROL_RADIUS_M)
    team = np.full(len(d), NONE, dtype=np.int8)
    team[home], team[away] = H, A
    slot = np.full(len(d), -1, dtype=np.int16)
    slot[home] = F["near_slot"][home, 0]
    slot[away] = F["near_slot"][away, 1]
    return team, slot


def evidence(F, team):
    """(n, 3) H/A/D probabilities per frame, 0,0,0 where approach 1 has no idea."""
    other = (1 - EVIDENCE_CONFIDENCE) / 2
    probs = np.zeros((len(team), 3))
    for t in (H, A):
        row = np.full(3, other)
        row[t] = EVIDENCE_CONFIDENCE
        probs[team == t] = row
    row = np.full(3, other)
    row[D] = EVIDENCE_CONFIDENCE
    probs[F["ball_out"]] = row
    return probs


# ---------------------------------------------------------------------------------------------
# Step 2: touches and the possession rules
# ---------------------------------------------------------------------------------------------

def touches(team, slot, xy, lo, hi):
    """Touches inside frames [lo, hi): list of dicts with team, slot, start, end, n, pos.

    A touch is one slot's control frames, bridging breaks of up to TOUCH_BREAK_FRAMES. It ends at
    a longer break or as soon as another slot has the ball.
    """
    out, cur = [], None
    for i in np.flatnonzero(team[lo:hi] != NONE) + lo:
        s = int(slot[i])
        if cur is not None and s == cur["slot"] and i - cur["end"] <= TOUCH_BREAK_FRAMES + 1:
            cur["end"] = int(i)
            cur["n"] += 1
            continue
        if cur is not None:
            out.append(cur)
        cur = {"team": int(team[i]), "slot": s, "start": int(i), "end": int(i), "n": 1,
               "pos": xy[i].astype(np.float64)}
    if cur is not None:
        out.append(cur)
    return out


def out_starts(ball_out, lo, hi):
    """First frame of each stretch of ball-out frames inside [lo, hi)."""
    b = ball_out[lo:hi].astype(np.int8)
    return list(np.flatnonzero(np.diff(np.r_[0, b]) == 1) + lo)


def confirmed(k, tl, out_frames):
    """Does touch k confirm its team: held long enough, or completed a pass to a teammate?"""
    t = tl[k]
    if t["n"] >= CONFIRM_FRAMES:
        return True
    if k + 1 < len(tl):
        u = tl[k + 1]
        if (u["team"] == t["team"] and u["slot"] != t["slot"]
                and u["start"] - t["end"] <= MAX_PASS_FRAMES
                and np.linalg.norm(u["pos"] - t["pos"]) >= MIN_PASS_DISTANCE_M
                and not any(t["end"] < o < u["start"] for o in out_frames)):
            return True
    return False


def switch_frame(t, state, last_evidence, F):
    """Where a confirmed change to touch t's team starts (see the module docstring)."""
    start = t["start"]
    if state == D:
        speed = F["ball_speed"][start:start + RESTART_SEARCH_FRAMES]
        moving = np.flatnonzero(speed > RESTART_SPEED_MS)
        return start + int(moving[0]) if len(moving) else start
    gap = start - last_evidence - 1
    if gap >= BLIND_MIN_FRAMES:
        missing = 1 - F["ball_seen"][last_evidence + 1:start].mean()
        if missing > BLIND_MISSING_SHARE:
            return (last_evidence + 1 + start) // 2
    return start


def decode_half(F, team, slot, lo, hi):
    """Possession codes for frames [lo, hi): the switching rules, applied in time order."""
    tl = touches(team, slot, F["ball_xy"], lo, hi)
    outs = out_starts(F["ball_out"], lo, hi)

    # Merge touches and ball-out starts into one timeline.
    events = [(t["start"], 0, k) for k, t in enumerate(tl)] + [(o, 1, o) for o in outs]
    events.sort()

    state, last_evidence = D, lo  # the half opens with a kick-off: a restart from D
    switches = [(lo, D)]
    for frame, kind, ref in events:
        if kind == 1:  # ball went out
            if state != D:
                switches.append((frame, D))
                state = D
            last_evidence = frame
            continue
        t = tl[ref]
        if t["team"] == state:
            last_evidence = t["end"]
            continue
        if confirmed(ref, tl, outs):
            b = max(switch_frame(t, state, last_evidence, F), switches[-1][0] + 1)
            switches.append((b, t["team"]))
            state = t["team"]
            last_evidence = t["end"]
        # an unconfirmed touch by the other side: sticky possession, nothing changes

    codes = np.empty(hi - lo, dtype=np.int8)
    bounds = [s for s, _ in switches] + [hi]
    for (s, c), e in zip(switches, bounds[1:]):
        codes[s - lo:e - lo] = c
    return codes


def predict(F):
    """(labels, probs): the approach-1 submission and its per-frame evidence."""
    team, slot = control_frames(F)
    labels = np.full(len(team), "X", dtype="<U1")
    for h in (1, 2):
        idx = np.flatnonzero(F["half"] == h)
        lo, hi = int(idx[0]), int(idx[-1]) + 1
        labels[lo:hi] = LETTERS[decode_half(F, team, slot, lo, hi)]
    return labels, evidence(F, team)


def describe(F, labels):
    """Possession count and label shares on graded frames -- the label-free sanity checks."""
    for h in (1, 2):
        lab = labels[F["half"] == h]
        change = np.flatnonzero(lab[1:] != lab[:-1]) + 1
        runs = np.split(lab, change)
        poss = [r for r in runs if r[0] in "HA"]
        print("  half {}: {} possessions, mean {:.1f} s, {} dead-ball stretches".format(
            h, len(poss), np.mean([len(r) for r in poss]) / FPS,
            sum(r[0] == "D" for r in runs)))
    g = labels[F["half"] > 0]
    print("  graded frames: " + "  ".join("{} {:.1f}%".format(c, 100 * np.mean(g == c))
                                          for c in "HAD"))


def stats(F):
    """The tracking measurements quoted next to the constants above. Uses no labels."""
    g = F["half"] > 0
    d = F["d_near"]
    dmin = np.fmin(d[:, 0], d[:, 1])
    bs = F["ball_speed"]
    xy, sp = F["players_xy"], F["players_speed"]
    step = np.linalg.norm(xy[1:] - xy[:-1], axis=2)
    still = (sp[1:] < 0.3) & (sp[:-1] < 0.3) & g[1:, None]
    print("standing player, frame-to-frame step: median {:.3f} m".format(
        np.nanmedian(step[still])))
    for lo, hi in ((0, 3), (3, 8), (8, 15), (15, 99)):
        m = g & np.isfinite(dmin) & (bs >= lo) & (bs < hi)
        print("ball {:>2}-{:<2} m/s: nearest player median {:.2f} m, within {} m: {:.0f}%".format(
            lo, hi, np.median(dmin[m]), CONTROL_RADIUS_M,
            100 * np.mean(dmin[m] <= CONTROL_RADIUS_M)))
    m = g & np.isfinite(dmin) & (bs < 3)
    hist, _ = np.histogram(dmin[m], bins=np.arange(0, 3.01, 0.25))
    print("slow ball, nearest player, 0.25 m bins from 0 to 3 m:", hist.tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stats", action="store_true", help="print the measurements behind the "
                                                             "constants and exit")
    args = parser.parse_args()

    F = load_frames()
    if args.stats:
        stats(F)
        sys.exit(0)

    labels, probs = predict(F)
    path = write_scores(os.path.join(SCORES_DIR, "approach1.csv"), probs)
    print("wrote", os.path.relpath(path, ROOT))
    path = write_submission(os.path.join(OUT_DIR, "approach1.csv"), labels)
    print("wrote", os.path.relpath(path, ROOT), "\n")
    describe(F, labels)
    print()
    validate(path)
