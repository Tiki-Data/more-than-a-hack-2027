"""Approach 3: dead balls -- when the ball is out of play, and who restarts it.

    python src/approach3.py      # -> out/scores/approach3.csv  (per-frame evidence, for the glue)
                                 #    out/restarts.csv          (one row per restart: frame,team,...)

What it does
------------
1. CLEAN THE TAGS, BY RULE. Tags are automatic and can be wrong, so a tagged goal is only trusted
   if the tracking CONFIRMS it: after a goal, play restarts with the ball kicked from the centre
   spot. A "Kick Off (After Goal)" tag where the ball is not seen leaving the centre spot is not a
   goal restart, and the goal tag before it is not trusted. That stoppage is kept as a long
   stoppage of its own: dead from the first ball-missing gap of 60 s+ after the goal tag to the
   first time the ball is seen moving after that gap, restarted by whoever is nearest the ball.
   (On this match the rule rejects the "goal" near 31', which the team plan already flagged as
   phantom; the stoppage it marks there is the cooling break. Nothing in the code names a frame.)
   The penalty is tagged as a free kick; it is a restart like any other, so it stays.

2. FIND THE EXACT RESTART FRAME. Tag times are 35 s phase windows, not moments. In each window we
   look for the ball being kicked or thrown from where that restart is taken (touchline, corner,
   goal area, centre spot), after a second in which it was not in live play. Fallbacks, in order:
   the first frame the ball is seen moving in the window, then the tag's start frame.

3. WHO RESTARTS. The tag's team when it has one (corners, goal kicks, free kicks). After a goal,
   the team that conceded. Otherwise (throw-ins, the opening kick-offs) the team of the player
   nearest the ball at the restart, cross-checked for throw-ins against "the team that touched it
   last before it went out did NOT get the throw".

4. DEAD-BALL STRETCHES.
   STRONG (D = 0.85): from where play stopped to the restart frame. "Where play stopped" is the
     last frame the ball was seen moving in the pitch before the restart, bounded by how long that
     kind of stoppage can last. After a goal: from the last time the ball was seen near the goal
     it went into. Before a half's kick-off: from the start of the half.
   WEAK (D = 0.65): the ball missing for a long time WHILE the players are slow. Catches fouls,
     offsides, injuries and VAR checks, where play stops without the ball going out. Missing ball
     alone is NOT dead (the tracker loses it 41% of the time), and slow players alone is NOT dead
     (slow build-up), so both are required.
   After each restart, the restarting team gets 0.8 for RESTART_TEAM_FRAMES.
   Everything else is 0,0,0: no opinion.

Constants were set from football reasoning or from this match's tracking and tags (no labels),
never from the public 5 minutes. `python src/approach3.py` prints the checks behind them.
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES  # noqa: E402
from src.load import load_frames, load_tags  # noqa: E402
from src.submission import OUT_DIR, SCORES_DIR, write_scores  # noqa: E402

# Params
FPS = GAMES[GAME]["fps"]
PITCH_L, PITCH_W = GAMES[GAME]["pitch"]
H, A, D = 0, 1, 2
SIDE_CODE = {"H": H, "A": A}

# --- Step 1: tag cleaning -----------------------------------------------------------------------
# A goal that the tracking does not confirm still marks a stoppage. Long stoppages (a cooling
# break, an injury, a review) last 1-3 minutes, and players walk off the ball, so it is the first
# ball-missing gap of 60 s+ after the tag. Open-play tracker dropouts are much shorter: gaps of
# 5 s+ away from set-piece tags have a median of 10.6 s and 75% are under 18.5 s.
LONG_STOPPAGE_GAP = 60 * 25

# --- Step 2: restart frame ----------------------------------------------------------------------
# The ball "moves" for a restart when it goes faster than walking pace with it: 3 m/s. Same value
# and reason as approach 1's RESTART_SPEED_MS, so the two pieces agree on what a restart is.
RESTART_SPEED_MS = 3.0
# Where to look around a tag's start. Over all 59 set-piece tags, the ball reappears at the
# restart spot within the first ~10 s of the window, and sometimes just before it starts (the
# taker places it early), so: 5 s before to 15 s after.
SEARCH_BEFORE = 5 * FPS
SEARCH_AFTER = 15 * FPS
# The ball must not have been in live play during the second before the restart kick: otherwise a
# ball running along the touchline before going out would count as the throw-in.
QUIET_BEFORE = FPS
# Restart spots, in metres. A thrower stands at or just behind the touchline; a corner is taken
# inside the 1 m arc; a goal kick anywhere in the goal area (5.5 m deep, 18.32 m wide) -- each with
# a 2-3 m margin for tracking error on a ball in hands or at feet.
THROW_IN_MARGIN = 3.0
CORNER_RADIUS = 4.0
GOAL_AREA_DEPTH = 5.5 + 2.0
GOAL_AREA_HALF_WIDTH = 9.16 + 2.0
CENTRE_RADIUS = 4.0

# --- Step 3: who restarts -----------------------------------------------------------------------
# A player "has" the ball at the restart if within 3 m: the thrower holds it over his head (the
# tracker places it at his feet or a step in front), a kicker stands a stride behind it.
TAKER_RADIUS = 3.0
# Last touch before a throw-in: the same control radius as approach 1.
TOUCH_RADIUS = 1.5
# How long after the restart frame to wait for the ball to be seen, if it is not seen on it.
TAKER_SEARCH = 2 * FPS

# --- Step 4: dead-ball stretches ----------------------------------------------------------------
# Longest stoppage we will claim before each kind of restart, from typical football timings: a
# throw-in is taken within ~10-20 s, a goal kick ~20-30 s, a corner ~30-40 s (players walk up),
# a free kick can take a minute (foul, card, wall). If the ball was never seen in play within this
# span we do not know when play stopped, and the stretch is downgraded to WEAK.
MAX_DEAD = {"Throw-in": 20 * FPS, "Goal Kick": 30 * FPS, "Corner Kick": 40 * FPS,
            "Free Kick": 60 * FPS}
# After a goal: the ball counts as "near the goal it went into" within 20 m of that goal line.
GOAL_NEAR_M = 20.0
# A goal's stoppage starts at a ball-missing gap of 10 s+: the celebration. Open-play tracker
# dropouts are mostly shorter (median 10.6 s for gaps of 5 s+ away from tags), and the gap must also
# follow a sighting near that goal.
GOAL_GAP = 10 * FPS
# Weak stretches: a ball-missing gap of at least 8 s while the median player speed over the gap is
# below 1.0 m/s. Measured on this match: missing-ball gaps of 5 s+ next to a set-piece tag have a
# median player speed of 1.08 m/s, other gaps 1.30 m/s, and all graded frames 1.37 m/s (p10 0.60);
# the three goal stoppages measure 0.77-1.0. 1.0 m/s sits at the dead-ball median, and 8 s is
# longer than most tracker dropouts in open play (median untagged gap 10.6 s, so many are cut).
WEAK_MIN_GAP = 8 * FPS
WEAK_SLOW_MS = 1.0
# The restarting team holds the ball for at least the first 2 s after nearly every restart (a
# throw or a goal kick reaches a teammate; contested long balls are the exception).
RESTART_TEAM_FRAMES = 2 * FPS

STRONG_D = 0.85
WEAK_D = 0.65
RESTART_TEAM_P = 0.8


def row(winner, p):
    """A probability row with p on the winner and the rest split evenly."""
    r = np.full(3, (1 - p) / 2)
    r[winner] = p
    return r


# ---------------------------------------------------------------------------------------------
# Step 1: clean the tags
# ---------------------------------------------------------------------------------------------

def clean_tags(tags):
    """(set_pieces, goals), sorted by frame. Goals are confirmed or rejected later, in restarts(),
    from the tracking -- never by frame number."""
    sp = tags[tags.code == "SET PIECES"]
    goals = tags[tags.code == "GOALS"]
    return (sp.sort_values("start_frame").reset_index(drop=True),
            goals.sort_values("start_frame").reset_index(drop=True))


# ---------------------------------------------------------------------------------------------
# Step 2: the exact restart frame
# ---------------------------------------------------------------------------------------------

def in_spot(kind, x, y):
    """Is the ball where a restart of this kind is taken? Arrays in, bool array out."""
    if kind == "Throw-in":
        return (np.abs(y) < THROW_IN_MARGIN) | (np.abs(y - PITCH_W) < THROW_IN_MARGIN)
    if kind == "Corner Kick":
        cx = np.minimum(np.abs(x), np.abs(x - PITCH_L))
        cy = np.minimum(np.abs(y), np.abs(y - PITCH_W))
        return np.hypot(cx, cy) < CORNER_RADIUS
    if kind == "Goal Kick":
        gx = np.minimum(np.abs(x), np.abs(x - PITCH_L))
        return (gx < GOAL_AREA_DEPTH) & (np.abs(y - PITCH_W / 2) < GOAL_AREA_HALF_WIDTH)
    if kind.startswith("Kick Off"):
        return np.hypot(x - PITCH_L / 2, y - PITCH_W / 2) < CENTRE_RADIUS
    return np.ones_like(x, dtype=bool)  # free kick: anywhere


def live_frames(F):
    """Ball seen in the pitch and moving faster than a restart: play is on."""
    return F["ball_seen"] & ~F["ball_out"] & (F["ball_speed"] > RESTART_SPEED_MS)


def find_restart(F, live, kind, tag_start):
    """(frame, how): where the restart kick/throw happens around tag_start."""
    n = len(F["frame"])
    lo, hi = max(0, tag_start - SEARCH_BEFORE), min(n, tag_start + SEARCH_AFTER)
    x, y = F["ball_xy"][lo:hi, 0], F["ball_xy"][lo:hi, 1]
    moving = F["ball_seen"][lo:hi] & (F["ball_speed"][lo:hi] > RESTART_SPEED_MS)
    cand = np.flatnonzero(moving & in_spot(kind, x, y)) + lo
    for f in cand:
        if not live[max(0, f - QUIET_BEFORE):f].any():
            return int(f), "spot"
    # Not seen leaving the spot: the first time the ball is seen moving after the tag starts.
    later = np.flatnonzero(moving[tag_start - lo:]) + tag_start
    if len(later):
        return int(later[0]), "first_moving"
    return int(tag_start), "tag"


# ---------------------------------------------------------------------------------------------
# Step 3: who restarts
# ---------------------------------------------------------------------------------------------

def nearest_team(F, f):
    """Team of the player nearest the ball on f, or the first frame within TAKER_SEARCH where the
    ball is seen and someone is within TAKER_RADIUS. None if nobody."""
    for i in range(f, min(len(F["frame"]), f + TAKER_SEARCH)):
        d = F["d_near"][i]
        if not np.isfinite(d).any():
            continue
        t = int(np.nanargmin(d))
        if d[t] <= TAKER_RADIUS:
            return t
    return None


def last_touch_team(F, before, lookback):
    """Team in control (within TOUCH_RADIUS) on the last frame before `before` where anyone was."""
    lo = max(0, before - lookback)
    d = F["d_near"][lo:before]
    dmin = np.fmin(d[:, 0], d[:, 1])
    close = np.flatnonzero(dmin <= TOUCH_RADIUS)
    if not len(close):
        return None
    i = close[-1]
    return int(np.nanargmin(d[i]))


# ---------------------------------------------------------------------------------------------
# Step 4: where each stoppage started
# ---------------------------------------------------------------------------------------------

def dead_start_walkback(F, live, kind, r, max_len, floor):
    """(start, bounded): the frame after the last live frame before r, within max_len of r.
    bounded is False when no live frame was found, i.e. we do not know when play stopped.

    A fast ball AT the restart spot is not live play: it is the ball being tossed or rolled to the
    taker (seen on 5 restarts whose stoppage otherwise came out at 0.5-1.5 s). Free kicks have no
    spot, so nothing is excluded for them."""
    # floor is the previous restart; tags are processed in tag order, so it can lie after r.
    lo = min(max(floor, r - max_len), r)
    x, y = F["ball_xy"][lo:r, 0], F["ball_xy"][lo:r, 1]
    at_spot = in_spot(kind, x, y) if kind != "Free Kick" else np.zeros(r - lo, dtype=bool)
    idx = np.flatnonzero(live[lo:r] & ~at_spot)
    if len(idx):
        return int(lo + idx[-1] + 1), True
    return int(lo), False


def goal_moment(F, goal, lo, kickoff):
    """Where the stoppage after a goal starts: the first ball-missing gap of GOAL_GAP+ frames in
    [lo, kickoff) whose last sighting before it was near the goal the ball went into. lo is the
    later of the goal tag and the previous restart, so the attack itself is not swallowed.

    Why not just "last sighting near the goal": during the celebration someone kicks the ball back
    toward the centre and the tracker briefly sees it (goal 1), which would start D a minute late.
    Falls back to the last sighting near the goal, then to lo."""
    hi = int(kickoff)
    scorer_right = F["h_attacks_right"][lo] if goal.side == "H" else ~F["h_attacks_right"][lo]
    line = PITCH_L if scorer_right else 0.0
    seen = F["ball_seen"][lo:hi]
    near = seen & (np.abs(F["ball_xy"][lo:hi, 0] - line) < GOAL_NEAR_M)
    edge = np.diff(np.r_[0, (~seen).astype(np.int8), 0])
    for a, b in zip(np.flatnonzero(edge == 1), np.flatnonzero(edge == -1)):
        if b - a >= GOAL_GAP and a > 0 and near[a - 1]:
            return lo + int(a)
    idx = np.flatnonzero(near)
    return lo + int(idx[-1]) + 1 if len(idx) else lo


def long_stoppage(F, tag_frame, until):
    """(dead_start, restart, how): the first LONG_STOPPAGE_GAP+ ball-missing gap in
    [tag_frame, until), and the first frame after it with the ball seen moving (within
    SEARCH_AFTER), else the gap's end. None if there is no such gap."""
    missing = ~F["ball_seen"][tag_frame:until]
    edge = np.diff(np.r_[0, missing.astype(np.int8), 0])
    for a, b in zip(np.flatnonzero(edge == 1), np.flatnonzero(edge == -1)):
        if b - a >= LONG_STOPPAGE_GAP:
            a, b = tag_frame + int(a), tag_frame + int(b)
            moving = F["ball_seen"][b:b + SEARCH_AFTER] & (
                F["ball_speed"][b:b + SEARCH_AFTER] > RESTART_SPEED_MS)
            idx = np.flatnonzero(moving)
            return a, (b + int(idx[0]) if len(idx) else b), ("first_moving" if len(idx) else "tag")
    return None


def weak_stretches(F):
    """Ball-missing gaps of WEAK_MIN_GAP+ frames inside a half with slow players."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        speed = np.nanmedian(F["players_speed"], axis=1)
    missing = (~F["ball_seen"]) & (F["half"] > 0)
    edge = np.diff(np.r_[0, missing.astype(np.int8), 0])
    out = []
    for a, b in zip(np.flatnonzero(edge == 1), np.flatnonzero(edge == -1)):
        if b - a >= WEAK_MIN_GAP and np.nanmean(speed[a:b]) < WEAK_SLOW_MS:
            out.append((int(a), int(b)))
    return out


# ---------------------------------------------------------------------------------------------
# Putting it together
# ---------------------------------------------------------------------------------------------

def restarts(F, set_pieces, goals):
    """One row per restart: frame, team, type, how, dead_start, strong, plus checks."""
    live = live_frames(F)
    rows = []
    for tag in set_pieces.itertuples():
        kind = tag.type
        r, how = find_restart(F, live, kind, int(tag.start_frame))
        half = int(F["half"][min(r, len(F["half"]) - 1)])
        check = None

        before = goals[goals.start_frame < tag.start_frame]
        if kind == "Kick Off (After Goal)" and len(before) == 0:
            continue  # a kick-off after a goal with no goal tag: nothing to anchor it to
        if kind == "Kick Off (After Goal)" and how != "spot":
            # Not seen leaving the centre spot: the goal before it is not confirmed. Keep the
            # stoppage it marks, from the tracking alone.
            goal = before.iloc[-1]
            found = long_stoppage(F, int(goal.start_frame), int(tag.end_frame) + SEARCH_AFTER)
            if found is None:
                continue
            start, r, how = found
            team = nearest_team(F, r)
            rows.append({"frame": r, "team": "HA"[team] if team is not None else "N",
                         "type": "Stoppage (goal not confirmed)",
                         "tag_frame": int(goal.start_frame), "how": how, "team_how": "nearest",
                         "dead_start": start, "strong": True, "last_touch_says": ""})
            continue
        if kind == "Kick Off (After Goal)":
            goal = before.iloc[-1]
            team = A if goal.side == "H" else H  # the team that conceded kicks off
            team_how = "conceded"
            lo = max(int(goal.start_frame), rows[-1]["frame"] if rows else 0)
            start, strong = goal_moment(F, goal, lo, r), True
        elif kind == "Kick Off (Start)":
            team = nearest_team(F, r)
            team_how = "nearest"
            idx = np.flatnonzero(F["half"] == half) if half else np.array([r])
            start, strong = int(idx[0]), True
        else:
            if tag.side in SIDE_CODE:
                team, team_how = SIDE_CODE[tag.side], "tag"
            else:
                team, team_how = nearest_team(F, r), "nearest"
            prev = rows[-1]["frame"] if rows else 0
            start, strong = dead_start_walkback(F, live, kind, r, MAX_DEAD[kind], prev)
            if kind == "Throw-in":
                last = last_touch_team(F, start, 5 * FPS)
                check = None if last is None else 1 - last
                if team is None and check is not None:
                    team, team_how = check, "last_touch"

        rows.append({"frame": r, "team": "HA"[team] if team is not None else "N",
                     "type": kind, "tag_frame": int(tag.start_frame), "how": how,
                     "team_how": team_how, "dead_start": min(start, r), "strong": strong,
                     "last_touch_says": "HA"[check] if check is not None else ""})
    table = pd.DataFrame(rows).sort_values("frame").reset_index(drop=True)
    # Two tags for one stoppage (detected within 2 s of each other): keep the first.
    dup = table.frame.diff().fillna(np.inf) < 2 * FPS
    return table[~dup].reset_index(drop=True), int(dup.sum())


def predict(F, table, weak):
    """(n, 3) H/A/D evidence."""
    n = len(F["frame"])
    probs = np.zeros((n, 3))
    for a, b in weak:
        probs[a:b] = row(D, WEAK_D)
    for t in table.itertuples():
        probs[t.dead_start:t.frame] = row(D, STRONG_D if t.strong else WEAK_D)
    for t in table.itertuples():
        if t.team in SIDE_CODE:
            probs[t.frame:t.frame + RESTART_TEAM_FRAMES] = row(SIDE_CODE[t.team], RESTART_TEAM_P)
    return probs


def describe(F, table, weak, probs, n_dup):
    """Label-free sanity checks, aggregates only."""
    g = F["half"] > 0
    print("restarts: {}  (duplicate tags merged: {})".format(len(table), n_dup))
    print("  restart frame found by: " + ", ".join(
        "{} {}".format(k, v) for k, v in table.how.value_counts().items()))
    print("  restarting team from:   " + ", ".join(
        "{} {}".format(k, v) for k, v in table.team_how.value_counts().items()))
    ti = table[(table.type == "Throw-in") & (table.last_touch_says != "")]
    agree = (ti.team == ti.last_touch_says).sum()
    print("  throw-ins where 'nearest at restart' and 'not the last toucher' agree: {}/{}".format(
        agree, len(ti)))
    dur = (table.frame - table.dead_start) / FPS
    for kind, d in dur.groupby(table.type):
        print("  {:<22} dead stretch: median {:5.1f} s, max {:5.1f} s".format(kind, d.median(),
                                                                            d.max()))
    print("  strong stretches: {}, downgraded to weak (play stop not seen): {}".format(
        int(table.strong.sum()), int((~table.strong).sum())))
    print("weak (slow + missing) stretches: {}, {:.0f} s in total".format(
        len(weak), sum(b - a for a, b in weak) / FPS))
    has = probs.sum(axis=1) > 0
    lab = probs.argmax(axis=1)
    print("graded frames with an opinion: {:.1f}%   D {:.1f}%  H {:.1f}%  A {:.1f}%  (of graded)"
          .format(100 * has[g].mean(), *(100 * np.mean(has[g] & (lab[g] == c)) for c in (D, H, A))))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args()

    F = load_frames()
    set_pieces, goals = clean_tags(load_tags())
    table, n_dup = restarts(F, set_pieces, goals)
    weak = weak_stretches(F)
    probs = predict(F, table, weak)

    path = write_scores(os.path.join(SCORES_DIR, "approach3.csv"), probs)
    print("wrote", os.path.relpath(path, ROOT))
    rpath = os.path.join(OUT_DIR, "restarts.csv")
    table.to_csv(rpath, index=False, lineterminator="\n")
    print("wrote", os.path.relpath(rpath, ROOT), "\n")
    describe(F, table, weak, probs, n_dup)


if __name__ == "__main__":
    main()
