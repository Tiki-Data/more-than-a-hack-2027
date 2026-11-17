"""One shared table of the match, so nobody re-parses the 195 MB ATD file.

    python src/load.py            # build (about 30 s) and print a summary
    python src/load.py --force    # rebuild even if the cache is fresh

Then, from any piece of the pipeline:

    from src.load import load_frames, frame_table, load_tags
    F = load_frames()          # dict of numpy arrays, see FIELDS below
    df = frame_table()         # pandas, one row per video frame, scalar columns only
    tags = load_tags()         # pandas, one row per Smart Tagging instance, in video frames

Everything is indexed by VIDEO frame (row i = submission row i), already shifted from the ATD's
frame_id = video_frame + 1. Everything is in METRES, on a 105.3 x 68.0 pitch with the origin at a
corner. Missing observations are NaN.

The cache lives in data/derived/ which is gitignored like the rest of data/.

FIELDS (n = 199319 frames, 46 track slots)
------------------------------------------
frame            (n,)      int32    0..n-1
half             (n,)      int8     1 or 2 on graded frames, 0 elsewhere (halftime, pre/post match)
h_attacks_right  (n,)      bool     True where Barcelona attack toward x = 105.3
ball_xy          (n,2)     float32  metres, NaN when the tracker has no ball
ball_seen        (n,)      bool
ball_speed       (n,)      float32  m/s, centred difference over frames i-2 .. i+2; NaN unless the
                                    ball is seen on i-2, i and i+2
ball_out         (n,)      bool     ball seen and more than OUT_MARGIN_M outside the pitch
players_xy       (n,46,2)  float32  metres
players_speed    (n,46)    float32  speed channel of the ATD, in m/s (see SPEED_UNIT_TO_MS)
team_of_slot     (46,)     int8     0 = H (Barcelona), 1 = A (Athletic)
n_visible        (n,2)     int8     tracked players per team [H, A]
d_near           (n,2)     float32  distance in metres from the ball to the nearest H / A player;
                                    NaN when no ball
near_slot        (n,2)     int8     slot of that player, -1 when no ball
"""

import argparse
import io
import os
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402

# Params
N_SLOTS = 46
HOME, AWAY = 0, 1

# A ball this far outside a line counts as out. Ball diameter is 0.22 m and tracking jitter on a
# rolling ball is a few tens of cm, so 1 m means "clearly over the line", not "on it". Approach 3
# (P3) replaces this with proper dead-ball detection; this is only the cheap signal.
OUT_MARGIN_M = 1.0

# The XML declares the speed channel as "speed_kph"; DATA.md says m/s. check_speed_unit() compares
# it with speed computed from positions: the ratio is 0.99 over 321k player samples, so the channel
# IS m/s and the XML unit label is wrong. Re-checked on every `python src/load.py`.
SPEED_UNIT_TO_MS = 1.0

# Ball speed is measured between frames i - k and i + k, k = 2: a 4-frame (0.16 s) span. Long
# enough to divide 1-frame jitter by 4, short enough that a pass or a kick changes the speed within
# a few frames, which approach 1 needs to tell a ball in flight from one at someone's feet.
BALL_SPEED_HALF_SPAN = 2

CACHE_DIR = os.path.join(ROOT, "data", "derived")
CACHE_VERSION = 2  # bump when the set of fields or their meaning changes


def _cache_path(game):
    return os.path.join(CACHE_DIR, "{}_frames_v{}.npz".format(game, CACHE_VERSION))


# ---------------------------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------------------------

def team_of_slot(meta_path):
    """0/1 per track slot. The first team in the metadata is home (Barcelona, per DATA.md)."""
    root = ET.parse(meta_path).getroot()
    teams = [(t.get("id"), t.findtext("Name")) for t in root.iter("Team")]
    if len(teams) != 2:
        raise ValueError("expected 2 teams in {}, got {}".format(meta_path, teams))
    home_id = teams[0][0]
    slots = [p.get("teamId") for p in root.iter("Player")]
    if len(slots) != N_SLOTS:
        raise ValueError("expected {} track slots, got {}".format(N_SLOTS, len(slots)))
    return np.array([HOME if t == home_id else AWAY for t in slots], dtype=np.int8), teams


def attack_direction_first_half(meta_path):
    """True if the home team attacks left to right in the first half (read from the metadata)."""
    root = ET.parse(meta_path).getroot()
    home = next(root.iter("Team"))
    for param in home.iter("ProviderParameter"):
        if param.findtext("Name") == "attack_direction_first_half":
            return param.findtext("Value") == "left_to_right"
    raise ValueError("no attack_direction_first_half for the home team in " + meta_path)


def parse_raw(raw_path, n_frames):
    """(frame_id, values[n, 46*3 + 3]) straight from the raw ATD text.

    Each line is "<frame_id>:<x,y,s>;<x,y,s>;...(46 slots):<x,y,s ball>". Turning both separators
    into commas makes it a plain numeric CSV, which pandas' C parser reads in seconds -- far faster
    than splitting 199319 x 46 groups in Python.
    """
    with open(raw_path) as handle:
        text = handle.read()
    text = text.replace(":", ",").replace(";", ",")
    width = 1 + N_SLOTS * 3 + 3
    table = pd.read_csv(io.StringIO(text), header=None, dtype=np.float64,
                        na_values=["NaN"], engine="c")
    if table.shape[1] != width:
        raise ValueError("expected {} numbers per line, got {}".format(width, table.shape[1]))
    values = table.to_numpy()

    frame_id = values[:, 0].astype(np.int64)
    if len(frame_id) != n_frames or frame_id[0] != 1 or np.any(np.diff(frame_id) != 1):
        raise ValueError("ATD is not one record per video frame numbered 1..{}".format(n_frames))
    return frame_id, values[:, 1:]


# ---------------------------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------------------------

def build(game=GAME, data_dir=None):
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    n = source_frames("tactical", game)
    length, width = GAMES[game]["pitch"]
    scale = np.array([length, width], dtype=np.float64)

    teams, team_names = team_of_slot(meta_path)
    frame_id, values = parse_raw(raw_path, n)

    players = values[:, :N_SLOTS * 3].reshape(n, N_SLOTS, 3)
    ball = values[:, N_SLOTS * 3:]

    # Row i must be video frame i. frame_id = video_frame + 1, checked contiguous above.
    frame = (frame_id - 1).astype(np.int32)
    assert frame[0] == 0 and frame[-1] == n - 1

    players_xy = players[:, :, :2] * scale
    players_speed = players[:, :, 2] * SPEED_UNIT_TO_MS
    # The ball's own speed channel (ball[:, 2]) is ignored. It is filled on only 10.5% of frames,
    # in 51 long blocks, more than half of them on frames with no ball position, and never exceeds
    # 7.3 m/s -- not a measurement of a kicked ball. Undocumented in the repo; kloppy drops it too.
    ball_xy = ball[:, :2] * scale
    ball_seen = ~np.isnan(ball_xy).any(axis=1)

    # Half of each video frame, from the published blocks (games.half_video_blocks).
    half = np.zeros(n, dtype=np.int8)
    for h, first, last in half_video_blocks(game):
        half[first:last + 1] = h

    home_right_first = attack_direction_first_half(meta_path)
    h_attacks_right = np.where(half == 2, not home_right_first, home_right_first)

    # Ball speed on frame i, from a centred difference over 2 * BALL_SPEED_HALF_SPAN frames:
    #     speed[i] = |ball[i + k] - ball[i - k]| / (2k / fps)
    # A 1-frame difference multiplies position jitter by 25 (10 cm of jitter reads as 2.5 m/s on a
    # still ball). Spreading it over 2k frames divides that error by 2k.
    # Frame i gets a speed only if the ball is seen on i, i - k and i + k. Frames in between may be
    # missing; the speed is then the average over the span, which is fine for a ball in flight or
    # rolling. NaN otherwise, including the first and last k frames of the video.
    fps = GAMES[game]["fps"]
    k = BALL_SPEED_HALF_SPAN
    ball_speed = np.full(n, np.nan)
    disp = np.linalg.norm(ball_xy[2 * k:] - ball_xy[:-2 * k], axis=1)  # NaN if an end is missing
    ball_speed[k:n - k] = disp * fps / (2 * k)
    ball_speed[~ball_seen] = np.nan

    ball_out = ball_seen & (
        (ball_xy[:, 0] < -OUT_MARGIN_M) | (ball_xy[:, 0] > length + OUT_MARGIN_M)
        | (ball_xy[:, 1] < -OUT_MARGIN_M) | (ball_xy[:, 1] > width + OUT_MARGIN_M))

    # Nearest player of each team to the ball.
    seen_player = ~np.isnan(players_xy).any(axis=2)
    n_visible = np.stack([(seen_player & (teams == t)).sum(axis=1) for t in (HOME, AWAY)],
                         axis=1).astype(np.int8)

    dist = np.linalg.norm(players_xy - ball_xy[:, None, :], axis=2)  # NaN where either missing
    d_near = np.full((n, 2), np.nan)
    near_slot = np.full((n, 2), -1, dtype=np.int8)
    for t in (HOME, AWAY):
        d = np.where(teams[None, :] == t, dist, np.nan)
        d = np.where(np.isnan(d), np.inf, d)
        slot = d.argmin(axis=1)
        best = d[np.arange(n), slot]
        ok = np.isfinite(best)
        d_near[ok, t] = best[ok]
        near_slot[ok, t] = slot[ok]

    return {
        "frame": frame,
        "half": half,
        "h_attacks_right": h_attacks_right.astype(bool),
        "ball_xy": ball_xy.astype(np.float32),
        "ball_seen": ball_seen,
        "ball_speed": ball_speed.astype(np.float32),
        "ball_out": ball_out,
        "players_xy": players_xy.astype(np.float32),
        "players_speed": players_speed.astype(np.float32),
        "team_of_slot": teams,
        "n_visible": n_visible,
        "d_near": d_near.astype(np.float32),
        "near_slot": near_slot,
        "team_names": np.array([name for _, name in team_names]),
    }


def load_frames(game=GAME, data_dir=None, force=False):
    """The shared table as a dict of numpy arrays. Built once, then read from data/derived/."""
    path = _cache_path(game)
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    fresh = (os.path.isfile(path)
             and os.path.getmtime(path) > max(os.path.getmtime(meta_path),
                                              os.path.getmtime(raw_path)))
    if fresh and not force:
        with np.load(path) as cached:
            return {key: cached[key] for key in cached.files}

    data = build(game, data_dir)
    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez(tmp, **data)
    os.replace(tmp, path)
    return data


def frame_table(frames=None, game=GAME):
    """Per-frame scalar columns as a DataFrame -- the easy view for quick analysis."""
    F = frames if frames is not None else load_frames(game)
    return pd.DataFrame({
        "frame": F["frame"],
        "half": F["half"],
        "h_attacks_right": F["h_attacks_right"],
        "ball_x": F["ball_xy"][:, 0],
        "ball_y": F["ball_xy"][:, 1],
        "ball_seen": F["ball_seen"],
        "ball_speed": F["ball_speed"],
        "ball_out": F["ball_out"],
        "n_vis_H": F["n_visible"][:, 0],
        "n_vis_A": F["n_visible"][:, 1],
        "d_near_H": F["d_near"][:, 0],
        "d_near_A": F["d_near"][:, 1],
        "near_slot_H": F["near_slot"][:, 0],
        "near_slot_A": F["near_slot"][:, 1],
    })


def load_tags(game=GAME, data_dir=None):
    """Smart Tagging as a DataFrame in video frames, with the team mapped to H / A / N.

    Unfiltered on purpose: the phantom goal near 31', the penalty-as-free-kick and the N/A
    throw-ins are all still here. Cleaning them is P3's job and should happen in one place.
    end_frame is exclusive.
    """
    sys.path.insert(0, os.path.join(ROOT, "starter"))
    import load_smart_tagging as lst  # noqa: E402

    events = lst.load(game, data_dir)
    meta_path = require(["atd_meta"], game, data_dir)[0]
    _, teams = team_of_slot(meta_path)
    home_name = teams[0][1]
    rows = []
    for e in events:
        team = e["tags"].get("Team")
        side = "N" if team in (None, "N/A") else ("H" if team == home_name else "A")
        rows.append({
            "code": e["code"],
            "type": e["tags"].get("Type"),
            "team": team,
            "side": side,
            "half_tag": e["tags"].get("Half"),
            "start_frame": e["start_frame"],
            "end_frame": e["end_frame"],
            "start_s": e["start_s"],
            "end_s": e["end_s"],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------------------------

def check_speed_unit(F, max_frames=20000):
    """Median of (ATD speed channel / speed computed from positions) for players.

    1.0 means the channel is m/s; 3.6 means km/h. Positions are smoothed over 5 frames first,
    because frame-to-frame jitter inflates the computed speed.
    """
    fps = 25
    graded = np.flatnonzero(F["half"] > 0)[:max_frames]
    xy = F["players_xy"][graded].astype(np.float64)
    k = 5
    # Displacement over k frames, centred, divided by elapsed time.
    v = np.linalg.norm(xy[k:] - xy[:-k], axis=2) * fps / k
    s = F["players_speed"][graded][k // 2: -k + k // 2].astype(np.float64)
    ok = np.isfinite(v) & np.isfinite(s) & (v > 1.0)
    return float(np.median(s[ok] / v[ok])), int(ok.sum())


def summary(F):
    n = len(F["frame"])
    graded = F["half"] > 0
    print("frames: {}  graded: {}  (half 1: {}, half 2: {})".format(
        n, int(graded.sum()), int((F["half"] == 1).sum()), int((F["half"] == 2).sum())))
    print("teams: H = {}, A = {}   slots H/A: {}/{}".format(
        F["team_names"][0], F["team_names"][1],
        int((F["team_of_slot"] == HOME).sum()), int((F["team_of_slot"] == AWAY).sum())))
    print("ball seen: {:.1f}% of all frames, {:.1f}% of graded   (DATA.md: 56.9% / 59.3%)".format(
        100 * F["ball_seen"].mean(), 100 * F["ball_seen"][graded].mean()))
    print("ball out (> {} m past a line): {} graded frames".format(
        OUT_MARGIN_M, int(F["ball_out"][graded].sum())))
    vis = F["n_visible"][graded].sum(axis=1)
    print("tracked players on graded frames: mean {:.1f}  (DATA.md: 20.5)".format(vis.mean()))
    bs = F["ball_speed"][graded]
    bs = bs[np.isfinite(bs)]
    print("ball speed (m/s) on graded frames: median {:.1f}, p90 {:.1f}, p99 {:.1f}".format(
        *np.percentile(bs, [50, 90, 99])))
    dn = F["d_near"][graded]
    dn = np.fmin(dn[:, 0], dn[:, 1])  # fmin ignores a single NaN; both NaN stays NaN
    dn = dn[np.isfinite(dn)]
    print("nearest player to ball (m): median {:.2f}, p25 {:.2f}, p75 {:.2f}".format(
        *np.percentile(dn, [50, 25, 75])))
    ratio, count = check_speed_unit(F)
    print("speed channel / speed from positions: {:.2f}  over {} samples  "
          "(1.0 = m/s, 3.6 = km/h; SPEED_UNIT_TO_MS = {})".format(ratio, count, SPEED_UNIT_TO_MS))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="rebuild the cache")
    args = parser.parse_args()

    t0 = time.time()
    F = load_frames(force=args.force)
    print("loaded in {:.1f} s -> {}\n".format(time.time() - t0,
                                             os.path.relpath(_cache_path(GAME), ROOT)))
    summary(F)
    tags = load_tags()
    print("\nsmart tagging: {} instances; side counts {}".format(
        len(tags), tags["side"].value_counts().to_dict()))
