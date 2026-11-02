"""Profile the tactical match data and print aggregate summaries only.

    python analysis/profile_data.py            # first run parses the ATD (~1-2 min) and caches it

Everything printed here is a rate, a count or a distribution -- never a row of tracking, a list of
frames or anything that would reconstruct the data. That keeps the output safe to paste into an
issue, a chat or an LLM conversation (see OVERVIEW.md, "Use of Large Language Models").

The parsed arrays are cached to data/cache/, which is gitignored along with the rest of data/.

Sections
--------
1. integrity      record count, alignment, slots per team
2. ball           visibility, gap lengths, speed, out-of-pitch, static episodes
3. players        objects per frame, tracklet lifetimes, teleports, duplicates, speeds
4. ball-players   distance from the ball to the nearest player of each team
5. smart tagging  codes, durations, coverage, conflicts
6. homography     validity and reprojection error
7. public window  how each signal behaves inside H, A and D on the public 5 minutes. For
                  understanding the labels only -- nothing here is used to tune anything.
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES, DATA_DIR, half_video_blocks, require, source_frames  # noqa: E402
from baseline import team_of_each_track  # noqa: E402
import load_homography  # noqa: E402
import load_smart_tagging  # noqa: E402

# Params
CACHE_PATH = os.path.join(DATA_DIR, "cache", "atd_tactical.npz")
PUBLIC_GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
FPS = GAMES[GAME]["fps"]
LENGTH, WIDTH = GAMES[GAME]["pitch"]
N_SLOTS = 46
VALUES_PER_LINE = N_SLOTS * 3 + 3
SECOND_HALF_KICKOFF = 113742       # FAQ.md: graded from 110808, all D until this frame
TELEPORT_M = 3.0                   # one-frame jump that no player can make (75 m/s)
DUPLICATE_M = 0.5                  # two same-team tracks this close are probably one player
BALL_SPEED_MAX = 40.0              # m/s, above any real shot
STATIC_SPAN = FPS                  # ball moves < STATIC_M over this many frames -> static
STATIC_M = 0.75
SLOW_TEAM = 2.0                    # m/s, both teams' mean speed below this -> "slow" frame
HOME, AWAY = 0, 1
TAG_TEAMS = {"Barcelona": HOME, "Atletic Club": AWAY}   # "Atletic" is the file's spelling
# Phases come in mirrored pairs on the same interval (Progression + Defending in Middle Third,
# ...), each tagged with its own team. A defending phase's Team is the side WITHOUT the ball.
DEFENDING_CODES = {"DEFENDING IN ATTACKING THIRD", "DEFENDING IN MIDDLE THIRD",
                   "DEFENDING IN DEFENSIVE THIRD", "DEFENSIVE TRANSITION"}
NOT_A_PHASE = {"SET PIECES", "GOALS"}


def load_atd():
    """(ball (T,3), players (T,46,3), slot_team (46,), bad_lines) in video frames, metres not applied."""
    if os.path.isfile(CACHE_PATH):
        cached = np.load(CACHE_PATH)
        return cached["ball"], cached["players"], cached["slot_team"], int(cached["bad_lines"])

    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    n_frames = source_frames()
    values = np.full((n_frames, VALUES_PER_LINE), np.nan, dtype=np.float32)
    bad_lines = 0
    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                bad_lines += 1
                continue
            row = np.fromstring(parts[1].replace(";", ",") + "," + parts[2], sep=",")
            if row.size != VALUES_PER_LINE:
                bad_lines += 1
                continue
            values[int(parts[0]) - 1] = row

    ball = values[:, N_SLOTS * 3:]
    players = values[:, :N_SLOTS * 3].reshape(n_frames, N_SLOTS, 3)
    slot_team = team_of_each_track(meta_path)
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    np.savez(CACHE_PATH, ball=ball, players=players, slot_team=slot_team, bad_lines=bad_lines)
    return ball, players, slot_team, bad_lines


def runs(mask):
    """(starts, lengths) of the True runs in a boolean array."""
    padded = np.concatenate(([False], mask, [False])).astype(np.int8)
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1)
    return starts, np.flatnonzero(edges == -1) - starts


def rolling_nanmean(x, half_width):
    """Centred mean over +/- half_width frames, ignoring NaN."""
    ok = ~np.isnan(x)
    sums = np.concatenate(([0.0], np.cumsum(np.where(ok, x, 0.0))))
    counts = np.concatenate(([0], np.cumsum(ok)))
    index = np.arange(len(x))
    low = np.maximum(0, index - half_width)
    high = np.minimum(len(x), index + half_width + 1)
    n = counts[high] - counts[low]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (sums[high] - sums[low]) / n, np.nan)


def pct(part, whole):
    return "{:5.1f}%".format(100.0 * part / whole) if whole else "  n/a"


def quantiles(x, qs=(10, 50, 90, 99)):
    x = np.asarray(x, dtype=np.float64)
    x = x[~np.isnan(x)]
    if not len(x):
        return "n/a"
    return "  ".join("p{}={:.2f}".format(q, v) for q, v in zip(qs, np.percentile(x, qs)))


def heading(title):
    print("\n" + "=" * 78 + "\n" + title + "\n" + "=" * 78)


def main():
    ball, players, slot_team, bad_lines = load_atd()
    n_frames = len(ball)
    frames = np.arange(n_frames)

    graded = np.zeros(n_frames, dtype=bool)
    half_of = np.zeros(n_frames, dtype=np.int8)
    for half, first, last in half_video_blocks(GAME):
        graded[first:last + 1] = True
        half_of[first:last + 1] = half
    live = graded & ~((frames >= 110808) & (frames < SECOND_HALF_KICKOFF))

    # Metres everywhere from here on.
    bx, by = ball[:, 0] * LENGTH, ball[:, 1] * WIDTH
    px, py, ps = players[:, :, 0] * LENGTH, players[:, :, 1] * WIDTH, players[:, :, 2]
    ball_seen = ~np.isnan(bx)
    present = ~np.isnan(px)

    # ------------------------------------------------------------------------------------------
    heading("1. INTEGRITY")
    print("records parsed            {:>8}   malformed lines: {}".format(n_frames, bad_lines))
    print("graded frames             {:>8}   (live after 2nd-half kickoff gap: {})".format(
        int(graded.sum()), int(live.sum())))
    print("track slots               {:>8}   home {}  away {}".format(
        N_SLOTS, int((slot_team == HOME).sum()), int((slot_team == AWAY).sum())))
    partial = present & np.isnan(ps)
    print("player obs with position but no speed: {} of present".format(
        pct(partial[graded].sum(), present[graded].sum())))
    ball_partial = ball_seen & np.isnan(ball[:, 2])
    print("ball obs with position but no speed:   {} of visible".format(
        pct(ball_partial[graded].sum(), ball_seen[graded].sum())))

    # ------------------------------------------------------------------------------------------
    heading("2. BALL")
    print("visible, all frames       {}".format(pct(ball_seen.sum(), n_frames)))
    print("visible, graded           {}".format(pct(ball_seen[graded].sum(), graded.sum())))
    for half in (1, 2):
        m = live & (half_of == half)
        print("visible, live half {}      {}".format(half, pct(ball_seen[m].sum(), m.sum())))
    live_idx = np.flatnonzero(live)
    blocks = [ball_seen[live_idx[i:i + 7500]].mean()
              for i in range(0, len(live_idx) - 7499, 7500)]
    print("visible per 5-min block of live play: {}   (the public window is the first block)".format(
        quantiles(np.array(blocks) * 100, (0, 25, 50, 75, 100))))

    gap_lengths = []
    for _, first, last in half_video_blocks(GAME):
        segment = ~ball_seen[first:last + 1]
        gap_lengths.append(runs(segment)[1])
    gap_lengths = np.concatenate(gap_lengths)
    missing_total = gap_lengths.sum()
    print("\ngaps in graded play       {} gaps, {} missing frames".format(
        len(gap_lengths), int(missing_total)))
    print("  {:<16} {:>7} {:>16}".format("gap length", "gaps", "% missing frames"))
    for low, high, label in ((1, 5, "1-5 fr (<0.2 s)"), (6, 25, "6-25 (<=1 s)"),
                             (26, 125, "26-125 (<=5 s)"), (126, 750, "126-750 (<=30 s)"),
                             (751, 10 ** 9, ">750 (>30 s)")):
        m = (gap_lengths >= low) & (gap_lengths <= high)
        print("  {:<16} {:>7} {:>16}".format(label, int(m.sum()),
                                             pct(gap_lengths[m].sum(), missing_total)))

    step = np.hypot(np.diff(bx), np.diff(by)) * FPS          # m/s between consecutive frames
    step_ok = ball_seen[1:] & ball_seen[:-1] & graded[1:]
    speeds = step[step_ok]
    print("\nball speed from positions (m/s): {}".format(quantiles(speeds, (50, 90, 99, 99.9))))
    print("  steps faster than {} m/s (implausible): {}".format(
        BALL_SPEED_MAX, pct((speeds > BALL_SPEED_MAX).sum(), len(speeds))))
    provided = ball[1:, 2][step_ok]
    both = ~np.isnan(provided)
    if both.sum() > 100:
        corr = np.corrcoef(provided[both], speeds[both])[0, 1]
        print("  provided speed channel: {}   corr with computed: {:.2f}".format(
            quantiles(provided[both], (50, 90, 99)), corr))

    out = ball_seen & ((ball[:, 0] < 0) | (ball[:, 0] > 1) | (ball[:, 1] < 0) | (ball[:, 1] > 1))
    print("\nball outside pitch (0-1 box), of visible live frames: {}".format(
        pct(out[live].sum(), ball_seen[live].sum())))
    out_runs = runs(out & live)[1]
    if len(out_runs):
        print("  out-of-pitch episodes: {}   duration s: {}".format(
            len(out_runs), quantiles(out_runs / FPS, (50, 90))))

    ahead = np.zeros(n_frames, dtype=bool)
    moved = np.hypot(bx[STATIC_SPAN:] - bx[:-STATIC_SPAN], by[STATIC_SPAN:] - by[:-STATIC_SPAN])
    ahead[:-STATIC_SPAN] = (ball_seen[:-STATIC_SPAN] & ball_seen[STATIC_SPAN:]
                            & (moved < STATIC_M))
    static_runs = runs(ahead & live)[1]
    long_static = static_runs[static_runs >= FPS]                 # static for >= 2 s in total
    print("\nstatic-ball episodes >= 2 s: {}   duration s: {}   (59 set pieces are tagged)".format(
        len(long_static), quantiles((long_static + STATIC_SPAN) / FPS, (50, 90))))

    # ------------------------------------------------------------------------------------------
    heading("3. PLAYERS")
    home_n = present[:, slot_team == HOME].sum(axis=1)
    away_n = present[:, slot_team == AWAY].sum(axis=1)
    for name, counts in (("home", home_n), ("away", away_n)):
        c = counts[live]
        print("{} tracked per live frame: mean {:.1f}   <9: {}   >11: {}".format(
            name, c.mean(), pct((c < 9).sum(), len(c)), pct((c > 11).sum(), len(c))))
    print("total tracked per live frame: mean {:.1f}".format((home_n + away_n)[live].mean()))

    lifetimes = np.concatenate([runs(present[:, s] & graded)[1] for s in range(N_SLOTS)])
    print("\ntracklets (continuous presence of one slot): {}".format(len(lifetimes)))
    print("  length s: {}".format(quantiles(lifetimes / FPS, (10, 50, 90))))
    print("  tracklets < 1 s: {}   player-frames in tracklets >= 1 min: {}".format(
        pct((lifetimes < FPS).sum(), len(lifetimes)),
        pct(lifetimes[lifetimes >= 60 * FPS].sum(), lifetimes.sum())))

    jump = np.hypot(np.diff(px, axis=0), np.diff(py, axis=0))
    jump_ok = present[1:] & present[:-1] & live[1:, None]
    teleports = (jump > TELEPORT_M) & jump_ok
    live_minutes = live.sum() / FPS / 60
    print("\none-frame jumps > {} m (identity switch / glitch): {}  = {:.1f} per minute".format(
        TELEPORT_M, int(teleports.sum()), teleports.sum() / live_minutes))
    print("player speed channel (m/s): {}".format(quantiles(ps[live][present[live]], (50, 90, 99))))

    same_team = (slot_team[:, None] == slot_team[None, :])
    upper = np.triu(np.ones((N_SLOTS, N_SLOTS), dtype=bool), 1)
    dup_frames = opp_close_frames = 0
    for start in range(0, len(live_idx), 10000):
        idx = live_idx[start:start + 10000]
        dx = px[idx][:, :, None] - px[idx][:, None, :]
        dy = py[idx][:, :, None] - py[idx][:, None, :]
        close = (np.hypot(dx, dy) < DUPLICATE_M) & upper
        dup_frames += int((close & same_team).any(axis=(1, 2)).sum())
        opp_close_frames += int((close & ~same_team).any(axis=(1, 2)).sum())
    print("\nframes with a same-team pair < {} m (likely duplicate): {}".format(
        DUPLICATE_M, pct(dup_frames, len(live_idx))))
    print("frames with an opposing pair < {} m (contact/occlusion): {}".format(
        DUPLICATE_M, pct(opp_close_frames, len(live_idx))))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        home_speed = rolling_nanmean(np.nanmean(np.where(slot_team == HOME, ps, np.nan), axis=1), FPS)
        away_speed = rolling_nanmean(np.nanmean(np.where(slot_team == AWAY, ps, np.nan), axis=1), FPS)
    slow = (home_speed < SLOW_TEAM) & (away_speed < SLOW_TEAM)
    print("\nteam mean speed, 1 s smoothed (m/s): home {}   away {}".format(
        quantiles(home_speed[live], (10, 50, 90)), quantiles(away_speed[live], (10, 50, 90))))
    print("live frames with BOTH teams < {} m/s (stoppage-like): {}".format(
        SLOW_TEAM, pct(slow[live].sum(), live.sum())))
    faster = np.fmax(home_speed, away_speed)
    gap_starts, gap_lengths_live = runs(~ball_seen & live)
    long_gap = np.zeros(n_frames, dtype=bool)
    for s, n in zip(gap_starts, gap_lengths_live):
        if n > 5 * FPS:
            long_gap[s:s + n] = True
    print("faster team's speed (m/s): ball visible {}   ball gone > 5 s {}".format(
        quantiles(faster[live & ball_seen], (50,)), quantiles(faster[long_gap], (50,))))
    edge = np.fmin(np.fmin(bx, LENGTH - bx), np.fmin(by, WIDTH - by))
    last_seen = gap_starts[(gap_lengths_live > 5 * FPS) & (gap_starts > 0)] - 1
    last_seen = last_seen[ball_seen[last_seen]]
    print("long gaps whose last-seen ball is < 3 m from a line or outside: {}".format(
        pct((edge[last_seen] < 3).sum(), len(last_seen))))

    # ------------------------------------------------------------------------------------------
    heading("4. BALL TO NEAREST PLAYER (live frames with a visible ball)")
    dist = np.hypot(px - bx[:, None], py - by[:, None])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        near_home = np.nanmin(np.where(slot_team == HOME, dist, np.nan), axis=1)
        near_away = np.nanmin(np.where(slot_team == AWAY, dist, np.nan), axis=1)
    nearest = np.fmin(near_home, near_away)
    m = live & ball_seen & ~np.isnan(nearest)
    print("nearest player distance (m): {}".format(quantiles(nearest[m], (10, 50, 90, 99))))
    for low, high in ((0, 1), (1, 2), (2, 5), (5, 10), (10, 1e9)):
        b = m & (nearest >= low) & (nearest < high)
        print("  {:>4}-{:<4} m  {}".format(low, "" if high > 1e8 else high, pct(b.sum(), m.sum())))
    margin = np.abs(near_home - near_away)
    print("|home - away| nearest-distance margin < 1 m: {}  (genuinely contested)".format(
        pct((margin[m] < 1).sum(), m.sum())))

    # ------------------------------------------------------------------------------------------
    heading("5. SMART TAGGING")
    events = load_smart_tagging.load()
    tags = pd.DataFrame([{"code": e["code"], "team": e["tags"].get("Team"),
                          "type": e["tags"].get("Type"), "start": e["start_frame"],
                          "end": e["end_frame"]} for e in events])
    tags["dur_s"] = (tags["end"] - tags["start"]) / FPS
    summary = tags.groupby("code")["dur_s"].describe(percentiles=[.5])[["count", "50%", "max"]]
    print(summary.rename(columns={"50%": "median_s", "max": "max_s"}).round(1).to_string())
    print("\nTeam values: {}".format(tags["team"].value_counts(dropna=False).to_dict()))
    print("Set-piece types: {}".format(tags.loc[tags["code"] == "SET PIECES", "type"]
                                       .value_counts().to_dict()))

    twins = tags.groupby(["start", "end"]).size()
    print("\nintervals shared by two codes (mirrored attack/defend pairs): {} of {}".format(
        int((twins > 1).sum()), len(twins)))

    # cover[t] = frames where some phase says team t has the ball.
    cover = np.zeros((2, n_frames), dtype=bool)
    set_piece = np.zeros(n_frames, dtype=bool)
    for row in tags.itertuples():
        if row.code == "SET PIECES":
            set_piece[row.start:row.end] = True
        if row.team in TAG_TEAMS and row.code not in NOT_A_PHASE:
            team = TAG_TEAMS[row.team]
            cover[1 - team if row.code in DEFENDING_CODES else team, row.start:row.end] = True
    any_cover = cover.any(axis=0)
    conflict = cover.all(axis=0)
    print("live frames where a phase names the team in possession: {}".format(
        pct(any_cover[live].sum(), live.sum())))
    print("  of those, phases disagree on the team: {}".format(
        pct(conflict[live].sum(), any_cover[live].sum())))
    print("live frames inside a set-piece window: {}".format(pct(set_piece[live].sum(), live.sum())))
    print("ball visible inside set-piece windows: {}   static: {}".format(
        pct(ball_seen[set_piece & live].sum(), (set_piece & live).sum()),
        pct(ahead[set_piece & live].sum(), (set_piece & live).sum())))

    # ------------------------------------------------------------------------------------------
    heading("6. HOMOGRAPHY")
    _, error, valid = load_homography.load()
    print("valid: {}   reprojection error on graded frames: {}".format(
        pct(valid.sum(), len(valid)), quantiles(error[graded & valid], (50, 90, 99))))

    # ------------------------------------------------------------------------------------------
    heading("7. PUBLIC WINDOW: SIGNALS BY GROUND-TRUTH CLASS (understanding only)")
    gt = pd.read_csv(PUBLIC_GT)["label"].str.strip().str.upper().values
    window = gt != "X"
    print("public window frames: {}".format(int(window.sum())))

    seg_starts, seg_lengths = runs(window)
    labels_seq, durations = [], []
    for s, n in zip(seg_starts, seg_lengths):
        block = gt[s:s + n]
        change = np.flatnonzero(block[1:] != block[:-1]) + 1
        bounds = np.concatenate(([0], change, [n]))
        for a, b in zip(bounds[:-1], bounds[1:]):
            labels_seq.append(block[a])
            durations.append((b - a) / FPS)
    labels_seq, durations = np.array(labels_seq), np.array(durations)
    direct = sum(1 for a, b in zip(labels_seq[:-1], labels_seq[1:]) if {a, b} == {"H", "A"})
    print("segments {}   possessions {}   dead-ball runs {}   direct H<->A changes {}".format(
        len(labels_seq), int(np.isin(labels_seq, ["H", "A"]).sum()),
        int((labels_seq == "D").sum()), direct))
    print("possession duration s: {}".format(
        quantiles(durations[np.isin(labels_seq, ["H", "A"])], (10, 50, 90))))
    print("dead-ball duration s:  {}".format(quantiles(durations[labels_seq == "D"], (10, 50, 90))))

    print("\n{:<34} {:>8} {:>8} {:>8}".format("signal (share of frames)", "H", "A", "D"))
    rows = (("ball visible", ball_seen), ("ball outside pitch", out),
            ("ball static (moves <0.75 m in 1 s)", ahead), ("both teams slow (<2 m/s)", slow),
            ("inside a set-piece window", set_piece),
            ("a phase says home has the ball", cover[HOME]),
            ("a phase says away has the ball", cover[AWAY]))
    for name, signal in rows:
        print("{:<34} {:>8} {:>8} {:>8}".format(name, *[
            pct((signal & (gt == c)).sum(), (gt == c).sum()) for c in ("H", "A", "D")]))
    print("{:<34} {:>8} {:>8} {:>8}".format("faster team's speed, median m/s", *[
        "{:.2f}".format(np.nanmedian(faster[gt == c])) for c in ("H", "A", "D")]))

    alive = np.isin(gt, ["H", "A"]) & ball_seen & ~np.isnan(nearest)
    nearest_team = np.where(near_home <= near_away, "H", "A")
    print("\nnearest-player team == GT team, live frames with a visible ball:")
    for low, high in ((0, 1), (1, 2), (2, 5), (5, 1e9)):
        b = alive & (nearest >= low) & (nearest < high)
        print("  nearest {:>2}-{:<3} m   agree {}   (n={})".format(
            low, "" if high > 1e8 else high, pct((nearest_team[b] == gt[b]).sum(), b.sum()),
            int(b.sum())))
    tagged = np.isin(gt, ["H", "A"]) & (cover[HOME] ^ cover[AWAY])
    tag_team = np.where(cover[HOME], "H", "A")
    print("phase team == GT team (where phases name exactly one team): {}  (covers {} of H/A)"
          .format(pct((tag_team[tagged] == gt[tagged]).sum(), tagged.sum()),
                  pct(tagged.sum(), np.isin(gt, ["H", "A"]).sum())))
    print("\nper code, on public H/A frames:")
    for code in sorted(set(tags["code"]) - NOT_A_PHASE):
        team_of = np.full(n_frames, -1, dtype=np.int8)
        for row in tags[tags["code"] == code].itertuples():
            if row.team in TAG_TEAMS:
                team = TAG_TEAMS[row.team]
                team_of[row.start:row.end] = 1 - team if code in DEFENDING_CODES else team
        k = np.isin(gt, ["H", "A"]) & (team_of >= 0)
        if k.sum():
            print("  {:<30} frames {:>5}   agree {}".format(
                code, int(k.sum()), pct((np.where(team_of[k] == HOME, "H", "A") == gt[k]).sum(),
                                        k.sum())))


if __name__ == "__main__":
    main()
