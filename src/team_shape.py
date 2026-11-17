"""Team-shape evidence (approach 4): who has the ball, judged from how the two teams are spread out.

    python src/team_shape.py                    # writes out/evidence/team_shape.csv
                                                #    and out/evidence/ball_short_gaps_filled.csv
    python src/team_shape.py --no-diagnostics   # skip the checks at the end

The idea: a team with the ball spreads out, long and wide, to make passing lanes. A team without
it squeezes up compact. So when Barcelona look more spread out than Athletic, that is evidence
Barcelona have the ball. Pressing points the same way: opponents close down the ball carrier, so
the team whose most-surrounded player is under more pressure probably has it. This matters most
when the ball itself is missing, which it is on 41% of graded frames -- the players are still
tracked.

team_shape.csv follows the team format, frame,H,A,D, as probabilities: H is the chance Barcelona
have the ball, A = 1 - H, D = 0. H stays between about 0.27 and 0.73, and 0.5,0.5 means no opinion.
Shape changes over seconds, so this says WHO probably has the ball, not WHEN it changed hands.

ball_short_gaps_filled.csv is the ball in metres with tiny gaps (up to 0.5 s) filled in with a
straight line.

The reasoning behind every constant is in TEAM_SHAPE_NOTES.md (team-hq repo, working_docs/),
under the [N#] tag shown next to it here. Everything printed is an aggregate, never rows of data.
"""

import argparse
import hashlib
import os
import sys
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES  # noqa: E402
from src.load import CACHE_DIR, CACHE_VERSION  # noqa: E402
from src.submission import OUT_DIR  # noqa: E402

# Params
CACHE_PATH = os.path.join(CACHE_DIR, "{}_frames_v{}.npz".format(GAME, CACHE_VERSION))  # [N1]
EVIDENCE_DIR = os.path.join(OUT_DIR, "evidence")
SHAPE_CSV = os.path.join(EVIDENCE_DIR, "team_shape.csv")
BALL_CSV = os.path.join(EVIDENCE_DIR, "ball_short_gaps_filled.csv")

FPS = GAMES[GAME]["fps"]
PITCH_LENGTH = GAMES[GAME]["pitch"][0]
BARCELONA, ATHLETIC = 0, 1                 # how src.load codes the teams

SECOND_HALF_KICKOFF = 113742               # [N2] frames before this in the 2nd half are all D
MIN_OUTFIELD_PLAYERS = 6                   # [N4] fewer than this visible -> no opinion
SPREAD_PERCENTILES = (10, 90)              # [N5] ignore the most extreme player at each end
SMOOTHING_FRAMES = FPS                     # [N6] average over 1 s either side
FEATURE_WEIGHTS = {"length": 1.0, "width": 1.0,  # [N7]
                   "line": 0.0,                   # [N7] switched off: points the wrong way
                   "pressure": 0.0}               # [N12] switched off: too weak, lowers the score
PRESSURE_RADIUS = 5.0                      # [N12] metres; opponents further away add no pressure
MAX_SCORE = 1.0                            # [N8] about 73% at most -- shape is weak evidence
MAX_GAP_TO_FILL = 12                       # [N9] frames, just under 0.5 s
MAX_BALL_SPEED = 40.0                      # [N9] m/s, faster than any real shot
NEAR_BALL = 1.0                            # [N10] metres, only used by the checks


def load_cached_frames():
    """The shared table exactly as src/load.py cached it, so we use the same data as everyone. [N1]"""
    if not os.path.isfile(CACHE_PATH):
        raise FileNotFoundError(CACHE_PATH + " not found -- run `python src/load.py` first")
    with np.load(CACHE_PATH) as cached:
        return {key: cached[key] for key in cached.files}


def frames_we_judge(F):
    """Graded frames, minus the stretch of the 2nd half before kickoff."""
    frame = F["frame"]
    before_second_kickoff = (frame >= 110808) & (frame < SECOND_HALF_KICKOFF)
    return (F["half"] > 0) & ~before_second_kickoff


def check_attack_direction(F, judged):
    """Make sure src.load's attack direction agrees with where the players actually stand.

    Each team's deepest player is nearly always their keeper, so whichever team's deepest player
    sits closer to x = 0 is defending the left goal. Getting this wrong would flip every feature,
    so we stop rather than carry on.
    """
    x = F["players_xy"][:, :, 0]
    print("[N3] attack direction: median of each team's deepest x (m)")
    for half in (1, 2):
        rows = judged & (F["half"] == half)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            barca_deepest = np.nanmedian(np.nanmin(x[rows][:, F["team_of_slot"] == BARCELONA], axis=1))
            athletic_deepest = np.nanmedian(np.nanmin(x[rows][:, F["team_of_slot"] == ATHLETIC], axis=1))

        barca_attack_right_by_data = barca_deepest < athletic_deepest
        barca_attack_right_by_load = F["h_attacks_right"][rows][0]
        print("  half {}: Barcelona {:5.1f}  Athletic {:5.1f}  -> Barcelona attack {}   "
              "(src.load says {})".format(half, barca_deepest, athletic_deepest,
                                         "right" if barca_attack_right_by_data else "left",
                                         "right" if barca_attack_right_by_load else "left"))
        if barca_attack_right_by_data != barca_attack_right_by_load:
            raise ValueError("attack direction in half {} disagrees with src.load".format(half))


def distance_from_own_goal(F):
    """Every player's x, measured forward from his own goal line. Removes the half-time switch."""
    x = F["players_xy"][:, :, 0].astype(np.float64)
    is_barca = (F["team_of_slot"] == BARCELONA)[None, :]
    barca_right = F["h_attacks_right"][:, None]
    attacks_right = np.where(is_barca, barca_right, ~barca_right)
    return np.where(attacks_right, x, PITCH_LENGTH - x)


def measure_team_shape(forward_x, y):
    """Length, width and back-line height of one team's outfield players, per frame.

    forward_x and y are (frames, slots) for one team. The deepest player in each frame is taken
    out first: that is almost always the keeper, and slots are not people, so we cannot find him
    any other way.
    """
    forward_x, y = forward_x.copy(), y.copy()
    seen = ~np.isnan(forward_x)
    frames_with_players = np.flatnonzero(seen.any(axis=1))
    keeper = np.argmin(np.where(seen, forward_x, np.inf), axis=1)[frames_with_players]
    forward_x[frames_with_players, keeper] = np.nan
    y[frames_with_players, keeper] = np.nan

    low, high = SPREAD_PERCENTILES
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # frames with nobody visible -> NaN
        x_low, x_high = np.nanpercentile(forward_x, [low, high], axis=1)
        y_low, y_high = np.nanpercentile(y, [low, high], axis=1)
        back_line = np.nanmin(forward_x, axis=1)

    shape = {"length": x_high - x_low, "width": y_high - y_low, "line": back_line}
    too_few_players = (~np.isnan(forward_x)).sum(axis=1) < MIN_OUTFIELD_PLAYERS
    for values in shape.values():
        values[too_few_players] = np.nan
    return shape


def pressure_on_most_pressed_player(own_xy, opponent_xy):
    """Per frame: the pressure on whichever of this team's players is most closed down. [N12]

    Pressure on one player = sum over opponents of (1 - distance / PRESSURE_RADIUS), for opponents
    inside the radius: an opponent at 0 m adds 1, at 2.5 m adds 0.5, at 5 m or more nothing.
    Opponents press the ball carrier, so the team whose player is most surrounded probably has it.
    """
    pressure = np.full(own_xy.shape[:2], np.nan)                       # (frames, own slots)
    for slot in range(own_xy.shape[1]):
        distance = np.linalg.norm(opponent_xy - own_xy[:, slot:slot + 1], axis=2)
        closeness = np.clip(1 - distance / PRESSURE_RADIUS, 0, None)  # NaN (not tracked) stays NaN
        pressure[:, slot] = np.nansum(closeness, axis=1)
    pressure[np.isnan(own_xy[:, :, 0])] = np.nan                       # this player not tracked
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)                # nobody visible -> NaN
        return np.nanmax(pressure, axis=1)


def rolling_mean(values, half_width):
    """Centred moving average over +/- half_width frames that skips NaNs."""
    ok = ~np.isnan(values)
    running_sum = np.concatenate(([0.0], np.cumsum(np.where(ok, values, 0.0))))
    running_count = np.concatenate(([0], np.cumsum(ok)))
    i = np.arange(len(values))
    start = np.maximum(0, i - half_width)
    end = np.minimum(len(values), i + half_width + 1)
    count = running_count[end] - running_count[start]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, (running_sum[end] - running_sum[start]) / count, np.nan)


def shape_score(F, judged):
    """The score s per frame, plus each feature on a common scale (for the checks).

    For each feature we take Barcelona minus Athletic, so positive always favours Barcelona. We
    smooth it inside each half, divide by its typical size so metres of length and metres of width
    count the same, average the features, and squash the result into [-MAX_SCORE, MAX_SCORE].
    """
    forward_x = distance_from_own_goal(F)
    y = F["players_xy"][:, :, 1].astype(np.float64)
    barca = F["team_of_slot"] == BARCELONA
    athletic = F["team_of_slot"] == ATHLETIC
    barca_shape = measure_team_shape(forward_x[:, barca], y[:, barca])
    athletic_shape = measure_team_shape(forward_x[:, athletic], y[:, athletic])

    xy = F["players_xy"].astype(np.float64)
    barca_shape["pressure"] = pressure_on_most_pressed_player(xy[:, barca], xy[:, athletic])
    athletic_shape["pressure"] = pressure_on_most_pressed_player(xy[:, athletic], xy[:, barca])
    # length is NaN where a team has too few players visible [N4]; pressure gets no opinion there too.
    for shape in (barca_shape, athletic_shape):
        shape["pressure"][np.isnan(shape["length"])] = np.nan

    print("\n[N7] typical Barcelona-minus-Athletic difference (metres; pressure has no unit)")
    scaled = {}
    for feature in FEATURE_WEIGHTS:
        difference = barca_shape[feature] - athletic_shape[feature]

        smoothed = np.full_like(difference, np.nan)
        for half in (1, 2):
            rows = np.flatnonzero(F["half"] == half)
            smoothed[rows] = rolling_mean(difference[rows], SMOOTHING_FRAMES)

        # Typical size around zero, not around the median: 0 means "the teams look alike", and
        # centring would quietly penalise whichever team has more of the ball over the match.
        typical = 1.4826 * np.nanmedian(np.abs(smoothed[judged]))
        scaled[feature] = smoothed / typical
        print("  {:<7} {:6.2f}   weight {}".format(feature, typical, FEATURE_WEIGHTS[feature]))

    weighted_sum = np.zeros(len(judged))
    weight_present = np.zeros(len(judged))
    for feature, weight in FEATURE_WEIGHTS.items():
        if weight == 0:
            continue
        present = ~np.isnan(scaled[feature])
        weighted_sum[present] += weight * scaled[feature][present]
        weight_present[present] += weight

    with np.errstate(invalid="ignore", divide="ignore"):
        average = weighted_sum / weight_present
    score = MAX_SCORE * np.tanh(np.nan_to_num(average, nan=0.0))
    score[~judged] = 0.0
    return score, scaled


def fill_short_ball_gaps(F):
    """The ball with gaps of up to MAX_GAP_TO_FILL frames filled by a straight line.

    Only inside a half, only when the ball is seen on both sides of the gap, and only if getting
    from one side to the other doesn't need an impossible speed. Longer gaps stay empty: nearly
    every missing frame is in a gap over 5 s, and a straight line across that would be made up.
    """
    ball = F["ball_xy"].astype(np.float64)
    filled = np.zeros(len(ball), dtype=bool)

    for half in (1, 2):
        rows = np.flatnonzero(F["half"] == half)
        first, last = rows[0], rows[-1]

        missing = (~F["ball_seen"][first:last + 1]).astype(np.int8)
        edges = np.diff(np.concatenate(([0], missing, [0])))
        gap_starts = np.flatnonzero(edges == 1) + first
        gap_ends = np.flatnonzero(edges == -1) + first          # first frame after the gap

        for start, end in zip(gap_starts, gap_ends):
            length = end - start
            if length > MAX_GAP_TO_FILL or start == first or end > last:
                continue
            before, after = ball[start - 1], ball[end]
            speed_needed = np.hypot(*(after - before)) * FPS / (length + 1)
            if speed_needed > MAX_BALL_SPEED:
                continue
            fraction = np.arange(1, length + 1)[:, None] / (length + 1)
            ball[start:end] = before + fraction * (after - before)
            filled[start:end] = True

    return ball, filled


def print_checks(F, score, scaled, filled, judged):
    """Coverage, gap filling, and a direction check against the nearest player's team.

    The direction check uses frames where a player is within NEAR_BALL of the ball and assumes his
    team has it -- right about 91% of the time. Those labels are made by code, never by hand, and
    nothing is fitted to them; they only tell us whether each feature points the right way.
    """
    def pct(part, whole):
        return "{:5.1f}%".format(100.0 * part / whole) if whole else "  n/a"

    ball_missing = judged & ~F["ball_seen"]
    has_opinion = score != 0
    print("\ncoverage")
    print("  judged frames with an opinion        {}".format(
        pct((judged & has_opinion).sum(), judged.sum())))
    print("  ...where the ball is missing         {}".format(
        pct((ball_missing & has_opinion).sum(), ball_missing.sum())))
    print("  |score|                              p10 {:.2f}  p50 {:.2f}  p90 {:.2f}".format(
        *np.percentile(np.abs(score[judged]), (10, 50, 90))))

    print("\n[N9] short-gap filling")
    print("  ball missing on judged frames        {}".format(pct(ball_missing.sum(), judged.sum())))
    print("  missing frames filled                {}  ({} frames)".format(
        pct((filled & judged).sum(), ball_missing.sum()), int((filled & judged).sum())))

    distance = np.where(np.isnan(F["d_near"]), np.inf, F["d_near"])     # (frames, [Barca, Athletic])
    near_ball = judged & (distance.min(axis=1) <= NEAR_BALL)
    barca_nearest = np.argmin(distance, axis=1)[near_ball] == BARCELONA
    print("\n[N10] direction check on {} frames with a player within {:.0f} m of the ball "
          "({} of them Barcelona)".format(int(near_ball.sum()), NEAR_BALL,
                                          pct(barca_nearest.sum(), near_ball.sum()).strip()))
    for name, values in list(scaled.items()) + [("score", score)]:
        v = values[near_ball]
        usable = ~np.isnan(v) & (v != 0)
        agrees = (v[usable] > 0) == barca_nearest[usable]
        print("  {:<7} points the same way on      {}".format(name, pct(agrees.sum(), usable.sum())))


def write_csv(table, path):
    """Fixed float format and '\\n' line endings, so a rerun gives a byte-identical file. [N11]"""
    table.to_csv(path, index=False, float_format="%.4f", na_rep="", lineterminator="\n")
    with open(path, "rb") as handle:
        fingerprint = hashlib.sha256(handle.read()).hexdigest()[:16]
    print("wrote", os.path.relpath(path, ROOT), " sha256", fingerprint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-diagnostics", action="store_true", help="skip the checks")
    args = parser.parse_args()

    F = load_cached_frames()
    judged = frames_we_judge(F)
    check_attack_direction(F, judged)

    score, scaled = shape_score(F, judged)
    ball, filled = fill_short_ball_gaps(F)

    if not args.no_diagnostics:
        print_checks(F, score, scaled, filled, judged)

    print()
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    barca_probability = 1 / (1 + np.exp(-score))       # score is log-odds -> P(Barcelona has it)
    write_csv(pd.DataFrame({"frame": F["frame"], "H": barca_probability, "A": 1 - barca_probability,
                            "D": 0.5}), SHAPE_CSV)
    write_csv(pd.DataFrame({"frame": F["frame"], "ball_x": ball[:, 0], "ball_y": ball[:, 1],
                            "filled": filled.astype(np.int8)}), BALL_CSV)
