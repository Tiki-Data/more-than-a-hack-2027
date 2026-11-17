"""Phase tags: who has the ball, from the Smart Tagging's phases of play.

    python src/phases.py      # -> out/scores/phases.csv  (per-frame evidence, for the glue)

Besides its 59 set pieces, the Smart Tagging holds 579 phases of play, each an interval with a
team: Build Up, Progression, Attacking Transition, Creating Chances, Ball in Final Third, Long Ball,
Ball in the Box, Players in the Box -- and, for the first four, a mirror code on exactly the same
intervals with the other team (Defending in Attacking Third, Defending in Middle Third, Defensive
Transition, Defending in Defensive Third). Approach 3 uses the set pieces and goals; this piece
uses the phases.

1. CALIBRATE EACH CODE, WITHOUT LABELS. For every code, over the frames inside its intervals where
   approach 1 sees a player in control of the ball, the share where that player's team is the
   tagged team. That share is the code's confidence, and its direction: a code below 1/2 speaks
   for the OTHER team (the Defending... codes), with confidence 1 - share. Measured on every run,
   not assumed: the ball-control frames are made by code (src/approach1.py), never by hand.
2. PER FRAME, the tags covering it vote. Nested tags describe the same attack (Progression inside
   Ball in Final Third inside Players in the Box) and mirrors repeat each other, so they are not
   independent evidence: per team only the single strongest tag counts, and the frame's log-odds
   for Barcelona is logit(strongest for H) - logit(strongest for A).
3. EVIDENCE: D = 1/3 always -- phases run through stoppages (they cover 47% of the frames approach
   3 is sure are dead), so they say nothing about whether the ball is in play -- and H and A share
   the other 2/3 by the log-odds. A frame no phase covers is 1/3,1/3,1/3: no opinion.

A phase is an interval, not a possession: it can span a turnover or stop short of one. That is
what the measured confidence prices in, and why the glue gives this piece a low weight.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from src.approach1 import control_frames  # noqa: E402
from src.load import load_frames, load_tags  # noqa: E402
from src.submission import SCORES_DIR, write_scores  # noqa: E402

# Params
# Left to approach 3: set pieces are restarts, and a goal tag is a stoppage, not a phase of play.
SKIP_CODES = {"SET PIECES", "GOALS"}

H, A = 0, 1
SIDE_CODE = {"H": H, "A": A}


def calibrate(F, tags):
    """{code: (speaks_for_tagged_team, confidence, control frames measured)} for every phase code."""
    team, _ = control_frames(F)
    out = {}
    for code, grp in tags.groupby("code"):
        if code in SKIP_CODES:
            continue
        same = other = 0
        for t in grp.itertuples():
            if t.side not in SIDE_CODE:
                continue
            in_control = team[int(t.start_frame):int(t.end_frame)]
            same += int((in_control == SIDE_CODE[t.side]).sum())
            other += int((in_control == 1 - SIDE_CODE[t.side]).sum())
        if same + other == 0:
            continue
        share = same / (same + other)
        out[code] = (share >= 0.5, max(share, 1 - share), same + other)
    return out


def evidence(F, tags, conf):
    """(n, 3) H/A/D probabilities per frame; uniform where no phase covers the frame."""
    n = len(F["frame"])
    strongest = np.zeros((n, 2))  # per frame, the strongest logit speaking for H and for A
    for t in tags.itertuples():
        if t.code not in conf or t.side not in SIDE_CODE:
            continue
        for_tagged, q, _ = conf[t.code]
        team = SIDE_CODE[t.side] if for_tagged else 1 - SIDE_CODE[t.side]
        a, b = int(t.start_frame), int(t.end_frame)
        np.maximum(strongest[a:b, team], np.log(q / (1 - q)), out=strongest[a:b, team])
    p_home = 1 / (1 + np.exp(-(strongest[:, H] - strongest[:, A])))
    return np.column_stack([2 / 3 * p_home, 2 / 3 * (1 - p_home), np.full(n, 1 / 3)])


def describe(F, conf, probs):
    """Label-free summary: each code's measured direction and confidence, and coverage."""
    print("{:<30} {:>9} {:>11} {:>9}".format("code", "speaks for", "confidence", "frames"))
    for code, (for_tagged, q, m) in sorted(conf.items(), key=lambda kv: -kv[1][1]):
        print("{:<30} {:>9} {:>11.3f} {:>9}".format(code, "tagged" if for_tagged else "other", q, m))
    g = F["half"] > 0
    has = np.abs(probs[:, H] - probs[:, A]) > 1e-9
    blind = g & ~F["ball_seen"]
    print("\ngraded frames with an opinion: {:.1f}%   ...where the ball is missing: {:.1f}%".format(
        100 * has[g].mean(), 100 * has[blind].mean()))


if __name__ == "__main__":
    F = load_frames()
    tags = load_tags()
    conf = calibrate(F, tags)
    probs = evidence(F, tags, conf)
    path = write_scores(os.path.join(SCORES_DIR, "phases.csv"), probs)
    print("wrote", os.path.relpath(path, ROOT), "\n")
    describe(F, conf, probs)
