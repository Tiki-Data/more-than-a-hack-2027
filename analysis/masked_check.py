"""Label-free check: does each piece help where the ball cannot be seen?

    python src/glue.py && python src/phases.py     # the score files it reads
    python analysis/masked_check.py

Hides approach 1's ball-control evidence on every third 30 s block of each half, combines what is
left exactly as the glue does, decodes it with the HMM, and asks: on the hidden frames where
approach 1 had seen a player in control, how often is the decoded team that player's team?

Ball control is made by code, so this uses no labels. It measures how well the other pieces carry
possession through a stretch with no ball -- the position the 41% of graded frames without a ball
put us in -- for each combination of pieces. It checks the pieces; it was not used to fit them.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES  # noqa: E402
from src.approach1 import control_frames  # noqa: E402
from src.hmm import PROB_FLOOR, decode  # noqa: E402
from src.load import load_frames  # noqa: E402
from src.submission import SCORES_DIR, read_scores  # noqa: E402

# Params
FPS = GAMES[GAME]["fps"]
BLOCK_FRAMES = 30 * FPS   # long enough to hold several possessions, like a real blind stretch
HIDE_EVERY = 3            # one block in three hidden: a third of the match tested

VARIANTS = [
    ("control + dead balls", {"approach1": 1.0, "approach3": 1.0}),
    ("  + team shape", {"approach1": 1.0, "approach3": 1.0, "approach4": 1 / 3}),
    ("  + phase tags", {"approach1": 1.0, "approach3": 1.0, "phases": 1 / 3}),
    ("  + shape + phase tags", {"approach1": 1.0, "approach3": 1.0, "approach4": 1 / 3,
                                "phases": 1 / 3}),
]


def hidden_blocks(half):
    """Every HIDE_EVERY-th BLOCK_FRAMES block of each half, starting with the second."""
    hidden = np.zeros(len(half), dtype=bool)
    for h in (1, 2):
        idx = np.flatnonzero(half == h)
        for k, start in enumerate(range(idx[0], idx[-1] + 1, BLOCK_FRAMES)):
            if k % HIDE_EVERY == 1:
                hidden[start:min(start + BLOCK_FRAMES, idx[-1] + 1)] = True
    return hidden


if __name__ == "__main__":
    F = load_frames()
    half = F["half"]
    team, _ = control_frames(F)
    hidden = hidden_blocks(half)
    target = hidden & (team >= 0)
    truth = np.where(team[target] == 0, "H", "A")

    pieces = {p for _, w in VARIANTS for p in w}
    scores = {p: read_scores(os.path.join(SCORES_DIR, p + ".csv")) for p in pieces}
    scores["approach1"][hidden] = 1 / 3

    print("hidden: {:.1f}% of graded frames; scored on {} hidden ball-control frames\n".format(
        100 * hidden[half > 0].mean(), int(target.sum())))
    print("{:<26} {:>14} {:>13}".format("pieces", "team correct", "called dead"))
    for name, weights in VARIANTS:
        log_p = sum(w * np.log(np.maximum(scores[p], PROB_FLOOR)) for p, w in weights.items())
        p = np.exp(log_p - log_p.max(axis=1, keepdims=True))
        labels = decode(p / p.sum(axis=1, keepdims=True), half)[target]
        print("{:<26} {:>13.1f}% {:>12.1f}%".format(
            name, 100 * np.mean(labels == truth), 100 * np.mean(labels == "D")))
