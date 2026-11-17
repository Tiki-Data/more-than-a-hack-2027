"""Approach 2: an HMM with minimum durations, decoded with Viterbi. P2's decoder.

    from src.hmm import decode
    labels = decode(probs, F["half"])     # same interface as src/decode.py

    python src/hmm.py                                # decodes out/scores/approach1.csv
    python src/hmm.py out/scores/<piece>.csv         # or any score file -> out/hmm_<piece>.csv

The model
---------
Three hidden states, H, A and D, each half decoded on its own (nothing carries across halftime).

- MINIMUM DURATIONS. Each state is a chain of sub-states that must be walked through before the
  state can be left, so a possession lasts at least MIN_POSSESSION_FRAMES and a stoppage at least
  MIN_DEAD_FRAMES. This is what removes the micro-possessions that spellF1 punishes: a plain HMM
  allows a one-frame possession whenever a single frame of evidence is strong enough.
- TRANSITIONS, only from the end of a chain. Leaving a state is a per-frame chance set from its
  mean duration; a possession that ends goes to the other team or to D (both hand-overs, H->A
  directly and H->D->A, can be produced); a stoppage restarts to either team equally, and the
  evidence decides which.
- EMISSIONS are the pieces' probabilities: EMISSION_WEIGHT * log(p), with p floored at PROB_FLOOR
  so no frame is ever certain. A "no idea" row (uniform) adds the same to every state, i.e.
  nothing, so missing evidence lets the transitions carry the state through.

Every constant comes from football reasoning, or is calibrated to approach 1's own confirmation
rule. None was fitted to the public 5 minutes.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES  # noqa: E402
from src.decode import A, D, H, LETTERS  # noqa: E402

# Params
FPS = GAMES[GAME]["fps"]

# Shortest possession: 1 s. Control means receiving the ball and doing something with it; a first
# touch alone takes 0.3-0.5 s (approach 1, CONFIRM_FRAMES), so anything under a second is a touch,
# a deflection or a block -- which the ground truth's definition says is not a change.
MIN_POSSESSION_FRAMES = 1 * FPS

# Shortest stoppage: 2 s. The quickest restart, a throw-in with the ball at hand, still needs the
# taker to pick it up and set their feet.
MIN_DEAD_FRAMES = 2 * FPS

# Mean durations, from typical top-flight figures, not from this match's labels: of the ~106 graded
# minutes roughly 60 are live, split into ~200 possessions (about 18 s each), and the ~45 dead
# minutes into ~130 stoppages (about 20 s each, stretched by goals, injuries and substitutions).
MEAN_POSSESSION_S = 18.0
MEAN_DEAD_S = 20.0

# Of the possessions that end, the share that end with the ball going dead rather than being won
# by the opponent in play. ~130 stoppages against ~200 possessions puts it near a half.
SHARE_ENDING_DEAD = 0.5

# No single frame is ever certain: probabilities are floored here before taking logs.
PROB_FLOOR = 0.05

# Consecutive frames are not independent observations (the tracker repeats itself at 25 fps), so
# each frame counts for a fraction of one. Calibrated to approach 1: a run of CONFIRM_FRAMES = 10
# frames of opponent control at EVIDENCE_CONFIDENCE = 0.8 should just outweigh the cost of one
# change of possession, as it does in approach 1's own rule. 10 * log(0.8 / 0.1) * w ~ 7 nats
# -> w ~ 1/3.
EMISSION_WEIGHT = 1.0 / 3.0

NEG_INF = -np.inf


def model():
    """(min duration per state, log P(stay) at the end of each chain, log P(s -> s') between states)."""
    k = np.array([MIN_POSSESSION_FRAMES, MIN_POSSESSION_FRAMES, MIN_DEAD_FRAMES])
    mean = np.array([MEAN_POSSESSION_S, MEAN_POSSESSION_S, MEAN_DEAD_S]) * FPS
    leave = 1.0 / (mean - k)  # geometric tail after the chain: total mean = mean
    log_stay = np.log1p(-leave)
    T = np.zeros((3, 3))
    T[H, A] = T[A, H] = 1 - SHARE_ENDING_DEAD
    T[H, D] = T[A, D] = SHARE_ENDING_DEAD
    T[D, H] = T[D, A] = 0.5
    with np.errstate(divide="ignore"):
        log_T = np.log(T * leave[:, None])
    return k, log_stay, log_T


def emissions(probs):
    """Per-frame log-evidence for H/A/D. Uniform and 0,0,0 rows contribute nothing."""
    p = np.asarray(probs, dtype=np.float64)
    s = p.sum(axis=1, keepdims=True)
    p = np.where(s > 0, p / np.where(s > 0, s, 1), 1.0 / 3)
    p = np.maximum(p, PROB_FLOOR)
    p /= p.sum(axis=1, keepdims=True)
    e = EMISSION_WEIGHT * np.log(p)
    return e - e.max(axis=1, keepdims=True)  # shift per frame: changes no decision, keeps scores small


def viterbi(e):
    """Most likely H/A/D sequence for one half, given per-frame log-evidence e (n, 3)."""
    k, log_stay, log_T = model()
    n = len(e)
    # v[s][j]: best log-score of being j frames into state s's chain (the last slot self-loops).
    v = [np.full(k[s], NEG_INF) for s in range(3)]
    for s in range(3):  # no prior: the first half is graded from a live kick-off, the second
        v[s][0] = e[0, s]  # from well before one, so the evidence decides how a half opens
    came_from = np.zeros((n, 3), dtype=np.int8)   # chain entries: previous state
    stayed = np.zeros((n, 3), dtype=bool)         # chain ends: self-loop rather than arriving

    for t in range(1, n):
        last = [v[s][-1] for s in range(3)]
        new = []
        for s in range(3):
            entry = [last[r] + log_T[r, s] for r in range(3)]
            r = int(np.argmax(entry))
            came_from[t, s] = r
            stay, arrive = last[s] + log_stay[s], v[s][-2]
            stayed[t, s] = stay >= arrive
            w = np.empty(k[s])
            w[0] = entry[r]
            w[1:-1] = v[s][:-2]
            w[-1] = max(stay, arrive)
            new.append(w + e[t, s])
        v = new

    # The half can end anywhere, mid-chain included.
    s = int(np.argmax([v[r].max() for r in range(3)]))
    j = int(np.argmax(v[s]))
    path = np.empty(n, dtype=np.int8)
    for t in range(n - 1, 0, -1):
        path[t] = s
        if j == 0:
            s = came_from[t, s]
            j = k[s] - 1
        elif j < k[s] - 1 or not stayed[t, s]:
            j -= 1
    path[0] = s
    return path


def decode(probs, half):
    """Labels H/A/D inside each half, X outside. half: per-frame 0/1/2 from src/load.py."""
    half = np.asarray(half)
    labels = np.full(len(half), "X", dtype="<U1")
    e = emissions(probs)
    for h in (1, 2):
        idx = np.flatnonzero(half == h)
        lo, hi = idx[0], idx[-1] + 1  # each half is one contiguous block
        labels[lo:hi] = LETTERS[viterbi(e[lo:hi])]
    return labels


if __name__ == "__main__":
    from src.approach1 import describe
    from src.load import load_frames
    from src.submission import OUT_DIR, SCORES_DIR, read_scores, validate, write_submission

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("scores", nargs="?", default=os.path.join(SCORES_DIR, "approach1.csv"),
                        help="score file to decode (default: out/scores/approach1.csv)")
    args = parser.parse_args()

    F = load_frames()
    labels = decode(read_scores(args.scores), F["half"])
    name = os.path.splitext(os.path.basename(args.scores))[0]
    path = write_submission(os.path.join(OUT_DIR, "hmm_{}.csv".format(name)), labels)
    print("wrote", os.path.relpath(path, ROOT), "\n")
    describe(F, labels)
    print()
    validate(path)
