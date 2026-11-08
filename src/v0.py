"""v0: the published baseline with its two biggest mistakes fixed. Our safety-net submission.

    python src/v0.py          # -> out/v0.csv, validated

The baseline (starter/baseline.py) gives the ball to the team of the nearest player, calls every
frame without a ball D, and smooths with a centred majority vote. v0 changes only this:

  1. Missing ball is NOT dead. The tracker loses the ball on 41% of graded frames, mostly while
     it is in play. A frame with no ball keeps the last label we had (sticky possession, which is
     also how the ground truth is defined).
  2. D comes from the ball being out of the pitch (load.OUT_MARGIN_M past a line), and is held
     through the missing-ball frames that follow, until the ball is seen in play again.
  3. Frames outside the two halves are X (never graded), and nothing is carried across halftime.

Everything else is the baseline's: the nearest-player rule and the +/-50-frame majority vote.
The window is the published baseline's default, NOT tuned by us -- tuning it on the public window
is exactly what the rules forbid and what baseline.py shows to mislead.

This is deliberately crude. Approach 1 (src/approach1.py) replaces it.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from src.load import load_frames  # noqa: E402
from src.submission import OUT_DIR, validate, write_submission  # noqa: E402

# Params
SMOOTH_HALF_WINDOW = 50   # frames each side; the published baseline's default, see docstring

H, A, D, UNKNOWN = 0, 1, 2, -1
LETTERS = np.array(["H", "A", "D"])


def raw_codes(F):
    """Per-frame evidence: H/A from the nearest player, D if the ball is out, UNKNOWN otherwise."""
    n = len(F["frame"])
    codes = np.full(n, UNKNOWN, dtype=np.int8)
    d = F["d_near"]
    has_player = np.isfinite(d).any(axis=1)
    nearest = np.where(np.fmin(d[:, 0], np.inf) <= np.fmin(d[:, 1], np.inf), H, A)
    codes[has_player] = nearest[has_player]
    codes[F["ball_out"]] = D  # out of the pitch overrides who is nearest
    return codes


def hold(codes):
    """Fill UNKNOWN with the last known code. Before the first observation: D (not yet in play)."""
    idx = np.where(codes != UNKNOWN, np.arange(len(codes)), -1)
    np.maximum.accumulate(idx, out=idx)
    return np.where(idx >= 0, codes[np.maximum(idx, 0)], D).astype(np.int8)


def majority(codes, w):
    """Centred majority vote over +/- w frames among H, A, D. Ties: H, then A, then D."""
    n = len(codes)
    counts = []
    for c in (H, A, D):
        cum = np.concatenate(([0], np.cumsum(codes == c)))
        i = np.arange(n)
        counts.append(cum[np.minimum(n, i + w + 1)] - cum[np.maximum(0, i - w)])
    return np.argmax(np.stack(counts, axis=1), axis=1).astype(np.int8)  # first max wins


def predict(F, w=SMOOTH_HALF_WINDOW):
    labels = np.full(len(F["frame"]), "X", dtype="<U1")
    raw = raw_codes(F)
    for h in (1, 2):
        idx = np.flatnonzero(F["half"] == h)
        lo, hi = idx[0], idx[-1] + 1  # each half is one contiguous block
        labels[lo:hi] = LETTERS[majority(hold(raw[lo:hi]), w)]
    return labels


if __name__ == "__main__":
    F = load_frames()
    labels = predict(F)
    path = write_submission(os.path.join(OUT_DIR, "v0.csv"), labels)
    print("wrote", os.path.relpath(path, ROOT), "\n")
    validate(path)
