"""Turn per-frame H/A/D probabilities into labels. The stand-in decoder until P2's HMM lands.

    from src.decode import decode
    labels = decode(probs, F["half"])     # probs (n, 3) H/A/D, half from src/load.py

This is the agreed interface: decode(probs, half) -> labels. P2's HMM replaces this function with
one of the same signature, and nothing that calls it changes.

What it does, per half (nothing is carried across halftime):
  1. Each frame votes for its most likely class. A frame with no single winner (0,0,0 / uniform,
     or a tie at the top) does not vote: "no idea" must never be read as a guess for H.
  2. Frames that do not vote keep the last vote (sticky possession). Before the first vote of a
     half: D, the ball is not yet in play.
  3. Centred majority vote over +/- SMOOTH_HALF_WINDOW frames, to remove flicker.
Frames outside the two halves are X.
"""

import numpy as np

# Params
# +/- 50 frames (2 s) is the published baseline's default window, NOT tuned by us: tuning it on
# the public window is forbidden and starter/baseline.py shows it picks the wrong value anyway.
SMOOTH_HALF_WINDOW = 50

H, A, D, UNKNOWN = 0, 1, 2, -1
LETTERS = np.array(["H", "A", "D"])


def votes(probs):
    """Argmax per frame, or UNKNOWN when the top probability is not strictly above the second."""
    probs = np.asarray(probs, dtype=np.float64)
    top2 = np.sort(probs, axis=1)[:, -2:]
    decided = top2[:, 1] > top2[:, 0]
    return np.where(decided, probs.argmax(axis=1), UNKNOWN).astype(np.int8)


def hold(codes):
    """Fill UNKNOWN with the last known code. Before the first known code: D."""
    idx = np.where(codes != UNKNOWN, np.arange(len(codes)), -1)
    np.maximum.accumulate(idx, out=idx)
    return np.where(idx >= 0, codes[np.maximum(idx, 0)], D).astype(np.int8)


def majority(codes, w):
    """Centred majority vote over +/- w frames among H, A, D. Ties: H, then A, then D."""
    n = len(codes)
    i = np.arange(n)
    counts = []
    for c in (H, A, D):
        cum = np.concatenate(([0], np.cumsum(codes == c)))
        counts.append(cum[np.minimum(n, i + w + 1)] - cum[np.maximum(0, i - w)])
    return np.argmax(np.stack(counts, axis=1), axis=1).astype(np.int8)  # first max wins


def decode(probs, half, w=SMOOTH_HALF_WINDOW):
    """Labels H/A/D inside each half, X outside. half: per-frame 0/1/2 from src/load.py."""
    half = np.asarray(half)
    labels = np.full(len(half), "X", dtype="<U1")
    raw = votes(probs)
    for h in (1, 2):
        idx = np.flatnonzero(half == h)
        lo, hi = idx[0], idx[-1] + 1  # each half is one contiguous block
        labels[lo:hi] = LETTERS[majority(hold(raw[lo:hi]), w)]
    return labels
