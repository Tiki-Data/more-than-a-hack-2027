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

Everything else is the baseline's: the nearest-player rule and the +/-50-frame majority vote, now
done by the shared decoder in src/decode.py (see there for why the window is not tuned).

This is deliberately crude. Approach 1 (src/approach1.py) replaces it.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from src.decode import A, D, H, UNKNOWN, decode  # noqa: E402
from src.load import load_frames  # noqa: E402
from src.submission import OUT_DIR, validate, write_submission  # noqa: E402


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


def predict(F):
    """One-hot evidence from raw_codes (UNKNOWN -> 0,0,0), decoded by the shared decoder."""
    raw = raw_codes(F)
    probs = np.zeros((len(raw), 3))
    known = raw != UNKNOWN
    probs[known, raw[known]] = 1.0
    return decode(probs, F["half"])


if __name__ == "__main__":
    F = load_frames()
    labels = predict(F)
    path = write_submission(os.path.join(OUT_DIR, "v0.csv"), labels)
    print("wrote", os.path.relpath(path, ROOT), "\n")
    validate(path)
