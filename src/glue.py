"""The glue: every piece's evidence, combined and decoded into the submission. One command.

    python src/glue.py               # runs the pieces, combines, decodes -> out/submission.csv
    python src/glue.py --no-build    # reuse the score files already in out/scores/

1. BUILD. Runs each piece in PIECES (python src/<piece>.py), which writes out/scores/<piece>.csv.
   Running them here, not trusting whatever is lying in out/scores/, is what makes this one
   command regenerate the same file from the raw data.
2. COMBINE. A weighted product of experts: per frame, sum over pieces of weight * log(p), with p
   floored at hmm.PROB_FLOOR so no single piece can veto a state, then renormalised. A piece's
   "no idea" row is uniform, so it adds nothing.
3. DECODE with the HMM (src/hmm.py). It masks frames outside the two halves to X.
4. WRITE out/submission.csv, validate it, and print its sha256 so two runs can be compared.

Team shape (src/team_shape.py) is not in PIECES yet: it writes to out/evidence/ and gives D = 0 on
every frame, which in a product of experts would rule dead balls out. It goes in once it writes a
score file to the spec in docs/TEAM_PLAN.md.
"""

import argparse
import hashlib
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from src.hmm import PROB_FLOOR, decode  # noqa: E402
from src.load import load_frames  # noqa: E402
from src.submission import OUT_DIR, SCORES_DIR, read_scores, validate, write_submission  # noqa: E402

# Params
# Piece -> weight. Equal weights: each piece already calibrates its own confidence (approach 1 at
# 0.8 per control frame, approach 3 at 0.85 / 0.65 on dead stretches) and the two speak mostly on
# different frames -- approach 1 when a player has the ball, approach 3 when nobody does -- so
# neither needs to be scaled against the other. Not tuned: the public window is too small to tune
# on and is not allowed to be used for it.
PIECES = {
    "approach1": 1.0,
    "approach3": 1.0,
}


def build():
    """Run every piece, so its score file comes from the current code and data."""
    for piece in PIECES:
        print("running src/{}.py".format(piece))
        subprocess.run([sys.executable, os.path.join(HERE, piece + ".py")], check=True,
                       stdout=subprocess.DEVNULL)


def combine():
    """(n, 3) H/A/D probabilities: the weighted product of every piece's score file."""
    log_p = 0
    for piece, weight in PIECES.items():
        p = read_scores(os.path.join(SCORES_DIR, piece + ".csv"))
        log_p = log_p + weight * np.log(np.maximum(p, PROB_FLOOR))
    p = np.exp(log_p - log_p.max(axis=1, keepdims=True))
    return p / p.sum(axis=1, keepdims=True)


if __name__ == "__main__":
    from src.approach1 import describe

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-build", action="store_true",
                        help="skip running the pieces; use the score files already in out/scores/")
    args = parser.parse_args()

    if not args.no_build:
        build()
    F = load_frames()
    labels = decode(combine(), F["half"])
    path = write_submission(os.path.join(OUT_DIR, "submission.csv"), labels)
    print("wrote", os.path.relpath(path, ROOT), "\n")
    describe(F, labels)
    print()
    validate(path)
    with open(path, "rb") as f:
        print("\nsha256", hashlib.sha256(f.read()).hexdigest())
