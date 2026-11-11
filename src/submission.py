"""The two file formats of the pipeline, in one place.

Submission (what we upload):    frame,label           label in H/A/D/X
Score file (what pieces write): frame,H,A,D           probabilities, 0,0,0 = no idea

Every piece writes its score file with write_scores() and the glue reads it with read_scores(), so
a malformed file fails loudly at the piece that wrote it, not as a quietly wrong submission.
Spec: docs/TEAM_PLAN.md, "Score file spec".

    from src.submission import write_scores, read_scores, labels_to_scores
    write_scores("out/scores/approach3.csv", probs)     # probs: (199319, 3) array, H/A/D
    probs = read_scores("out/scores/approach3.csv")    # (199319, 3), rows sum to 1
"""

import os
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, source_frames  # noqa: E402

OUT_DIR = os.path.join(ROOT, "out")
SCORES_DIR = os.path.join(OUT_DIR, "scores")
VALID_LABELS = {"H", "A", "D", "X"}
SCORE_COLUMNS = ["frame", "H", "A", "D"]
N_FRAMES = source_frames("tactical", GAME)

# A row of a score file must sum to 1 within this. Loose enough for 6-decimal rounding (3 values
# each off by up to 5e-7) and for pieces that write 4 decimals; tight enough to catch a piece that
# writes raw scores instead of probabilities.
ROW_SUM_TOLERANCE = 1e-3


def write_submission(path, labels):
    """frame,label CSV, one row per video frame. Refuses anything the scorer would reject."""
    labels = np.asarray(labels)
    bad = set(np.unique(labels)) - VALID_LABELS
    if bad:
        raise ValueError("labels outside H/A/D/X: {}".format(sorted(bad)))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    # newline="\n" so the file is byte-identical on Windows and Linux -- reproducibility.
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, label) for i, label in enumerate(labels))
    return path


def validate(path):
    """Run evaluation/validate_submission.py on the file. Raises if it fails."""
    script = os.path.join(ROOT, "evaluation", "validate_submission.py")
    result = subprocess.run([sys.executable, script, path], capture_output=True, text=True)
    print(result.stdout.rstrip())
    if result.returncode != 0:
        print(result.stderr.rstrip())
        raise RuntimeError("validation failed for " + path)


# ---------------------------------------------------------------------------------------------
# Score files
# ---------------------------------------------------------------------------------------------

def _check_probs(probs, where):
    """Shape (N_FRAMES, 3), finite, in [0, 1], each row summing to 1 or all zero. Raises if not."""
    if probs.shape != (N_FRAMES, 3):
        raise ValueError("{}: expected shape ({}, 3), got {}".format(where, N_FRAMES, probs.shape))
    if not np.isfinite(probs).all():
        bad = np.flatnonzero(~np.isfinite(probs).all(axis=1))
        raise ValueError("{}: NaN/inf on {} frames, first at frame {}".format(
            where, len(bad), bad[0]))
    if probs.min() < 0 or probs.max() > 1 + ROW_SUM_TOLERANCE:
        raise ValueError("{}: values outside [0, 1] (min {:.4g}, max {:.4g})".format(
            where, probs.min(), probs.max()))
    sums = probs.sum(axis=1)
    bad = np.flatnonzero((sums != 0) & (np.abs(sums - 1) > ROW_SUM_TOLERANCE))
    if len(bad):
        raise ValueError("{}: {} rows neither sum to 1 nor are 0,0,0; first at frame {} "
                         "(sum {:.4f})".format(where, len(bad), bad[0], sums[bad[0]]))


def write_scores(path, probs):
    """Write a frame,H,A,D score file. probs: (N_FRAMES, 3) array, columns H, A, D.

    Rows must be probabilities (sum to 1) or 0,0,0 for "no idea". 6 decimals and '\\n' line
    endings, so the same input always gives the same bytes on any machine.
    """
    probs = np.asarray(probs, dtype=np.float64)
    _check_probs(probs, path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    table = pd.DataFrame({"frame": np.arange(N_FRAMES), "H": probs[:, 0], "A": probs[:, 1],
                          "D": probs[:, 2]})
    table.to_csv(path, index=False, float_format="%.6f", lineterminator="\n")
    return path


def read_scores(path):
    """Read a score file into a (N_FRAMES, 3) array of H, A, D probabilities.

    Checks the header, the frame column (0..N_FRAMES-1 in order -- an off-by-one is an error, not
    a quietly shifted file), blanks, and the probability rules. Then:
      - 0,0,0 rows ("no idea") become uniform 1/3, 1/3, 1/3, so they pull the result nowhere;
      - every row is renormalised to sum to exactly 1, absorbing rounding.
    """
    table = pd.read_csv(path)
    columns = [c.strip() for c in table.columns]
    if columns != SCORE_COLUMNS:
        raise ValueError("{}: header must be {}, got {}".format(
            path, ",".join(SCORE_COLUMNS), ",".join(columns)))
    if len(table) != N_FRAMES:
        raise ValueError("{}: expected {} rows, got {}".format(path, N_FRAMES, len(table)))
    if table.isna().any().any():
        rows = np.flatnonzero(table.isna().any(axis=1).to_numpy())
        raise ValueError("{}: blank cells on {} rows, first at row {}".format(
            path, len(rows), rows[0]))
    frames = table.iloc[:, 0].to_numpy()
    if not np.array_equal(frames, np.arange(N_FRAMES)):
        first = int(np.flatnonzero(frames != np.arange(N_FRAMES))[0])
        raise ValueError("{}: frame column must be 0..{} in order; row {} says {}".format(
            path, N_FRAMES - 1, first, frames[first]))

    probs = table.iloc[:, 1:4].to_numpy(dtype=np.float64, copy=True)  # pandas may hand a read-only view
    _check_probs(probs, path)
    sums = probs.sum(axis=1, keepdims=True)
    no_idea = sums[:, 0] == 0
    probs[no_idea] = 1.0 / 3
    sums[no_idea] = 1.0
    return probs / sums


def labels_to_scores(labels):
    """Hard labels -> one-hot score rows. H/A/D become 1,0,0 / 0,1,0 / 0,0,1; X becomes 0,0,0.

    For rule-based pieces that decide a label outright (and for turning a submission back into a
    score file).
    """
    labels = np.asarray(labels)
    bad = set(np.unique(labels)) - VALID_LABELS
    if bad:
        raise ValueError("labels outside H/A/D/X: {}".format(sorted(bad)))
    return np.stack([labels == "H", labels == "A", labels == "D"], axis=1).astype(np.float64)
