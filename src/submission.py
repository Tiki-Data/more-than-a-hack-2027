"""Write a submission CSV and check it with the official validator. Shared by every script that
produces a submission (v0, approach 1, the glue), so the format lives in one place."""

import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_DIR = os.path.join(ROOT, "out")
VALID_LABELS = {"H", "A", "D", "X"}


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
