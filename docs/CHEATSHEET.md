# Cheatsheet

Run everything from the repo root. Data goes in `data/tactical/` (see [data/README.md](../data/README.md)).

## Commands

### Setup
| Command | What it does |
|---|---|
| `pip install -r requirements.txt` | numpy + pandas: all our pipeline and the scorer need |
| `pip install -r requirements-starter.txt` | extras for the starter notebooks (kloppy, matplotlib, ...) |

### Our pipeline
| Command | What it does |
|---|---|
| `python src/load.py` | builds the shared frame table (~3 s), caches it, prints a sanity summary |
| `python src/load.py --force` | rebuilds the cache even if it looks fresh |
| `python src/v0.py` | writes and validates the safety-net submission `out/v0.csv` |
| `python src/approach1.py` | writes `out/scores/approach1.csv` (evidence) and `out/approach1.csv` (submission) |
| `python src/approach1.py --stats` | prints the tracking measurements behind approach 1's constants |

### Checking and scoring
| Command | What it does |
|---|---|
| `python evaluation/validate_submission.py out/X.csv` | the official format check, plus label counts and possession count |
| `cd evaluation && python run_evaluation.py --results ../out --no-detail` | scores every CSV in `out/` on the public 5 minutes |
| `... --bootstrap 1000` | adds the noise band: on 5 minutes, gaps under ~0.3 spellF1 mean nothing |
| `sha256sum out/X.csv` | fingerprint: run twice, the hashes must match (reproducibility) |

**The public 5 minutes are for checking nothing is broken. Never tune on them.**

### In Python
```python
from src.load import load_frames, frame_table, load_tags
F = load_frames()                  # dict of numpy arrays, one row per video frame, metres
df = frame_table()                 # same, scalar columns only, as pandas
tags = load_tags()                 # Smart Tagging in video frames, raw (not cleaned)

from src.submission import write_scores, read_scores, labels_to_scores, write_submission, validate
write_scores("out/scores/<piece>.csv", probs)   # probs: (199319, 3) H/A/D, rows sum to 1 or 0,0,0
probs = read_scores("out/scores/<piece>.csv")    # checked, 0,0,0 -> uniform, renormalised

from src.decode import decode
labels = decode(probs, F["half"])  # stand-in decoder until P2's HMM; same signature
```

## Files

### Our code (`src/`)
| File | One sentence |
|---|---|
| [load.py](../src/load.py) | Parses the raw tracking once into a cached per-frame table (ball, players, half, nearest player per team) in metres, indexed by video frame. |
| [submission.py](../src/submission.py) | The two file formats: writes/validates submissions (`frame,label`) and writes/reads score files (`frame,H,A,D`). |
| [decode.py](../src/decode.py) | Turns per-frame H/A/D probabilities into labels (vote, hold, majority smoothing); P2's HMM replaces it. |
| [v0.py](../src/v0.py) | The safety-net submission: the baseline with "missing ball = keep the owner" and "ball out = D". |
| [approach1.py](../src/approach1.py) | Rule-based possession: control radius, confirm before switching, backdate the switch. |

### Our docs (`docs/`)
| File | One sentence |
|---|---|
| [TEAM_PLAN.md](TEAM_PLAN.md) | Who does what, the score file spec, the timeline and the rules. |
| [APPROACHES_DETAILED.md](APPROACHES_DETAILED.md) | The long explanation of each approach, its constants and pitfalls. |
| [CHEATSHEET.md](CHEATSHEET.md) | This page. |

### Outputs (gitignored)
| Path | One sentence |
|---|---|
| `data/derived/BAR-ATH_frames_v2.npz` | The cached frame table; delete it or use `--force` to rebuild. |
| `out/scores/<piece>.csv` | Each piece's per-frame H/A/D evidence, for the glue. |
| `out/<name>.csv` | Submissions: `v0.csv`, `approach1.csv`, later `submission.csv`. |

### From the organisers
| File | One sentence |
|---|---|
| [README.md](../README.md) | Overview of the challenge and the repository. |
| [CHALLENGE.md](../CHALLENGE.md) | The task: label every frame H, A, D or X. |
| [DATA.md](../DATA.md) | What each data file contains and its quirks (ball missing 41%, 46 track slots, ...). |
| [EVALUATION.md](../EVALUATION.md) | How spellF1 and macroF1 work and why spellF1 ranks the board. |
| [SUBMISSION.md](../SUBMISSION.md) | The CSV format and the three deliverables (file, public repo, technical document). |
| [RULES.md](../RULES.md) | What is allowed: no hand labels on this match, no tuning on the public window. |
| [FAQ.md](../FAQ.md) | Answers to common questions. |
| [OVERVIEW.md](../OVERVIEW.md) | The competition as a whole: what it is, why the format changed, and what participants receive. |
| [games.py](../games.py) | Per-match constants (frame count, halves, pitch size) and data file paths. |
| [evaluation/validate_submission.py](../evaluation/validate_submission.py) | The official format check. |
| [evaluation/run_evaluation.py](../evaluation/run_evaluation.py) | Scores submissions against the ground truth and prints a ranked table. |
| [evaluation/possession_metrics.py](../evaluation/possession_metrics.py) | The metric code itself (spellF1, macroF1, transition timing). |
| [evaluation/gt/](../evaluation/gt/) | The public 5-minute ground truth (frames 11185–18684); everything else is X. |
| [evaluation/results/](../evaluation/results/) | The four published baseline submissions. |
| [starter/baseline.py](../starter/baseline.py) | The reference baseline: nearest player has the ball, smoothed. |
| [starter/load_atd.py](../starter/load_atd.py) | Loads the tracking with kloppy and checks `video_frame = frame_id - 1`. |
| [starter/load_smart_tagging.py](../starter/load_smart_tagging.py) | Reads the Smart Tagging XML and converts seconds to video frames. |
| [starter/load_homography.py](../starter/load_homography.py) | Reads the per-frame pitch-to-video matrices, for projecting into the video. |
| [starter/make_dummy_submission.py](../starter/make_dummy_submission.py) | Writes three trivial valid submissions (all D, all H, all X), each scoring spellF1 = 0. |
| [starter/utils.py](../starter/utils.py) | Plotting helpers (draw a pitch, plot a frame) for the notebooks. |
| `starter/01–03_*.ipynb` | Walkthroughs: the tagging, the tracking, and a first submission. |

## Facts worth remembering
- **Indexing:** row i of everything = video frame i. ATD `frame_id` = video frame + 1, and `load.py` already shifts it.
- **Graded frames:** 11185–90651 and 110808–190197 (158,857). The 2nd-half kick-off is at 113742, so about 2 minutes before it are graded.
- **Player speed is m/s**, despite the XML saying km/h. **The ATD's ball speed field is junk**, so use `ball_speed` from `load.py`.
- **Ball gaps are long:** median 10–12 s, and gaps of 0.5 s or less are only 0.1% of missing frames. Interpolating short gaps barely helps.
- **Never** commit `data/`, hand-label this match, or tune on the public 5 minutes.
