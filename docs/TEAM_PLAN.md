# Game plan

Due **17 Nov**. Let's aim to submit on the **16th** so we're not panicking.

The approach numbers refer to [APPROACHES_DETAILED.md](APPROACHES_DETAILED.md).

## The one thing to agree on

Everyone works on their own piece, and one script glues them together. So that the glue works,
every piece spits out the same thing:

> **a CSV with `frame,H,A,D`**: one row per video frame, the probability that your piece
> thinks the frame is H, A or D. Write `0,0,0` when your piece has no idea.

P3 also produces a list of restarts: `frame, team`.

### Score file spec

```
frame,H,A,D
0,0.000000,0.000000,0.000000
1,0.620000,0.300000,0.080000
```

- **Rows:** exactly 199,319, one per video frame, `frame` = 0..199318 in order. Row i is frame i
  everywhere (same indexing as `src/load.py` and the submission).
- **Values:** floats in [0, 1], finite, no blanks, 6 decimals. Each row should sum to 1 (the
  reader renormalises, so small rounding is fine).
- **No idea:** write `0,0,0`. The reader treats it as uniform (1/3, 1/3, 1/3), i.e. "no
  information", so an abstaining piece never pulls the result anywhere.
- **Score every frame**, including halftime and pre/post match. The glue masks frames outside the
  two halves to `X` using `half` from `src/load.py`, so you don't need to.
- **Hard rules are fine:** a rule-based piece just writes one-hot rows (`1,0,0`).
- **Use the helpers**, don't hand-roll the CSV: `write_scores(path, probs)` and
  `read_scores(path)` in `src/submission.py`. They check shape, range, NaN and frame order.
- **Name it** `out/scores/<piece>.csv` (e.g. `approach1.csv`, `approach3.csv`).

**How the glue combines them:** a weighted product of experts, i.e. a weighted sum of
`log(p + eps)` per piece, renormalised per frame, then decoded (HMM when P2 has it, otherwise
argmax plus majority vote). Uniform rows add nothing, so abstaining is free. The weight of each
piece and the reason for it goes in the write-up.

That's it. Do the rest however you like.

## Who does what

- **P1: the glue, plus the quick win.**
  - Load the data into one table everyone can use.
  - Build approach 1: the baseline, but smarter.
  - Write the script that combines everyone's CSVs into a submission.
  - Get a v0 submission out in the first couple of days so we always have something.
- **P2: the brain.**
  - Build approach 2: the HMM that turns everyone's scores into clean possessions.
  - Can start straight away on fake data, without waiting for anyone.
  - Also pulls the write-up together at the end.
- **P3: dead balls.**
  - Build approach 3: ball out of play, players standing around.
  - Clean up the set-piece tags (bin the fake 31' goal) and find who takes each restart.
- **P4: when the ball's missing.**
  - Fill the short ball gaps.
  - Build approach 4: work out who has it from how the teams are shaped.
- **P5: the fancy stuff.**
  - First, get some *other* match data (Metrica's open sample games) so we have something to
    measure against. We can't tune on the 5 public minutes.
  - Then either train a model on it (approach 6) or build our own ball detector (approach 5).
    Whichever sounds more fun.

## Rough timeline

| When | What |
|---|---|
| **5–6 Nov** | Quick call to agree the CSV format. P1 gets v0 out. Everyone starts on their piece |
| **7–9 Nov** | First versions of everyone's pieces, glued together |
| **10 Nov** | Submit whatever we have, as a safety net |
| **11–14 Nov** | Make it better |
| **15 Nov** | Stop adding stuff. Check the code regenerates the exact same file |
| **16 Nov** | Finish the write-up, submit |

## Status and next steps (as of 11 Nov)

**Done (P1):** shared frame table (`src/load.py`), submission writer and validator
(`src/submission.py`), v0 safety-net submission (`src/v0.py` -> `out/v0.csv`).

**Not done:** the score-file helpers, approach 1, the glue, and (as far as the repo shows) pieces
P2–P5.

Order matters: 1 unblocks everyone, 2 and 3 are the P1 critical path.

1. **P1, today: add `write_scores` / `read_scores` to `src/submission.py`** per the spec above, plus
   a `labels_to_scores` helper (hard labels -> one-hot rows) and a shared decoder. Tell everyone
   when it is pushed.
2. **P1: submit `out/v0.csv` if it hasn't been submitted yet.** The plan said 10 Nov for the safety
   net, so this is already late. It is validated and ready.
3. **P1: approach 1 in `src/approach1.py`.** Split it into an evidence step that outputs scores and
   a smoothing step. Add ball speed (`ball_speed` in `load.py`) to tell a ball in flight from one
   at someone's feet. Pick the constants without tuning on the public minutes, and write the reason
   for each one in a comment.
4. **P1: the glue, `src/glue.py`.** Read every file in `out/scores/`, combine as above, mask to
   `X` outside the halves, write `out/submission.csv`, validate. One command, deterministic.
5. **Everyone, by 12 Nov: a first `out/scores/<piece>.csv`**, however rough, so the glue has
   something real to combine. A piece with no result yet writes all zeros.
   - **P2:** HMM on fake scores first, then swap in real ones. Owns the decoder.
   - **P3:** clean the set-piece tags (drop the fake 31' goal), dead-ball scores, restart list.
   - **P4:** fill short ball gaps, team-shape scores for frames without a ball.
   - **P5:** the other-match data (Metrica) and a way to measure against it, then approach 5 or 6.
6. **12–14 Nov: improve.** Use P5's other-match data to choose the weights, not the 5 public minutes.
7. **15 Nov: freeze.** One command regenerates the byte-identical submission from raw data. Add a
   `.gitignore` for `data/`, `out/` and `.venv/` before this.
8. **16 Nov: write-up and submit.** P2 pulls it together. Each owner supplies their thresholds and
   the reason for them.

## The only rules that actually matter

- **Don't hand-label anything** on this match.
- **Don't tune numbers on the 5 public minutes.** Use them only to check nothing's broken.
- **Write down why you picked each threshold.** We need it for the write-up.
- **Don't push the match data to the public repo.**
