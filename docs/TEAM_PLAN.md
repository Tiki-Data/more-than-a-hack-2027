# Game plan

Due **17 Nov**. Let's aim to submit on the **16th** so we're not panicking.

The approach numbers refer to [APPROACHES_DETAILED.md](APPROACHES_DETAILED.md).

## The one thing to agree on

Everyone works on their own piece, and one script glues them together. So that the glue works,
every piece spits out the same thing:

> **a CSV with `frame, H, A, D`**: one row per video frame (0–199318), a score for how much your
> piece thinks the frame is H, A or D. Leave it at 0, 0, 0 when your piece has no idea.

P3 also produces a list of restarts: `frame, team`.

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

## The only rules that actually matter

- **Don't hand-label anything** on this match.
- **Don't tune numbers on the 5 public minutes.** Use them only to check nothing's broken.
- **Write down why you picked each threshold.** We need it for the write-up.
- **Don't push the match data to the public repo.**
