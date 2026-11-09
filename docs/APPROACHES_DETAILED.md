# Approaches: detailed guide

This is the long version of [APPROACHES.md](APPROACHES.md). Each approach below covers the idea,
why it suits the metric, how to build it with the files we have, how to choose its constants
without breaking the rules, the pitfalls, and the effort involved.

---

## 0. Background that every approach depends on

### What we are predicting

One label per video frame of the tactical recording, 199,319 rows in total:

- `H`: Barcelona in possession, ball alive
- `A`: Athletic in possession, ball alive
- `D`: dead ball
- `X`: abstain

About 158,857 frames are graded.

The ground truth comes mechanically from Metrica's *corrected* tracking. A frame is `D` where that
tracking marks the ball dead. Otherwise the frame gets the team the tracking names as the owner.
Possession is **sticky**: while the ball is loose or contested, it stays with the last team that
had control. It changes only when the other team gains *controlled* possession. A touch, a
deflection, or a header won without follow-up control does not change it.

### How the metric shapes the design

- **`spellF1` ranks the board.** A possession is a maximal run of `H` or of `A`. Our possessions
  are matched one-to-one with the true ones by temporal IoU, and the score averages F1 over IoU
  thresholds 0.3–0.7. Two failures matter most:
  - **Over-segmentation.** A false flicker such as `H H A H H` splits one true possession into
    three predicted ones. That wrecks the IoU of the real one and adds two false positives.
  - **Under-segmentation.** Merging two real possessions loses one of them entirely.
- **Boundary timing.** A one-frame offset costs nothing, 8 frames late gives 0.89, and 1 s late
  gives 0.65. Boundaries need to land within a few frames. Roughly right is not enough.
- **`X` behaviour.** Before matching, the scorer replaces each `X` with the last label we gave in
  that half. Abstaining through an occlusion in the middle of a possession is therefore free and
  safe. Near a hand-over, abstaining drags the old label across it.
- **`macroF1`** is only the tie-break, and it treats `X` as wrong. Ignore it while designing, but
  don't produce absurd output on it.

### Data facts we will rely on

| Fact | Consequence |
|---|---|
| ATD coordinates are normalised 0–1 | Multiply by 105.3 × 68.0 (from `games.py`) before computing distances in metres |
| ATD `frame_id` is `video_frame + 1` | Use `starter/load_atd.py` → `video_frame_of()` and never index by hand |
| 46 track slots, each with a fixed team (24 Barcelona, 22 Athletic in the metadata) | We get team membership for free. Slots are not people, and identities switch |
| The ball is missing on 41% of graded frames | Every component must cope without a ball |
| Halves in video frames: 1st half 11185–90651, 2nd half 110808–190197 | Treat each half as its own sequence. Everything outside them is ungraded, so write `X` |
| Barcelona attack left→right in the 1st half | Directions flip at half-time. Needed for "moving toward goal" features |
| Smart tagging: 638 automatic phases and set pieces, times in seconds (`frame = round(s*25)`) | Useful evidence with known errors: a phantom Athletic goal at about 31' plus its kick-off at about 34', the penalty tagged as a free kick, and every throw-in with `Team = N/A` |
| Tactical homography is valid on every frame | We can project the ball and players into the video at any time |

### What the baseline does, and why it is weak

`starter/baseline.py` takes the nearest tracked player to the ball as the owner, labels a frame
`D` whenever the ball is missing, and then applies a centred majority vote. Its weaknesses:

- "Ball missing" is not the same as "dead".
- There is no idea of control: the nearest player is not necessarily in control.
- A majority vote smears hand-overs instead of placing them precisely.

Every approach below addresses at least one of these.

---

## 1. Better rule-based baseline

### Idea
Keep the baseline's intuition that the player nearest the ball probably has it, but add the
football rules the ground truth follows: control has to be established before possession changes,
and possession is sticky while the ball is loose.

### How to build it
1. **Per frame, find the candidate controller.** Take the nearest player to the ball, in metres.
   Count them as a candidate only if they are within a control radius *r*. Otherwise the frame
   is "loose".
2. **Confirm control before switching.** Switch possession to team T only when one of:
   - T has a candidate for at least *N* consecutive frames (the player keeps the ball), or
   - T has two touches in a row by different T players (a completed pass).

   Then **backdate the switch to the first frame of that confirming run**. This keeps the boundary
   accurate even though we only decide after the fact.
3. **Hold through loose frames.** Loose frames keep the current owner.
4. **Handle missing ball frames:**
   - Short gaps, under about 0.5 s: interpolate the ball linearly and treat it as observed.
   - Long gaps: keep the current owner, or write `X`. Under `spellF1` the two behave the same,
     but `X` is more honest.
   - Do **not** label a frame `D` just because the ball is missing. That is the baseline's
     biggest mistake.
5. **Basic `D` detection** for now: ball outside the touchlines or goal lines by more than a
   small margin. Approach 3 replaces this.

### Choosing constants without tuning on the public window
- **Control radius *r*:** roughly 1–2 m, given player reach and tracking noise. Use the
  tracking's own noise level, the jitter of a stationary player, to set the lower end.
- ***N* frames:** "keeps it for a few seconds" is the rules' own suggested proxy, but a clean
  first touch can take about 0.3–0.5 s. Start with 8–12 frames.
- **Write the reasoning down.** The technical document has to explain each constant.

### Pitfalls
- **Tracking duplicates and identity switches** can create a phantom "candidate" from the wrong
  team. Requiring several frames in a row protects against this.
- **Contested duels** where both teams are within *r*: keep the current owner.
- **Ball in the air.** The ball's 2D position passes over players who are not touching it.
  Speed helps: a ball moving faster than about 15 m/s is in flight, not controlled.

### Effort and payoff
A few hours. Fixing flicker and the "missing means `D`" error should beat the 0.24 floor by a
large margin.

---

## 2. HMM or semi-Markov model with Viterbi decoding

### Idea
Treat the match as a hidden sequence of states `H`, `A` and `D`, where each frame gives noisy
evidence about the state. Viterbi decoding finds the single most likely state sequence given all
the evidence and the transition costs. The model takes the whole match into account at once,
rather than relying on hand-written switching logic.

### Why it suits the metric
- **Transition penalties control segment count.** `spellF1` mostly measures whether we produce
  the right number of possessions with the right boundaries. A penalty on each `H↔A` switch
  directly sets how readily the model starts a new possession.
- **Decoding is global.** Viterbi can place a boundary at the best-supported frame inside an
  ambiguous stretch, so boundaries land more precisely than with a sliding majority vote.
- **Missing evidence is handled naturally.** A frame with no ball simply gives a flat (or weaker)
  likelihood, and the transition model carries the state through.

### How to build it
1. **States:** `H`, `A`, `D`. Run each half separately.
2. **Emissions,** the per-frame probability of what we observe under each state, built from
   approaches 1, 3 and 4 or from a learned model (approach 6). Examples:
   - The ball is close to a Barcelona player → evidence for `H`.
   - The ball is out of bounds and players are slow → evidence for `D`.
   - No ball → weak evidence from team shape only.
3. **Transitions:**
   - `H→A` and `A→H`: rare, a change of possession.
   - `H→D` and `A→D`: the ball goes dead.
   - `D→H` and `D→A`: a restart. These can be forced by a set-piece tag together with which team
     takes it.
4. **Duration priors (semi-Markov version):** a plain HMM implies geometric durations, which
   allow very short possessions. A semi-Markov model, or a chain of sub-states that enforces a
   minimum duration, says "a possession lasts at least about 1 s" and "a stoppage lasts at least
   about 2 s". That removes the micro-possessions that ruin `spellF1`.
5. **Decode with Viterbi.** Optionally use forward–backward posteriors to write `X` where the
   model is unsure, but only in the middle of a possession.

### Choosing constants
- **Durations and transition rates:** take them from football statistics or other matches. For
  example, a typical La Liga match has roughly 150–250 possessions under this kind of definition.
  Check this against a definition close to ours.
- **Don't fit them to the 17 possessions in the public window.** That is forbidden, and the
  baseline sweep shows it misleads anyway.
- **Use the public window for sanity checks:** our possession count over those 5 minutes should be
  in a plausible range, and the boundaries should look right.

### Pitfalls
- **Over-confident emissions** overpower the transition penalties and bring flicker back. Keep
  per-frame probabilities modest, never 0.0 or 1.0.
- **A hand-over has to be typed.** `H→A` directly and `H→D→A` are different stories. Make sure
  the model can produce both.

### Effort and payoff
About a day for a working version, with numpy only (Viterbi over 3 states is roughly 20 lines).
This is the backbone of the recommended pipeline.

---

## 3. Dead-ball detection

### Idea
`D` is the class the baseline handles worst. Dead-ball stretches have strong, recognisable
signatures, and the smart tagging gives us the moments when play restarts.

### Signals
| Signal | How | Notes |
|---|---|---|
| Ball out of bounds | Ball outside `[0,1]` beyond a small margin, or projected outside the pitch with the homography | Strongest single signal when the ball is tracked |
| Players' overall speed collapses | Mean or median player speed (the ATD gives speed per slot) falls below walking pace for a while | Play stopping shows up quickly. Can be confused with slow build-up in our own half |
| Set-piece shape | Players lining up for a corner or free kick, a wall, players gathering for a throw-in near the touchline | Detect with simple geometry around the ball's last position |
| Set-piece tags | Throw-ins, corners, goal kicks, free kicks and kick-offs are each a restart, so the **end of a `D` stretch** | Use as soft anchors. Drop the phantom goal and kick-off (about 31' and 34'). The penalty is tagged as a free kick |
| Goals | A goal always produces a long `D` stretch up to the kick-off | The 3 real goals are at frames ~45947, ~73710 and ~156757 (from the tags) |
| Ball missing for a long time **together with** low player speed | Combined evidence | Missing alone is not dead |
| Possibly the referee's whistle in the video audio | Detect a whistle from the audio track, if the tactical mp4 has one | Not verified. Check whether the file has usable audio |

### Who restarts
After a stoppage, the team taking the restart is the possessor. Ways to find it:
- **From the tag, when it has a team.** Throw-ins are always `N/A`.
- **From geometry:** the player nearest the ball at the restart frame, or the side the ball went
  out on (the team that did *not* touch it last gets the throw-in).
- **From the tracking:** the team the tracking associates with the ball just after the restart.

### Pitfalls
- **Tags are automatic.** Treat them as evidence with a confidence, not as truth.
- **Offsides, fouls and injuries** stop play without the ball going out. The player-speed signals
  are what catch these.
- **Tag times are phase windows,** so the exact restart frame must come from the tracking (the
  ball starting to move).

### Effort and payoff
Medium. Fixing `D` improves both metrics, and accurate stoppages also give clean boundaries
between possessions.

---

## 4. Team-shape features for frames without a ball

### Idea
When the ball is missing, which happens on 41% of graded frames and disproportionately in key
moments, the 22 players still give possession away through how they are arranged and how they
move.

### Features, computed per frame for each team
- **Spread:** convex-hull area, width and length. The attacking team stretches out; the defending
  team stays compact.
- **Centroid movement:** speed toward the opponent's goal (adjusted for half-time direction). The
  team in possession tends to advance, and the defending team retreats or presses.
- **Pressing:** how many opponents are within a few metres of the nearest player. Pressure points
  at the player who probably has the ball.
- **Defensive line height,** and **rank of each team's last defender.**
- **Convergence:** where several players' velocity vectors meet tends to be where the ball is.
  This can also give a rough estimate of the ball's position.

### How to use them
- As **extra emission evidence** in the HMM (approach 2), weighted lower than direct ball evidence.
- As **inputs to a learned model** (approach 6) trained on other matches, where both the ball and
  the labels are available.

### Choosing weights
Hand-set them from football reasoning, or learn them on other matches. Do not learn them on the
public window.

### Pitfalls
- **Shape changes slowly,** so it tells us *who* has the ball, but it cannot pin down *when*
  possession changed. Use it to hold the state and to break ties, not to place boundaries.
- **Tracking gaps** change hull areas artificially. Normalise by how many players are visible, or
  use features that tolerate missing players (medians, for example).

### Effort and payoff
Medium. Most valuable during long ball gaps, where otherwise we can only hold or guess.

---

## 5. Improve the ball track ourselves

### Idea
The ball is the strongest signal and the least reliable one. Recovering more ball positions from
the video raises the quality of everything downstream.

### How to build it
1. **Detect the ball** in tactical-video frames with a model trained on public data, for example
   a YOLO-type detector trained on SoccerNet or another public football dataset. Do *not* train
   it on hand labels of this match.
2. **Project detections to pitch coordinates** with the tactical homography
   (`starter/load_homography.py` → `unproject()`). This assumes the ball is on the ground; a
   ball in the air will be off.
3. **Combine with the ATD ball** and smooth with a Kalman filter or physics model: straight,
   slowing roll on the ground, a parabola in the air.
4. **Process only frames where the ATD ball is missing,** so compute stays manageable.

### Pitfalls
- **Compute:** about 160k graded frames of 1920×1080 video. Cut the cost with a crop around where
  the ball is expected, or by detecting only inside gaps.
- **False detections:** heads, line markings, the second ball beside the pitch. Reject
  detections that don't fit the motion model.
- **The videos are not in our `data/` folder yet.** Download them first.
- **Reproducibility:** the repository must regenerate the submission exactly, so pin model
  weights and versions and make inference deterministic.

### Effort and payoff
High effort, possibly the largest single gain. Best attempted once approaches 1–3 are in place.

---

## 6. A learned model trained on other matches

### Idea
Learn the mapping from tracking features to `H` / `A` / `D` on matches where labels exist, then
apply it to this match. Use the model's probabilities as HMM emissions.

### Training data options (all on other matches, which the rules allow)
- **Metrica Sports open sample data:** a few anonymised matches with tracking (ball included) and
  event data. It comes from the **same provider** as our data, which lowers the mismatch. Derive
  possession labels from the events (passes, receptions, ball-out events).
- **SkillCorner open data,** and other public tracking with event data. Check what each includes
  and its licence before using it.
- **Our own hand labels on other matches** are allowed in any amount. The technical document
  must record what was labelled, by whom and how much.

### Model choices
- **Gradient-boosted trees** on per-frame features (ball–player distances, ball speed, approach
  4's team-shape features, player speeds). Fast, easy to explain, and good with missing values.
- **A small sequence model** (GRU, temporal CNN or transformer) over windows of a few seconds.
  It captures dynamics such as passes and pressing.
- **Always decode with the HMM afterwards** (approach 2). Raw per-frame argmax flickers.

### Pitfalls
- **Definition mismatch.** Possession built from other datasets' events will not match "the
  corrected tracking's owner" exactly. Align the definitions: sticky possession, controlled
  hand-overs, and dead-ball conventions.
- **Domain shift.** Other matches' data may be corrected or cleaner. Inject our noise when
  training: drop the ball on about 40% of frames, add jitter, fragment tracks.
- **Validation** has to happen on other matches as well. Hold out a full match from the training
  data.

### Effort and payoff
High. It pays off most when combined with approach 2, and it gives the technical document a solid
story.

---

## 7. Self-training on this match

### Idea
Some frames are easy: the ball clearly at one player's feet, nobody else near, or a confirmed
restart. Label those automatically, train a model on them, and let it fill the hard frames.

### How to build it
1. **Generate high-confidence pseudo-labels with a rule,** for example a ball within 1 m of one
   team's player for at least 1 s with no opponent within 3 m.
2. **Train a classifier on features** (approaches 3 and 4) from those frames.
3. **Predict on the uncertain frames,** then decode with the HMM.
4. **Optionally repeat** a few rounds.

### Rules check
Allowed **only** if no person decides any label. The rule that selects confident frames must be
written in code, and not adjusted by looking at specific frames.

### Pitfalls
- **Confirmation bias.** Easy frames are not representative of hard ones: easy frames are mostly
  in the middle of a possession, hard frames are at hand-overs. The model may do worst exactly
  where it matters.
- **It can only be as good as the pseudo-label rule.**

### Effort and payoff
Medium. A sensible addition if approach 6 is not available.

---

## Other ideas worth noting

- **The broadcast video** is not aligned, and only 66% of its frames have a valid homography. It is
  probably not worth it for Phase 1. It could help recover the ball in occlusions if aligned with
  audio or visual cross-correlation, but that is a project in itself.
- **Smart-tagging phases** (Build Up, Progression and so on) each carry a team. As **weak**
  evidence for that team's possession during the phase they are useful. Copied across as labels
  they are not: phases are not possessions.
- **Writing `X` deliberately:** in the middle of a long possession with no ball, abstaining is
  free under `spellF1` and protects against invented hand-overs.

---

## Recommended pipeline

```
ATD + homography + tagging
        │
        ├─ ball cleaning (interpolate short gaps; later approach 5)
        ├─ per-frame features (ball–player distances, speeds, team shape, out of bounds)
        ├─ restart anchors from set-piece tags (filtered)
        │
        ▼
per-frame evidence for H / A / D   (hand-weighted rules first, approach 6 model later)
        │
        ▼
semi-Markov decoding per half (duration priors, restart constraints)
        │
        ▼
labels for graded halves, X elsewhere  →  validate_submission.py  →  submission.csv
```

### Order of work
1. **Approach 1:** get a valid submission and check it with `evaluation/validate_submission.py`.
2. **Approach 3:** dead-ball detection.
3. **Approach 2:** HMM or semi-Markov decoding over approaches 1 and 3's evidence.
4. **Approach 4:** team-shape evidence.
5. **Approach 5 or 6,** depending on the team's skills and time.

### What to record as we go, for the technical document
- **Every constant and where it came from:** football reasoning, a cited statistic, or a value
  measured on another match.
- **Every dataset used,** with its source, who labelled it and how much.
- **Approaches tried and dropped,** and why.
- **Sanity checks against the public window:** used only to confirm alignment and that the output
  looks plausible, never to tune.
