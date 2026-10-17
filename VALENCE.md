# Valence: giving the write a sign

Design note, 2026-09-21; **built and run 2026-09-22 - see the RESULT section at the bottom
before acting on anything above it.** It exists because the `world` run (`results/world/`)
failed in a way that looked structural rather than incidental.

**Short version of the outcome: the central claim is FALSIFIED.** A signed write does not
prevent the constant collapse. The v1 runs appeared to show that it did, but they were
damaged - an unparseable-output penalty acted as a runaway and pushed the model off the
valid-move manifold, and the "conditioning" measured there was wreckage, not a disposition.
With the runaway fixed (v2) and the training signal intact for the full run, the signed arm
collapses to a single move exactly like the unsigned one, in both seeds, TVD 0.00.

Everything below the RESULT heading at the bottom still describes the design accurately. The
ARGUMENT for it - that an unsigned integrator must collapse because it can only add - is the
part that did not survive: signing makes the process non-monotone and it collapses anyway. Do
not build on the reasoning in this document without reading that section first.

## The gap

The consolidation loop has two outcomes:

    verified      ->  write   (dW += eta * post (x) x)
    not verified  ->  skip, or teach the correct answer

**There is no "bad".** A move that scored just under the bar and a move that was catastrophic
are handled identically: discarded. And `teach` is not the missing negative - it needs an
oracle that already knows the right answer, which is the regime where a template is cheaper
than a fast weight.

So the system has *good* and *neutral*, and neutral is doing double duty as the bin for
everything that is not a clear win.

## Why an unsigned integrator must collapse

`dW_t = (1-lambda) dW_{t-1} + eta * (post_t (x) x_t)` only ever makes an output more likely.
Nothing in the consolidation path ever makes anything less likely. Over a stream, every
candidate action is reinforced at roughly its own success rate, and nothing is ever
suppressed - so the loop does not select the *best* action, it amplifies the **most frequent
action that is sometimes right**. That is the model's prior, not a policy.

This is not a hypothesis about the `world` run; it is what the run did.

    disposition   compiled   best constant   compiled emitted
    trader          0.44         0.50        REFUSE x48
    diplomat        0.15         0.70        REFUSE x48
    hardliner       0.52         0.58        REFUSE x48

One move across 48 situations and all three dispositions. And it was not a failure to learn:
diplomat held **0.81** for twelve consecutive probes - above every constant policy measured
(max 0.71) - before dropping to 0.15 at t=652 and freezing. The loop found above-floor
conditional behaviour and then accumulated over the top of it.

A monotone process in a bounded store has exactly one stable outcome: whatever is written
most often eventually occupies everything. Valence is what makes the process non-monotone.

## Three categories

An organism's read on an outcome is coarse even when its causes are not: good, bad, or
neither. The operationalisation is free, because the engine already computes it and throws it
away - `evaluate()` scores eight sampled alternatives to set the bar, then discards them.

    advantage = score(action) - mean(score(sampled alternatives))

    advantage >  eps   GOOD      reinforce, magnitude scaled by advantage
    advantage < -eps   BAD       damage, magnitude scaled by advantage
    |advantage| <= eps NEUTRAL   write nothing; let decay handle it

`eps` is a real design knob, not a formality: it sets how decisive an outcome must be before
the system spends capacity on it. Capacity is the scarce resource (~0.45 % of parameters per
sustained skill, [[hdelta-capacity-limit]]), so "this taught me nothing" should cost zero.

**Implemented as of 2026-09-21, after one wrong turn worth recording.**

The first build normalised each chunk by its own median and forced `sum(|w|) = 1`. That gave
exact control of the effective learning rate - valence redistributing a fixed amount of
plasticity rather than adding more - which seemed like the right control for the first
comparison. It was not, because it made an item's write depend on its accidental chunk-mates:
over 2000 draws, an identical item (advantage +5.0) received weights spanning **38.5x**
depending only on which three puzzles shared its batch of four. A batching artifact landing
directly in the learned policy, and far larger than the learning-rate difference the
renormalisation was protecting against.

The build now normalises against a **running EMA of |advantage|, global to the run**, and
divides by chunk size to keep magnitude comparable to the unsigned case. Same measurement:
**1.33x**, and smaller in practice since a slow EMA moves ~1 % per chunk.

Neutrality then becomes absolute for free, instead of needing an untuned parameter:

    advantages                        chunk mass written
    [0.1, -0.05, 0.02, -0.01]              0.011      nothing happened -> nothing written
    [5.0, -3.0,  0.2, -130.0]              1.014      mixed -> normal
    [9.0,  8.0,  7.0,   10.0]              1.928      decisive -> writes more

Per-chunk mass is now logged (`adv_chunk_mass`) rather than enforced, so the effective-LR
question is answerable after the fact instead of being assumed away.

**Neutral is already implemented, by accident.** Passive decay is exactly the "spend nothing
and let it fade" channel. The change is not to build it but to route to it deliberately,
instead of dumping every non-win into it.

## The sign turns a running total into a running balance

With a signed write the same loop can be written *and unwritten*. Forgetting stops being
damage limitation and becomes half the arithmetic. This is the property the collapse needed
and did not have: REFUSE would have been pushed back down on the situations where it is
wrong, and the conditional structure that existed at t=600 would have had something holding
it in place.

## Valence-segregated groups, and asymmetric decay

Only `hdeltaH` (attention) and `hdeltaG` (MLP) were ever implemented. `hdeltaA` and `hdeltaI`
never existed, and the anatomical labels are recorded as untested decoration
([[hdelta-biological-analogy]]). Valence supplies a reason to have separate groups that is
*functional* rather than anatomical:

    route GOOD writes and BAD writes to different groups, with different decay constants.

The argument for asymmetry is that the costs are not symmetric. Forgetting which berry was
poisonous is not the same error as forgetting which one was pleasant, and an organism that
decays both at the same rate is miscalibrated for any world where mistakes are expensive.

`GroupConfig` already carries per-group `eta`, `decay`, `budget` and `rem_eta`, so this is
configuration rather than new machinery.

## What this gives REM

REM currently feeds **Gaussian noise** through the model and shrinks whatever responds to it.
That is a specificity filter - "do you fire when there is nothing there?" - not a training
ground. It has no content, replays nothing, and has no notion of correct or incorrect.

Valence gives offline replay an **addressable target**: replay from the negative store. The
store that decays slowest is the one that keeps being re-tested, and it keeps being re-tested
until replaying it stops producing a negative result.

Two rules matter more than the selection heuristic:

**Replay must RE-TEST, never RE-APPLY.** Regenerate the action with current weights, score it
now, and write on that score. If replay instead re-applies the stored outcome, then replaying
whatever is dominant deepens the rut - and under re-testing it does the opposite, because the
dominant habit is the one being applied everywhere and therefore the one whose errors are
worth finding. Frequency of replay then decides *where you look*, not which way you move.

**The objective is global, not per-item.** "Get this input right" is local. Integration means
getting it right *without distorting responses to everything else* - which matters precisely
because capacity is bounded. The test is available now: replay an item, then re-score the whole
probe set. X improves and Y, Z degrade -> overwriting. X improves and the rest holds ->
integration. Nothing in the current REM distinguishes these. This may also be what homeostasis
was reaching for; it is implemented as a global Frobenius cap, a blunt "do not get too big",
and it is the one mechanism recorded as not having earned its keep.

## What is stored, and why that is the next mechanism's input

Chunk aggregates cannot say *which* situations were punished, so they support neither the
re-acquisition prediction nor REM replay. Each run now writes `experience.jsonl`, one record
per decision rather than per chunk:

    t, rule, input, move, adv, weight, kind (own|alt), verified, better

`adv` and `weight` are both kept because normalisation makes them differ, and only `weight`
says what was actually written. `input` is what makes a punished situation reproducible.

About 200 KB per 1080-puzzle run, which is nothing - and it **is** the replay buffer REM
needs. The design above says "replay from the negative store"; this file is that store, and
without it the offline mechanism has nothing to read.

## What pays rent

This project's rule is that a biological analogy must produce a falsifiable prediction before
it touches a design choice. Three, in descending order of how cheaply they can be checked:

1. **Signed writes prevent constant collapse.** Re-run `configs/world.yaml` with a negative
   term. The compiled reflex should emit more than one distinct move and should hold above
   `CONSTANT_FLOOR`. Checked by the action distribution, which already exists.
2. **A punished habit is not re-learnable at the same rate as a novel one.** If negative
   memory decays more slowly, re-acquiring a previously punished action should cost measurably
   more verified writes than acquiring a fresh one. This is a WriteBench measurement: `W` for
   a naive structure against `W` for a punished one. **If they come out equal, asymmetric
   decay buys nothing and valence segregation is decoration.**
3. **Replay beats no replay at holding a skill that is drifting.** The population to replay is
   already computed and discarded every probe: the monitor generates both the compiled and the
   deliberate answer and records where they disagree (`agree_wrong_idx`).

## What could kill it

- **Scale.** The REM anti-Hebbian rate is normalised against noise statistics; real replayed
  inputs produce much larger responses. Too strong and damage erases the skill along with the
  error. Perplexity and the action distribution both catch this.
- **Advantage magnitude.** Raw advantage in the `world` spans tens of utility points and would
  dwarf `eta`. It needs per-chunk normalisation, and the normalisation is a free parameter
  nobody has tuned.
- **It may be a verifier fix, not an architecture fix.** The `world` collapse happened in a
  task whose optimal move *was* computable by enumeration ([[hdelta-outcome-verifier-collapse]]).
  Signed writes might be fixing a bad verifier rather than a missing mechanism. Prediction 1
  does not distinguish these; prediction 2 does, because asymmetric decay has no verifier story.

## Implementation hooks (verified 2026-09-21, not written)

- `response_loss` is a flat mean cross-entropy over the batch. Weighting each item's loss by a
  signed advantage before reducing makes the outer product advantage-weighted automatically,
  **and a negative weight flips that item's gradient** - so reinforcement and damage come
  through one path, with no second code mode.
- `PlasticityController.anti_hebbian()` already performs `dW <- dW - eta * (post (x) x)`. It is
  currently pointed only at noise, during REM.
- `on_wrong` has a single branch point (`hdelta/experiment.py:477`). A third mode - `punish` -
  goes there.
- `TradeWorld.evaluate()` already computes and discards the sampled alternatives that the
  advantage needs.


## RESULT: the chain ran 2026-09-22 (4 arms, 2 seeds)

`results/world*`, analysed by `analyse_valence.py` and `check_conditioning.py`.

**Prediction 1 holds. Signing prevents the constant collapse, and the surviving behaviour is
genuinely conditioned rather than noise.**

    arm                distinct moves (48 situations, 3 dispositions)   TVD(hostile|friendly)
    unsigned    s0     1 / 1 / 1   REFUSE x48                            -
    unsigned    s1     1 / 1 / 1   ACCEPT x48                            -
    reward only s0     1 / 1 / 1   WALK   x48                           0.00 / 0.00 / 0.00
    signed      s0     2 / 2 / 3                                        1.00 / 1.00 / 1.00
    signed      s1     2 / 2 / 2                                        0.53 / 0.53 / 0.62

TVD is the total-variation distance between the move distribution under `hostile` and under
`friendly`, on matched situations differing ONLY in standing. 0.00 means the reflex ignores
the feature. `distinct > 1` alone could not separate "learned to discriminate" from "shoved
around by a broken signal"; this does.

**Punishment is the active ingredient, gradation is not.** The reward-only arm grades positive
writes by advantage and never pushes anything down. It collapsed as completely as unsigned -
one move, TVD 0.00 - while running perfectly cleanly. So the missing force was never the
inability to tell a good win from a marginal one. It was the absence of a negative term.

### Three things wrong with the current implementation

**The advantage is mis-centred, and this is the root cause.** It is measured against the MEAN
of the sampled alternatives, and a 3B plays below that mean most of the time: 75 % (seed 0) and
67 % (seed 1) of the model's own moves received negative writes. Punishing three quarters of
all output is demolition, not correction. Centre it so roughly half the outcomes are positive.

**It destroys the slow path.** Invalid output over the run, in fifths:

    signed s0     1% -> 1% -> 72% -> 100% -> 100%
    signed s1     0% -> 0% -> 11% -> 100% -> 100%
    reward only   0% -> 0% ->  0% ->   0% ->   0%

By the halfway mark the model that generates the training signal is emitting gibberish, so the
second half of both signed runs trained on garbage. Whatever they learned, they learned early.

**The damage is surgical, which is the most surprising part.** Perplexity and cloze are FLAT in
every arm (ppl +0.0 % to +0.4 %, cloze unchanged). A 2.75 % plastic slice was enough to
annihilate the task behaviour and nowhere near enough to dent general language ability. Base-
skill checks would have reported all four runs as perfectly healthy.

### What it learned, and what it did not

One policy keyed on `standing`, shared across all three dispositions - trader and diplomat are
byte-identical (WALK when hostile, ACCEPT when friendly). The disposition tag is in the prompt
and the reflex ignores it. It also OVER-conditions: the reference's trader TVD is 0.19, because
a trader should care about goods rather than standing, and the reflex sits at 1.00.

So it found the most robust single axis in the situation and keyed everything on it - which is
what a system under indiscriminate punishment would do.


## CORRECTION: v2, with the runaway fixed (2026-09-22)

The v1 result above is an artifact and should not be quoted.

`contrast()` returned -99 for an unparseable move. That made the format penalty a runaway:
slip off the valid-move manifold once, take an enormous negative write, get pushed further
off. Both v1 signed seeds went from ~1 % invalid output to 100 % within two fifths of the
stream and trained on garbage thereafter, while the running scale inflated 20x (4.4 -> 91.8).

Fixed by never writing an unparseable output at all - it is not a bad decision, it is the
absence of one, so the gradient lands on arbitrary tokens at ~40x the magnitude of a real
policy error.

    run                    invalid output over the stream        neg share   scale
    world_contrast   (v1)  1% -> 1% -> 72% -> 100% -> 100%          75%      4.4 -> 91.8
    world_contrast_v2      0% -> 0% ->  0% ->   0% ->   0%          33%      4.1 ->  4.1
    world_contrast_s1(v1)  0% -> 0% -> 11% -> 100% -> 100%          67%      0.1 -> 81.0
    world_contrast_s1_v2   0% -> 0% ->  0% ->   0% ->   0%          42%      0.1 ->  2.1

With the signal intact for the full run, both signed seeds collapse:

    arm                  distinct moves   profile        TVD(hostile|friendly)
    unsigned      s0           1          REFUSE x48       -
    unsigned      s1           1          ACCEPT x48       -
    reward only   s0           1          WALK   x48      0.00
    signed v2     s0           1          ACCEPT x48      0.00
    signed v2     s1           1          REFUSE x48      0.00

**Valence changes which constant it collapses to. It does not prevent the collapse.**

Two further corrections that matter more than the result:

**The monotone-integrator argument is not the explanation.** This document's central claim is
that collapse follows from the write being unsigned, because a process that can only add ends
up occupied by whatever is added most. Signing makes it non-monotone. It still collapses. So
monotone accumulation was at best incomplete, and the real cause is unknown.

**"Damage is situation-dependent too."** `check_conditioning.py` was built precisely to
separate learned discrimination from noise, and it returned TVD 0.53-1.00 for the v1 signed
arms - which was read as evidence of learning. It was detecting a model being pushed off the
manifold, which varies with the situation for reasons that have nothing to do with a learned
policy. A conditioning test only means what it appears to mean when the run that produced the
weights was healthy. Check the training signal FIRST, then read the behavioural test.
