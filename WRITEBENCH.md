# WriteBench: measuring how hard a logical structure is to write into weights

A protocol for quantifying the *sample complexity* of continual learning as a function of the
**structure of the rule being learned**, rather than its size or its subject.

This document is the method. The numbers it produced are in `RESULTS.md` Part E.

---

## Why not accuracy at a fixed budget

The obvious metric - train every structure for N steps, compare final accuracy - is what the
first version of this ladder used, and it produced six false conclusions in two days. Every
one was the measurement window being read as a property of the architecture:

| claimed | actually |
|---------|----------|
| the homeostatic clamp erases minority rules | the clamp never fired (peak norm 0.0284 of a 0.03 budget) |
| capacity is set by slice size, not amplitude | the equilibrium decides whether the ceiling is reached; 3 rules went 0.38 -> 2.50 on lowered decay |
| conditional precedence cannot be written | it compiles to 1.00 in isolation with a competent teacher |
| non-decomposable joint mappings resist | they compile to +0.89 given 3x the exposure |
| a sequential structure suffers a terminal collapse | it dipped and recovered; the run ended on the dip |
| the failure is representational | the write locus changes the rate ~3x, and either locus gets there |

The common failure is that **a fixed budget measures the budget**. A structure that needs more
exposure looks like a structure that cannot be learned, and the two are not distinguishable
from a single cut point.

## The metric

For a structure `S`, the **write complexity** `W(S, theta)` is the number of *verified writes*
after which the compiled reflex first reaches lift `>= theta` **and sustains it for `k`
consecutive probes** (default `k = 3`).

Report `W` at several thresholds, with error bars over seeds, alongside a stability class.

### lift, not accuracy

    lift = (accuracy - baseline) / (1 - baseline)

where `baseline` is the majority-class share of the structure's own answer distribution.
Structures differ in how guessable they are - in our set the priors ran 0.171 to 0.666 - and
raw accuracy ranks them by prior, not by difficulty. Check the prior before running anything:
a family whose answers are 96 % one label (an early version of our conjunction) measures
nothing at all, because a constant predictor scores 0.96.

### verified writes, not steps or puzzles

Count what actually reached the weights. Structures differ in how often the teacher produces a
correct demonstration, and a rejected attempt teaches nothing, so puzzles are not a common
currency. In our set the teacher ranged 0.806 to 1.000 across families - and in an earlier
task it sat at 0.142, which made a structure look unwritable when it was simply never being
shown correctly.

### sustained, not first crossing

Define the crossing as the first probe beginning a stretch of at least `H` **puzzles** at or
above `theta` (we use `H = 150`). Three definitions fail, all for the same reason - the
measurement apparatus leaking into the number:

- *first crossing* rewards a single lucky probe;
- *holds until the end of the run* makes the score depend on the budget;
- *k consecutive probes* makes it depend on the PROBE INTERVAL - `k = 3` is 150 puzzles of
  stability at `probe_every = 50` and 15 at `probe_every = 5`, so measuring a structure more
  finely would lower its `W` for free. We hit this one while preparing a fine-grained run to
  de-alias the fast families, which would have made them look cheaper than they are.

Specify the hold as a duration and convert to probes per run. One structure in our set peaked at 1.00 and dipped twice over 1800 puzzles; under
"holds to the end" it would have scored `never`, and under first-crossing it would have scored
its luckiest probe.

### several thresholds, because the ordering inverts

Our `precedence` family is the **second cheapest** structure at `theta = 0.5` (73 writes) and
the **most expensive** at `theta = 0.8` (363). It reaches mediocrity fast and then crawls. A
single cut point would have told either story depending on where it was cut. Report at least
`theta in {0.5, 0.8, 0.95}`, and prefer publishing the whole curve.

### seeds, with error bars

Runs are **not** bit-reproducible: bf16 matmuls vary run to run and self-training amplifies the
divergence. Same config, same seed, two runs:

    W(conjunction, 0.8)   1370  vs  906    1.5x   (the headline coefficient)
    W(composition, 0.5)    100  vs   52    2.0x
    W(precedence,  0.8)    363  vs  363    exact

The conjunction figure was about to be published as "26.3x the cheapest structure". The
replicate makes it 17.4x. Same config. A single-seed coefficient is a noisy draw, not a
measurement. **Minimum three seeds, report mean and spread.**

Note what *did* reproduce: the trajectory SHAPE (a dead phase of ~12 probes near the prior,
then a noisy climb) appeared in both conjunction runs, and neither reached `theta = 0.95`.
Shape and ordering are far more stable than the crossing numbers - report both, and do not let
a stable qualitative finding lend false confidence to an unstable quantitative one.

Variance may itself be structure-dependent, which would mean the seed count cannot be a flat
number: in our set `precedence` reproduced `W(0.8) = 363` exactly across two runs while
`composition` swung 100 -> 52 at `theta = 0.5`. **That is n=2 per structure and could easily be
chance** - two runs cannot distinguish a stable coefficient from a lucky pair. Treat
structure-dependent variance as an open question, and until it is settled, budget seeds by the
observed spread per structure rather than assuming any family is cheap to pin down.

### reliability: do not characterise it from three runs either

An earlier draft of this section reported that reliability degrades as theta rises - W@0.5
replicating at 1.19x while W@0.8 spread 1.51x - and drew the conclusion that reaching the
"gist" of a structure costs a predictable amount of exposure while carving it out precisely
does not. That was n=3. The fourth run inverted it:

    n=3     W@0.5  663 / 779 / 653              1.19x
            W@0.8  906 / 1370 / 1025            1.51x

    n=4     W@0.5  663 / 779 / 653 / 369        2.11x   <- now the WORST
            W@0.8  906 / 1370 / 1025 / 889      1.54x
            W@0.95 never / never / 1403 / 1232  1.14x, but only 2 of 4 runs reached it

Two lessons, both of which cost a claim here:

**Reliability is a statistic like any other and needs the same n.** Characterising the spread
from three runs is the same error as characterising a coefficient from one, moved up a level.

**Conditioning on success fakes precision.** W@0.95 looks the tightest of the three at 1.14x,
but that is computed only over the runs that reached it. The two that never did contribute
nothing to the spread. A threshold that half your runs fail to reach has a *yield*, and the
yield is the number that matters - quoting a tight interval over the survivors describes the
lucky half and hides the rest.

Report, per threshold: the values, the spread, AND how many runs produced a value at all.

### probe spacing sets the resolution

`W` is quantised to the probe interval. At `probe_every = 50` three of our families all read
exactly 100 writes and are indistinguishable; their "1.9x" is really "between 1x and 2x". Fast
structures need fine probing (`probe_every = 5`) far more than they need long runs. Choose the
interval from the fastest structure in the set, not the slowest.

## Hard rule: do not characterise a trajectory before the run completes

Not a guideline. In two days of building this protocol, reading the edge of the available data
as the terminal state produced seven wrong conclusions - including two written *after* the
warning was already in this document. The failure is not ignorance of the rule, it is momentum:
a striking partial number invites a mechanical story before the data that would refute it
exists.

Concretely, from this work:
- a family read `-0.148` three probes in and `+0.798` on completion (called a "sharp cliff")
- a family read `1.00 -> 0.14` at the final probe and was called a phase transition; the longer
  run showed it dipping and recovering twice, ending at 1.00
- a family was called "never settles" from a mid-run snapshot and ended with four consecutive
  perfect probes

Every run here ramps over roughly five probes and several oscillate afterwards. **A mid-run
value is not a noisy estimate of the final value - it is a different quantity.** Quote
trajectory shape, stability class and `W` only from completed runs.

## Preconditions: what to check before spending a training run

Each of these is cheap, read-only, and each caught a fatal problem in our work:

1. **Teacher gate.** With the rule stated in context, can the frozen model execute the
   structure *in generated output*? Require lift `>= 0.5`. A structure the teacher cannot
   demonstrate measures the teacher. (Cost us two runs before we measured it.)
2. **Zero-shot floor.** With nothing in context, is accuracy at the prior? If it is already
   high, the answer is recoverable from the input surface and the structure is not being
   learned. Keep the answer vocabulary disjoint from the input vocabulary.
3. **Prior headroom.** Majority-class share well below 1. See above.
4. **Teacher-forced vs generated.** Score generated output. Teacher-forced scoring hid a
   real failure three separate times here; in one case a family read 0.208 teacher-forced and
   0.854 generated.
5. **Feature availability.** If a structure fails, probe whether the target is linearly
   decodable from the residual stream at the layers being written. If it is, the barrier is
   not the representation, and a plan to relocate the plastic slice is already refuted - ten
   minutes of probing saved an hour of training for us.

## Controls that make a structural comparison mean anything

Vary the structure, hold everything else:

- **surface form and prompt length.** Our `map` vs `map_wide` pair has identical structure and
  rule texts of 149 vs 552 characters; the difference was `-0.003`. Without that control, any
  structure needing a longer rule statement is confounded with verbosity.
- **table size, label count, and prior.** `map_wide` vs `conjunction` share all three and
  differ only in whether the answer factors.
- **parameter count**, when comparing write locations. MLP `down_proj` is 2.4x attention per
  layer; comparing 9 layers of each would confound locus with capacity.

## Write locus is a second comparison the protocol can make

Everything above compares structures at a fixed locus. The same machinery compares loci at a
fixed structure, and it answers a design question rather than a descriptive one: where should
the plastic slice go? Measured on `conjunction`, 1800 puzzles, parameter-matched (attention
9L = 84.9M, MLP 4L = 90.2M), three distinct seeds each.

    W to first sustain lift >= theta for 150 puzzles
      locus    theta=0.5           theta=0.8             theta=0.95
      attn9    369-779 (mean 616)  889-1370 (mean 1048)  1232 / 1403, 2 of 4 never
      mlp4     382-472 (mean 439)  569- 723 (mean  668)  723 / 759 / 1202, 3 of 3

    base-model drift, same runs
      locus    ppl final   ppl max     n
      attn9      +0.46%     +1.05%     5 runs / 3 seeds
      mlp4       +3.06%     +8.57%     4 runs / 3 seeds

**MLP is faster, and the separation holds only above theta = 0.8.** At 0.8 the ranges do not
overlap (569-723 against 889-1370). At 0.5 they do - one attention run (369) beats every MLP
run, while the other three sit above the entire MLP range. So the speed claim survives at the
same threshold the ordinal hierarchy needs, and below it the ordering is *not established*,
which is not the same as the loci being equivalent. Do not convert a null result into a claim
of equivalence; the point estimate still favours MLP by 29 %.

**The drift coefficient was published internally as ~40x and is really ~7x.** The 40x came
from pairing attention's best run (+0.08 %) against MLP's worst (+3.85 %), both from the same
`_long` pair. Like-for-like means give 6.7x on final drift and 8.2x on max. The direction was
never in doubt; the magnitude was a best-vs-worst artifact of exactly the kind that produced
"conjunction is 26.3x the cheapest structure". **Compare means to means, or report the pairing
you used.** Note also that `mlp9` (202.9M, 6.6 %) must be excluded from that mean: it is not
parameter-matched, and pooling it is how an earlier audit reported "MLP drifts 165 %".

**"MLP buys deterministic convergence" is dead, and it died to this document's own rule.** The
MLP arm was n=1 at theta=0.95 (759 writes) against attention's 1-of-3, which invited a story
about a mechanically reliable locus. At n=3 the MLP spread at 0.95 is 1.66x (723-1202) and the
original 759 sits near the *bottom* of that range - the usual single-draw-at-the-lucky-end
pattern. Attention's 1.14x looks tighter but is computed over the 2 of 4 runs that reached the
threshold at all, which is the conditioning-on-success trap from the reliability section
above. **Spreads measured over different yields are not comparable.** Report the yield first.

**Yield is what actually separates the loci at theta = 0.95**: 3 of 3 MLP seeds reached it
against 2 of 3 distinct attention seeds. Consistent with the speed result, and far too small
an n to quote as a rate.

### hard tier: constructions validated, writability untested

`conj3`, `comp3` and `prec5` exist so the teacher gate can still find a ceiling above 3B,
where all seven ordinary families saturate. Their constructions now check out end to end
(CPU-only): no family is degenerate, and `conj3`'s restricted 24-cell domain is honoured by
the sampler and enumerated in full by the rule text, so it is hard rather than unanswerable.

    family   n_answers  majority  chance   unsolvable/500
    conj3        6        0.251    0.167        0
    comp3        3        0.338    0.333        0
    prec5        5        0.423    0.200        0

The floors differ enough that raw accuracy is not comparable across them; `probe_cloud.py`
scores chance-corrected lift instead. **No hard family has ever been run with plasticity**, so
nothing is known about their writability and they have no place on the ladder yet.

## Stability is a second axis, not part of the score

Write complexity says how long a structure takes to learn. It says nothing about whether the
structure *stays* learned, and these came apart in our data. Classify separately, by the
post-convergence trajectory:

- **locked** - reaches the ceiling and holds it (our unary and decomposable families)
- **excursion** - holds the ceiling with rare isolated dips that fully recover
- **wobble** - never settles; oscillates in a band indefinitely
- **dead phase then noisy climb** - long stretch at the prior, then a climb that does not settle

> **These are observed trajectory shapes, not mechanisms.** It is tempting to read them as
> basin geometry - deep basins, adversarial boundaries, routing geometry being rearranged - and
> those stories may be right, but nothing here measures weight-space structure. The one time we
> did look for a mechanism behind a shape (a supposed discontinuous collapse) the mechanism was
> not there and the shape was a windowing artifact. Treat the taxonomy as descriptive until
> somebody measures the landscape.

## Reporting template

For each structure: prior, teacher ceiling, zero-shot floor, `W` at each threshold with mean
and spread over >=3 seeds, **the yield at each threshold** (how many runs produced a value at
all), stability class, and the full lift trajectory. Report `never` as "not within budget B",
never as "cannot be learned" - that distinction is the entire reason this protocol exists.

When comparing loci rather than structures, add the parameter count of each slice and the
base-model drift, and state the pairing behind any ratio you quote.

## What this protocol will and will not emit

It emits an **ordinal hierarchy** at `theta >= 0.8` with wide intervals, a **yield** per
threshold, a **stability class**, and a **locus trade-off** (speed against drift). Across our
set that is: `composition < precedence < conjunction`, separated at 0.8 and 0.95, overlapping
at 0.5; and MLP faster than attention above 0.8 at roughly 7x the perplexity cost.

It does **not** emit coefficients. Every family spreads 1.5-2.3x across seeds, and no scalar
here has survived its own replication - 26.3x became 17-26x, 40x became 7x, and a "reliability
degrades with theta" law inverted on its fourth run. Any use of WriteBench that requires a
number like "budget 1,500 verified writes to carve in this rule" is asking for the one thing
the measurements have repeatedly refused to support. Budget from the observed spread of the
nearest measured family, and treat the ordering as the finding.
