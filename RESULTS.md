# Results

**Read Part F first (2026-09-24): it changes the direction.** Tested against the cheapest
alternative in every regime - rule tables, non-table patterns with and without a useful
pretrained prior - the fast-weight slice never won: a reason cache, a tiny learner, or a
low-rank LoRA adapter did at least as well, and LoRA did it without damaging the base model.
The one edge found - the aggressive fast weights kept the first of two sequential skills
better than plain LoRA in both seeds, at a large base-model cost (F.3) - disappears once LoRA
gets one replay step per fresh step: it then keeps more, learns the second skill as well, and
costs ppl +4.5 % instead of +27 % to +311 % (F.5). **Part G** records the design around the
learner that survives - confidence gate, reason-keyed memory that answers directly, and what
each costs. The parts after it remain valid as measurements of the mechanism; Part F is about
whether to build on it.

**Part C, because it bounds the mechanism: capacity is set by the SIZE of the
plastic slice - which sets a hard ceiling - together with the deposition/decay EQUILIBRIUM,
which decides whether that ceiling is ever reached (see the amendment in Part C; the original
"not by amplitude" reading was confounded by raising `eta` rather than lowering `decay`). At
3.1 % plastic the ladder as originally run holds about one skill and eight rules produce
nothing - but the same slice reaches **2.5 of 3** once the equilibrium is raised by halving
decay, and still fails at four. At 9.3 % it sustains four to five of eight concurrent skills
for a cloze cost of about 0.05 and no perplexity cost, i.e. roughly 2 % of the model per
skill. Pushing harder via `eta` fails and damages the base model; pushing via `decay` does
not, and is the lever to reach for. Routing is an efficiency argument, not a necessity.**
See "Part C: capacity" below.

Six experiments. **Part E** (2026-09-21) measures how much exposure different logical
STRUCTURES need before they compile into fast weights - a rate spectrum, not a map of
what is possible: everything tested compiles eventually, differing by ~3x in cost, and
the write locus trades speed against base-model damage. It also overturns Part D's
attribution to conditional precedence, and gives what looked like the first working audit for the
self-verification blind spot (retracted 2026-09-23: it detects exception INPUTS, see Part E §1). **Part D** (2026-09-20) is the first application built on the Part C
recipe, a generalising tool-call cache, and the one that bounds deployment: the reflex
compiles and generalises, but a ground-truth-free monitor's blind spot GROWS with skill
quality and per-query confidence does not cover it. Its mechanism study also corrects
Part C's framing - what looked like a capacity ceiling is an energy limit, and the rule
the reflex never learned is ~50x harder to write per update, not under-supplied. **Part 0** (2026-09-18) is the puzzle-level hand-over measurement on
Qwen2.5-3B-Instruct across three seeds: the statistical core of the crutch-to-reflex
claim. **Part 1** (2026-09-17) is the single-seed run that first produced the hand-over
and the design work behind it. **Part 2** (2026-09-16) is the original few-shot design on
SmolLM2-135M with the full ablation suite, which established what each forgetting
mechanism buys.

---

# Part F: does the fast-weight slice beat the alternatives? (3B, 2026-09-24)

Every earlier task was a small rule table, and an LLM-free cache of the tool's reasons solves
each one outright: workspace scores 1.000 with 32 tool calls out of 800 (its 32 rules),
conjunction the same with 24 (`pipeline_ws.py`; model-free check). So "the reflex generalises"
had only ever meant "it learned to ignore irrelevant fields". Part F asks the question the
build depends on: on tasks a cache cannot solve, does the fast-weight slice beat (a) a tiny
learner with no pretrained knowledge and (b) ordinary low-rank gradient training (LoRA, r=16,
lr 2e-4, the same q/k/v/o locus in the last 9 layers, the same stream seen once)? Every learner
in a task is scored on the IDENTICAL held-out probe set, with means over the second half of
probes. The base-model health of LoRA (ppl, cloze with the adapter on) is measured exactly as
for the fast weights.

## F.1 Four regimes

| regime | task (held-out = disjoint by construction) | best | fast weights |
|---|---|---|---|
| small rule table | workspace / conjunction | reason cache: 1.000, no model | unnecessary |
| non-table, no useful prior | Connect-4 value, 5x4 board (`connect4.py`) | MLP on cells: 0.616 in one pass, 0.705 in four, 0.78 at 32k | 0.44-0.57, below the strongest reason cache (0.570) |
| non-table, code, prior unused | will the stdlib test suite catch this mutant? (`mutants.py`, `mutgen.py`) | three-way tie on AUC (~0.72-0.75); MLP on diff tokens has the best accuracy | tie; +14-42 % ppl when pushed |
| non-table, prior matters | does this Python snippet raise? (`snippets.py`, `snipgen.py`) | **LoRA 0.878 / AUC 0.957, ppl +1-2 %** | see F.2 |
| sequential two skills | snippets -> mutants -> snippets (`composite.py`) | **LoRA** | see F.3 |

Connect-4 detail: LoRA reaches ~0.6 and barely moves with repetition (0.606 -> 0.619 over four
passes, while the MLP goes 0.616 -> 0.705), so the limit is the LLM reading a board written as
text, not the single pass. Fast-weight levers, in order of effect: removing decay
(0.439 -> 0.521, at which point the homeostatic clamp binds), budget 0.12 (0.547, ppl +34 %),
replay of earlier pairs (0.574), budget 0.24 (no gain, cloze 0.73 -> 0.23). A second seed
confirmed "below the cache".

Mutants: the seed-0 result looked like a reversal (fast weights AUC 0.789 vs LoRA 0.700) and
did not replicate: over three seeds AUC was 0.746 / 0.718 / 0.716 (fast weights / LoRA / MLP),
inside the ~0.04 seed spread. Showing the whole mutated function in the prompt moved nothing
for any learner. The missing signal is probably coverage - whether any test executes the
mutated line - which none of them can see.

## F.2 Where the prior matters, LoRA dominates the fast weights (snippets, 3 seeds)

The frozen model is at chance zero-shot (AUC 0.52), but a little supervision unlocks its
knowledge fast: LoRA reaches 0.79 / 0.88 after 1000 examples. That is the regime where an LLM
should earn its place, and it does: LoRA beats the MLP on code tokens in every seed.

| learner | acc | AUC | ppl (base 9.34) | cloze (base 0.73) |
|---|---|---|---|---|
| LoRA (3 seeds) | 0.878 | 0.957 | 9.4-9.5 | 0.73 |
| fast weights, no decay, budget .12 (3 seeds) | 0.890 | 0.964 | 21.5 / 12.5 / 46.4 | 0.33 / 0.80 / 0.27 |
| fast weights, baseline recipe (3 seeds) | 0.748 | 0.903 | 9.4 | 0.77-0.80 |
| MLP on code tokens, N=8000 (3 seeds) | 0.815 | 0.902 | - | - |

The skill-vs-damage frontier (seed 0, acc / AUC / ppl change): baseline 0.738 / 0.907 / +0.9 %;
decay .001 budget .06 0.825 / 0.939 / +4.7 %; no decay budget .06 0.859 / 0.952 / +16 %; the same
with replay 0.864 / 0.958 / +11.5 %; no decay budget .12 0.884 / 0.956 / +130 %. **LoRA: 0.883 /
0.958 / +1.6 %.** Every fast-weight setting lies below LoRA's trade-off. The fast-weight update
is gradient-based on a full-rank attention slice; it buys skill by moving many weights, and
that is what damages the base model.

## F.3 The sequential case (the one the forgetting mechanisms were built for)

Snippets 4000 -> mutants 4000 -> snippets 2000, both skills probed every 250; each cell is the
mean over the last 1000 draws of that phase (acc / AUC). Seed 0:

| learner | snip learned | snip kept after mutants | mut learned | snip after refresh | ppl / cloze |
|---|---|---|---|---|---|
| fast weights, baseline recipe | 0.767 / 0.91 | 0.540 / 0.65 | 0.594 / 0.67 | 0.721 / 0.90 | 9.40 / 0.80 |
| fast weights, no decay budget .12 | 0.845 / 0.94 | 0.809 / 0.92 | 0.612 / 0.71 | 0.875 / 0.94 | 38.4 / 0.10 |
| LoRA | 0.865 / 0.94 | 0.752 / 0.84 | 0.722 / 0.77 | 0.863 / 0.94 | 9.63 / 0.73 |

Seed 1 (same columns): baseline recipe 0.741 / 0.495 / 0.616 / 0.789, ppl 9.40; no decay budget
.12 0.881 / **0.827** / 0.680 / 0.893, ppl 11.84, cloze 0.80; LoRA 0.860 / 0.745 / 0.675 / 0.900, ppl
9.56.

Decay does what it was designed to do - forget, then relearn fast - and LoRA relearns as fast.
**But the aggressive fast weights kept the first skill better than LoRA in BOTH seeds** (0.809 vs
0.752, 0.827 vs 0.745): the one consistent fast-weight advantage found. Mutant acquisition was
mixed (LoRA ahead on seed 0, a tie on seed 1), and the damage ranged from ppl +27 % to +311 %
against LoRA's +2-3 %. The obvious confound is capacity: LoRA r=16 trains 1.8M parameters, the
fast-weight slice 85M. F.5 tests it, and gives plain LoRA the standard fix for forgetting.

## F.5 The retention edge disappears once the alternative gets replay

Same sequential stream, both seeds (snip learned / snip kept / mut learned / snip after
refresh; ppl):

| learner | seed 0 | seed 1 |
|---|---|---|
| fast weights, no decay budget .12 | 0.845 / 0.809 / 0.612 / 0.875; ppl 38.4 | 0.881 / 0.827 / 0.680 / 0.893; ppl 11.8 |
| LoRA r=16 | 0.865 / 0.752 / 0.722 / 0.863; ppl 9.63 | 0.860 / 0.745 / 0.675 / 0.900; ppl 9.56 |
| LoRA r=64 (lr 2e-4) | 0.845 / 0.788 / 0.630 / 0.798; ppl 22.6 | 0.853 / 0.653 / 0.710 / 0.875; ppl 48.2 |
| LoRA r=256 (lr 2e-4) | diverged (ppl 1e7) | diverged (ppl 4e4) |
| **LoRA r=16 + replay 1** | **0.873 / 0.860 / 0.670 / 0.875; ppl 9.76** | **0.895 / 0.895 / 0.620 / 0.910; ppl 9.75** |

Raising LoRA's rank at a fixed lr is not a clean capacity test - it destabilises training -
so whether the fast weights' edge over plain LoRA is capacity stays open. It does not matter
for the build: **LoRA plus one replay step per fresh step (earlier items, no new tool calls)
keeps the first skill better than the aggressive fast weights in both seeds (0.860 vs 0.809,
0.895 vs 0.827), learns the second about as well (0.645 vs 0.646 mean), and costs ppl +4.5 %
instead of +27 % to +311 %.** The fast weights had no replay, so this compares practical
systems, not learning rules - but it is the comparison a build decision needs.

## F.4 What this means for the build

The fast-weight slice, as built, is dominated. What survives is the design AROUND the learner,
all of it measured in the last week: a reason-keyed cache that answers directly (workspace
exceptions 0.913, zero harm, `pipeline_ws.py`), a confidence gate, the tool as teacher and
verifier - with LoRA, trained online on verified outcomes and with a replay buffer, as the
plastic part. The model is only worth having where its pretrained knowledge helps (snippets);
elsewhere a cache or a tiny learner is cheaper and at least as good.

**Scope.** One model (Qwen2.5-3B); single-pass online streams; LoRA at r=16, lr 2e-4 (lr 5e-4
collapsed on Connect-4; higher rank at that lr is unstable); fast-weight eta held at 1e-4;
stationary tasks 3 seeds where a comparison was close, the sequential test 2 seeds. The fast
weights were never given replay in the sequential test. Do not restate the verdict beyond this.

## Reproduce (Part F)

    .venv/Scripts/python.exe c4_preflight.py ; mut_preflight.py ; mutgen.py ; snipgen.py --n 16000
    run_c4_phase{A..E}.ps1 ; run_mut_phase{F,G,H}.ps1 ; run_snip_phase{I,J,K}.ps1 ; run_seq_phase{L,M}.ps1
    .venv/Scripts/python.exe c4_analyse.py ; mut_analyse.py [results/snip] ; seq_analyse.py
    .venv/Scripts/python.exe lora_control.py --config configs/<task>.yaml [--phases] [--replay 1]

---

# Part G: the design around the learner (3B, 2026-09-21 to 09-23)

The parts of the system that Part F leaves standing, measured on conjunction (192 inputs, 24
real circumstances) and workspace. Scripts: `predict_failure.py`, `failure_memory.py`,
`compare_pain.py`, `pipeline.py`, `cost.py`, `pipeline_ws.py`, `probe_delusion_ctrl.py`.

## G.1 The confidence gate

The instinct's probability for the answer it gives predicts its own failures at **AUC
0.96-0.98** on all four pain-free conjunction baselines (24-40 real failures each), with zero
parameters. Deferring the least-confident 30 % to deliberation lifts decisions from ~84 % to
~100 %. A ridge probe on the hidden state, split by circumstance, does NOT transfer (AUC
0.28-0.76): with 12 training circumstances it memorises their identity. Failure becomes
predictable as soon as the skill exists - scaling dW by alpha, accuracy is 0 up to ~0.5 and
switches on by 0.75. **On workspace the gate is blind to the errors that matter:** 90 of 92
exception failures had p_min >= 0.9 (Part D's gate collapse, seen from the pipeline).

## G.2 Failure memory: retrieval precision is everything

Notes like "On south/gamma/spar you answered #q4; it was #q3, because zone south with cluster
gamma -> #q3". The model **copies any note's answer**: a misleading note (similar input,
different circumstance) takes accuracy from ~1.0 to ~0.04, irrelevant notes to chance. Surface
retrieval (2 of 3 fields shared) sent a misleading note to ~47 % of inputs. **Retrieving on the
fields the note's own REASON names** - a note fires only if the input matches every one -
took precision 0.41 -> 1.00 at full coverage; memory then never hurt in 6 of 6 cells, helped an
imperfect instinct (0.786 -> 0.828) and rescued a broken one (0.00 -> 0.46). Once retrieval is
precise, showing the reason to the model barely matters: its value is as the KEY.

## G.3 End to end, and what it costs (conjunction)

Instinct -> (confidence < tau) keyed memory -> (no note) deliberation, one pass, memory filled
from the system's own verified failures (`pipeline.py`, 7 runs x 5 tau x 5 orders). Memory
never harmed (0 cases where it was wrong and deliberation right). At the tau where gate->delib
first reaches >= 0.99: same accuracy with ~45 % fewer deliberate calls (36 vs 71, 49 vs 70,
21 vs 48, 32 vs 58). Most remaining calls are unsure-but-RIGHT answers, which a failures-only
memory never covers. Per call (`cost.py`, one at a time): instinct 136 ms; memory with 2 notes
139 ms (4x the compute, no extra latency - unbatched time is decode steps, ~34 ms/token);
deliberation stopped at the answer line 715 ms; as the testbed ran it (to 128 tokens) 3913 ms.
Per 192 decisions the full pipeline cost 0.74-0.84x the GPU time of gate->delib.

## G.4 On workspace, memory must answer DIRECTLY

With the learned delta on, the reflex **ignores notes** - 0/12 copied even when the note was
about the same file; with the delta off the frozen model copies 4-5/12. So a note that fires
must answer itself. Keys must include the workspace (the same glob maps to different targets in
different tables), and the reason must state what did NOT match: positive-only keys are
0.90-0.97 precise, because an ordinary note ("belongs to T via pattern P") also fires on test
files in the same package. The **decisive** key - the winning pattern plus the later patterns
that did not match - gave, in all 4 runs: 0.987-0.990 overall, exceptions 0.913 (the misses are
first sightings), **zero inputs made worse**, against gate->delib 0.825-0.891. Notes the model
had to read were harmful (31-216 inputs made worse).

**And an LLM-free cache of the tool's reasons scores 1.000 with 32 tool calls** - the whole
workspace is 32 rules, conjunction 24. That finding is what sent Part F looking for tasks a
reason cache cannot solve.

## G.5 Two dead ends

- **Localized pain** (a negative write on only the wrong tokens, never on unparseable output):
  modestly better on average but not established (p~0.11-0.20, one seed carries it), and in 1
  of 3 seeds it DAMAGED the deliberate path (0.76-0.88, 8-19 unparseable answers; fine with the
  weights off). The full gated system scored better without it (1.000 vs 0.948). A pain response
  to deferral itself would punish correct answers: 38-66 % of deferrals are the instinct right
  but unsure.
- **A hidden-state "delusion detector"** is an exception detector: "wrong iff exception input"
  scores 1.000 on its own label cross-seed (Part E §1 amendment).

---

# Part C: capacity of one global h-delta state (3B, 2026-09-18)

`results/capacity/`, drivers `run_capacity.sh` and `run_capacity_controls.sh`, configs from
`make_capacity_config.py`, analysis `hdelta/capacity.py`. N cipher rules are interleaved in
short blocks that cycle, each rule carrying its own tag in the prompt, so the compiled path
must learn `(tag, input) -> answer` for every rule at once. Every rule gets 320 puzzles in
every run, so per-rule exposure is held constant and only the competition varies.

The eight rules are four additive (`d -> d + c`, c odd) and four reflective (`d -> m - d`,
m even). A collision would need `2d = m - c (mod 10)`; an even minus an odd is odd and `2d`
is always even, so **no two rules agree on any digit**. Attribution is structural signal
with zero chance agreement.

> **AMENDMENT (2026-09-20): the ladder below understates capacity, because the energy axis was
> confounded with the size axis.** The headline "capacity is set by the SIZE of the plastic
> slice, not by amplitude" rests on the energy sweep in this part, which scales energy through
> **`eta`** - and `eta` moves the deposition/decay equilibrium *and* the per-update step size
> together. Raising the equilibrium the other way, by **halving `decay`**, separates them.
>
> The two rungs where this ladder failed were re-run at `decay` 0.002 with `budget` 0.06 so the
> clamp stayed non-binding (`run_capacity_decay.ps1` -> `results/capacity_decay/`). Means over
> the second half of probes:
>
> | rung | this ladder (decay .004) | decay .002 | sd | max | final |
> |------|--------------------------|------------|------|------|-------|
> | 3 of 3 | 0.38 | **1.404** | 0.748 | 2.500 | 2.500 |
> | 4 of 4 | 0.28 | **0.237** | 0.183 | 0.688 | 0.594 |
>
> **The cliff moved from between-2-and-3 to between-3-and-4; it did not disappear.** At 3.1 %
> plastic the true ceiling is three skills, and the configuration used below was leaving two of
> them unrealised. At four skills, more energy does not help.
>
> So the corrected statement is: **slice size sets a hard ceiling, and the equilibrium decides
> whether you reach it.** Below the ceiling, energy binds; at the ceiling, size binds. The
> sentence in this part that does *not* survive is "capacity versus energy is non-monotonic and
> peaks early" - that is a property of raising `eta`, not of energy. Compare, all at this same
> 3.1 % slice with three rules:
>
> | route to 2x equilibrium | summed compiled (of 3) | ppl worst / final | cloze mean |
> |-------------------------|------------------------|-------------------|------------|
> | 1x baseline | 0.38 | +14.6 % / +6.3 % | 0.69 |
> | 2x via `eta` | 0.59 | +40.3 % / +18.2 % | 0.59 |
> | 4x via `eta` | 0.09 (collapse) | +325 % / +80.7 % | 0.41 |
> | **2x via `decay`** | **1.404 mean, 2.500 final** | +21.6 % / +10.7 % | 0.63 |
>
> Same equilibrium, two routes: ~4x the capacity at about half the perplexity cost. Large steps
> damage the base model; longer-lived traces do not. **Prefer lowering `decay` over raising
> `eta`.** The cost is real but different in kind - cloze mean 0.73 -> 0.63.
>
> Part D independently found the homeostatic clamp never fires in the normal configuration
> (peak norm 0.0284 against a 0.03 budget), consistent with this part's own note that the
> eta/decay equilibrium, not the clamp, is what caps the fast weights.
>
> Caveats: these rungs plasticise **both** groups, including the MLP group this part finds
> harmful, so an attention-only re-run may do better still (untested). The 3-rule figure has a
> wide spread (sd 0.748) because it inflected late and was still climbing at t=960 - the mean
> understates the state reached, the final overstates its stability.

## The result

| rules | puzzles per visit | summed compiled accuracy | per rule | share of capacity | mean deliberate |
|---|---|---|---|---|---|
| 2 | 10 | 1.38 of 2 | 0.69 | 69 % | 0.55 |
| 2 | 40 | 1.16 of 2 | 0.58 | 58 % | 0.73 |
| 3 | 40 | 0.38 of 3 | 0.13 | 12 % | 0.44 |
| 4 | 20 | 0.28 of 4 | 0.07 | 7 % | 0.52 |
| 4 | 40 | 0.16 of 4 | 0.04 | 4 % | 0.58 |
| 8 | 10 | 0.06 of 8 | 0.01 | 1 % | 0.40 |

**At a 3.1 % plastic slice the cliff is between two and three skills.** (The sweeps below
show this cliff is a property of the slice size, not of the architecture: at 6.2 % it moves
past three, and at 9.3 % the same eight-rule schedule that yields 0.06 here sustains
four to five skills.) At two rules the reflex reaches 0.94 to 0.97
on the strongest rule; at three, no rule ever clears the 0.8 threshold; by eight, the
compiled path never leaves zero.

**Competition is destructive, not divisive.** If the fast weights were a fixed-capacity
store shared among N skills, summed accuracy would stay near 1.16 and each rule would get
1.16/N. Instead the *sum itself* collapses: 1.16, 0.38, 0.16. Adding a third skill does not
divide the budget, it destroys most of it.

**Only consolidation fails; competence does not.** The deliberate path stays healthy in
every run, 0.40 to 0.73 mean per rule even with eight rules live, and individual rules reach
0.88. The model can always execute a stated rule. What a 3.1 % slice cannot do is compile
more than two of them into reflexes at the same time -- and the ceiling test below shows
that is a property of the slice, not of the mechanism.

## Two confounds, both eliminated

The first, which the researcher identified before launch: with fixed block lengths the gap
before a rule is revisited grows with N, so decay alone would erase more of each trace at
larger N (85 %, 62 %, 33 % of a trace surviving at N = 2, 4, 8). Holding the cycle at 80
puzzles instead keeps the surviving fraction at 76 to 85 % across the ladder.

The second was introduced by that fix and caught afterwards: a matched cycle forces the
block length down as N rises (40, 20, 10), confounding "more rules" with "shorter visits".
Two controls settle it. Two rules with ten-puzzle visits **learn** (1.38, the best result in
the table), so short visits are not the problem. Four rules with forty-puzzle visits still
**fail** (0.16), so long visits do not rescue four rules. The decisive comparison is two
versus three versus four rules at identical forty-puzzle visits and identical per-rule
exposure: 58 %, 12 %, 4 %.

## A mechanistic detail worth keeping

At two rules the *schedule* changes the character of the interference. With forty-puzzle
blocks the two reflexes oscillate in antiphase (pairwise correlation -0.19) and the most
recently trained rule has the strongest reflex at 14 of 15 probes (93 %, chance 50 %,
p = 0.0005): a single recency-biased slot being rewritten. With ten-puzzle blocks they rise
*together* (+0.76) and recency dominance falls to chance. Rapid alternation appears to find
a shared solution that serves both rules; long blocks let each rule's consolidation dominate
in turn. This also reframes Part 0: what the phased design called clean "unhooking" and
treated as a feature is the same single-slot overwrite seen from the other side.

## Can more plastic energy buy the third skill?  No.

`results/capacity/budget_sweep.md`. Three rules, forty-puzzle visits, parameter count held
at 3.1 % plastic throughout.

**A correction first.** Raising `budget` alone does nothing, because the clamp is not what
caps the fast weights in the normal configuration. With homeostasis switched off entirely
(`suite3b_v2/no_homeo`) the weights still plateau at 0.030, the same as the capped runs;
the clamp only truly binds when decay is off (`no_decay` sits pinned at exactly 0.0300,
`none` runs away to 0.065). The ceiling is the equilibrium between deposition and decay,
scaling as eta/decay. Energy was therefore scaled through `eta`, with `budget` raised in
step purely so it never became binding. The equilibrium moved as predicted, 0.029 to 0.054
to 0.106, which is itself a check on the mechanism.

| energy | eta | fast-weight plateau | summed compiled (of 3) | rules held | perplexity worst / final | cloze final |
|---|---|---|---|---|---|---|
| 1x | 1e-4 | 0.029 | 0.38 | 0 of 3 | +14.6 % / +6.3 % | 0.70 |
| 2x | 2e-4 | 0.054 | **0.59** | 1 of 3 | +40.3 % / +18.2 % | 0.50 |
| 4x | 4e-4 | 0.106 | **0.09** | 0 of 3 | +325.5 % / +80.7 % | 0.53 |

**Capacity versus energy is non-monotonic and peaks early.** Doubling the energy lifts
summed accuracy from 0.38 to 0.59, which is still only half the two-rule benchmark of 1.16,
and it costs nearly triple the perplexity drift. Cloze is best compared as a mean over
probes rather than a final draw, since the eval is 30 items: 0.69 at 1x, 0.59 at 2x and
0.41 at 4x. Quadrupling the energy *destroys* capacity, down to 0.09.

**The 4x collapse is self-inflicted, and the mechanism is visible in the logs.** At 4x the
model's own generated slow path degrades over the run, from 0.32 to 0.28, where the 1x and
2x runs hold or improve (0.49 to 0.52, 0.42 to 0.50). It produces 374 verified loops against
514 at 1x. The plasticity damages the base model enough that it can no longer reason
correctly, which starves the verifier of the correct traces consolidation needs: a
self-reinforcing collapse. Note that the *teacher-forced* deliberate score stays at 0.43 in
all three runs, so it completely hides this; only the generated path reveals it. The same
lesson as the 1.5B positional-drift finding, recurring at a different scale.

**Conclusion: there is no energy setting that holds three skills.** The optimum sits near
the current configuration, reaches half of two-rule performance, and already costs triple
the base-skill drift. Amplitude cannot buy capacity.

## But dimensionality can, and this reverses the routing verdict

`results/capacity/dimension_sweep.md`. Same three rules, same schedule, per-synapse
dynamics held *exactly* at the 1x settings (eta 1e-4, decay 0.004, budget 0.03). The only
change is how many synapses are plastic, by extending the same module pattern down more
layers.

| axis | setting | plastic | summed compiled (of 3) | rules held at once | perplexity worst / final | cloze final |
|---|---|---|---|---|---|---|
| dimensionality | 3 layers | 3.1 % | 0.38 | 0 | +14.6 % / +6.3 % | 0.70 |
| dimensionality | 6 layers | 6.2 % | **2.03** | **2** | +25.1 % / +8.2 % | 0.57 |
| dimensionality | 9 layers | 9.3 % | **2.47** | **2** | +20.2 % / +5.9 % | 0.57 |
| amplitude | 3 layers, 2x rate | 3.1 % | 0.59 | 1 | +40.3 % / +18.2 % | 0.50 |
| amplitude | 3 layers, 4x rate | 3.1 % | 0.09 | 0 | +325.5 % / +80.7 % | 0.53 |

**Doubling the subspace works where doubling the amplitude does not**, and it is the
cheaper of the two in base-skill terms: 6 layers reaches 2.03 against 2x amplitude's 0.59,
at +8.2 % final perplexity against +18.2 %. Nine layers reaches 2.47 of 3 with per-rule
finals of 1.00, 1.00 and 0.47, and *lower* final perplexity drift than the 3-layer baseline
(+5.9 % vs +6.3 %). These are also the **first and only runs in the whole study to hold two
skills above threshold simultaneously**; every earlier configuration peaked at one.

The deliberate path improves with the subspace too (mean 0.44 at 3 layers, 0.72 at 6, 0.95
at 9), so a larger plastic slice does not damage the model's ability to reason; at 9 layers
it is executing the stated rules nearly perfectly.

**So the limit was parameter volume, not architecture.** The earlier conclusion in this
document that "routing is necessary" was wrong, and the dimensionality axis is what
falsified it. Capacity scales roughly linearly with the plastic slice, at about 3 to 4 % of
the model's parameters per concurrent skill.

## The ceiling test: 9 layers with eight rules

The 6- and 9-layer runs above were ceiling-limited: only three rules were on offer and two
were already pinned at 1.00, so 2.47 of 3 could not distinguish "holds two and a half" from
"holds six, but three was all we asked". Repeating the original eight-rule run
(`configs/capacity_c80_n8.yaml`, ten-puzzle visits, 320 puzzles per rule) with the slice
changed from 3 layers to 9, and nothing else, removes the ceiling.

| plastic slice | rules | summed compiled, final | summed, mean over last quarter | rules >= 0.8, final / mean / peak | perplexity worst / final | cloze final |
|---|---|---|---|---|---|---|
| 3 layers, 3.1 % | 8 | 0.06 of 8 | 0.07 | 0 / 0.0 / 0 | +15.8 % / +8.9 % | 0.70 |
| 9 layers, 9.3 % | 8 | **6.72 of 8** | **4.75** | 6 / 3.2 / 8 | +21.5 % / +5.9 % | 0.70 |

**Tripling the slice takes capacity from nothing to roughly five skills**, at identical
schedule, identical rule count and identical per-synapse dynamics. Per-rule finals are
1.00, 0.97, 0.88, 0.84, 0.84, 0.81, 0.72, 0.66. This is a hundredfold change in summed
accuracy from a threefold change in parameters, so the earlier "3-4 % per skill" estimate
was too pessimistic: the sustained figure is about **2 % of the model per concurrent
skill**.

**It is not a stable eight, and the distinction matters.** All eight rules were above
threshold simultaneously at exactly one probe of 78. The sustained state oscillates
heavily, 4, 4, 2, 0, 0, 3, 6, 3, 4, ... , averaging 3.2 rules above threshold and 4.75
summed over the final quarter. So the system holds four or five skills in a churning
equilibrium rather than eight securely, and the single-probe peak of 8 should not be quoted
as the capacity.

**At this slice the base-skill cost is small, and the earlier cloze story was an artifact.**
Final perplexity drift is +5.9 %, lower than the 3-layer run's +8.9 %. On cloze, quoting the
final probe was a mistake: the eval is only 30 items, so one probe carries a standard error
of about 0.08, and cloze varies by 0.03 to 0.08 across probes *within* every run. Comparing
means rather than final probes: 0.69 and 0.69 for the two 3-layer runs, against 0.64, 0.62
and 0.64 for the three larger-slice runs. So the dramatic "0.70 -> 0.57 drop" and its
equally dramatic "recovery to 0.70" were both single-draw noise. What survives is a small,
consistently-signed penalty of roughly 0.05 (about 1.5 items of 30) at larger slices, which
is below the resolution of this eval and would need a bigger cloze set to confirm.

The contrast with the amplitude route is the part that does survive the noise analysis:
comparing means, cloze goes 0.69 -> 0.64 along the dimensionality axis but 0.69 -> 0.59 ->
0.41 along the amplitude axis. Buying capacity with more synapses is cheap; buying it with
harder writes is not.

**Schedule interacts strongly with slice size.** At 9 layers, three rules on forty-puzzle
visits gave 2.47 of 3, while eight rules on ten-puzzle visits gave 6.72 of 8. More skills
on a faster cycle did better than fewer skills on a slower one. That is the "rapid
interleaving finds a shared subspace" effect from the two-rule comparison, now visible at a
much larger scale, though rule count and block length differ between those two runs so it
is suggestive rather than isolated.

**With enough dimensions the skills separate themselves, with no router at all.** At 9.3 %
plastic with eight rules, the mean and maximum cross-rule answer rate are both **0.000**:
never once does asking for one rule return another rule's mapping, against a mean own-rule
accuracy of 0.84. Compare 0.141 at 3.1 % plastic with two rules. This corrects the earlier
claim in this document that "the prompt tag does not gate the synaptic loop": the 28 %
bleed seen at a 3.1 % slice was capacity starvation, not a failure of tag-based gating.
Given adequate capacity, a tag in the prompt is sufficient for clean separation. Caveat:
these eight rules are maximally distinguishable by construction (no two agree on any digit,
and each has a distinct tag), so this bounds the easy case, not the case of semantically
overlapping skills.

## Where the plasticity has to live: attention, not MLP

`results/capacity/final_two.md`. Each group run alone at 9 layers with the other group's
layer list emptied, everything else identical to the run above. The groups are deliberately
not parameter-matched: `hdeltaH` is q/k/v/o (about 30 % of the slice) and `hdeltaG` is
`down_proj` alone (about 70 %). If capacity tracked parameter count rather than which
modules are plastic, there would be no functional split to find.

| plastic modules | params | % of model | skills sustained (of 8) | M params per skill | perplexity final | cloze mean |
|---|---|---|---|---|---|---|
| both groups, 3 layers | 96 M | 3.1 % | 0.07 | n/a | +8.9 % | 0.69 |
| **attention only, 9 layers** | **85 M** | **2.8 %** | **5.92** | **14** | **+1.2 %** | **0.76** |
| MLP only, 9 layers | 203 M | 6.6 % | 0.61 | 334 | +5.2 % | 0.70 |
| both groups, 9 layers | 288 M | 9.3 % | 4.75 | 61 | +5.9 % | 0.64 |

**The split is real, large, and the labels were backwards.** Attention-only reaches 5.92
sustained skills on 85 M parameters. MLP-only reaches 0.61 on 203 M, i.e. a tenth of the
capacity from 2.4x the parameters. `hdeltaG` was commented in the config as the
"procedural" group, the basal-ganglia analogue, "what to do"; it is the group that cannot
compile a procedure. `hdeltaH`, the "relational" group, is where all of the compilation
happens.

**Including the MLP group actively hurts.** Both groups together give 4.75 skills from
288 M, against 5.92 from the 85 M attention slice alone. Adding 203 M of MLP plasticity
*subtracts* about a skill, triples the perplexity cost (+5.9 % against +1.2 %) and drags
cloze down (0.64 against 0.76, the latter being at or above the 0.73 baseline). The
attention-only configuration is better on every axis measured.

**This revises the cost per skill downward by a factor of four.** 14 M parameters per
sustained skill, or about 0.45 % of the model, rather than the 2 % estimated from the
mixed-group runs. Eight skills cost under 3 % of the model, and the 95 %-frozen constraint
from the design spec survives after all.

**One confound is not isolated.** Attention-only is 36 modules of four different
projection types; MLP-only is 9 modules of one type. So "attention beats MLP" could also be
"many small diverse matrices beat few large ones". Making `up_proj`/`gate_proj` plastic, or
`q_proj` alone, would separate those.

## Do confusable cues break the isolation?  Partly, and gracefully

Identical to the 9-layer eight-rule run except that the tags become `K1`..`K8`, which share
a token and differ by one character, instead of `A7`/`M6`/`A3`/... The mappings are
untouched, so attribution stays exact.

| cue | skills sustained (of 8) | cross-rule bleed, mean / max |
|---|---|---|
| distinct (`A7`, `M6`, ...) | 4.75 | 0.000 / 0.000 |
| confusable (`K1`..`K8`) | 3.87 | 0.043 / 0.094 |

Confusable cues cost roughly one skill and introduce real but small bleed: on the worst
rule, 3 of 32 probe puzzles are answered with a different rule's mapping. So the
self-organised isolation is not an artifact of maximally distinctive tags, but it is not
free either. It degrades gradually rather than collapsing, which argues against needing a
hard firewall and in favour of treating cue design as a first-class concern.

## How fallible can the verifier be?

`results/capacity/verifier_noise.md`. Everything else in this document assumed a
ground-truth oracle. Here the verifier wrongly accepts a fraction of the model's *wrong*
answers, and a falsely accepted answer is written back as the model's own -- on the slow
path and in the zero-shot replay -- so the system reinforces its own mistakes. Attention
only, 9 layers, eight interleaved rules.

| verifier false-positive rate | contamination of what is written | skills sustained (of 8) | reinforced errors reproduced, final / peak | recovers truth on poisoned inputs | perplexity final |
|---|---|---|---|---|---|
| 0 % (oracle) | 0.0 % | 5.92 | n/a | n/a | +1.2 % |
| 5 % | 0.3 % | 6.13 | 0.000 / - | 0.78 | +1.2 % |
| 15 % | 1.6 % | 5.40 | 0.000 / - | 0.94 | +1.3 % |
| 30 % | 3.2 % | 5.48 | 0.000 / 0.167 | 0.88 | +1.0 % |

**The architecture is far more robust to a fallible verifier than expected.** At a 30 %
false-positive rate, with 92 wrong answers consolidated as if correct, capacity falls only
7 % (5.92 to 5.48), base skills are untouched (+1.0 % perplexity, cloze 0.77), and by the
end of the run the compiled path reproduces *none* of the errors written into it while
returning the correct answer for 88 % of the poisoned inputs.

**Errors are written, surface briefly, then wash out.** The final figure of 0.000 hides the
dynamics: mid-run the poison probe peaks at 0.167, so about one in six poisoned inputs did
temporarily return its reinforced error. Passive decay plus the weight of consistent
evidence removes them. This is the asymmetry the mechanism depends on: a skill is
*systematic*, reinforced identically by hundreds of examples that agree, while false
positives are *unsystematic*, each pointing in its own random direction. Consolidation
integrates, so coherent evidence accumulates and incoherent evidence cancels.

**The headline number needs an important qualification.** A 30 % false-positive rate is not
30 % contamination. The rate applies only to answers the model got wrong, and in this
configuration its own slow path is right about 91 % of the time, so the pool of candidates
for false acceptance is small: 30 % of them is a 3.2 % contamination of the consolidation
stream. Verifier precision and slow-path accuracy multiply. **What this experiment
establishes is tolerance of roughly 3 % contamination, not 30 %.** In a domain where the
model is right only half the time, the same verifier would contaminate around 23 %, which
is untested.

**The other untested case is systematic error.** The injected noise is random: a given
input is rarely poisoned twice with the same wrong answer. Real verifiers fail in
correlated ways -- a test suite that misses an edge case accepts the *same* wrong behaviour
every time -- which is precisely the regime where "incoherent evidence cancels" stops
holding. That, not the raw rate, is the risk worth testing before deployment.

**Where this leaves routing.** Dense scaling works, and works better than estimated: about
2 % of parameters per sustained skill, with no measurable base-skill cost at 9.3 % plastic.
Fifty skills would still imply most of the model being plastic, so sparse addressable
modules remain the way to avoid spending O(N) of the network. But that is now firmly an
efficiency argument. The claim earlier in this document that dense gradients interfere
structurally, and that routing is therefore unavoidable, is **false**.

## What this means for the agentic loop

Routing is an efficiency measure, not a prerequisite. A single global fast-weight state can
back N skills, at roughly 2 % of the model's parameters per sustained skill, so a dense
approach remains viable into the low tens of skills and only exhausts the model beyond that. The verifier gate and canary-rollback design
still stand, but they must sit on top of per-skill plastic state with addressable routing.
The tag in the prompt is *not* sufficient to gate the synaptic loop: at two rules with long
blocks, asking for one rule returned the other rule's mapping 28 % of the time.

---

# Part D: the tool-call cache, and why it cannot be deployed unguarded (3B, 2026-09-20)

`results/gate3_cold/` (plasticity off), `results/gate3_gated/` (plasticity on, per-query
confidence captured, `delta.pt` saved). Task `hdelta/tasks/workspace.py`, configs
`workspace_gate3.yaml` / `workspace_gated.yaml`, safety code `hdelta/toolcache.py`,
analysis `analyse_gate.py` and `check_bloom.py`.

The first application built on the Part C recipe: distil an expensive authoritative tool
into the fast weights, using the tool as both teacher and verifier. The tool is a hermetic
simulator of a build-graph query (file path -> build target), so the baseline is reproducible
without a live workspace. Four workspaces are interleaved in 56 blocks of 25, tagged with
deliberately confusable cues (`core-auth` / `core-audit` / `core-audio` / `core-author`) and
pinned to one top-level area so the path never reveals which workspace applies. Target labels
come from a vocabulary independent of the directory names, which holds the zero-shot baseline
at the floor by construction.

## The physics claim holds

Zero-shot direct accuracy, **mean over probes** in the second half:

| rule | cold baseline | mean t>=700 | sd | max |
|------|---------------|-------------|------|------|
| A | 0.00 | 0.505 | 0.165 | 0.83 |
| B | 0.00 | 0.559 | 0.211 | 0.88 |
| C | 0.00 | 0.523 | 0.166 | 0.81 |
| D | 0.00 | 0.538 | 0.192 | 0.88 |

The cold run measured exactly 0.00 on all four rules across 16 probes, so this is not
baseline leakage. **84.4 % of scored inputs never appeared in the training stream**, and
accuracy splits seen 0.572 / never-seen 0.524, a gap of **+0.048** against the prior
+0.007 +- 0.068. The reflex generalises; it is not a hash cache.

Safety during learning was clean: 1220 writes, **0 bad writes, contamination 0.0**,
perplexity 9.340 -> 9.386 (+0.50 % final, +1.03 % max), cloze flat at 0.733.

Nothing sustained at these settings: sd ~0.19 with peaks at 0.88 and means near 0.53;
monitor-healthy fractions were A 1/17, B 3/17, C 5/17, D 7/17. Which rule would starve was
predicted wrong twice - the probe set said D, the stream said A, and all four behaved alike.

**That oscillation is an energy limit, not a capacity ceiling, and the first draft of this
section got it wrong.** The fast-weight norm peaked at 0.0284 against a 0.03 budget: the
homeostatic clamp never fired, so nothing was being clamped away. Halving decay (below)
doubles the equilibrium and puts **three of four rules at 1.00** on the ordinary branch
simultaneously. The slice was never full.

## The deployment claim does not hold

Coverage at fixed accuracy, threshold **fitted and scored on disjoint halves** (in-sample
selection inflates this: on pure noise it reads 0.025 against 0.000 held out):

| target | best stat | coverage | acc@cov |
|--------|-----------|----------|---------|
| 0.70 | p_geo | 0.688 | 0.716 |
| 0.80 | p_min | 0.350 | 0.803 |
| 0.90 | p_min | **0.001** | 1.000 |
| 0.99 | p_min | **0.001** | 1.000 |

The gate does not degrade, it **collapses between 0.80 and 0.90**. 18 of 19 probes had zero
queries pass at the 0.99 threshold, so this is not an averaging artefact.

## The blind spot grows with skill quality

Grouping every (probe, rule) state by its compiled accuracy:

| accuracy band | mean agree_wrong |
|---------------|------------------|
| 0-20 % | 0.011 |
| 20-40 % | 0.029 |
| 40-60 % | 0.062 |
| 60-80 % | 0.078 |
| 80-100 % | **0.117** |

Monotonic. Inside states already >=0.80 accurate (n=1152) confidence separation collapses
from +0.174 pooled to **+0.036**, with agree_wrong p_min .894 against correct .897.

On the 263 agree_wrong cases (17.4 % of all errors) confidence sits with the **correct**
answers on every statistic:

```
p_disc : agree_wrong .887   other wrong .757   correct .896   <- indistinguishable
p_tail : agree_wrong .960   other wrong .875   correct .940   <- HIGHER than correct
p_min  : agree_wrong .837   other wrong .677   correct .850
p_seq  : agree_wrong .800   other wrong .603   correct .811
```

**Divergence and confidence fail on the same inputs.** Stacking them buys nothing, and
adding capacity moves the system further into the regime where the signal is weakest. A
model cannot grade its own compiled homework: the frozen reference executes the same rule
over the same representations, so its errors are correlated with the reflex's by
construction, not by accident.

## The mechanism: it failed structurally, not randomly

Splitting by whether the **input** matches an override pattern (`**/*_test.ext`) - a regex
decision needing no model and no tool call. Two samples, because they differ: the **final
compiled state** (192 queries, the state a deployment would ship) and **pooled over every
late probe** (3648 queries, which averages across the oscillation):

| input class | share | error rate (final / pooled) | agree_wrong (final / pooled) |
|-------------|-------|-----------------------------|------------------------------|
| override-eligible | 15 % | **100 %** / 99.8 % | 62 % / 47.5 % |
| ordinary | 85 % | 10.4 % / 31.0 % | **0 %** / 0.03 % |

The ordinary error rate is much worse pooled (31 %) than at the final state (10 %) because
the reflex oscillates; the override numbers barely move, because that failure is structural.
Ordinary `agree_wrong` is one query in 3097 pooled, and zero at the final state.

**100 % of blind-spot errors sit on override-eligible inputs.** The reflex never learned the
last-match-wins exception at all (accuracy 0.002 on those inputs across every late probe) -
it learned the package mapping, which is right 85 % of the time, and ignored the exception.
The frozen reference misses the same exception, which is exactly why the two paths agree.
Why it never learned it is the subject of the mechanism study below; the short answer is that
it is not a data problem.

This also disposes of an existence check as the fix. Against the saved `delta.pt`, a bloom
filter of valid targets catches 8.7 % of errors and a per-workspace filter 41.3 %, but of the
blind-spot errors it catches **0.0 %** - all 18 are valid targets of the *correct* workspace.
91 % of wrong answers are real labels applied to the wrong input.

## Why the exception never compiled: three candidates, two eliminated

> **Partly superseded by Part E.** The structure blamed below - conditional precedence -
> compiles to 1.00 when trained alone with a competent teacher. The measurements here stand;
> the attribution does not. The structure that actually resists is the non-decomposable joint
> mapping. See Part E.


`probe_teacher.py`, `run_mechanism.ps1` -> `results/mech_h2_ratio/`, `results/mech_h1_decay/`.

The exception branch sat at ~0.00 while the ordinary branch compiled. Three explanations were
live: the teacher supplied too few correct demonstrations, the sparse updates were eroded
between exposures, or the mapping is simply hard to write into these weights. Each arm below
moves exactly one variable from the `workspace_exc` run.

**First, a free diagnostic.** Teacher-forced accuracy on the exception branch is **0.0073,
identical to generated (0.0073)**. The mapping is not in the weights at all, so this was never
a decoding failure - the weights contain no exception mapping to decode.

**Teacher quality is real but insufficient.** The frozen teacher - rule stated, two worked
examples - is wrong on **86 %** of override-eligible inputs (0.142 accuracy, per-rule
A 0.000, B 0.333, C 0.069, D 0.167). Every earlier run used that formulation, so the reflex
saw roughly **8 correct override demonstrations per rule** against ~333 ordinary. Guaranteeing
one override example in the shots triples it (0.142 -> 0.424, paired +0.281 +- 0.250, positive
in 10/12 pairs) at no detectable cost to ordinary inputs (-0.035 +- 0.183). Reordering the
table specific-to-general does **not** help (0.177); reading order is not the problem, example
coverage is. But tripling the signal changed nothing downstream: 8 -> 25 demonstrations per
rule moved the compiled exception branch from 0.0018 to 0.0073.

| arm | what moved | exception | n_exc | ordinary | peak fw | ppl drift |
|-----|-----------|-----------|-------|----------|---------|-----------|
| natural | - | 0.0073 | 551 | 0.6432 | 0.0284 | +0.40 % |
| **H2 ratio 1:1** | `exception_fraction` 0.15 -> 0.68 | 0.0072 | 2774 | 0.3387 | 0.0266 | +0.77 % |
| **H1 decay halved** | `decay` 0.004 -> 0.002, budget -> 0.06 | **0.0000** | 551 | 0.7678 | 0.0417 | +0.90 % |

**H2, gradient dominance: eliminated.** Raising the exception share to 0.68 targets parity of
*verified writes* rather than of traffic (the teacher is 0.42 on exceptions against 0.88 on
ordinary, so 0.68 x 0.42 ~= 0.32 x 0.88). Realised split: ~172 exception writes against ~170
ordinary, from 1:15.4. The exception branch went 0.0073 -> 0.0072, measured on five times the
observations.

**The matched-write control is the decisive measurement.** Inside that one arm, **~170
ordinary writes produced 0.339 and ~172 exception writes produced 0.007**. Same run, same
slice, same decay, same competition, same teacher, the same number of verified writes - and a
**~50x gap per write**. Everything H1 and H2 appeal to is held constant across that comparison.

**H1, decay erosion: eliminated.** Halving decay doubled the equilibrium as intended (peak
fast-weight norm 0.0284 -> 0.0417; the budget was raised to 0.06 in this arm so the clamp
stayed non-binding and decay alone moved). The exception branch went to **exactly 0.0000**
across 551 observations while the ordinary branch rose to 0.7678 - and on the final probes
ordinary is **1.00 on three of four rules** with the exception still at 0.0000.

**H3, representational difficulty: supported directly**, not merely by elimination. The
exception is not being overwritten and not being eroded; it is roughly fifty times harder to
write per update than a lookup of the same nominal size.

**What remains untested is absolute volume.** The ordinary branch needed ~1134 writes to reach
0.64; the exception has never received more than ~172. "Harder per write" is established.
"Impossible at any volume" is not - though at 50x the cost, and with the exception at 15 % of
traffic, the deployment runway required makes the distinction academic.

**A practical heuristic falls out.** Reaching 2x equilibrium by *halving decay* cost +0.90 %
perplexity; Part C reached the same equilibrium by *raising eta* and paid +18 %. Same
destination, far gentler path, because the step size is unchanged. Prefer lowering decay.

**One confound to record.** Pushing the stream to 68 % exceptions also biased the *shot* pool,
so the teacher saw multiple override examples, inferred that the exception was the rule, and
its accuracy fell 0.770 -> 0.218 (total writes 1208 -> 342). The matched-write control is
internal to that run and survives intact, but the arm is not a clean test of abundant
exposure, and it is a sharp reminder that the teacher's in-context accuracy is a function of
the distribution it is shown.

## What is deployable

Route override-eligible inputs straight to the tool, then confidence-gate the remainder,
where the blind spot is empty and divergence monitoring is valid again. **Three seeds**
(0/1/2, `aggregate_seeds.py`); the threshold is refitted per seed:

| target | total tool calls avoided | per seed |
|--------|--------------------------|----------|
| 0.90 | **41.9 % +- 9.6 %** | 41.9 / 30.2 / 53.7 |
| 0.95 | **24.1 % +- 4.6 %** | 28.1 / 17.7 / 26.6 |
| 0.99 | **3.1 % +- 2.6 %** | 2.8 / 0.2 / 6.5 |

Usable at a 95 % accuracy target, useless at 99 %. The spread is the point: at 0.95 the
seeds range 17.7-26.6 %, so quote the interval, not a point estimate. Compiled accuracy
itself is stable at 0.614 +- 0.020. The cheapest effective safety mechanism
was a regex on the **input**; confidence and divergence were both inspecting the wrong end of
the pipe.

Where the fast weights are treated as a branch predictor rather than a cache - speculate
immediately, fire the tool asynchronously, drop the speculation on mismatch - the blind spot
stops mattering, because the tool result always arrives before anything is committed. That
captures the latency win on the hit rate and degrades to baseline tool speed on a miss.

## What replicates across seeds, and what does not

Three seeds, same workspaces (fixed by the per-rule config seeds), different probe sets,
stream ordering, REM noise and torch init.

| claim | mean +- sd | per seed |
|-------|------------|----------|
| override-eligible error rate | **99.9 % +- 0.1 %** | 99.8 / 100.0 / 99.7 |
| ordinary error rate (pooled) | **30.4 % +- 0.7 %** | 31.0 / 30.8 / 29.4 |
| agree_wrong on ordinary inputs | **0.0 %** | 0.03 / 0.00 / 0.03 |
| share of errors that are agree_wrong | 15.2 % +- 1.8 % | 17.4 / 15.2 / 12.9 |
| compiled accuracy | 0.614 +- 0.020 | 0.586 / 0.627 / 0.629 |

The blind spot grows with skill quality in **every** seed:

| accuracy band | mean agree_wrong | per seed |
|---------------|------------------|----------|
| 0-20 % | 0.023 +- 0.003 | .027 / .021 / .021 |
| 20-40 % | 0.036 +- 0.003 | .031 / .039 / .037 |
| 40-60 % | 0.050 +- 0.007 | .060 / .043 / .047 |
| 60-80 % | 0.063 +- 0.011 | .077 / .061 / .049 |
| 80-100 % | **0.081 +- 0.026** | .117 / .073 / .054 |

So the structural findings are seed-independent to within a fraction of a percent, while the
coverage figures swing by a factor of ~1.5. The override rule fails in every seed; the blind
spot is confined to override inputs in every seed (one query in ~3000 outside them, twice out
of three runs); and the monotonic relationship between skill quality and invisible error
holds three times over. Only the deployment arithmetic is a single-draw quantity, and it is
now reported with its spread.

## Three measurement traps, all of which changed a conclusion

1. **`p_first` carries exactly zero signal**: correct .997 vs wrong .997, separation
   **-0.001**. Every label tokenises as `[' //', head, ':', tail]`, so the first token is
   shared by all 33 of them. Sequence probability read without checking which position
   discriminates would have rubber-stamped the system.
2. **`instructed_scratch_acc` is exact match on the whole trace, not capability.** Rule B
   reads 0.208 there and **0.854** on generated answers - it reaches the right target by a
   different route.
3. **Runs are not bit-reproducible.** bf16 GPU matmuls vary run to run and self-training
   amplifies it: the same config diverged by t=52 (D direct .04 vs .00, ppl 9.42 vs 9.43).
   Trajectory equality is not a valid correctness check; compare distributions.

Two accounting rules: quote `verifier/verifier_calls` (agent-side, 1568) and never
`task.tool_calls()` (13358, which includes dataset construction and probe instrumentation);
and keep the monitor's calibration verifier separate from the training one, or its checks
inflate `writes` and `contamination`.

## Reproduce

```
.venv\Scripts\python run_experiment.py --config configs/workspace_gate3.yaml  --out results/gate3_cold   # ~27 min, plasticity off
.venv\Scripts\python run_experiment.py --config configs/workspace_gated.yaml  --out results/gate3_gated  # ~52 min
.venv\Scripts\python analyse_gate.py results/gate3_gated
.venv\Scripts\python check_bloom.py  results/gate3_gated
powershell -ExecutionPolicy Bypass -File run_gated_seeds.ps1     # seeds 1 and 2, ~105 min, SEQUENTIAL
.venv\Scripts\python aggregate_seeds.py results/gate3_gated results/gate3_gated_s1 results/gate3_gated_s2

.venv\Scripts\python probe_teacher.py                          # teacher ceiling on exceptions, ~7 min
.venv\Scripts\python run_experiment.py --config configs/workspace_exc.yaml --out results/gate3_exc
powershell -ExecutionPolicy Bypass -File run_mechanism.ps1      # both mechanism arms, ~105 min, SEQUENTIAL
```

Launch long runs detached with `Start-Process`, not `nohup ... &` - a nohup'd run was killed
at phase 55/56 when the shell session that launched it ended. Check free VRAM first: a run started with 8.6 GB
already in use ran ~4x slower without ever reporting OOM.

---

# Part E: what can be written into fast weights, and how to audit it (3B, 2026-09-21)

`hdelta/tasks/logic.py`, `probe_families.py`, `run_ladder.ps1`, `run_doseresp.ps1`,
`probe_hidden_xseed.py`, analysis `analyse_ladder.py` / `analyse_doseresp.py`.
Results in `results/ladder/`, `results/doseresp/`, `results/families/`.

Part D left two questions. The safety question: is the shared delusion detectable at all,
given that output divergence and per-query confidence are both blind to it? And the mechanism
question: Part D attributed the uncompilable branch to "conditional precedence", on two data
points from one task. Both are answered here, and the second answer overturns Part D.

## 1. The geometric signature: the delusion is visible in the residual stream

> **AMENDMENT (2026-09-23): this probe is an exception detector, not a delusion detector.**
> It was never controlled for input structure, and on this task every shared delusion sits
> on an override input. `probe_delusion_ctrl.py` (same three runs; states taken one prompt at
> a time; reference given 160 tokens instead of 64) scores the rule "wrong iff the input is
> an exception" at **1.000 accuracy, precision and recall on all six cross-seed pairs**,
> against the probe's 0.76-0.98 AUC. The instinct is wrong on 481/481, 482/482 and 469/477
> exception inputs, so there is no within-exception variance for a model-state signal to
> explain. On confident-wrong answers (p_min >= 0.9) the probe matches or trails the same
> rule. Cross-seed transfer was expected either way: every seed has the same exceptions.
> Retract "the first working audit for the blind spot". The posture is readable from the
> INPUT, which is what keyed memory and a structural route already use, with no labels.

A linear probe on the hidden state at the final prompt position, fitted on one run's
activations and **scored on a different run's fast weights**:

| fit -> test | layer | accuracy | baseline | precision | recall |
|-------------|-------|----------|----------|-----------|--------|
| s0 -> s1 | 27 | 0.976 | 0.908 | 0.914 | 0.821 |
| s0 -> s2 | 27 | 0.971 | 0.890 | 0.943 | 0.786 |
| s1 -> s0 | 26 | 0.978 | 0.949 | 1.000 | 0.571 |
| s1 -> s2 | 26 | 0.974 | 0.890 | 0.971 | 0.786 |
| s2 -> s0 | 16 | 0.976 | 0.949 | 0.704 | 0.905 |
| s2 -> s1 | 16 | 0.957 | 0.908 | 0.714 | 0.897 |

A hyperplane fitted on one seed flags shared delusion in `dW` states it has never seen. The
signal is therefore a property of the phenomenon, not a memorised artefact of one run - which
is exactly what the within-run version (0.986 against a 0.877 baseline, but on 9 held-out
minority points) could not establish.

**The boundary matters as much as the result.** Fitting the probe needs labels, and labels
need the tool. This is **ground-truth-cheap, not ground-truth-free**: pay N tool calls once to
calibrate, then detect for free. That is a weaker claim than the divergence monitor made for
itself, and it is the honest one. The operating point is also unstable - which run you
calibrate on moves recall between 0.571 and 0.905 - so a deployment would need to choose the
precision/recall trade deliberately rather than inherit it.

## 2. The writability ladder, and a correction to Part D

> **MAJOR AMENDMENT (2026-09-21, later the same day).** The headline below - that
> non-decomposable joint mappings *resist* being written - is **wrong**, and was an artifact of
> the 600-puzzle budget every family was given. Run to 1800 puzzles, `conjunction` compiles
> under **both** loci:
>
> | locus | puzzles | lift | ppl drift | cloze |
> |-------|---------|------|-----------|-------|
> | attention 9L (84.9M) | 600 | +0.056 | +0.6 % | 0.73 |
> | attention 9L (84.9M) | 1800 | **+0.803** | **+0.1 %** | 0.77 |
> | MLP down_proj 4L (90.2M) | 600 | +0.404 | +2.3 % | 0.70 |
> | MLP down_proj 4L (90.2M) | 1800 | **+0.890** | +3.9 % | 0.70 |
>
> **Writability is a RATE, not a possibility.** The ladder below measures how much verified
> exposure each structure needs at a fixed budget - which is a real and useful quantity - but
> it must not be read as a map of what can and cannot be written. Everything tested here can
> be written; they differ by roughly 3x in how long it takes.
>
> The locus matters, and it trades speed against damage: MLP plasticity reaches the joint
> mapping about 3x faster and costs ~40x the perplexity drift (+3.9 % vs +0.1 %). More MLP is
> not better - a 9-layer MLP slice (202.9M, 6.6 % of the model) scores +0.083, *worse* than the
> 4-layer one, consistent with Part C on MLP plasticity being destructive at scale.
>
> Two further hypotheses died on the way to this. **Sparse expansion**: the conjunction is
> linearly decodable at 1.000 across the whole existing plastic slice (`probe_decodable.py`),
> so the slice was never positioned where the feature had been lost, and moving it to middle
> layers would have moved it somewhere the feature is *less* available (0.93). **Attention as
> mere routing**: it does express the joint mapping, just slowly.
>
> What survives from this section: the family ordering as a rate spectrum, the controlled
> verbosity and arity contrasts *at fixed budget*, gate 1 passing for every family, and the
> Part D correction below - `precedence` compiles, so Part D's attribution was wrong either way.


Seven logical structures, same slice (attention-only, 9 layers), same eta/decay/budget, each
**trained alone** for 600 puzzles so capacity and inter-rule competition are excluded. Scored
as lift over each family's own answer prior, because the priors range 0.171 to 0.666 and raw
accuracy would rank families by how guessable they are:

| family | baseline | teacher | lift | structure |
|--------|----------|---------|------|-----------|
| map | 0.171 | 0.993 | **+1.000** | unary lookup |
| disjunction | 0.615 | 1.000 | **+1.000** | OR of two unary predicates |
| negation | 0.666 | 0.917 | **+1.000** | unary predicate + default |
| map_wide | 0.171 | 0.993 | **+0.997** | unary, 24-entry table |
| composition | 0.338 | 1.000 | +0.815 | g(f(x)), chained unary |
| precedence | 0.534 | 0.806 | +0.808 | ordered clauses, last match wins |
| conjunction | 0.172 | 0.965 | **+0.056** | irreducibly bivariate |

Every family passes gate 1 (teacher 0.806-1.000 on **generated** output, zero-shot exactly
0.000 everywhere), so no family's failure is attributable to the teacher - the confound that
cost Part D two runs before anyone measured it.

**Part D's attribution was wrong.** `precedence` - the structure blamed for the 0.0000
override branch - compiles to 1.00 here. The Part D failure was real but required a
combination that was never separated: inter-rule competition among four tagged rules,
deliberately confusable cues, a teacher at 0.142, and the glob-pattern surface form. "Fast
weights cannot hold conditional precedence" is falsified.

**The barrier is decomposability.** The controlled contrasts:

| contrast | held constant | delta |
|----------|---------------|-------|
| map -> map_wide | structure; rule text 149 -> 552 chars | **-0.003** |
| map_wide -> conjunction | rule text 552 chars, 24-entry table, 6 labels, prior 0.171 | **-0.941** |

Prompt length, table size, label count and answer prior are all eliminated. What remains is
whether the answer **factors into independent per-feature contributions**. Disjunction needs
two features and compiles perfectly, because OR of two unary predicates decomposes.
Conjunction-as-joint-lookup does not decompose, and does not compile. Chaining does not
resist either: `composition` reaches 1.00.

**A verified student can sometimes beat its teacher - on two families, not generally.**
`audit_claims.py` re-tests this against every completed run. An early version compared the
reflex's PEAK to the teacher's MEAN and reported 6/15, biased upward by construction;
like-for-like (second-half mean vs teacher mean) read 3/15, and at 88 completed runs it reads
**5/22** - `negation` once (1.00 v 0.92) and `precedence` four times (0.91-0.96 v 0.81).

The fraction is not the finding. Its denominator is "runs on disk that have a teacher
number", so it drifts every time the corpus grows - it went 3/15 -> 5/22 without a single new
fact. What has been stable across both audits is the **family list**: only `negation` and
`precedence` ever do it, and eleven other families never have. The verifier consolidates only
*correct* demonstrations, so on those two the student learns from a cleaner distribution than
the teacher's average output. **On `negation` and `precedence`** verified self-training acts
as a denoiser rather than a copier; nothing here shows it does so in general.

## 3. The resistance gradient

Start from a 24-cell table that factors as `f(cluster)` and break exactly **k** cells so they
depend on the `(zone, cluster)` pair. Rule text is 552 characters with 6 labels at every k, so
decomposability is the only variable; k=0 reproduces `map_wide` and k=24 reproduces
`conjunction`.

| k | non-decomposable | lift | n |
|---|------------------|------|---|
| 0 | 0 % | **+1.000 +- 0.000** | 3 |
| 4 | 17 % | +0.929 | 1 |
| 8 | 33 % | +0.798 | 1 |
| 16 | 67 % | +0.668 | 1 |
| 24 | 100 % | **+0.187 +- 0.108** | 3 |

Monotonic and smooth: resistance is **graded, not a threshold**. The endpoints are replicated
three times each and anchor the curve; the interior points are single runs and should be read
as a provisional shape rather than as calibrated coefficients. Note also that the two ends
differ in how reproducible they are - k=0 has sd 0.000 while k=24 has sd 0.108 (per-seed
0.129 / 0.339 / 0.120). The hard end is genuinely noisy, and a single seed there would
mislead in either direction.

There is **no poisoning effect**: partial non-decomposability degrades writability smoothly
and never drives lift below the answer prior. (An earlier reading of this data claimed a
sharp cliff with negative lift at k=8. That was taken from a run three probes in and still
ramping; completed, it reached +0.798. Every run here ramps over roughly five probes, and a
mid-run value is not a noisy estimate of the final one - it is a different quantity.)

## 4. An unexplained instability

Two families collapsed on their final probe after long stable stretches: `composition`
1.00 -> 0.14, `precedence` 1.00 -> 0.78. Both are the structures requiring an ordered
sequence of operations; the four purely unary families did not collapse.

Two candidate explanations are ruled out by the logs. It is **not norm accumulation**:
`composition` collapsed at a fast-weight norm of 0.0057, the *lowest* in the set, while
`map_wide` sits at the *highest* (0.0282) and is still at 0.98. It is **not REM pruning**:
the probe is taken before the sleep in the same chunk, so the collapsing probe precedes the
REM that follows it.

What is left is striking on its own terms: `composition` moved from 1.00 to 0.14 while its
norm moved only 0.0049 -> 0.0057. A near-stationary weight change produced a catastrophic
accuracy change - a cliff in weight space rather than a drift. Unexplained, and it should be
understood before any of this is built on.

## Caveats

One model (Qwen2.5-3B-Instruct), one slice geometry, 600 puzzles per family, single-rule
training throughout. The `map_wide` / `conjunction` contrast and the dose-response endpoints
are the load-bearing measurements and are replicated; the rest of the family ordering is a
coarse first map. Whether the decomposability barrier survives multi-epoch gradient
fine-tuning is **untested here** - though Part D's fine-tune-like control (decay off, clamp
off, 4x replay, ~688 writes) left its exception branch at 0.0078, which is suggestive.

## Reproduce

```
.venv\Scripts\python probe_families.py                          # gate 1 per family, ~10 min
powershell -ExecutionPolicy Bypass -File run_ladder.ps1         # 7 families, ~1 h, SEQUENTIAL
powershell -ExecutionPolicy Bypass -File run_doseresp.ps1       # 9 jobs, ~75 min, SEQUENTIAL
.venv\Scripts\python analyse_ladder.py
.venv\Scripts\python analyse_doseresp.py
.venv\Scripts\python probe_hidden_xseed.py                      # needs delta.pt in each run
```

---

# Part 0: the hand-over, measured per puzzle (3B, three seeds, 2026-09-18)

`results/suite3b_v2/`, driver `run_suite3b.sh`, analysis `hdelta/handover.py`. Three seeds
of the full mechanism set at probe cadence 20 with all 48 held-out puzzles scored on every
path, plus five ablations at cadence 40. Every probe records, per puzzle, whether the
**deliberate** path solved it (rule stated, four worked examples, steps, *generated by the
model*), whether the **compiled** path solved it (input only, greedy), whether the
open-ended **instant** answer was right, and which rule the compiled answer followed.

## The ordering claim

Averages cannot show that the same puzzle is first solved with the crutch and later
without it, so the claim is tested three ways. Mean ± sd over seeds; the per-puzzle lag is
the pre-registered sign test, anchored on the model's own generated reasoning.

| phase (rule) | deliberate t50 | compiled t50 | compiled accuracy when deliberation reaches 0.5 | deliberate accuracy when the reflex first shows life (0.1) | pooled median lag [CI] | pooled sign test |
|---|---|---|---|---|---|---|
| 0 (A, first learning) | 100 ± 20 | 153 ± 23 | **0.10 ± 0.12** | **0.61 ± 0.14** | 40 [40, 60] (n=142) | 96 + / 36 − , p < 1e-4 |
| 1 (B, after the switch) | 147 ± 31 | 160 ± 20 | 0.47 ± 0.31 | 0.41 ± 0.14 | 20 [0, 40] (n=143) | 82 + / 51 − , p = 0.0045 |
| 2 (A again, savings) | 100 ± 40 | 127 ± 42 | 0.38 ± 0.38 | 0.37 ± 0.14 | 40 [20, 40] (n=123) | 79 + / 21 − , p < 1e-4 |

The clearest statement is the third column for phase 0: **at the moment the model can
execute the procedure on half the puzzles with the crutch, it can do essentially none of
them without it** (0.10 ± 0.12). The fourth column is the converse: when the reflex first
shows any sign of life, deliberation is already at 0.61 ± 0.14. The compiled path is
exactly 0.00 at the start of every phase in every seed.

The figure that shows the mechanism is the joint-state occupancy panel of
`full_s*/handover.png`: puzzles move from *neither* through *deliberate-only* into *both*.
In the 135M validation run the same panel shows mass going straight from *neither* to
*compiled-only*, with no deliberate-only phase, because that model's slow path never
worked. That contrast is the control for the whole claim.

## Honest reading of the statistics

* **Per seed, the sign test is significant in 5 of 9 phase-seeds** (p = 0.000, 0.003,
  0.000, 0.052, 0.383, 0.035, 0.011, 0.111, 0.000). All nine median lags are ≥ 0. The
  pooled test is significant in all three phases, but pooling treats puzzles within a seed
  as independent when they share the same fast weights, so the pooled p-values are
  anti-conservative. The mean ± sd across seeds is the conservative summary.
* **The effect is strongest on first acquisition and weakest on re-acquisition.** Phase 0
  (learning rule A from scratch) gives lead 53 ± 42 puzzles; phase 1 gives 13 ± 23. By
  phase 1 the fast weights already carry general digit-arithmetic improvements from phase
  0, so the reflex has less ground to make up. In phase 2 the reflex sometimes *leads*
  deliberation, which is the savings effect, not a hand-over.
* **Consolidation supply is the most stable quantity in the study.** The compiled path
  reaches 0.5 after 47 ± 18, 50 ± 6 and 43 ± 13 verified self-generated loops in the three
  phases. Roughly 45 to 50 verified solutions buy a reflex, regardless of rule or seed.
* **Data quality**: generated and teacher-forced scoring of the compiled path agree on
  0.963 to 0.987 of puzzle-probes.

## The cadence check, and what it invalidates

Re-analysing the cadence-20 full runs on a synthetic cadence-40 grid (the ablations' grid)
changes the population crossings by at most one probe interval, but destroys the
per-puzzle lag statistic: phase 1's median lag falls from 60 to 0 (p 0.003 → 0.215) and
phase 2's from 50 to 0 (p 0.000 → 0.500), because lags of 20 to 60 puzzles quantise to
zero. Phase 0 survives (80, p < 1e-4).

Consequence for the tables below: **per-puzzle lag and sign-test values from the
cadence-40 ablations are not comparable to the cadence-20 full runs**, and are reported
only for completeness. Cross-run comparison uses the population crossings, the two onset
statistics and the occupancy, all of which are robust to cadence.

## The frozen control calibrates the acquisition criterion

`frozen` keeps the h-delta modules in place but never writes to them, and runs the same
1000-puzzle stream with the same self-generation. Its curves are flat to the digit: the
deliberate path reads 0.03 and the compiled path 0.00 at **every one of the eleven probes
in every phase**, because greedy decoding through unchanged weights is deterministic. It
acquires 1 puzzle of 32 on the deliberate path (the single puzzle it can genuinely solve,
solved at every probe) and 0 of 32 on the compiled path, and no curve ever crosses 0.5.

So the acquisition criterion produces no spurious acquisitions and the probe pipeline
contributes no drift of its own. Every rise in every other run is caused by the fast
weights.

## The mechanisms do not create the hand-over; they make it survivable

Across the seven learning configurations (three full seeds plus four ablations, 21
phase-runs), the deliberate path's lead is **never negative**, and positive in 16 of 21.
Restricting to the phases where a rule is met for the first time (phase 0, all seven
configurations):

| | leads by | compiled accuracy when deliberation reaches 0.5 |
|---|---|---|
| range over the seven configurations | 20 to 120 puzzles | 0.00 to 0.23 |
| mean | 80 | 0.06 |

The ordering therefore does not depend on decay, on the homeostatic clamp or on REM
sleep. It is a property of the excitation-then-consolidation protocol itself: the reflex
is built from verified self-generated solutions, so it cannot precede them. What the
forgetting mechanisms decide is whether the model that acquires the reflex is still a
language model afterwards.

| run | phase-0 lead | compiled acc at delib 0.5 (phase 0) | perplexity worst / final | cloze min / final | max fast-weight ratio | noise response, final |
|---|---|---|---|---|---|---|
| full (3 seeds) | 100 / 40 / 20 | 0.00 / 0.06 / 0.23 | +12 to +15 % / +7 to +8 % | 0.60 to 0.67 / 0.70 | 0.028 to 0.029 | ~1e-3 |
| no_rem | 120 | 0.00 | +12.1 % / +4.2 % | 0.63 / 0.70 | 0.030 | 2.6e-3 |
| no_decay | 80 | 0.06 | **+19.0 % / +10.7 %** | **0.57 / 0.60** | 0.030 (pinned at budget) | 1.6e-3 |
| no_homeo | 120 | 0.00 | +12.3 % / +5.7 % | 0.67 / 0.70 | 0.030 | 9.0e-4 |
| none | 80 | 0.06 | **+97.0 % / +85.3 %** | **0.27 / 0.33** | **0.065, still growing** | **2.5e-2** |
| frozen | never learns | n/a | +0.0 % / +0.0 % | 0.73 / 0.73 | 0.000 | 0 |

* **`none` learns fastest and shows the ordering most strongly** (deliberation is at 0.78
  to 0.94 when the reflex first stirs, the highest of any configuration) while nearly
  doubling perplexity and cutting factual cloze accuracy from 0.73 to 0.33. It is
  task-specific overfitting of a model that is being destroyed underneath the task.
* **`no_decay` is the worst of the configurations that still work**: perplexity +19 %
  worst and +10.7 % final, cloze down to 0.57, fast weights pinned at the budget. Without
  decay, capacity is never recycled and the clamp rescales every later update.
* **`no_homeo` is indistinguishable from `full`** at this learning rate; the budget never
  binds when decay is on. It is a safety rail, not a daily constraint.
* **`no_rem` matches `full` on the task and on perplexity** (final +4.2 % versus +6.6 to
  +8.1 % across the full seeds, i.e. within seed spread at n=1) but leaves the modules
  three times more noise-reactive (2.6e-3 versus ~1e-3). Sleep buys rigidity hygiene, not
  task performance and not unlearning.

Generated tables: `results/suite3b_v2/suite_handover.md` (per-phase hand-over statistics
at native cadence, the synthetic cadence-40 view, and the seed aggregation),
`suite_summary.md` (mechanism table plus hand-over summary),
`handover_seeds_full.md`. Per-run JSON: `<run>/handover.json`, `<run>/summary.json`.
Figures: `<run>/handover.png`, `ablation_compare.png`.

---

# Part 1: the hand-over, on Qwen2.5-3B-Instruct (2026-09-17)

Config `configs/instructed_3b.yaml`, results in `results/suite3b/`. Frozen bf16 weights
(96.9 %), 15 h-delta modules with fp32 fast weights (95.9 M parameters, 3.1 %). Digit
cipher, rules A = `add 3, then 9 - d` and B = `add 4, then add 3`, stream 400 A / 400 B /
200 A. For every stream puzzle the model **generates its own steps and answer** with the
rule stated and four worked examples in the prompt; a verified answer is written back
as the model's own reasoning and replayed zero-shot into the h-delta groups, a wrong one
gets the correct demonstration. Probes every 40 puzzles, REM sleep every 100.

## The hand-over

| | rule A (first) | rule B (after the switch) | rule A (return) |
|---|---|---|---|
| slow path (rule stated, thinking out loud) reaches 0.5 at | 80 | 120 | 80 |
| compiled path (nothing but the input) reaches 0.5 at | **200** | **160** | **80** |
| compiled path reaches 0.8 at | 200 | 200 | 80 |
| final accuracy, compiled / slow | 0.94 / 0.94 | 0.98 / 0.86 | 1.00 / 0.86 |
| old-rule answers after the switch, peak / gone by | - | 17 % / 80 puzzles | 0 % / - |

The ordering the design spec asks for is now visible and repeatable (`compilation_unhooking.png`):
the model first solves the puzzle by executing the stated procedure step by step (0.52 at
80 puzzles, 0.77 at 120, while the direct path is at 0.00 and 0.06), and only then does
the compiled reflex appear and overtake it (0.42 at 160, 0.92 at 200). The open-ended
probe, which gives the model nothing but "Input: x", goes from 0.00 to 0.95 correct
instant answers on the same timetable (`t_instant` = 200, 200, 80) and the model never
emits a Step line in that setting: the reflex answers directly. On the return to rule A
the reflex reforms in 80 puzzles instead of 200 (savings).

Unhooking is again fast: A's compiled path falls from 0.92 to 0.17 within 40 puzzles of
the switch and to 0.00 by 80, and 17 % of early answers follow the old rule (four times the
135M figure). A striking side effect: A's *instructed* slow path also collapses (0.94 to
0.21) while B is being learned, so the plastic modules interfere with executing a rule
even when it is spelled out in the prompt. It recovers to 0.86 once A returns.

## Base skills and sleep

| metric | baseline | worst | final |
|---|---|---|---|
| perplexity, held-out prose | 7.92 | 8.88 (+12 %, at puzzle 40) | 8.21 (+3.6 %) |
| cloze accuracy (30 items) | 0.73 | 0.67 | 0.70 |

Ten REM sleeps: zero task-accuracy cost at every sleep (mean and max 0.000), perplexity
down 0.04 per sleep on average, noise response cut from 1.9e-3 to 7.2e-4 each time. The
fast weights stayed at 2.3 to 2.9 % of the frozen norms under a 3 % budget.

Caveat: in this run the `free_tokens` metric counts post-answer rambling when the model
answers without a label (fixed in `trim_to_answer` afterwards), so use `free_steps`
(always 0) and `free_acc` for the open-ended probe.

## Ablations on the 3B (`results/suite3b/`, table in `suite_summary.md`)

| run | A: slow -> compiled reach 0.5 at | B: slow -> compiled | A again: compiled at 0.8 | ppl worst / final | cloze min / final | max fast-weight ratio | noise response, final |
|---|---|---|---|---|---|---|---|
| full | 80 -> 200 | 120 -> 160 | 80 | +12 % / +3.6 % | 0.67 / 0.70 | 0.029 | 5e-4 to 8e-4 |
| no_rem | 80 -> 200 | 120 -> 120 | 120 | +12 % / +4.2 % | 0.63 / 0.70 | 0.030 | 1.8e-3 to 2.6e-3 |
| none | 80 -> 160 | 80 -> 120 | 80 | **+97 % / +85 %** | **0.27 / 0.33** | **0.065, growing** | 1.6e-2 to 2.5e-2 |

* **No mechanisms**: the fastest compiler and, on a 3B model, catastrophic for base
  skills: perplexity nearly doubles, cloze falls from 0.73 to 0.33, and while rule B is
  being learned the model's unscaffolded output breaks completely (open-ended accuracy
  0.00 even though the "Answer:"-prompted probe reads 1.00). Habit persistence is not the
  failure mode here; erosion of the base model is.
* **No REM**: identical task dynamics to the full run (the hand-over ordering, the
  timings, the unhooking) and slightly worse base skills (final perplexity +4.2 % versus
  +3.6 %, cloze minimum 0.63 versus 0.67), with the noise response three times higher
  throughout. Sleep is hygiene: it keeps the modules from becoming noise-reactive at zero
  cost to the compiled loops, but decay plus overwriting is what unlearns the old rule.
* The hand-over ordering (slow path first, compiled path 40 to 120 puzzles later) holds in
  all three configurations; it is a property of the excitation-then-consolidation
  protocol, not of the forgetting mechanisms.

## Why 1.5B was not enough: positional drift, not arithmetic

Qwen2.5-1.5B-Instruct scored 0.89 token accuracy but 0.00 exact match on the instructed
digit scratchpad. If its errors were independent at that rate, about a third of 9-symbol
puzzles would come out exactly right (0.89^9 = 0.35); a flat zero across 48 puzzles means
the errors are systematic. Reading its free generations (rule stated, four worked
examples, 48 held-out puzzles) shows what the system is:

| step 1 ("add 3"), scored on the model's own output | |
|---|---|
| input digit does not wrap (0 to 6) / wraps (7 to 9) | 64 % / 66 % |
| by position in the 3-digit string | **100 % / 69 % / 25 %** |
| error signature, (model minus expected) mod 10 | +8 in 37 of 51 errors (i.e. input + 1) |
| step 2 ("9 - d") scored against the model's own step 1, by position | 67 % / 48 % / 52 % |
| answer copies the model's own step 2 | 48 / 48 |
| exact match, generated | 2 / 48 |

Wrap-around is irrelevant. The model applies the stated operator perfectly to the first
symbol, holds it about two thirds of the time on the second, and by the third has mostly
replaced "add 3" with "add 1": the same digit 8 is wrapped correctly at position 2 and
mishandled at position 3 ("7 8 8" -> "0 1 9"). As the generated sequence grows, the
local context of the digits it just emitted overrides the instruction and the model
slides into the most common digit continuation in its pretraining, counting up. The
second operation is unreliable at every position (about 55 %). Copying and the digit
vocabulary are flawless (no out-of-range symbols, unlike the letter task), so this is a
working-memory failure, not a lookup failure.

Two consequences for the testbed. Teacher-forced token accuracy overstates a small
model's grasp of a procedure, because every token is conditioned on a corrected prefix;
free generation is the honest measure of the slow path (64 % and 56 % per digit here,
compounding). And fast-weight plasticity amplifies whatever loop the excitation path
produces: had we run the 1.5B, the h-delta modules would have consolidated the +1
attractor. The 3B's 0.10 exact / 0.95 token baseline was the thin but genuine signal that
consolidation needed, and its per-position decay is mild enough that verified traces
accumulate.

## What it took to get here

* Few-shot induction of the cipher is beyond every frozen model up to 3B (all 0.00 on the
  8-shot scratchpad); the rule had to be *stated*.
* Even with the full letter-mapping table in the prompt, letter lookups fail for every
  model up to 3B (0.00 exact). Digits, where each step is single-digit arithmetic, are
  what small instruct models can execute: 3B reaches 0.95 token accuracy at baseline;
  1.5B reaches 0.89 but cannot sustain the operator across positions (see above).
* A 152k-token vocabulary makes full-sequence logits the dominant cost; scoring only the
  response span cut probe time from 175 s to 22 s.
* Qwen tokenises " 6 5 9" as six tokens; a fixed 5-token generation budget silently zeroed
  the generated direct metric until the budget was derived from the tokenizer.

---

# Part 2: ablation suite on SmolLM2-135M (2026-09-16)

All numbers come from `results/suite/<run>/summary.json`; figures are in the same
folders. Every run uses the same seed, the same held-out probe sets (48 puzzles per
rule) and the same stream: 400 puzzles of rule A, 400 of rule B, 200 of rule A again.
Rule A is `shift 3` then `mirror` (compiles to the reflection x -> 6 - x over a 10-letter
alphabet), rule B is `shift 4` then `shift 3` (compiles to the shift x -> x + 7); the two
rules never agree on any letter, so an answer that follows the old rule is unambiguous.

Frozen: 96.05 % of the 134.5 M parameters. Plastic: 15 h-delta modules (q/k/v/o
projections and MLP down-projections of the top three layers), 5.31 M parameters.

## Headline: the full mechanism set does what Phase 3 asks for

| measurement | rule A (first) | rule B (after the switch) | rule A (return) |
|---|---|---|---|
| puzzles until zero-shot direct accuracy >= 0.8 | 320 | 260 | 0.75 reached at 200 (phase ended) |
| best direct accuracy in phase | 0.85 | 0.98 | 0.75 |
| old-rule answers, peak after the switch | - | 4.2 % | 2.1 % |
| old-rule answers below 10 % after | - | 40 puzzles | 40 puzzles |
| previous rule's residual direct accuracy at phase end | - | 0.00 | 0.00 |

Base language skills over the whole 1000-puzzle stream (ten REM sleeps):

| metric | baseline | worst | final |
|---|---|---|---|
| perplexity, 6 k tokens of held-out prose | 17.94 | 18.66 (+4.0 %) | 18.14 (+1.1 %) |
| factual cloze top-1 (30 items) | 0.73 | 0.63 | 0.70 |

REM sleep: mean task-accuracy cost per sleep 0.008 (worst 0.062), mean perplexity
change per sleep -0.12 (sleep *improves* the language model), noise-response ratio
cut from about 1e-3 to about 2e-4 at every sleep, regrowing between sleeps (the
sawtooth in `fast_weight_health.png`). The fast-weight energy settled at 3.1 % and
2.3 % of the frozen norms for the relational and procedural groups, below the 4 %
budget.

**Cognitive compilation** shows as the zero-shot direct path rising from 0.02 at
puzzle 100 to 0.42 at 200 and 0.79 at 400: a sigmoid, not a linear ramp. **Unhooking**
shows as that same curve collapsing to 0.07 within 20 puzzles of the switch and the
old rule's answers never exceeding 4 % of probes, while rule B compiles on the same
timetable A did. **Savings**: on the return to A the model reaches 0.75 in 200
puzzles where it originally needed 300.

## Ablations: what each forgetting mechanism buys

| run | A: t_compile | B: t_compile | A again: t_compile | old-rule peak (B / A again) | max fast-weight ratio | noise response, final | ppl drift, worst | cloze, min |
|---|---|---|---|---|---|---|---|---|
| full | 320 | 260 | (0.75 @ 200) | 4 % / 2 % | 0.031 | 2.8e-4 | +4.0 % | 0.63 |
| no_rem | 260 | 280 | (0.56 @ 200) | 4 % / 2 % | 0.030 | 3.0e-3 | +4.3 % | 0.63 |
| no_homeo | 260 | 260 | (0.60 @ 200) | 4 % / 0 % | 0.032 | 2.9e-4 | +4.0 % | 0.63 |
| no_decay | 260 | **360** | 140 | 4 % / **25 %** | 0.040 (pinned at budget) | 7.1e-4 | +6.8 % | 0.63 |
| none | 220 | 160 | 100 | **52 % / 44 %** | **0.072, still growing** | **1.5e-2** | **+15.5 %** | **0.57** |
| hebbian | never | never | never | 0 % | 0.040 | 1.9e-3 (peak 8e-2) | +11.0 % | 0.67 |
| frozen (control) | never | never | never | 0 % | 0 | 0 | 0 % | 0.73 |

(The generated table with every column is `results/suite/suite_summary.md`; the
comparison figure is `results/suite/ablation_compare.png`.)

What the table says:

* **No forgetting at all (`none`) is the fastest learner and the pathological one.**
  It compiles every rule sooner and reaches 1.00, but the fast weights grow without
  bound (7.2 % of the frozen norm after 1000 puzzles and rising), the noise response
  is 50 times the full run's, perplexity drifts +15 % and cloze falls to 0.57. It is
  also the only configuration with a real habit: after each switch roughly half of the
  probe answers still follow the old rule (52 % and 44 %, versus 4 % and 2 %). This is
  the "algorithmic obsession" regime from the design spec, produced on demand.
* **Passive decay is what keeps the modules fluid under a budget.** With decay off
  and the budget on (`no_decay`), rule A fills the energy budget, every later update
  is rescaled down by the clamp, and rule B takes 360 puzzles to compile instead of
  260; on the return to A a quarter of early answers still follow B. That is the
  saturation the spec anticipated ("saturated with infinite or conflicting loops").
* **The homeostatic clamp is a safety rail, not a daily constraint, at this learning
  rate.** With decay on, the energy never reaches the 4 % budget, so `no_homeo` is
  indistinguishable from `full`. The pilots show where it bites: at a 2x higher rate the
  clamp binds by puzzle 120 and caps the run near 0.5 accuracy, and without it (`none`)
  the norm runs away. The budget therefore sets the ceiling on how far the base model
  can be bent, and decay sets how quickly capacity is recycled.
* **REM sleep removes interference without touching the task loop.** With sleep off
  (`no_rem`) the task curves are the same, but the noise response never comes down
  (3.0e-3 versus 2.8e-4) and perplexity sits slightly higher throughout. Across the
  ten sleeps in `full`, task accuracy changed by less than one point on average while
  perplexity dropped each time. The anti-Hebbian rule is selective because the
  compiled cipher loop barely fires under structureless input.
* **The pure Hebbian rule, taken literally, learns nothing and is the most rigid.**
  `post = y` on a feed-forward projection only amplifies whatever the frozen layer
  already does. Accuracy stays at 0.00 for all 1000 puzzles, yet the noise response
  spikes to 8e-2 (the highest of any run) before REM prunes it back. Error-driven
  plasticity, not raw Hebbian plasticity, is what makes the h-delta modules learn.

## Things the testbed found that the design did not anticipate

1. **The raw leaky-integrator formula has no usable learning rate on this model.**
   With `post = -dL/dy` and the update applied as written, eta = 0.05 moves the fast
   weights 0.2 % of the frozen norm in 240 puzzles (no learning), eta = 0.5 destroys
   the language model in two chunks (perplexity 7 -> 46 000). A fixed-quantum
   normalisation (`frob`) fills the budget without reducing the loss (a random walk).
   Per-synapse normalisation with running moments (`adam`) is the rule that compiles;
   it is the default. See `results/_sweep`, `results/_pilot`, `results/_pilot2`.
2. **Even a 0.2 % perturbation of the top three layers costs measurable perplexity.**
   The top layers are the most sensitive place to put plasticity from a
   base-skills standpoint; the energy budget must be set in the low single digits of
   percent, not the 10-15 % first tried.
3. **The scratchpad is not a crutch for a 135 M base model.** The strict slow-path
   metric (nine letters correct across three mappings) never clears 0.8 and supplying
   one or two steps never beats supplying none, so compilation depth jumps from
   "unsolved" to 0. The hand-over from a mastered slow path to a compiled reflex needs
   a base model that can actually execute steps in context; the pipeline measures it
   (`icl_scratch_acc`, `partial{k}_acc`, `steps_needed`) but this model cannot show it.
4. **Unhooking is fast, and it is decay plus overwriting, not sleep.** The old loop
   collapses within 20 puzzles in every configuration with decay; `no_rem` unhooks as
   fast as `full`. Sleep's job here is hygiene (interference, rigidity), not unlearning
   the specific old rule.
5. **Habits only appear when both decay and the budget are removed.** In every other
   configuration the model's answers after a switch become *wrong* rather than *old*.
   If habit persistence is the phenomenon of interest, `none` (or a much lower decay)
   is the regime to study.

## Reproduce

```
.venv\Scripts\python run_suite.py --out results/suite          # ~40 min on an RTX 5070 Ti (7 runs)
.venv\Scripts\python run_suite.py --out results/suite --plots-only
```

Single runs take about 8 minutes at the default probe cadence (probes dominate; the
1000-puzzle stream itself is under a minute of GPU time).
