# h-delta testbed: compartmentalised fast-weight plasticity in a frozen LM

> **Status (2026-09-24).** The research is complete. Tested against the cheapest alternative
> on tasks a reason cache cannot solve, the fast-weight slice never won: a tiny learner did
> better where the model has no useful prior, and online LoRA (plus replay, for sequential
> skills) matched its skill without damaging the base model (`RESULTS.md` Part F). The parts
> worth building on are the design around the learner - a reason-keyed cache that answers
> directly, a confidence gate, the tool as teacher and verifier (Part G) - with LoRA as the
> plastic part (`lora_control.py`). This README still documents the fast-weight pipeline
> accurately; the task and script map at the end covers everything added since.
>
> "The design spec", here and in `RESULTS.md`, is the project's original planning document,
> which is not included.

An experimental pipeline for the "self-upgrade sandbox": a small open-weights
language model whose weights are frozen except for a handful of **h-delta
fast-weight modules** that learn online, forget in three controlled ways, and are
measured for *cognitive compilation* (slow, excitation-based reasoning becoming an
instant structural reflex), *unhooking* (the old reflex dissolving when the rules
change) and *catastrophic forgetting* of base language skills.

```
python run_experiment.py --config configs/base.yaml --out results/full      # one run
python run_suite.py --out results/suite                                      # all ablations + figures
python run_suite.py --out results/suite --only full no_rem --plots-only      # re-plot
```

Setup (Windows, Python 3.14, RTX 5070 Ti):

```
python -m venv .venv
.venv\Scripts\python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch
.venv\Scripts\python -m pip install -r requirements.txt
```

The base model (`HuggingFaceTB/SmolLM2-135M`) is fetched from the Hugging Face
hub on first use. Any HF causal LM with `nn.Linear` projections works (Llama,
Qwen, GPT-NeoX families); set `model.name` and the module patterns.

## Phase 1: the sandbox

| Spec item | Implementation |
|---|---|
| Freeze >= 95 % of the model | every parameter is frozen; `plastic.max_fraction` (0.05) is enforced at install time. The default puts 5.3 M of 134.5 M parameters (3.9 %) in h-delta modules. |
| Replace projection matrices with fast-weight modules | `HDeltaLinear` (`hdelta/fastweights.py`) wraps a frozen `nn.Linear`: `y = (W + dW) x + b`. `W` never changes, `dW` is the h-delta. |
| Functionally specialised groups | `hdeltaH` (relational: attention q/k/v/o of the top 3 layers) and `hdeltaG` (procedural: MLP down-projections of the top 3 layers). Each group has its own learning rate, decay, budget and REM rate. Add groups freely in `configs/base.yaml`. |
| Procedural logic puzzle | `hdelta/tasks/cipher.py`: a multi-step letter cipher whose rule set (an ordered list of primitives) changes between phases. `hdelta/tasks/routing.py`: routing through a small network whose topology changes. Both expose *scratchpad* ("Step 1 / Step 2 / Answer") and *direct* ("Answer:") formats. |

### The update rule

Every chunk of the stream (default 4 puzzles) triggers one online plasticity step:

```
dW_t = (1 - lambda) dW_{t-1} + eta * (post_t (x) x_t)
```

`post_t` is selectable (`plastic.postsynaptic`):

* `error` (default, "localised error-driven learning"): `post_t = -dL/dy_t`, the error
  signal at the module's output for the next-token loss over the response span. It is
  obtained with autograd restricted to `dW`; nothing else has gradients. Summed over
  the chunk this is exactly the delta rule's outer product.
* `hebbian`: `post_t = y_t`, the module's own output, i.e. the formula exactly as
  written. On a feed-forward projection this rule is self-reinforcing and carries no
  information about the target, so it cannot acquire a new input-to-output mapping.
  It is included as an ablation to demonstrate that.

`plastic.update_norm` sets how the outer-product term is scaled before it enters the
leaky integrator. `adam` (default) normalises per synapse with running first/second
moments of the outer product, `frob` rescales each chunk's update to a fixed fraction
of `||W||_F` (a fixed quantum of synaptic change per experience), `none` uses the raw
outer product. The pilots in `results/_sweep`, `results/_pilot`, `results/_pilot2`
show why the default is `adam`: the raw rule has no usable learning rate on this model
(0.05 barely moves the weights, 0.5 destroys the LM in two chunks), the fixed-quantum
rule random-walks (the energy budget fills without the loss improving), and the
per-synapse rule compiles the task within a few hundred puzzles at +2 % perplexity.

### Exposure

Each puzzle in the stream is experienced twice (`stream.exposure: both`):

1. **Slow path**: two in-context examples plus the scratchpad, the model "thinks out
   loud" token by token. h-delta updates on the step tokens and the answer.
2. **Consolidation**: the same puzzle replayed zero-shot in direct form. This is the
   hand-off in the design spec's item 4: the loop worked out by excitation is passed to
   the h-delta groups to be written structurally.

`exposure: slow` disables consolidation (tests whether direct answering emerges
from scratchpad exposure alone); `exposure: direct` disables the slow path. With
`exposure: direct` and `consolidate: all`, every stream item is written with the TOOL's answer:
tool-as-teacher distillation, used by the Connect-4, mutant and snippet configs.

`stream.replay: k` (default 0, which leaves the original code path untouched) adds k extra
update steps per chunk, each on a batch drawn at random from pairs consolidated earlier in the
stream - revisiting old verified data with no new tool calls. `consolidation_repeats` is
different: it rewrites the current batch.

### The instructed, self-generated variant (`configs/instructed_3b.yaml`)

The few-shot slow path above asks the frozen model to *induce* the cipher from
examples, which no model up to 3B can do (`probe_base.py`). The instructed variant
turns the slow path into *execution* of a stated procedure and makes it the model's
own work, which is what Phase 3 of the design spec describes:

* `stream.rule_text: true`: the prompt states the rule (`Task.describe`, e.g. "Step 1:
  add 3 to each digit, wrapping around after 9 (0->3, ...). Step 2: replace each digit
  d with 9 - d (...)"). Context plus steps is the excitation path; the zero-shot direct
  prompt with no rule text is the structural path.
* `stream.self_generate: true`: for every stream puzzle the model generates its own
  steps and answer with the current plastic network. A verified answer is written back
  as the model's own reasoning; a wrong one gets the correct demonstration
  (`on_wrong: teach`) or nothing (`skip`).
* `stream.consolidate: verified`: only verified answers are replayed zero-shot into the
  h-delta groups (the design spec's "verify, then write structurally").
* `stream.free_probe: true`: an open-ended probe after "Input: x" with nothing else. The
  model decides whether to think out loud. Logged per rule: `free_acc`, `free_steps`
  (Step lines emitted before the answer) and `free_tokens` (tokens spent to reach it);
  `free_rule_*` is the same with the rule stated and the stream's worked examples
  (the model's own slow path as generated). `t_instant` in the summary is when the
  open-ended answer clears the threshold.
* `task.alphabet: digits` swaps the letter alphabet for digits, so the steps are
  single-digit arithmetic (which small instruct models execute far better than letter
  table lookups).
* `model.dtype: bfloat16` keeps the frozen weights in bf16 with fp32 fast weights and
  fp32 Adam moments; scoring computes logits only over the response span
  (`logits_to_keep`), which is what makes 150k-vocabulary models affordable.

Base model for this variant: `Qwen/Qwen2.5-3B-Instruct` (3.1 % plastic). At baseline
its instructed slow path is at 0.95 token accuracy on the digit cipher while the
zero-shot direct path is at 0.

## Phase 2: controlled forgetting

| Mechanism | Where | Knobs |
|---|---|---|
| Passive decay (leaky integrator) | `PlasticityController.decay`: `dW <- (1-lambda)^n dW`, `lambda` per puzzle | `groups.*.decay`, `plastic.decay_enabled` |
| Homeostatic plasticity (limiter) | `HDeltaLinear.homeostatic_clamp`: the Frobenius norm of dW is capped at `budget` times that of W (global) or per output neuron (row = synaptic scaling). Strengthening one loop past the budget proportionally weakens everything else. | `groups.*.budget`, `homeo_mode`, `plastic.homeo_enabled` |
| Simulated REM sleep (active pruning) | `hdelta/rem.py`: every `rem.every` puzzles the model goes offline and receives Gaussian noise embeddings (per-dimension std matched to the embedding table). Each module records its input `x` and its fast-weight response `dW x`. The learning rate flips negative: dW is reduced in proportion to how strongly it responds to the noise. A loop that fires under structureless input is shrunk in proportion to how strongly it fires; selective loops are barely touched. | `rem.eta`, `rem.steps`, `rem.postsynaptic` (`delta` penalises only the fast weight's response, `full` penalises the whole output), `rem.threshold` |

The **noise-response ratio** `||dW x||^2 / ||W x||^2` under noise is the rigidity
gauge; it is logged at every probe and before/after every sleep.

## Phase 3: measuring success

Every `probe_every` puzzles (and immediately before and after each REM sleep) the
model is probed on held-out puzzles of *every* rule set with fast weights frozen:

| Metric (per rule R) | Meaning |
|---|---|
| `R/zero_direct_acc` | exact-match accuracy with no examples and no scratchpad, greedy decode. **The compiled path.** |
| `R/icl_scratch_acc` | exact match over all steps + answer with two in-context examples. **The slow, excitation-based path.** |
| `R/zero_scratch_acc`, `R/icl_direct_acc` | the two intermediate formats |
| `R/partial{k}_acc`, `R/steps_needed` | accuracy when k of the K steps are supplied; `steps_needed` is the smallest k reaching the threshold (0 = compiled, K+1 = unsolved even by copying). **Compilation depth.** |
| `R/rate_S` | fraction of probe answers that equal what rule S would produce. After a switch from A to B, `B/rate_A` is the **habit**: the old loop still firing. |
| `seen/zero_direct_acc` | direct accuracy on the last 48 stream puzzles (memorisation vs generalisation) |
| `lm_ppl`, `cloze_acc` | perplexity on 6 k tokens of a public-domain novel and top-1 accuracy on 30 factual cloze prompts. **Catastrophic-forgetting check.** |
| `fw_norm_ratio_group/*`, `noise_ratio_group/*` | fast-weight energy and rigidity per group |

`hdelta/report.py` derives the headline numbers written to `summary.json` /
`summary.md`:

* **Cognitive compilation**: `t_slow_path` (puzzles until the slow path clears the
  threshold), `t_compile` (until the compiled path does), `compile_lag`, and the
  trajectory of `steps_needed` (K -> 0).
* **Unhooking**: after a rule switch, `oldrule_rate_peak`, `t_dissolve` (old-rule
  answers fall below 10 %), `t_unhook` (new rule compiled), and the previous rule's
  residual direct accuracy.
* **Base skills**: perplexity drift (final and worst) and cloze accuracy.
* **REM**: task-accuracy cost and perplexity change across each sleep, and the
  noise-response ratio before/after.

Figures per run (`hdelta/plots.py`): `compilation_unhooking.png`,
`habit_persistence.png`, `compilation_depth.png`, `base_skills.png`,
`fast_weight_health.png`. The suite adds `ablation_compare.png` and
`suite_summary.md`.

## Ablations (`configs/ablations/`)

`full`, `no_decay`, `no_homeo`, `no_rem`, `none` (all three forgetting mechanisms
off), `hebbian` (postsynaptic term = output), `frozen` (h-delta present, never
written: the control), `routing` (the network-routing task).

## Hand-over tracking (`hdelta/handover.py`)

Mean accuracies cannot show that the *same* puzzle is solved first by deliberation and
later by reflex, so every probe also logs, for each fixed held-out puzzle, whether the
slow path solved it, whether the compiled path solved it, whether the open-ended instant
answer was right, and which rule the compiled answer followed (`per_puzzle` in
`probes.jsonl`), plus the number of verified loops consolidated so far
(`cum_consolidated`). `analyse_run` (called at the end of every run, or
`python -m hdelta.handover <run_dir>`) turns that into `handover.json / .md / .png`:

* population curves with Wilson 95 % intervals; crossing times at 0.5 and 0.8 for each
  path with bootstrap 95 % intervals over puzzles (1000 resamples, reporting the share of
  resamples that never cross); the crossing point `t_H` where the reflex first matches
  deliberation; the lead area between the curves;
* joint-state occupancy per probe: neither / slow-only / both / direct-only. The
  hand-over is mass flowing from slow-only into both;
* per-puzzle acquisition times (correct on 2 consecutive probes), the per-puzzle lag
  compiled minus slow with a bootstrap median and a one-sided sign test of "reflex
  follows deliberation", with censoring counts;
* consolidation supply: verified loops written by the time the compiled path crosses
  0.5 / 0.8, so runs that verify at different rates can be compared per loop;
* unhooking: share of compiled answers still following the old rule (with CI), the
  per-puzzle time the old answer disappears, and the old rule's own slow and compiled
  paths through the switch (interference);
* a data-quality check: agreement between generated and teacher-forced compiled scoring.

`run_suite.py` appends a hand-over table across ablations to `suite_summary.md`.

## Results

`RESULTS.md` has the numbers, newest first: **Part F** compares the fast weights with a reason
cache, tiny learners and LoRA across four regimes and a sequential stream (the direction);
**Part G** the design around the learner (confidence gate, reason-keyed memory, cost); Part C
capacity; Part D the tool-call cache and the self-verification blind spot; Part E writability
(protocol in `WRITEBENCH.md`); Part 0 the three-seed hand-over; Part 1 the first 3B hand-over
(`results/suite3b/`); Part 2 the ablation suite on SmolLM2-135M (`results/suite/`). The
negotiation-world valence study is in `VALENCE.md`.
`probe_base.py <model> ...` checks whether a candidate frozen model can execute the
slow path at all before you spend a run on it (`PROBE_ALPHABET=digits`,
`PROBE_DTYPE=bfloat16`).

## Layout

```
hdelta/fastweights.py   HDeltaLinear, GroupConfig, PlasticityController (decay, homeostasis, anti-Hebbian)
hdelta/modelwrap.py     load + freeze the LM, splice in the modules, enforce the 5 % cap
hdelta/rem.py           noise forward pass, noise-response measurement, REM phase
hdelta/encoding.py      prompt/response tokenisation with exact response spans
hdelta/metrics.py       teacher-forced exact match, greedy generation, perplexity, cloze
hdelta/experiment.py    the stream / probe / REM / replay loop
hdelta/toolcache.py     ToolVerifier (honest / random / systematic / corrupt), CompiledMonitor,
                        per-token confidence (generate_with_confidence)
hdelta/handover.py      per-puzzle hand-over analysis;  hdelta/capacity.py  capacity analysis
hdelta/report.py        summary.json + summary.md;      hdelta/plots.py     figures
configs/                base.yaml, instructed_3b.yaml, ablations/, and one yaml per study
probe_base.py           frozen-model slow-path probe (run before choosing a base model)
data/                   lm_reference.txt (Pride and Prejudice, public domain), cloze.json
```

### Tasks (`hdelta/tasks/`, all built through `make_task`)

| task | what | used in |
|---|---|---|
| `cipher.py`, `routing.py` | changeable multi-step rules | Parts 1, 2, C |
| `logic.py` | one surface, seven logical structures (map ... conjunction, precedence) | Part E, G |
| `workspace.py` | file -> build target with last-match-wins overrides; the tool is a resolver | Parts D, G |
| `world.py` | negotiation world with an outcome verifier (every arm collapsed to one move) | `VALENCE.md` |
| `repoidiom.py` | repo conventions compiled from verified work (works, but a template beats it) | `idiom.py` |
| `connect4.py` | 5x4 Connect-4 value, exact solver as teacher/verifier, held-out split by board | Part F |
| `mutants.py` | will a stdlib module's own tests catch this mutant? (data: `mutgen.py`) | Part F |
| `snippets.py` | does this Python snippet raise? (data: `snipgen.py`, labelled by execution) | Part F |
| `composite.py` | several single-rule tasks as one multi-rule stream (sequential tests) | Part F |

### Scripts added for Parts F and G

| script | purpose |
|---|---|
| `lora_control.py` | LoRA on the same locus, stream and probes as any h-delta config; `--phases` for sequential streams, `--replay k`, base-model health adapter-on |
| `mutgen.py`, `snipgen.py` | generate the mutant and snippet datasets locally (no downloads) |
| `c4_preflight.py`, `mut_preflight.py`, `mut_bars_on_probes.py`, `snip_bars.py`, `c4_mlp_*.py` | model-free bars (reason cache, logreg, MLP), scored on the runs' exact probe sets |
| `c4_analyse.py`, `mut_analyse.py [dir]`, `seq_analyse.py` | read the runs: 2nd-half means, per-class accuracy, AUC, base-model health |
| `predict_failure.py`, `failure_memory.py`, `pipeline.py`, `pipeline_ws.py`, `cost.py` | confidence gate, keyed failure memory, the end-to-end pipeline, per-call cost |
| `probe_delusion_ctrl.py` | the input-structure control that retracted Part E's delusion probe |
| `run_{c4,mut,snip,seq}_phase*.ps1` | resumable GPU chains (skip finished stages, write `.DONE` markers) |

## Notes and caveats

* Updates are applied once per chunk (after the forward pass over that chunk), so the
  fast weights used for puzzle n reflect puzzles 1..n-1: online, causal, but not
  token-by-token inside a chunk. Set `stream.batch_size: 1` for per-puzzle updates.
* Probes never write to the fast weights and do not advance decay: they are
  measurements outside the stream.
* The slow path is teacher-forced (the scratchpad tokens are given). A 135 M model
  cannot invent the cipher steps itself; a larger base model could self-generate the
  scratchpad and consolidate only verified answers.
* With this base model the scratchpad is not a crutch: the strict slow-path metric
  (nine letters over three mappings all correct) never clears the threshold and the
  partial formats are never better than the direct one, so `steps_needed` jumps from
  "unsolved" straight to 0. Cognitive compilation is therefore measured as the rise of
  the zero-shot direct path, not as a hand-over from a mastered slow path. A base model
  that can execute steps in context is needed to see the hand-over itself.
* `error` mode backpropagates through the frozen layers above each module to obtain
  the error signal. Plasticity is localised; the error *signal* is not strictly local.
  Placing modules only in the final layer makes it nearly local.
