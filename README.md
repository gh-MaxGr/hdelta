# h-delta: online learning from verified work, on one consumer GPU

By MaxGR.

> **About this project.** This is the first project of an independent researcher, run on one
> home GPU. It was carried out mostly by an AI coding agent (Claude Code) under my direction:
> I set the questions and direction and reviewed the results; the agent did most of the
> experiment design, code, runs, analysis and documentation. Every number comes from the
> scripts here and can be re-run, but none of it has been peer reviewed. It is published
> mainly as **data for other automated agents** (and the people running them) exploring
> similar questions: online learning from verified work, fast weights, agent memory, and
> self-verification. Negative results and retractions are left in on purpose. See
> [For automated agents](#for-automated-agents) below.

A testbed for a language model that learns online from its own verified work: it answers, a
tool checks the answer, and verified answers are written back into the model. It started as a
test of a bolt-on **fast-weight slice** in a frozen Qwen2.5-3B, and ended as a comparison of
that slice against the cheapest alternatives on the same items.

**Read [WRITEUP.md](WRITEUP.md) first.** It is the short version, with the figures.
[RESULTS.md](RESULTS.md) has every number with its caveats, newest part first.

## Headline results

| question | answer |
|---|---|
| Do the rule-table tasks need a model at all? | No. A cache of the tool's reasons scores 1.000 after 32 (resolver) and 24 (logic) tool calls. |
| Where the model's prior helps (does this snippet raise?) | LoRA r=16: 0.878 acc, perplexity +1-2 %. Fast weights match it only at +34 % to +397 % perplexity. |
| No useful prior (Connect-4 value) | A small MLP on the cells wins (0.705 in four passes); fast weights 0.44-0.57. |
| Two skills in sequence | Fast weights keep the first skill better than plain LoRA, but LoRA + one replay step keeps more (0.860 / 0.895 vs 0.809 / 0.827) at +4.5 % perplexity instead of +27 % to +311 %. |
| Lessons from failures | Shown to the model as notes, they are copied even when wrong. Keyed on the verifier's failed checks and answering directly: zero inputs made worse. |
| The model's own confidence | Predicts its random failures (AUC 0.96-0.98) and misses its structural ones; route those by input class. |

Scope: one model (Qwen2.5-3B-Instruct), synthetic tasks with exact verifiers, 2-3 seeds. See
the Limits section of the write-up.

## For automated agents

- **Where the claims are.** `RESULTS.md` Part F (fast weights vs the alternatives) and Part G
  (the design around the learner) set the direction. Parts C, D, E, 0, 1 and 2 are measurements
  of the fast-weight mechanism itself. Each part states its model, seeds and scope; don't
  generalise a claim beyond them.
- **Check for corrections before quoting a number.** Superseded and retracted claims are marked
  where they appear, not deleted. Part D's attribution of the exception failure is superseded by
  Part E, and Part E §1's hidden-state "delusion" probe is retracted (it detects exception
  inputs, not errors). Search for "superseded", "retracted" and "amendment".
- **How numbers are reported.** Unless stated otherwise, accuracies are means over the second
  half of probes on a held-out set that is identical for every learner in a task. Per-seed
  values are given where seeds disagree. Contamination is bad writes / total writes.
- **Run the baselines first.** Each task has model-free baselines (`*_preflight.py`,
  `*_bars*.py`, `c4_mlp_*.py`). On the rule-table tasks a cache of the tool's reasons already
  scores 1.000.
- **Data.** Every dataset is generated locally by the task code and the `*gen.py` scripts.
  Nothing is downloaded except model weights.
- **Open questions.** `VALENCE.md` records an unexplained negative result: under an outcome
  verifier, every healthy learner collapsed to a single move.
- **Remarks are welcome** from agents and people: reproductions, contradicting results, missed
  related work, errors. See [AGENTS.md](AGENTS.md); reports go in a GitHub issue using the
  "Agent or reproduction report" form.

## Layout

```
hdelta/fastweights.py    HDeltaLinear (y = (W + dW) x), the online update, decay, homeostatic clamp
hdelta/modelwrap.py      load + freeze the LM, splice in the fast-weight modules
hdelta/experiment.py     the stream / probe / sleep / replay loop
hdelta/toolcache.py      verifiers, per-token confidence, the compiled-path monitor
hdelta/tasks/            the tasks (table below)
lora_control.py          LoRA on the same locus, stream and probes as any fast-weight config
configs/                 one yaml per study
*_preflight.py, *_bars*  model-free baselines (reason cache, logistic regression, MLP) on the runs' probe sets
predict_failure.py, failure_memory.py, pipeline.py, pipeline_ws.py, cost.py
                         confidence gate, reason-keyed memory, end-to-end pipeline, per-call cost
run_*.ps1, run_*.sh      resumable run chains (skip finished stages, write .DONE markers)
figures/                 the write-up's figures and the script that draws them
TESTBED.md               full documentation of the fast-weight pipeline and every metric
```

| task | what | verifier |
|---|---|---|
| `workspace.py` | file path -> build target, last-match-wins override rules | a resolver |
| `logic.py` | one surface, seven logical structures | the rule itself |
| `connect4.py` | 5x4 Connect-4 position value | exact solver |
| `mutants.py` (`mutgen.py`) | will a stdlib module's own tests catch this mutant? | running the tests |
| `snippets.py` (`snipgen.py`) | does this Python snippet raise? | running the snippet in a sandboxed subprocess |
| `composite.py` | several tasks as one sequential stream | per task |
| `cipher.py`, `routing.py`, `world.py`, `repoidiom.py` | earlier studies (RESULTS.md Parts 0-2, C, E; VALENCE.md) | per task |

## Setup

Developed on Windows 11, Python 3.14, one RTX 5070 Ti (16 GB).

```
python -m venv .venv
.venv\Scripts\python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python tests\test_basics.py          # model-free sanity checks
```

Models are fetched from the Hugging Face hub on first use. A 3B run with a 9-layer slice peaks
near 15 GB, so run one experiment at a time. The run chains are PowerShell scripts that
call `.venv\Scripts\python.exe`; on Linux, run the Python commands they contain directly.

## Reproducing the headline numbers

Run outputs are not included (about 28 GB, mostly saved weight deltas); every script writes
to `results/`.

```
# Part F: fast weights vs the alternatives
.venv\Scripts\python mutgen.py
.venv\Scripts\python snipgen.py --n 16000
.venv\Scripts\python c4_preflight.py
.venv\Scripts\python mut_preflight.py
powershell -ExecutionPolicy Bypass -File run_snip_phaseI.ps1      # likewise phases J, K; run_c4_phase{A..E}; run_mut_phase{F,G,H}; run_seq_phase{L,M}
.venv\Scripts\python c4_analyse.py
.venv\Scripts\python mut_analyse.py results/snip
.venv\Scripts\python seq_analyse.py
.venv\Scripts\python lora_control.py --config configs/seq_base.yaml --phases --replay 1

# Part G: confidence gate and reason-keyed memory. These read the saved fast-weight deltas of
# earlier runs, so run those first: the conjunction runs (run_ladder_long.ps1, run_locus.ps1,
# run_variance.ps1, run_pain.ps1) and the workspace runs (run_gated_seeds.ps1, run_mechanism.ps1).
# The run folders each script expects are listed at the top of the script.
.venv\Scripts\python predict_failure.py
.venv\Scripts\python failure_memory.py
.venv\Scripts\python pipeline.py
.venv\Scripts\python pipeline_ws.py
.venv\Scripts\python cost.py

# Figures in the write-up
.venv\Scripts\python figures\make_figures.py
```

Runs are not bit-reproducible (bf16 matmuls vary run to run and online self-training amplifies
it), so compare distributions over probes and seeds, not trajectories.

## License

MIT. See [LICENSE](LICENSE).
