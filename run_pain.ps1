# Pain on a known baseline: ONE variable against the best-characterised result in this repo.
#
# The template is results/variance/conjunction_s1/config.json, the exact config the baseline
# ran with. The four baseline runs differ from each other ONLY in `seed` (checked), so these
# three differ from the baseline in exactly one field: stream.pain_lambda = 1.0.
#
# What pain_lambda does: on the existing `teach` path, when the model's own answer is wrong,
# it ALSO writes that answer with weight -1. The teach demonstration, verified consolidation,
# decay, REM, homeostasis, locus and every hyperparameter are untouched. With pain_lambda = 0
# no negative item is ever created and every chunk takes the original code path - the loss and
# its gradient were checked identical to the baseline's, max diff 0.00e+00.
#
# Baseline to compare against (n=4, attention 9L, 1800 puzzles):
#     W@0.5   369-779   mean 616
#     W@0.8   889-1370  mean 1048
#     W@0.95  1232/1403, 2 of 4 never reached it
#     ppl     +0.46 % final
#
#   stage                       ~mins   cumulative
#   1  pain conjunction s0        28        28
#   2  pain conjunction s1        28        56
#   3  pain conjunction s2        28        84
$ErrorActionPreference = "Continue"
$tpl = "results/variance/conjunction_s1/config.json"

function Invoke-Stage($n, $seed) {
    $out = "results/pain/conjunction_s$seed"
    Write-Output "=== stage $n : pain_lambda=1.0, seed $seed ($(Get-Date -Format HH:mm:ss)) ==="
    if (Test-Path "$out/summary.json") { Write-Output "    already complete, skipping"; return }
    & ".venv/Scripts/python.exe" run_experiment.py --config $tpl --out $out `
        --set "seed=$seed" --set "stream.pain_lambda=1.0" 2>&1 | Select-String -NotMatch "Loading weights"
    Write-Output "    stage $n done at $(Get-Date -Format HH:mm:ss)"
}

Invoke-Stage 1 0
Invoke-Stage 2 1
Invoke-Stage 3 2

New-Item -ItemType File -Force -Path "results/pain.DONE" | Out-Null
Write-Output "PAIN CHAIN COMPLETE at $(Get-Date -Format HH:mm:ss)"
