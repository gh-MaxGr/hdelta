# Seed-variance rig: is the friction coefficient a measurement or a draw?
#
# Observed so far, each n=2 and each SAME-seed (bf16 noise only):
#   W(conjunction, 0.8)  1370 vs 906   1.5x swing
#   W(composition, 0.5)   100 vs  52   2.0x swing
#   W(precedence,  0.8)   363 vs 363   exact
#
# Two runs cannot tell a stable coefficient from a lucky pair, and the tempting story -
# that sequential chaining is initialisation-sensitive while adversarial boundaries need a
# deterministic carve-out - is the same n=2 overreach as the windowing errors, moved to the
# variance axis. These runs vary the SEED (probe set + stream order), giving n=3 per family.
#
# SEQUENTIAL. ~28 min each, ~2.8 h total.
$ErrorActionPreference = "Continue"
Write-Output "=== conjunction seed 1 ($(Get-Date -Format HH:mm:ss)) ==="
& ".venv\Scripts\python.exe" run_experiment.py --config configs/var_conjunction_s1.yaml --out results/variance/conjunction_s1 2>&1 | Select-String -NotMatch "Loading weights"
Write-Output "=== conjunction seed 2 ($(Get-Date -Format HH:mm:ss)) ==="
& ".venv\Scripts\python.exe" run_experiment.py --config configs/var_conjunction_s2.yaml --out results/variance/conjunction_s2 2>&1 | Select-String -NotMatch "Loading weights"
Write-Output "=== precedence seed 1 ($(Get-Date -Format HH:mm:ss)) ==="
& ".venv\Scripts\python.exe" run_experiment.py --config configs/var_precedence_s1.yaml --out results/variance/precedence_s1 2>&1 | Select-String -NotMatch "Loading weights"
Write-Output "=== precedence seed 2 ($(Get-Date -Format HH:mm:ss)) ==="
& ".venv\Scripts\python.exe" run_experiment.py --config configs/var_precedence_s2.yaml --out results/variance/precedence_s2 2>&1 | Select-String -NotMatch "Loading weights"
Write-Output "=== composition seed 1 ($(Get-Date -Format HH:mm:ss)) ==="
& ".venv\Scripts\python.exe" run_experiment.py --config configs/var_composition_s1.yaml --out results/variance/composition_s1 2>&1 | Select-String -NotMatch "Loading weights"
Write-Output "=== composition seed 2 ($(Get-Date -Format HH:mm:ss)) ==="
& ".venv\Scripts\python.exe" run_experiment.py --config configs/var_composition_s2.yaml --out results/variance/composition_s2 2>&1 | Select-String -NotMatch "Loading weights"
Write-Output "VARIANCE RIG COMPLETE"
