# Locus test, round 2.
#
# Round 1: conjunction lift +0.056 (attention 84.9M) vs +0.404 (MLP 90.2M, matched params).
# But the MLP arm was still climbing at 600 puzzles while attention plateaued by t=200, so
# the comparison was not at matched convergence. These arms fix that and add the
# over-budget 9-layer MLP arm that tripped the 5% plastic guard.
#
# SEQUENTIAL.
 = "Continue"
Write-Output "=== conjunction_mlp9: 9 MLP layers, 202.9M, max_fraction raised to 8% ($(Get-Date -Format HH:mm:ss)) ==="
& ".venv\Scripts\python.exe" run_experiment.py --config configs/conjunction_mlp9.yaml --out results/locus/conjunction_mlp9 2>&1 | Select-String -NotMatch "Loading weights"
Write-Output "=== conjunction_mlp4_long: matched params, 1800 puzzles ($(Get-Date -Format HH:mm:ss)) ==="
& ".venv\Scripts\python.exe" run_experiment.py --config configs/conjunction_mlp4_long.yaml --out results/locus/conjunction_mlp4_long 2>&1 | Select-String -NotMatch "Loading weights"
Write-Output "=== conjunction_attn_long: attention, 1800 puzzles - equal run length ($(Get-Date -Format HH:mm:ss)) ==="
& ".venv\Scripts\python.exe" run_experiment.py --config configs/conjunction_attn_long.yaml --out results/locus/conjunction_attn_long 2>&1 | Select-String -NotMatch "Loading weights"
Write-Output "LOCUS ROUND 2 COMPLETE"
