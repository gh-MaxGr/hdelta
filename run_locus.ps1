# Locus test: is the decomposability barrier about WHERE we write, not what we write?
#
# probe_decodable.py showed the conjunction is linearly decodable at 1.000 across the whole
# current plastic slice - so the feature exists where we write and the barrier is not
# availability. Remaining hypothesis: attention q/k/v/o routes and mixes; it cannot act as a
# general function approximator mapping a joint feature onto logits. MLPs can.
#
# Three arms so the result cannot be explained by parameter count alone:
#   conjunction_mlp4  90.2M  matched to the 84.9M attention slice
#   conjunction_mlp9 202.9M  2.4x params, tests whether more capacity helps
#   map_mlp4          90.2M  control - MLP must compile the EASY family or the arm is void
#
# SEQUENTIAL. ~8 min each.
 = Continue
Write-Output "=== conjunction_mlp4 ($(Get-Date -Format HH:mm:ss)) ==="
& ".venv\Scripts\python.exe" run_experiment.py --config configs/conjunction_mlp4.yaml --out results/locus/conjunction_mlp4 2>&1 | Select-String -NotMatch "Loading weights"
Write-Output "=== conjunction_mlp9 ($(Get-Date -Format HH:mm:ss)) ==="
& ".venv\Scripts\python.exe" run_experiment.py --config configs/conjunction_mlp9.yaml --out results/locus/conjunction_mlp9 2>&1 | Select-String -NotMatch "Loading weights"
Write-Output "=== map_mlp4 ($(Get-Date -Format HH:mm:ss)) ==="
& ".venv\Scripts\python.exe" run_experiment.py --config configs/map_mlp4.yaml --out results/locus/map_mlp4 2>&1 | Select-String -NotMatch "Loading weights"
Write-Output "LOCUS TEST COMPLETE"
