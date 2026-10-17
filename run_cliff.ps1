# Does the composition weight-space cliff reproduce, and what shatters?
#
# Seed 0 of the ladder ran 1.00 for ten probes then collapsed to 0.14 on the last, with the
# fast-weight norm moving only 0.0049 -> 0.0057. Norm accumulation and REM are both ruled
# out by the logs. But runs are NOT bit-reproducible, so the first question is whether there
# is a phenomenon here at all. Three seeds answer that; per-probe singular-value spectra
# answer what shattered if there is.
#
# SEQUENTIAL. ~10 min each.
 = "Continue"
Write-Output "=== composition cliff seed 0 ($(Get-Date -Format HH:mm:ss)) ==="
& ".venv\Scripts\python.exe" run_experiment.py --config configs/comp_cliff_s0.yaml --out results/cliff/s0 2>&1 | Select-String -NotMatch "Loading weights"
Write-Output "=== composition cliff seed 1 ($(Get-Date -Format HH:mm:ss)) ==="
& ".venv\Scripts\python.exe" run_experiment.py --config configs/comp_cliff_s1.yaml --out results/cliff/s1 2>&1 | Select-String -NotMatch "Loading weights"
Write-Output "=== composition cliff seed 2 ($(Get-Date -Format HH:mm:ss)) ==="
& ".venv\Scripts\python.exe" run_experiment.py --config configs/comp_cliff_s2.yaml --out results/cliff/s2 2>&1 | Select-String -NotMatch "Loading weights"
Write-Output "CLIFF RUNS COMPLETE"
