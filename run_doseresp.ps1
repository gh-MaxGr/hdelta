# Writability dose-response. How much non-decomposability does it take to block a write?
#
# k of 24 table cells depend on the (zone, cluster) PAIR; the rest factor as f(cluster).
# k=0 reproduces map_wide (+0.997), k=24 reproduces conjunction (+0.056). Rule text is
# 552 chars and 6 labels at every k, so decomposability is the only variable. The last
# four jobs replicate the endpoints at two further run seeds, because the whole
# writability claim rests on that contrast and one seed is not enough for it.
#
# SEQUENTIAL. ~8 min per job, ~75 min total.
$ErrorActionPreference = "Continue"
Write-Output "=== k=0 seed=0 ($(Get-Date -Format HH:mm:ss)) ==="
& ".venv\Scripts\python.exe" run_experiment.py --config configs/jk_0_s0.yaml --out results/doseresp/k0_s0 2>&1 | Select-String -NotMatch "Loading weights"
Write-Output "=== k=4 seed=0 ($(Get-Date -Format HH:mm:ss)) ==="
& ".venv\Scripts\python.exe" run_experiment.py --config configs/jk_4_s0.yaml --out results/doseresp/k4_s0 2>&1 | Select-String -NotMatch "Loading weights"
Write-Output "=== k=8 seed=0 ($(Get-Date -Format HH:mm:ss)) ==="
& ".venv\Scripts\python.exe" run_experiment.py --config configs/jk_8_s0.yaml --out results/doseresp/k8_s0 2>&1 | Select-String -NotMatch "Loading weights"
Write-Output "=== k=16 seed=0 ($(Get-Date -Format HH:mm:ss)) ==="
& ".venv\Scripts\python.exe" run_experiment.py --config configs/jk_16_s0.yaml --out results/doseresp/k16_s0 2>&1 | Select-String -NotMatch "Loading weights"
Write-Output "=== k=24 seed=0 ($(Get-Date -Format HH:mm:ss)) ==="
& ".venv\Scripts\python.exe" run_experiment.py --config configs/jk_24_s0.yaml --out results/doseresp/k24_s0 2>&1 | Select-String -NotMatch "Loading weights"
Write-Output "=== k=0 seed=1 ($(Get-Date -Format HH:mm:ss)) ==="
& ".venv\Scripts\python.exe" run_experiment.py --config configs/jk_0_s1.yaml --out results/doseresp/k0_s1 2>&1 | Select-String -NotMatch "Loading weights"
Write-Output "=== k=0 seed=2 ($(Get-Date -Format HH:mm:ss)) ==="
& ".venv\Scripts\python.exe" run_experiment.py --config configs/jk_0_s2.yaml --out results/doseresp/k0_s2 2>&1 | Select-String -NotMatch "Loading weights"
Write-Output "=== k=24 seed=1 ($(Get-Date -Format HH:mm:ss)) ==="
& ".venv\Scripts\python.exe" run_experiment.py --config configs/jk_24_s1.yaml --out results/doseresp/k24_s1 2>&1 | Select-String -NotMatch "Loading weights"
Write-Output "=== k=24 seed=2 ($(Get-Date -Format HH:mm:ss)) ==="
& ".venv\Scripts\python.exe" run_experiment.py --config configs/jk_24_s2.yaml --out results/doseresp/k24_s2 2>&1 | Select-String -NotMatch "Loading weights"
Write-Output "DOSE RESPONSE COMPLETE"
