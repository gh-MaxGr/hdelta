# MLP-locus variance: is the MLP advantage at high theta real, or one lucky run?
#
#   attention sustained theta=0.95 in 1 of 3 runs (1403 writes)
#   MLP       sustained theta=0.95 in 1 of 1 runs ( 759 writes)
#
# 1-of-1 is not a reliability estimate. These make it n=3 at the MLP locus, which is the
# minimum for comparing YIELD rather than a single draw.
#
# Waits on a marker FILE written by watch_variance.ps1. The previous chain grepped a log
# path that had been mangled by escaping - "results\variance.log" lost the \v and became
# "resultsariance.log" - so the wait fell through and three 3B runs collided on a 16 GB card.
# Forward slashes throughout: PowerShell accepts them and they survive every layer of quoting.
$ErrorActionPreference = "Continue"
$marker = "results/variance.DONE"
Write-Output "waiting for $marker ..."
while (-not (Test-Path $marker)) { Start-Sleep -Seconds 60 }
Write-Output "marker seen, starting MLP variance runs"

Write-Output "=== conjunction MLP seed 1 ($(Get-Date -Format HH:mm:ss)) ==="
& ".venv/Scripts/python.exe" run_experiment.py --config configs/mlpvar_s1.yaml `
    --out results/variance/conj_mlp_s1 2>&1 | Select-String -NotMatch "Loading weights"

Write-Output "=== conjunction MLP seed 2 ($(Get-Date -Format HH:mm:ss)) ==="
& ".venv/Scripts/python.exe" run_experiment.py --config configs/mlpvar_s2.yaml `
    --out results/variance/conj_mlp_s2 2>&1 | Select-String -NotMatch "Loading weights"

Write-Output "MLP VARIANCE COMPLETE"
