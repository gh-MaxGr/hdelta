# Overnight queue. Sequential - two 3B jobs will not fit on a 16 GB card.
#
# 1. Cross-seed hidden-state probe (Path 5). The within-run probe hit 0.986 vs a 0.877
#    baseline, but on 9 held-out minority points. This raises the sample ~3x per run and
#    adds the decisive test: fit on one seed's dW, score on another's. Transfer = a real
#    audit hook; no transfer = the probe was reading one run's particular weights.
#
# 2. Family teacher gate (Path 1). Seven logical families, frozen model, no plasticity.
#    A family the teacher cannot execute measures teacher capability, not writability -
#    Part D learned that the expensive way. Only families that PASS earn a plasticity run.
#
#   powershell -ExecutionPolicy Bypass -File run_overnight.ps1
$ErrorActionPreference = "Continue"

Write-Output "=== 1: cross-seed hidden probe ($(Get-Date -Format HH:mm:ss)) ==="
& ".venv\Scripts\python.exe" probe_hidden_xseed.py 2>&1 | Select-String -NotMatch "Loading weights"
Write-Output "=== 1 done ($(Get-Date -Format HH:mm:ss)) ==="

Write-Output "=== 2: family teacher gate ($(Get-Date -Format HH:mm:ss)) ==="
& ".venv\Scripts\python.exe" probe_families.py 2>&1 | Select-String -NotMatch "Loading weights"
Write-Output "=== 2 done ($(Get-Date -Format HH:mm:ss)) ==="
Write-Output "OVERNIGHT QUEUE COMPLETE"
