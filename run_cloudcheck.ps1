# End-to-end validation of probe_cloud.py before anyone runs it on rented hardware.
#
# Uses the SMALLEST model and the hard tier: cheap, and the hard families are the ones the
# script exists to measure. A 0.5B should fail most of the hard tier - that is the expected
# result and it confirms the gate discriminates rather than rubber-stamping.
#
# Chained on a marker file. A previous version grepped a log path that lost a backslash to
# escaping, the wait fell through, and three 3B runs collided on a 16 GB card.
$ErrorActionPreference = "Continue"
$marker = "results/mlpvar.DONE"
Write-Output "waiting for $marker ..."
while (-not (Test-Path $marker)) { Start-Sleep -Seconds 60 }

Write-Output "=== validating probe_cloud.py on 0.5B, hard tier ($(Get-Date -Format HH:mm:ss)) ==="
& ".venv/Scripts/python.exe" probe_cloud.py `
    --models Qwen/Qwen2.5-0.5B-Instruct `
    --tier hard --n-eval 32 --draws 2 `
    --out results/cloud 2>&1 | Select-String -NotMatch "Loading weights"
Write-Output "CLOUD CHECK COMPLETE"
