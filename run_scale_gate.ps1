# Scale gate: can a SMALLER model act as teacher for the logic families?
#
# Gate 1 requires the frozen model to execute a structure with the rule in context, scored
# on generated output, at lift >= 0.5. A model that cannot teach cannot bootstrap the
# self-training loop - the plastic weights never see a correct demonstration. At 3B the
# teacher ran 0.806-1.000 across families; precedence was already the weakest at 0.806.
#
# Two scales, not one: 0.5B alone cannot distinguish a gradual decline from a cliff, and
# which it is decides whether edge deployment needs simpler TASKS or just a bigger model.
#
# Read-only, no plasticity. ~10 min each.
$ErrorActionPreference = "Continue"
Write-Output "waiting for the variance rig to finish..."
while (-not (Select-String -Path resultsariance.log -Pattern "VARIANCE RIG COMPLETE" -Quiet -ErrorAction SilentlyContinue)) { Start-Sleep -Seconds 60 }
Write-Output "variance rig done, starting scale gate"
Write-Output "=== teacher gate: Qwen/Qwen2.5-1.5B-Instruct ($(Get-Date -Format HH:mm:ss)) ==="
$env:PROBE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
& ".venv\Scripts\python.exe" probe_families.py 2>&1 | Select-String -NotMatch "Loading weights"
Write-Output "=== teacher gate: Qwen/Qwen2.5-0.5B-Instruct ($(Get-Date -Format HH:mm:ss)) ==="
$env:PROBE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
& ".venv\Scripts\python.exe" probe_families.py 2>&1 | Select-String -NotMatch "Loading weights"
Write-Output "SCALE GATE COMPLETE"
