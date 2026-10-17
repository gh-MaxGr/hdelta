# Sample-complexity runs: the three families the 600-puzzle budget could not resolve.
#
# Scored as VERIFIED WRITES to first sustain lift >= theta for 3 consecutive probes, which
# is budget-independent. The fast families already cross inside 600 puzzles, so their
# crossing times are already measured and do not need re-running.
#
# This also settles the composition cliff without a separate diagnostic.
#
# SEQUENTIAL. ~25 min each.
$ErrorActionPreference = "Continue"
Write-Output "=== composition 1800 ($(Get-Date -Format HH:mm:ss)) ==="
& ".venv\Scripts\python.exe" run_experiment.py --config configs/long_composition.yaml --out results/ladder_long/composition 2>&1 | Select-String -NotMatch "Loading weights"
Write-Output "=== precedence 1800 ($(Get-Date -Format HH:mm:ss)) ==="
& ".venv\Scripts\python.exe" run_experiment.py --config configs/long_precedence.yaml --out results/ladder_long/precedence 2>&1 | Select-String -NotMatch "Loading weights"
Write-Output "=== conjunction 1800 ($(Get-Date -Format HH:mm:ss)) ==="
& ".venv\Scripts\python.exe" run_experiment.py --config configs/long_conjunction.yaml --out results/ladder_long/conjunction 2>&1 | Select-String -NotMatch "Loading weights"
Write-Output "LONG LADDER COMPLETE"
