# Two extra seeds for the Part D coverage numbers.  SEQUENTIAL: a 3B run with a 9-layer
# plastic slice peaks near 15 GB, so two at once will not fit on the 16 GB card.
#
#   powershell -ExecutionPolicy Bypass -File run_gated_seeds.ps1
#
# The workspaces are fixed by the per-rule seeds in the config (11/22/33/44); the run seed
# varies the probe sets, the stream ordering, REM noise and torch init, which is the
# sampling variance the coverage figures need bounding by.
$ErrorActionPreference = "Continue"
foreach ($s in 1, 2) {
    $out = "results/gate3_gated_s$s"
    Write-Output "=== seed $s -> $out  ($(Get-Date -Format HH:mm:ss)) ==="
    & ".venv\Scripts\python.exe" run_experiment.py `
        --config configs/workspace_gated.yaml `
        --out $out `
        --set "seed=$s" 2>&1 | Select-String -NotMatch "Loading weights"
    Write-Output "=== seed $s done ($(Get-Date -Format HH:mm:ss)) ==="
}
Write-Output "ALL SEEDS COMPLETE"
