# Valence chain: does giving the write a SIGN stop the constant collapse, and which half does it?
#
# Runs sequentially in ONE script, so no marker files are needed - the stages are serialised
# by the script itself and two GPU jobs cannot collide on a single marker, which is the
# failure mode the marker chains elsewhere in this repo exist to prevent.
#
# Ordered so truncation degrades gracefully. Stop it after stage 2 and you still have the
# full three-arm ladder at seed 0; stages 3-4 only add seed replication.
#
#   stage                       arm                              ~mins   cumulative
#   1  world_contrast           signed, graded, contrastive        94        94
#   2  world_positive           graded reward, no punishment       60       154
#   3  world_contrast_s1        signed, seed 1                     94       248
#   4  world_s1                 unsigned control, seed 1           47       295
#
# The unsigned control at seed 0 already exists (results/world) and is the thing to beat:
# REFUSE x48 on all three dispositions, 0.44 / 0.15 / 0.52 against floors 0.50 / 0.70 / 0.58.
#
# SUCCESS IS THE ACTION DISTRIBUTION, NOT THE ACCURACY. A lucky constant policy can post a
# respectable number - diplomat's always-ACCEPT scores 0.70. Read `analyse_valence.py`, which
# reports distinct moves emitted per disposition alongside the accuracy.
#
# Launch detached, never from inside a tool call:
#   Start-Process -FilePath "powershell.exe" -ArgumentList @("-ExecutionPolicy","Bypass","-File","run_valence_chain.ps1") -RedirectStandardOutput "results/valence_chain.log" -NoNewWindow
$ErrorActionPreference = "Continue"

function Invoke-Stage($n, $name, $cfgPath, $outDir, $extra) {
    Write-Output "=== stage $n : $name ($(Get-Date -Format HH:mm:ss)) ==="
    if (Test-Path "$outDir/summary.json") {
        Write-Output "    already complete, skipping"
        return
    }
    $a = @("run_experiment.py", "--config", $cfgPath, "--out", $outDir) + $extra
    & ".venv/Scripts/python.exe" $a 2>&1 | Select-String -NotMatch "Loading weights"
    Write-Output "    stage $n done at $(Get-Date -Format HH:mm:ss)"
}

Invoke-Stage 1 "signed + contrastive, seed 0" "configs/world_contrast.yaml" "results/world_contrast" @()
Invoke-Stage 2 "graded reward only, seed 0"   "configs/world_positive.yaml" "results/world_positive" @()
Invoke-Stage 3 "signed + contrastive, seed 1" "configs/world_contrast.yaml" "results/world_contrast_s1" @("--set","seed=1")
Invoke-Stage 4 "unsigned control, seed 1"     "configs/world.yaml"          "results/world_s1"          @("--set","seed=1")

New-Item -ItemType File -Force -Path "results/valence_chain.DONE" | Out-Null
Write-Output "VALENCE CHAIN COMPLETE at $(Get-Date -Format HH:mm:ss)"
