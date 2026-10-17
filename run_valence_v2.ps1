# Rerun the two SIGNED arms with the format-penalty runaway fixed.
#
# Only these two need rerunning. The unsigned arms never call `contrast`, and in the
# reward-only arm an unparseable move already failed the `adv > 0` test, so all three of those
# results stand as measured.
#
# What changed: an unparseable output now returns (None, None) from `contrast` and is NOT
# written at all, rather than being punished at -99. In v1 that penalty was a runaway - once
# the model slipped off the valid-move manifold it took an enormous negative write, pushing it
# further off, and both seeds went from ~1 % invalid to 100 % invalid within two fifths of the
# stream and trained on garbage thereafter.
#
# The v1 belief that the advantage was mis-centred was WRONG and is worth not repeating: the
# early segments ran at 38-46 % negative with a median advantage of +0.3 to +0.8. The 75 %
# figure was an average dominated by the degenerate tail.
#
#   stage                         ~mins   cumulative
#   1  world_contrast_v2    s0      78        78
#   2  world_contrast_s1_v2 s1      52       130
#
# Compare against: results/world (unsigned s0, REFUSE x48), results/world_positive (reward
# only, WALK x48, TVD 0.00), and the v1 signed runs which conditioned on `standing` but
# learned ONE shared policy rather than three dispositions.
$ErrorActionPreference = "Continue"

function Invoke-Stage($n, $name, $outDir, $extra) {
    Write-Output "=== stage $n : $name ($(Get-Date -Format HH:mm:ss)) ==="
    if (Test-Path "$outDir/summary.json") {
        Write-Output "    already complete, skipping"
        return
    }
    $a = @("run_experiment.py", "--config", "configs/world_contrast.yaml", "--out", $outDir) + $extra
    & ".venv/Scripts/python.exe" $a 2>&1 | Select-String -NotMatch "Loading weights"
    Write-Output "    stage $n done at $(Get-Date -Format HH:mm:ss)"
}

Invoke-Stage 1 "signed, runaway fixed, seed 0" "results/world_contrast_v2"    @()
Invoke-Stage 2 "signed, runaway fixed, seed 1" "results/world_contrast_s1_v2" @("--set","seed=1")

New-Item -ItemType File -Force -Path "results/valence_v2.DONE" | Out-Null
Write-Output "VALENCE V2 COMPLETE at $(Get-Date -Format HH:mm:ss)"
