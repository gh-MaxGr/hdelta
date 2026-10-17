# Connect-Four PHASE C, queued behind phase B (waits for results/c4/phaseB.DONE).
# Phase A: decay is the first lever (0.439 -> 0.477 -> 0.521 as decay .004 -> .001 -> 0), and
# with decay off the homeostatic clamp BINDS - fw norm hit the 0.06 budget by t~4500 and sat
# there. So the next single variable is the budget, decay held off. Base-model damage grew
# with the norm (ppl +10.6 % at 0.06), so it is watched as closely as accuracy.
#   C1  no decay, budget 0.12
#   C2  no decay, budget 0.24
# Prediction: held-out mean rises with budget but stays below the MLP (0.73 @16k); ppl drift
# roughly doubles per doubling of budget. If C1 is no better than A3 (0.521), the clamp was
# not the limit after all and the learning rule is.
Set-Location $PSScriptRoot
$py = ".venv\Scripts\python.exe"
while (-not (Test-Path "results/c4/phaseB.DONE")) { Start-Sleep -Seconds 30 }
$common = @("-u", "run_experiment.py", "--config", "configs/c4_base.yaml",
            "--set", "stream.phases[0].puzzles=16000", "--set", "stream.probe_every=500",
            "--set", "plastic.decay_enabled=false")
$stages = @(
    @{ name = "C1_nodecay_b012"; extra = @("--set", "plastic.groups.hdeltaH.budget=0.12") },
    @{ name = "C2_nodecay_b024"; extra = @("--set", "plastic.groups.hdeltaH.budget=0.24") }
)
foreach ($s in $stages) {
    $out = "results/c4/$($s.name)"
    if (Test-Path "$out/summary.json") { Write-Output "skip $($s.name)"; continue }
    Write-Output "$(Get-Date -Format s) start $($s.name)"
    $p = Start-Process -FilePath $py -ArgumentList ($common + $s.extra + @("--out", $out)) `
        -RedirectStandardOutput "results/c4/$($s.name).log" -RedirectStandardError "results/c4/$($s.name).err" `
        -NoNewWindow -PassThru -Wait
    Write-Output "$(Get-Date -Format s) end $($s.name) exit $($p.ExitCode)"
}
Set-Content -Path "results/c4/phaseC.DONE" -Value "$(Get-Date -Format s)"
