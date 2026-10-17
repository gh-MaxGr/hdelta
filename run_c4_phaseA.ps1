# Connect-Four value, PHASE A: can the plastic slice learn a pattern that is not a table,
# and which lever limits it? 16000 tool-labelled positions per run, held-out probes of 256
# every 500. Sequential; each stage skips itself if its summary.json exists (resumable).
#
# Bars it has to clear (c4_preflight.py, same 5x4 task, held-out positions):
#   prior 0.43 | strongest reason cache 0.570 | logreg 0.56 | MLP 0.654 @4k, 0.675 @8k, 0.710 @16k
#
# Predictions, stated before the run (mean over the second half of probes):
#   A1 baseline recipe (decay .004, ~250-puzzle memory)  plateaus near the cache, 0.55-0.60
#   A2 decay .001, budget .06                           higher, 0.62-0.68
#   A3 no decay, budget .06                             highest of the three if capacity allows;
#                                                       watch ppl/cloze for base-model damage
#   A4 = A2 with 18 layers (max_fraction .10)           above A2 if slice size binds
#   A5 = A2 with REM off                                within noise of A2
# "Learns a non-table pattern" = held-out mean >= cache + 0.05 AND above logreg.
Set-Location $PSScriptRoot
$py = ".venv\Scripts\python.exe"
$common = @("-u", "run_experiment.py", "--config", "configs/c4_base.yaml", "--set", "stream.phases[0].puzzles=16000",
            "--set", "stream.probe_every=500")
$L18 = "plastic.groups.hdeltaH.layers=[-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-13,-14,-15,-16,-17,-18]"
$stages = @(
    @{ name = "A1_base";    extra = @() },
    @{ name = "A2_decay001"; extra = @("--set", "plastic.groups.hdeltaH.decay=0.001",
                                       "--set", "plastic.groups.hdeltaH.budget=0.06") },
    @{ name = "A3_nodecay";  extra = @("--set", "plastic.decay_enabled=false",
                                       "--set", "plastic.groups.hdeltaH.budget=0.06") },
    @{ name = "A4_L18";      extra = @("--set", "plastic.groups.hdeltaH.decay=0.001",
                                       "--set", "plastic.groups.hdeltaH.budget=0.06",
                                       "--set", "plastic.max_fraction=0.10", "--set", $L18) },
    @{ name = "A5_norem";    extra = @("--set", "plastic.groups.hdeltaH.decay=0.001",
                                       "--set", "plastic.groups.hdeltaH.budget=0.06",
                                       "--set", "rem.enabled=false") }
)
foreach ($s in $stages) {
    $out = "results/c4/$($s.name)"
    if (Test-Path "$out/summary.json") { Write-Output "skip $($s.name)"; continue }
    $args_ = $common + $s.extra + @("--out", $out)
    Write-Output "$(Get-Date -Format s) start $($s.name)"
    $p = Start-Process -FilePath $py -ArgumentList $args_ -RedirectStandardOutput "results/c4/$($s.name).log" `
        -RedirectStandardError "results/c4/$($s.name).err" -NoNewWindow -PassThru -Wait
    Write-Output "$(Get-Date -Format s) end $($s.name) exit $($p.ExitCode)"
}
Set-Content -Path "results/c4/phaseA.DONE" -Value "$(Get-Date -Format s)"
