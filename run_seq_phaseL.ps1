# SEQUENTIAL, PHASE L: snippets 4000 -> mutants 4000 -> snippets 2000, both skills probed every
# 250 (configs/seq_base.yaml). Waits for snippet phase K.
#   L1  fast weights, baseline recipe (decay .004, budget .03)
#   L2  fast weights, no decay, budget .12
#   L3  LoRA, same stream (lora_control.py --phases), eval every 250
# Readouts: snippet skill at the END of phase 1 (acquired), at the END of phase 2 (retained while
# mutants were learned), mutant skill at the end of phase 2, snippet skill after the 2000-draw
# refresh (relearning), and base-model health.
# Predictions: L1 loses most of its snippet skill during the mutant phase (a ~250-draw decay
# horizon) and regains it fastest in the refresh; LoRA retains the most; L2 retains more than L1
# but ends with the most base-model damage. Fast weights earn a niche here only if they relearn
# clearly faster than LoRA, or retain at least as much with less damage.
Set-Location $PSScriptRoot
$py = ".venv\Scripts\python.exe"
while (-not (Test-Path "results/snip/phaseK.DONE")) { Start-Sleep -Seconds 30 }
New-Item -ItemType Directory -Force "results/seq" | Out-Null
$runs = @(
    @{ name = "L1_base"; extra = @() },
    @{ name = "L2_nodecay_b012"; extra = @("--set", "plastic.decay_enabled=false", "--set", "plastic.groups.hdeltaH.budget=0.12") }
)
foreach ($s in $runs) {
    $out = "results/seq/$($s.name)"
    if (Test-Path "$out/summary.json") { continue }
    Write-Output "$(Get-Date -Format s) start $($s.name)"
    $p = Start-Process -FilePath $py -ArgumentList (@("-u", "run_experiment.py", "--config", "configs/seq_base.yaml") + $s.extra + @("--out", $out)) `
        -RedirectStandardOutput "$out.log" -RedirectStandardError "$out.err" -NoNewWindow -PassThru -Wait
    Write-Output "$(Get-Date -Format s) end $($s.name) exit $($p.ExitCode)"
}
if (-not (Test-Path "results/seq/L3_lora/lora_log.json")) {
    Write-Output "$(Get-Date -Format s) start L3_lora"
    $p = Start-Process -FilePath $py -ArgumentList @("-u", "lora_control.py", "--config", "configs/seq_base.yaml",
        "--phases", "--eval-every", "250", "--out", "results/seq/L3_lora") `
        -RedirectStandardOutput "results/seq/L3_lora.log" -RedirectStandardError "results/seq/L3_lora.err" -NoNewWindow -PassThru -Wait
    Write-Output "$(Get-Date -Format s) end L3_lora exit $($p.ExitCode)"
}
Set-Content -Path "results/seq/phaseL.DONE" -Value "$(Get-Date -Format s)"
