# SNIPPETS, PHASE J: replicate phase I on seeds 1 and 2 before believing it (the mutant task's
# seed-0 "reversal" did not survive this step). Waits for phase I.
# Seed 0: LoRA 0.851 / AUC 0.953; fast weights, base recipe 0.738 / AUC 0.907 (threshold swings
# +-0.10, base model untouched); bag-of-words MLP 0.785 / AUC 0.854 at N=4000.
# Prediction: on both seeds the base-recipe fast weights beat the per-seed MLP on AUC (>= +0.03)
# and stay below LoRA; their accuracy swings more than either.
Set-Location $PSScriptRoot
$py = ".venv\Scripts\python.exe"
while (-not (Test-Path "results/snip/phaseI.DONE")) { Start-Sleep -Seconds 30 }
foreach ($seed in 1, 2) {
    $runs = @(
        @{ name = "J_base_s$seed"; extra = @() },
        @{ name = "J_nodecay_b012_s$seed"; extra = @("--set", "plastic.decay_enabled=false", "--set", "plastic.groups.hdeltaH.budget=0.12") }
    )
    foreach ($s in $runs) {
        $out = "results/snip/$($s.name)"
        if (Test-Path "$out/summary.json") { continue }
        Write-Output "$(Get-Date -Format s) start $($s.name)"
        $p = Start-Process -FilePath $py -ArgumentList (@("-u", "run_experiment.py", "--config", "configs/snip_base.yaml", "--set", "seed=$seed") + $s.extra + @("--out", $out)) `
            -RedirectStandardOutput "$out.log" -RedirectStandardError "$out.err" -NoNewWindow -PassThru -Wait
        Write-Output "$(Get-Date -Format s) end $($s.name) exit $($p.ExitCode)"
    }
    $lo = "results/snip/J_lora_s$seed"
    if (-not (Test-Path "$lo/lora_log.json")) {
        Write-Output "$(Get-Date -Format s) start lora s$seed"
        $p = Start-Process -FilePath $py -ArgumentList @("-u", "lora_control.py", "--config", "configs/snip_base.yaml",
            "--seed", "$seed", "--n", "8000", "--eval-every", "1000", "--out", $lo) `
            -RedirectStandardOutput "$lo.log" -RedirectStandardError "$lo.err" -NoNewWindow -PassThru -Wait
        Write-Output "$(Get-Date -Format s) end lora s$seed exit $($p.ExitCode)"
    }
}
Set-Content -Path "results/snip/phaseJ.DONE" -Value "$(Get-Date -Format s)"
