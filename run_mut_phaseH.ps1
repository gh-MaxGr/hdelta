# MUTANTS, PHASE H: replicate the one surprising result before believing it. Waits for phase G.
# F2 (fast weights, no decay, budget 0.12, plain diff prompt) averaged acc 0.680 / AUC 0.791 on
# seed 0's probes - above one-pass LoRA (0.631 / 0.700) and the bag-of-words MLP on the SAME
# probes (0.68-0.70 / 0.72-0.73). The reverse of Connect-Four. Plausible cause: the pool is
# small, LoRA overfits it (train loss 0.08) and the fast-weight budget acts as a regulariser.
# Each seed changes the stream AND the probe set, so the MLP bars are recomputed per seed (CPU).
#   H1/H2  F2 config, seeds 1 and 2
#   H3/H4  LoRA control, seeds 1 and 2
# Prediction (stated before seeds 1-2 exist): fast weights beat LoRA on AUC in both seeds; the
# gap to the per-seed MLP is smaller than seed 0's +0.06, within noise in at least one seed.
Set-Location $PSScriptRoot
$py = ".venv\Scripts\python.exe"
while (-not (Test-Path "results/mut/phaseG.DONE")) { Start-Sleep -Seconds 30 }
foreach ($seed in 1, 2) {
    $out = "results/mut/H_fw_nodecay_b012_s$seed"
    if (-not (Test-Path "$out/summary.json")) {
        Write-Output "$(Get-Date -Format s) start fw seed $seed"
        $p = Start-Process -FilePath $py -ArgumentList @("-u", "run_experiment.py", "--config", "configs/mut_base.yaml",
            "--set", "seed=$seed", "--set", "plastic.decay_enabled=false", "--set", "plastic.groups.hdeltaH.budget=0.12",
            "--out", $out) -RedirectStandardOutput "$out.log" -RedirectStandardError "$out.err" -NoNewWindow -PassThru -Wait
        Write-Output "$(Get-Date -Format s) end fw seed $seed exit $($p.ExitCode)"
    }
    $lo = "results/mut/H_lora_s$seed"
    if (-not (Test-Path "$lo/lora_log.json")) {
        Write-Output "$(Get-Date -Format s) start lora seed $seed"
        $p = Start-Process -FilePath $py -ArgumentList @("-u", "lora_control.py", "--config", "configs/mut_base.yaml",
            "--seed", "$seed", "--n", "8000", "--eval-every", "1000", "--out", $lo) `
            -RedirectStandardOutput "$lo.log" -RedirectStandardError "$lo.err" -NoNewWindow -PassThru -Wait
        Write-Output "$(Get-Date -Format s) end lora seed $seed exit $($p.ExitCode)"
    }
}
Set-Content -Path "results/mut/phaseH.DONE" -Value "$(Get-Date -Format s)"
