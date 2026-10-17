# SEQUENTIAL, PHASE N: is the fast weights' retention edge just CAPACITY?
# In both seeds the aggressive fast weights (no decay, budget .12; 85M plastic params) kept the
# first skill better than LoRA r=16 (1.8M params): snip kept 0.809 vs 0.752 (s0), 0.827 vs 0.745
# (s1). Two skills competing in a small low-rank subspace would explain it. One variable - LoRA
# rank (lr unchanged at 2e-4; 5e-4 collapsed on Connect-Four) - on the same stream, both seeds.
# Prediction: retention rises with rank; r=256 (~32M params) keeps >= 0.80 with ppl within +5 %.
# If it does, capacity explains the edge and LoRA still dominates; if not, the full-rank
# fast-weight update has a genuine retention advantage.
Set-Location $PSScriptRoot
$py = ".venv\Scripts\python.exe"
foreach ($seed in 0, 1) {
    foreach ($rank in 64, 256) {
        $out = "results/seq/N_lora_r${rank}_s$seed"
        if (Test-Path "$out/lora_log.json") {
            $last = (Get-Content "$out/lora_log.json" | ConvertFrom-Json).log[-1].n
            if ($last -ge 10000) { continue }
        }
        Write-Output "$(Get-Date -Format s) start r$rank s$seed"
        $p = Start-Process -FilePath $py -ArgumentList @("-u", "lora_control.py", "--config", "configs/seq_base.yaml",
            "--phases", "--seed", "$seed", "--rank", "$rank", "--eval-every", "250", "--out", $out) `
            -RedirectStandardOutput "$out.log" -RedirectStandardError "$out.err" -NoNewWindow -PassThru -Wait
        Write-Output "$(Get-Date -Format s) end r$rank s$seed exit $($p.ExitCode)"
    }
}
Set-Content -Path "results/seq/phaseN.DONE" -Value "$(Get-Date -Format s)"
