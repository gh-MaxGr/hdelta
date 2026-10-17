# Connect-Four PHASE E, queued behind phase D (waits for results/c4/phaseD.DONE).
#   E1  LoRA r=64, lr 5e-4, one pass - is the ~0.6 LoRA ceiling a hyperparameter artefact?
#   E2  C1 (no decay, budget 0.12) with seed 1 - does "fast weights below the cache" replicate?
# Predictions: E1 within 0.03 of B1 (~0.57-0.61): the LLM reading the board as text is the limit.
# E2 within 0.03 of C1 (0.547), still below the cache (0.570).
Set-Location $PSScriptRoot
$py = ".venv\Scripts\python.exe"
while (-not (Test-Path "results/c4/phaseD.DONE")) { Start-Sleep -Seconds 30 }
if (-not (Test-Path "results/c4/E1_lora_r64/lora_log.json")) {
    Write-Output "$(Get-Date -Format s) start E1"
    $p = Start-Process -FilePath $py -ArgumentList @("-u", "c4_lora.py", "--n", "16000", "--rank", "64",
        "--lr", "5e-4", "--out", "results/c4/E1_lora_r64") -RedirectStandardOutput "results/c4/E1_lora_r64.log" `
        -RedirectStandardError "results/c4/E1_lora_r64.err" -NoNewWindow -PassThru -Wait
    Write-Output "$(Get-Date -Format s) end E1 exit $($p.ExitCode)"
}
$out = "results/c4/E2_nodecay_b012_s1"
if (-not (Test-Path "$out/summary.json")) {
    Write-Output "$(Get-Date -Format s) start E2"
    $p = Start-Process -FilePath $py -ArgumentList @("-u", "run_experiment.py", "--config", "configs/c4_base.yaml",
        "--set", "seed=1", "--set", "stream.phases[0].puzzles=16000", "--set", "stream.probe_every=500",
        "--set", "plastic.decay_enabled=false", "--set", "plastic.groups.hdeltaH.budget=0.12", "--out", $out) `
        -RedirectStandardOutput "$out.log" -RedirectStandardError "$out.err" -NoNewWindow -PassThru -Wait
    Write-Output "$(Get-Date -Format s) end E2 exit $($p.ExitCode)"
}
Set-Content -Path "results/c4/phaseE.DONE" -Value "$(Get-Date -Format s)"
