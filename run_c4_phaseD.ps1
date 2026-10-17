# Connect-Four PHASE D, queued behind phase C (waits for results/c4/phaseC.DONE).
# Calibration showed the MLP's lead is mostly REPETITION: one pass 0.616, four passes 0.705,
# against one-pass LoRA ~0.57 and the best fast weights 0.52. So the next variable is replay.
#   D1  LoRA, 4 passes over the same 16000 positions      (is the LLM's reading of the board
#                                                          the limit, or the single pass?)
#   D2  fast weights, no decay, budget 0.12, replay 3     (1 fresh + 3 replayed update steps per
#                                                          chunk = the MLP's 4x update budget; C1 is
#                                                          the identical run without replay)
# Predictions: D1 rises to ~0.62-0.66 (below the 4-pass MLP 0.705: text is a worse input than
# one-hot cells). D2 beats C1 by >= 0.03 if the single pass is what holds the fast weights back.
Set-Location $PSScriptRoot
$py = ".venv\Scripts\python.exe"
while (-not (Test-Path "results/c4/phaseC.DONE")) { Start-Sleep -Seconds 30 }
Write-Output "$(Get-Date -Format s) start D1_lora_4pass"
if (-not (Test-Path "results/c4/D1_lora_4pass/lora_log.json")) {
    $p = Start-Process -FilePath $py -ArgumentList @("-u", "c4_lora.py", "--n", "16000", "--passes", "4",
        "--eval-every", "4000", "--out", "results/c4/D1_lora_4pass") `
        -RedirectStandardOutput "results/c4/D1_lora_4pass.log" -RedirectStandardError "results/c4/D1_lora_4pass.err" `
        -NoNewWindow -PassThru -Wait
}
Write-Output "$(Get-Date -Format s) end D1_lora_4pass"
$out = "results/c4/D2_nodecay_b012_replay3"
if (-not (Test-Path "$out/summary.json")) {
    Write-Output "$(Get-Date -Format s) start D2"
    $p = Start-Process -FilePath $py -ArgumentList @("-u", "run_experiment.py", "--config", "configs/c4_base.yaml",
        "--set", "stream.phases[0].puzzles=16000", "--set", "stream.probe_every=500",
        "--set", "plastic.decay_enabled=false", "--set", "plastic.groups.hdeltaH.budget=0.12",
        "--set", "stream.replay=3", "--out", $out) `
        -RedirectStandardOutput "$out.log" -RedirectStandardError "$out.err" -NoNewWindow -PassThru -Wait
    Write-Output "$(Get-Date -Format s) end D2 exit $($p.ExitCode)"
}
Set-Content -Path "results/c4/phaseD.DONE" -Value "$(Get-Date -Format s)"
