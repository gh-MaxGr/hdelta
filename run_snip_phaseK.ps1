# SNIPPETS, PHASE K: the skill-vs-damage frontier (seed 0). Waits for phase J.
# Two points so far: I1 (decay .004, budget .03) acc 0.738 / AUC 0.907 / ppl 9.44 (no damage);
# I2 (no decay, budget .12) 0.884 / 0.956 / ppl 21.5, cloze 0.73 -> 0.33 (wrecked).
#   K0  LoRA seed 0 WITH base-model health (lora_control now measures ppl/cloze adapter-on),
#       the reference point: LoRA skill was 0.851 / 0.953 but its damage was never measured
#   K1  no decay, budget 0.06
#   K2  decay 0.001, budget 0.06
#   K3  no decay, budget 0.06, replay 3   (replay cut damage on Connect-Four: ppl 11.4 vs 12.6)
# Question it answers: is there a fast-weight setting with LoRA-level skill and small damage -
# and is LoRA itself undamaged at its skill level? If LoRA gets 0.95 AUC with little damage and
# no fast-weight setting does, the gradient adapter dominates on this task.
Set-Location $PSScriptRoot
$py = ".venv\Scripts\python.exe"
while (-not (Test-Path "results/snip/phaseJ.DONE")) { Start-Sleep -Seconds 30 }
if (-not (Test-Path "results/snip/K0_lora_health/lora_log.json")) {
    Write-Output "$(Get-Date -Format s) start K0"
    $p = Start-Process -FilePath $py -ArgumentList @("-u", "lora_control.py", "--config", "configs/snip_base.yaml",
        "--n", "8000", "--eval-every", "1000", "--out", "results/snip/K0_lora_health") `
        -RedirectStandardOutput "results/snip/K0_lora_health.log" -RedirectStandardError "results/snip/K0_lora_health.err" -NoNewWindow -PassThru -Wait
    Write-Output "$(Get-Date -Format s) end K0 exit $($p.ExitCode)"
}
$runs = @(
    @{ name = "K1_nodecay_b006"; extra = @("--set", "plastic.decay_enabled=false", "--set", "plastic.groups.hdeltaH.budget=0.06") },
    @{ name = "K2_decay001_b006"; extra = @("--set", "plastic.groups.hdeltaH.decay=0.001", "--set", "plastic.groups.hdeltaH.budget=0.06") },
    @{ name = "K3_nodecay_b006_replay3"; extra = @("--set", "plastic.decay_enabled=false", "--set", "plastic.groups.hdeltaH.budget=0.06", "--set", "stream.replay=3") }
)
foreach ($s in $runs) {
    $out = "results/snip/$($s.name)"
    if (Test-Path "$out/summary.json") { continue }
    Write-Output "$(Get-Date -Format s) start $($s.name)"
    $p = Start-Process -FilePath $py -ArgumentList (@("-u", "run_experiment.py", "--config", "configs/snip_base.yaml") + $s.extra + @("--out", $out)) `
        -RedirectStandardOutput "$out.log" -RedirectStandardError "$out.err" -NoNewWindow -PassThru -Wait
    Write-Output "$(Get-Date -Format s) end $($s.name) exit $($p.ExitCode)"
}
Set-Content -Path "results/snip/phaseK.DONE" -Value "$(Get-Date -Format s)"
