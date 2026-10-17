# SEQUENTIAL, PHASE M: seed 1 of phase L, to confirm it. Waits for phase L.
# Seed 0 (snip learned / snip kept after mutants / mut learned / snip after refresh; ppl, cloze):
#   FW gentle      0.767 / 0.540 / 0.594 / 0.721   ppl 9.40  cloze 0.80
#   FW aggressive  0.845 / 0.809 / 0.612 / 0.875   ppl 38.4  cloze 0.10
#   LoRA           0.865 / 0.752 / 0.722 / 0.863   ppl 9.63  cloze 0.73
# Prediction: the same ordering on seed 1 - LoRA learns the second skill best with no damage,
# aggressive fast weights retain the first skill best at the cost of the base model.
Set-Location $PSScriptRoot
$py = ".venv\Scripts\python.exe"
while (-not (Test-Path "results/seq/phaseL.DONE")) { Start-Sleep -Seconds 30 }
$runs = @(
    @{ name = "M1_base_s1"; extra = @() },
    @{ name = "M2_nodecay_b012_s1"; extra = @("--set", "plastic.decay_enabled=false", "--set", "plastic.groups.hdeltaH.budget=0.12") }
)
foreach ($s in $runs) {
    $out = "results/seq/$($s.name)"
    if (Test-Path "$out/summary.json") { continue }
    Write-Output "$(Get-Date -Format s) start $($s.name)"
    $p = Start-Process -FilePath $py -ArgumentList (@("-u", "run_experiment.py", "--config", "configs/seq_base.yaml", "--set", "seed=1") + $s.extra + @("--out", $out)) `
        -RedirectStandardOutput "$out.log" -RedirectStandardError "$out.err" -NoNewWindow -PassThru -Wait
    Write-Output "$(Get-Date -Format s) end $($s.name) exit $($p.ExitCode)"
}
if (-not (Test-Path "results/seq/M3_lora_s1/lora_log.json")) {
    Write-Output "$(Get-Date -Format s) start M3_lora_s1"
    $p = Start-Process -FilePath $py -ArgumentList @("-u", "lora_control.py", "--config", "configs/seq_base.yaml",
        "--phases", "--seed", "1", "--eval-every", "250", "--out", "results/seq/M3_lora_s1") `
        -RedirectStandardOutput "results/seq/M3_lora_s1.log" -RedirectStandardError "results/seq/M3_lora_s1.err" -NoNewWindow -PassThru -Wait
    Write-Output "$(Get-Date -Format s) end M3_lora_s1 exit $($p.ExitCode)"
}
Set-Content -Path "results/seq/phaseM.DONE" -Value "$(Get-Date -Format s)"
