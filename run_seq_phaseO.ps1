# SEQUENTIAL, PHASE O: can an UNDAMAGING setup keep the first skill as well as the aggressive
# fast weights? (They kept snippets 0.809 / 0.827 through the mutant phase vs LoRA r=16's
# 0.752 / 0.745, at ppl +311 % / +27 %.) Raising LoRA rank at lr 2e-4 was not a clean capacity
# test - r=64 was unstable (ppl 22.6 / 48.2), r=256 diverged. The standard practical fix for
# forgetting is replay, so: LoRA r=16 + 1 replay step per fresh step (earlier items, no new
# tool calls), both seeds. The fast weights had no replay: this compares practical SYSTEMS.
# Prediction: snippets kept >= 0.80 with ppl within +5 %, mutants learned no worse than LoRA alone.
Set-Location $PSScriptRoot
$py = ".venv\Scripts\python.exe"
while (-not (Test-Path "results/seq/phaseN.DONE")) { Start-Sleep -Seconds 20 }
foreach ($seed in 0, 1) {
    $out = "results/seq/O_lora_replay1_s$seed"
    if (Test-Path "$out/lora_log.json") { if (((Get-Content "$out/lora_log.json" | ConvertFrom-Json).log[-1].n) -ge 10000) { continue } }
    Write-Output "$(Get-Date -Format s) start replay s$seed"
    $p = Start-Process -FilePath $py -ArgumentList @("-u", "lora_control.py", "--config", "configs/seq_base.yaml",
        "--phases", "--seed", "$seed", "--replay", "1", "--eval-every", "250", "--out", $out) `
        -RedirectStandardOutput "$out.log" -RedirectStandardError "$out.err" -NoNewWindow -PassThru -Wait
    Write-Output "$(Get-Date -Format s) end replay s$seed exit $($p.ExitCode)"
}
Set-Content -Path "results/seq/phaseO.DONE" -Value "$(Get-Date -Format s)"
