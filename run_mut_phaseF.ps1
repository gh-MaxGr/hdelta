# MUTANTS, PHASE F: the first task where the LLM's pretraining should help (code), and a
# reason cache cannot (every mutant unique, reasons are test names/tracebacks, held-out = unseen
# functions). Waits for Connect-Four phase E AND for the mutant generator named by -GenPid.
#   F0  LoRA control (n=0 = the FROZEN model zero-shot), 8000 balanced draws
#   F1  fast weights, baseline recipe
#   F2  fast weights, no decay, budget 0.12 (best Connect-Four setting)
# Predictions: frozen zero-shot AUC 0.55-0.65; LoRA climbs past the model-free bars (logreg
# AUC ~0.66-0.70) since it can read code; fast weights land below LoRA, as on Connect-Four.
param([int]$GenPid = 0)
Set-Location $PSScriptRoot
$py = ".venv\Scripts\python.exe"
while (-not (Test-Path "results/c4/phaseE.DONE")) { Start-Sleep -Seconds 30 }
if ($GenPid -gt 0) { Wait-Process -Id $GenPid -ErrorAction SilentlyContinue }
New-Item -ItemType Directory -Force "results/mut" | Out-Null
if (-not (Test-Path "results/mut/F0_lora/lora_log.json")) {
    Write-Output "$(Get-Date -Format s) start F0"
    $p = Start-Process -FilePath $py -ArgumentList @("-u", "lora_control.py", "--config", "configs/mut_base.yaml",
        "--n", "8000", "--eval-every", "1000", "--out", "results/mut/F0_lora") `
        -RedirectStandardOutput "results/mut/F0_lora.log" -RedirectStandardError "results/mut/F0_lora.err" -NoNewWindow -PassThru -Wait
    Write-Output "$(Get-Date -Format s) end F0 exit $($p.ExitCode)"
}
$runs = @(
    @{ name = "F1_base"; extra = @() },
    @{ name = "F2_nodecay_b012"; extra = @("--set", "plastic.decay_enabled=false", "--set", "plastic.groups.hdeltaH.budget=0.12") }
)
foreach ($s in $runs) {
    $out = "results/mut/$($s.name)"
    if (Test-Path "$out/summary.json") { Write-Output "skip $($s.name)"; continue }
    Write-Output "$(Get-Date -Format s) start $($s.name)"
    $p = Start-Process -FilePath $py -ArgumentList (@("-u", "run_experiment.py", "--config", "configs/mut_base.yaml") + $s.extra + @("--out", $out)) `
        -RedirectStandardOutput "$out.log" -RedirectStandardError "$out.err" -NoNewWindow -PassThru -Wait
    Write-Output "$(Get-Date -Format s) end $($s.name) exit $($p.ExitCode)"
}
Set-Content -Path "results/mut/phaseF.DONE" -Value "$(Get-Date -Format s)"
