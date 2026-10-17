# SNIPPETS, PHASE I - the competent-prior quadrant. Waits for snipgen (-GenPid) to finish.
#   I0  LoRA control; its n=0 line is the FROZEN model zero-shot - the size of the prior
#   I1  fast weights, baseline recipe
#   I2  fast weights, no decay, budget 0.12
# Predictions: frozen zero-shot is well above chance (AUC >= 0.65) - unlike Connect-Four and
# mutants. A bag-of-words MLP (snip_bars.py, same probes) is weaker than LoRA here, because
# token counts cannot see semantics. Fast weights: the open question - do they improve the prior
# without eroding it (watch ppl/cloze), and do they land nearer LoRA or nearer the MLP?
param([int]$GenPid = 0)
Set-Location $PSScriptRoot
$py = ".venv\Scripts\python.exe"
if ($GenPid -gt 0) { Wait-Process -Id $GenPid -ErrorAction SilentlyContinue }
New-Item -ItemType Directory -Force "results/snip" | Out-Null
if (-not (Test-Path "results/snip/I0_lora/lora_log.json")) {
    Write-Output "$(Get-Date -Format s) start I0"
    $p = Start-Process -FilePath $py -ArgumentList @("-u", "lora_control.py", "--config", "configs/snip_base.yaml",
        "--n", "8000", "--eval-every", "1000", "--out", "results/snip/I0_lora") `
        -RedirectStandardOutput "results/snip/I0_lora.log" -RedirectStandardError "results/snip/I0_lora.err" -NoNewWindow -PassThru -Wait
    Write-Output "$(Get-Date -Format s) end I0 exit $($p.ExitCode)"
}
$runs = @(
    @{ name = "I1_base"; extra = @() },
    @{ name = "I2_nodecay_b012"; extra = @("--set", "plastic.decay_enabled=false", "--set", "plastic.groups.hdeltaH.budget=0.12") }
)
foreach ($s in $runs) {
    $out = "results/snip/$($s.name)"
    if (Test-Path "$out/summary.json") { Write-Output "skip $($s.name)"; continue }
    Write-Output "$(Get-Date -Format s) start $($s.name)"
    $p = Start-Process -FilePath $py -ArgumentList (@("-u", "run_experiment.py", "--config", "configs/snip_base.yaml") + $s.extra + @("--out", $out)) `
        -RedirectStandardOutput "$out.log" -RedirectStandardError "$out.err" -NoNewWindow -PassThru -Wait
    Write-Output "$(Get-Date -Format s) end $($s.name) exit $($p.ExitCode)"
}
Set-Content -Path "results/snip/phaseI.DONE" -Value "$(Get-Date -Format s)"
