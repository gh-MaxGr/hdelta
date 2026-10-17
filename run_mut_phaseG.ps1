# MUTANTS, PHASE G: does the LLM's knowledge of code show up once it can SEE the code?
# Phase F showed only a 5-line diff: LoRA tied a bag-of-words MLP (AUC 0.73 vs 0.72-0.73) and
# fast weights sat at the operator-rules level. One variable changes here - the prompt also
# shows the mutated function (configs/mut_ctx.yaml, task.context=true). Waits for phase F.
#   G0   LoRA control with context
#   G1a  fast weights, baseline recipe, with context      (reference F1)
#   G1b  fast weights, no decay budget 0.12, with context (reference F2)
# Prediction: if code understanding matters, LoRA+context AUC >= 0.78 while the bag-of-words
# MLP with the same context stays ~0.72; if LoRA+context ~= LoRA, context does not help here.
Set-Location $PSScriptRoot
$py = ".venv\Scripts\python.exe"
while (-not (Test-Path "results/mut/phaseF.DONE")) { Start-Sleep -Seconds 30 }
if (-not (Test-Path "results/mut/G0_lora_ctx/lora_log.json")) {
    Write-Output "$(Get-Date -Format s) start G0"
    $p = Start-Process -FilePath $py -ArgumentList @("-u", "lora_control.py", "--config", "configs/mut_ctx.yaml",
        "--n", "8000", "--eval-every", "1000", "--out", "results/mut/G0_lora_ctx") `
        -RedirectStandardOutput "results/mut/G0_lora_ctx.log" -RedirectStandardError "results/mut/G0_lora_ctx.err" -NoNewWindow -PassThru -Wait
    Write-Output "$(Get-Date -Format s) end G0 exit $($p.ExitCode)"
}
$runs = @(
    @{ name = "G1a_base_ctx"; extra = @() },
    @{ name = "G1b_nodecay_b012_ctx"; extra = @("--set", "plastic.decay_enabled=false", "--set", "plastic.groups.hdeltaH.budget=0.12") }
)
foreach ($s in $runs) {
    $out = "results/mut/$($s.name)"
    if (Test-Path "$out/summary.json") { Write-Output "skip $($s.name)"; continue }
    Write-Output "$(Get-Date -Format s) start $($s.name)"
    $p = Start-Process -FilePath $py -ArgumentList (@("-u", "run_experiment.py", "--config", "configs/mut_ctx.yaml") + $s.extra + @("--out", $out)) `
        -RedirectStandardOutput "$out.log" -RedirectStandardError "$out.err" -NoNewWindow -PassThru -Wait
    Write-Output "$(Get-Date -Format s) end $($s.name) exit $($p.ExitCode)"
}
Set-Content -Path "results/mut/phaseG.DONE" -Value "$(Get-Date -Format s)"
