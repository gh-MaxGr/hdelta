# Connect-Four PHASE B, queued behind phase A (waits for results/c4/phaseA.DONE).
#   B1  LoRA control: ordinary gradient training on the SAME locus (q/k/v/o, last 9 layers),
#       the SAME 16000 stream positions seen once, the SAME held-out probes. Separates "the
#       fast-weight rule can't learn this" from "a 3B model reading the board as text can't".
# Prediction: LoRA clears the reason cache (0.570) well before 16000 and ends near the MLP
# (0.65-0.72) - the representation is not the bottleneck.
Set-Location $PSScriptRoot
$py = ".venv\Scripts\python.exe"
while (-not (Test-Path "results/c4/phaseA.DONE")) { Start-Sleep -Seconds 30 }
$stages = @(
    @{ name = "B1_lora_s0"; args = @("-u", "c4_lora.py", "--n", "16000", "--out", "results/c4/B1_lora_s0") }
)
foreach ($s in $stages) {
    if (Test-Path "results/c4/$($s.name)/lora_log.json") {
        $done = (Get-Content "results/c4/$($s.name)/lora_log.json" | ConvertFrom-Json).log[-1].n
        if ($done -ge 16000) { Write-Output "skip $($s.name)"; continue }
    }
    Write-Output "$(Get-Date -Format s) start $($s.name)"
    $p = Start-Process -FilePath $py -ArgumentList $s.args -RedirectStandardOutput "results/c4/$($s.name).log" `
        -RedirectStandardError "results/c4/$($s.name).err" -NoNewWindow -PassThru -Wait
    Write-Output "$(Get-Date -Format s) end $($s.name) exit $($p.ExitCode)"
}
Set-Content -Path "results/c4/phaseB.DONE" -Value "$(Get-Date -Format s)"
