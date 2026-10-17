# Gate A: architectural sanity check on the PEFT adapter before Gate B measures transfer.
#
# Cheap and read-only as GPU work goes - loads the 3B once, runs six short forward passes
# and one monitor call. Nothing is trained.
#
# Chained on a marker FILE, like every other driver here. Forward slashes throughout:
# a mangled "results\variance.log" lost its \v once, the wait fell through, and three 3B
# runs collided on a 16 GB card.
#
# PREREQUISITE: peft must already be installed. It is deliberately NOT installed here -
# pip writing into site-packages while a queued run is about to import transformers is an
# avoidable risk, and a driver that mutates the environment is not a driver. Install it
# once the chain is clear:
#     .venv/Scripts/python.exe -m pip install peft
# A dry run confirmed this adds peft only; torch 2.11+cu128 and transformers 5.17 are
# already satisfied and do not move.
$ErrorActionPreference = "Continue"
$marker = "results/poison.DONE"
Write-Output "waiting for $marker ..."
while (-not (Test-Path $marker)) { Start-Sleep -Seconds 60 }

& ".venv/Scripts/python.exe" -c "import peft" 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Output "ABORT: peft is not installed - see the header of this script"
    exit 1
}

Write-Output "=== gate A: PEFT frozen-reference sanity ($(Get-Date -Format HH:mm:ss)) ==="
& ".venv/Scripts/python.exe" gate_a_lora.py `
    --config configs/gate_a_lora.yaml `
    --out results/gate_a 2>&1 | Select-String -NotMatch "Loading weights"
Write-Output "GATE A COMPLETE"
