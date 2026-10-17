# After the pain chain: three analyses, each answering one question, in order.
#
# Chained on results/pain.DONE, the marker the pain chain writes last. A marker FILE, never a
# log grep - a mangled log path once let a wait fall through and put three 3B runs on one card.
#
#   1  compare_pain.py       does pain change how the skill compiles?       CPU, seconds
#                            W at each threshold with YIELD, against the n=4 baseline
#   2  predict_failure.py    can the instinct tell it is about to be wrong, and does
#                            deferring when it is unsure improve decisions? GPU, 7 runs
#   3  failure_memory.py     does a small readable memory of past failures - with the
#                            reason - improve decisions, and does the reason matter?
#                                                                           GPU, 3 pain runs
#
# Each is read-only on weights already trained. None of them adds a training variable.
$ErrorActionPreference = "Continue"
$marker = "results/pain.DONE"
Write-Output "waiting for $marker ..."
while (-not (Test-Path $marker)) { Start-Sleep -Seconds 60 }

Write-Output "=== 1 compare_pain ($(Get-Date -Format HH:mm:ss)) ==="
& ".venv/Scripts/python.exe" compare_pain.py 2>&1 | Out-File -Encoding utf8 "results/compare_pain.log"

Write-Output "=== 2 predict_failure ($(Get-Date -Format HH:mm:ss)) ==="
& ".venv/Scripts/python.exe" predict_failure.py 2>&1 | Select-String -NotMatch "Loading weights" |
    Out-File -Encoding utf8 "results/predict_failure.log"

Write-Output "=== 3 failure_memory ($(Get-Date -Format HH:mm:ss)) ==="
& ".venv/Scripts/python.exe" failure_memory.py 2>&1 | Select-String -NotMatch "Loading weights" |
    Out-File -Encoding utf8 "results/failure_memory.log"

New-Item -ItemType File -Force -Path "results/analysis.DONE" | Out-Null
Write-Output "ANALYSIS CHAIN COMPLETE at $(Get-Date -Format HH:mm:ss)"
