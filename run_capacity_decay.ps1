# Does Part C's capacity ceiling survive a lower decay rate?
#
# Part C concluded capacity is set by SLICE SIZE "not by amplitude", on an energy sweep that
# scaled `eta` - which moves the deposition/decay equilibrium AND the per-update step size
# together. Part D separated them: the same 2x equilibrium via halved decay cost +0.90 %
# perplexity where raising eta cost +18 %, and lifted a stalled 4-rule run from 0.64 to 0.77
# with three of four rules at 1.00.
#
# These are the two rungs where Part C's ladder failed: 3 rules summed 0.38 of 3, 4 rules
# summed 0.28 of 4. If they lift at decay 0.002, that ceiling was energy, not slice size.
#
#   powershell -ExecutionPolicy Bypass -File run_capacity_decay.ps1
#
# SEQUENTIAL: two 3B runs will not fit on a 16 GB card.
$ErrorActionPreference = "Continue"
foreach ($cfg in "capacity_b40_n3_decay", "capacity_c80_n4_decay") {
    $out = "results/capacity_decay/$cfg"
    Write-Output "=== $cfg -> $out  ($(Get-Date -Format HH:mm:ss)) ==="
    & ".venv\Scripts\python.exe" run_experiment.py `
        --config "configs/$cfg.yaml" `
        --out $out 2>&1 | Select-String -NotMatch "Loading weights"
    Write-Output "=== $cfg done ($(Get-Date -Format HH:mm:ss)) ==="
}
Write-Output "CAPACITY DECAY RUNS COMPLETE"
