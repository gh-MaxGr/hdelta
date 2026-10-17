# Which of the three candidate mechanisms starves the exception branch?
#
# Teacher-forced accuracy on exceptions is 0.0073, identical to generated, so the mapping
# is not in the weights at all - this is a learning failure, not a decoding one. Three
# candidates remain, and each arm moves exactly one variable from the workspace_exc run:
#
#   h2_ratio   exception_fraction 0.68  -> ~1:1 verified writes instead of ~1:13
#   h1_decay   decay 0.004 -> 0.002     -> doubles the eta/decay equilibrium
#
# H3 (representational) is tested by elimination: if the exception branch stays at zero in
# both arms, neither exposure nor decay explains it. That is weaker than a direct test and
# should be reported as elimination, not proof.
#
#   powershell -ExecutionPolicy Bypass -File run_mechanism.ps1
#
# SEQUENTIAL: two 3B runs will not fit on a 16 GB card.
$ErrorActionPreference = "Continue"
foreach ($arm in "h2_ratio", "h1_decay") {
    $out = "results/mech_$arm"
    Write-Output "=== $arm -> $out  ($(Get-Date -Format HH:mm:ss)) ==="
    & ".venv\Scripts\python.exe" run_experiment.py `
        --config "configs/workspace_$arm.yaml" `
        --out $out 2>&1 | Select-String -NotMatch "Loading weights"
    Write-Output "=== $arm done ($(Get-Date -Format HH:mm:ss)) ==="
}
Write-Output "BOTH ARMS COMPLETE"
