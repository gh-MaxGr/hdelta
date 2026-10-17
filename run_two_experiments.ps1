# The two experiments that gate the remaining research paths.
#
# A) Fine-tune-like control. The update rule is ALREADY backprop (postsynaptic=error takes
#    autograd.grad wrt delta), so the differences from a LoRA-style fine-tune are passive
#    decay, the homeostatic clamp and single-pass exposure. This turns all three off and
#    gives the exception ~688 writes against the ~172 it has ever had. Decides whether the
#    representational barrier survives ordinary fine-tuning, or whether we measured a
#    property of the online regime.
#
# B) Hidden-state probe on the saved dW. Do agree_wrong inputs separate from agree_right in
#    the representations, even though their outputs are identical? Linear probe fitted and
#    scored on disjoint halves. Decides whether an audit hook exists at all.
#
#   powershell -ExecutionPolicy Bypass -File run_two_experiments.ps1
#
# SEQUENTIAL: two 3B jobs will not fit on a 16 GB card.
$ErrorActionPreference = "Continue"

Write-Output "=== A: fine-tune-like control ($(Get-Date -Format HH:mm:ss)) ==="
& ".venv\Scripts\python.exe" run_experiment.py `
    --config configs/workspace_finetune.yaml `
    --out results/mech_finetune 2>&1 | Select-String -NotMatch "Loading weights"
Write-Output "=== A done ($(Get-Date -Format HH:mm:ss)) ==="

Write-Output "=== B: hidden-state probe ($(Get-Date -Format HH:mm:ss)) ==="
& ".venv\Scripts\python.exe" probe_hidden.py results/gate3_gated 2>&1 |
    Select-String -NotMatch "Loading weights"
Write-Output "=== B done ($(Get-Date -Format HH:mm:ss)) ==="
Write-Output "BOTH EXPERIMENTS COMPLETE"
