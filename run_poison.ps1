# Does CORRELATED verifier failure degrade gracefully, or cannibalise the rule system?
#
# This is the load-bearing question for any UI-gated continual-learning loop - the
# implicit-feedback verifier of a chat client, where the signal is "the user did not
# hit regenerate". That verifier's failures are correlated by construction: a user who
# misses one hallucination misses similar ones, and nobody regenerates over a stylistic tic.
#
# Random contamination at ~3% cost ~7% capacity. Correlated failure has never been run,
# and it is listed in our own notes as the real deployment risk.
#
#   honest      0.0   control - map compiles to +1.000 in ~100 verified writes
#   systematic  0.3   the model's OWN wrong answers accepted on a fixed input subset
#   corrupt     0.1   the verifier BELIEVES the wrong label and teaches it
#   corrupt     0.3   same, three times the corrupted surface
#
# The two modes differ in when they fire. `systematic` needs the model to still be making
# errors, so it goes quiet once a skill compiles. `corrupt` keeps firing forever, which is
# what a permissive real verifier does - it never stops mislabelling the cases nobody checks.
#
# What separates graceful from catastrophic: accuracy on the CLEAN inputs. If poison occupies
# its own slots and clean mappings stay at 1.00, the cache is lossy but usable. If clean
# accuracy falls, the correlated error is cannibalising the valid rule and the design is dead.
#
# Chained on a marker file. ~15 min per run.
$ErrorActionPreference = "Continue"
# NOT mlpvar.DONE - run_cloudcheck.ps1 already waits on that one, and two jobs
# firing on a single marker is how three 3B runs ended up on one card earlier.
$marker = "results/cloudcheck.DONE"
Write-Output "waiting for $marker ..."
while (-not (Test-Path $marker)) { Start-Sleep -Seconds 60 }

foreach ($cfg in "poison_honest_00", "poison_systematic_03", "poison_corrupt_01", "poison_corrupt_03") {
    Write-Output "=== $cfg ($(Get-Date -Format HH:mm:ss)) ==="
    & ".venv/Scripts/python.exe" run_experiment.py --config "configs/$cfg.yaml" `
        --out "results/poison/$cfg" 2>&1 | Select-String -NotMatch "Loading weights"
}
Write-Output "POISON RUNS COMPLETE"
