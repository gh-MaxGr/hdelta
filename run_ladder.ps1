# The writability ladder: seven logical structures, same slice, same budget, trained ALONE.
#
# Part D found an associative map compiles (1.00) and a conditional with precedence does not
# (0.0000) at matched writes - two points on an unmapped axis. This fills it in.
#
# Controls that make the comparison mean something:
#   * all seven families pass gate 1 (teacher 0.81-1.00 on generated output, zero-shot 0.000),
#     so a failure to compile is not a failure of the teacher - the confound that cost Part D
#     two runs before anyone measured it
#   * trained one family at a time, so capacity and inter-rule competition are excluded
#   * map / map_wide differ ONLY in rule-text length (149 vs 552 chars) -> isolates verbosity
#   * map_wide / conjunction have IDENTICAL text length -> isolates arity
#   * answer priors differ by family, so score lift over baseline, never raw accuracy
#
#   powershell -ExecutionPolicy Bypass -File run_ladder.ps1
#
# SEQUENTIAL. ~20 min per family, ~2.5 h total.
$ErrorActionPreference = "Continue"
foreach ($fam in "map", "map_wide", "composition", "conjunction", "disjunction", "negation", "precedence") {
    $out = "results/ladder/$fam"
    Write-Output "=== $fam -> $out  ($(Get-Date -Format HH:mm:ss)) ==="
    & ".venv\Scripts\python.exe" run_experiment.py `
        --config "configs/fam_$fam.yaml" --out $out 2>&1 |
        Select-String -NotMatch "Loading weights"
    Write-Output "=== $fam done ($(Get-Date -Format HH:mm:ss)) ==="
}
Write-Output "LADDER COMPLETE"
