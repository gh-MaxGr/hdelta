# Writes results/mlpvar.DONE when the MLP variance runs finish.
# Detection is kept separate from launching so a single path check cannot start a GPU job.
$ErrorActionPreference = "Continue"
$log = "results/mlpvar.log"
$marker = "results/mlpvar.DONE"
while ($true) {
    if (Test-Path $log) {
        if (Select-String -Path $log -Pattern "MLP VARIANCE COMPLETE" -Quiet) {
            New-Item -ItemType File -Force -Path $marker | Out-Null
            Write-Output "mlp variance complete at $(Get-Date -Format HH:mm:ss)"
            break
        }
    }
    Start-Sleep -Seconds 60
}
