# Writes results/variance.DONE once the variance rig genuinely completes.
#
# Separating "detect completion" from "start the next job" means a single fragile path check
# cannot silently launch a concurrent 3B run - which is exactly what happened when a chained
# wait grepped a mangled path, fell through, and put three heavy runs on one 16 GB card.
$ErrorActionPreference = "Continue"
$log = "results/variance.log"
$marker = "results/variance.DONE"
while ($true) {
    if (Test-Path $log) {
        if (Select-String -Path $log -Pattern "VARIANCE RIG COMPLETE" -Quiet) {
            New-Item -ItemType File -Force -Path $marker | Out-Null
            Write-Output "variance rig complete at $(Get-Date -Format HH:mm:ss) - marker written"
            break
        }
    }
    Start-Sleep -Seconds 60
}
