# Writes results/cloudcheck.DONE when the cloud-probe validation finishes.
$ErrorActionPreference = "Continue"
while ($true) {
    if (Test-Path "results/cloudcheck.log") {
        if (Select-String -Path "results/cloudcheck.log" -Pattern "CLOUD CHECK COMPLETE" -Quiet) {
            New-Item -ItemType File -Force -Path "results/cloudcheck.DONE" | Out-Null
            Write-Output "cloud check complete at $(Get-Date -Format HH:mm:ss)"
            break
        }
    }
    Start-Sleep -Seconds 60
}
