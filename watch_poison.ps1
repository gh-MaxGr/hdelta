# Writes results/poison.DONE when the four poison runs finish.
# Detection is kept separate from launching so a single path check cannot start a GPU job.
$ErrorActionPreference = "Continue"
while ($true) {
    if (Test-Path "results/poison.log") {
        if (Select-String -Path "results/poison.log" -Pattern "POISON RUNS COMPLETE" -Quiet) {
            New-Item -ItemType File -Force -Path "results/poison.DONE" | Out-Null
            Write-Output "poison runs complete at $(Get-Date -Format HH:mm:ss)"
            break
        }
    }
    Start-Sleep -Seconds 60
}
