# Queued after pipeline_ws.py: wait for that process to exit, then run the delusion-probe
# controls on the GPU. Runs are sequential, never concurrent (two 3B models do not fit).
#   Start-Process powershell -ArgumentList @("-File", "run_probe_ctrl.ps1", "-AfterPid", "<pid>")
param([int]$AfterPid = 0)
Set-Location $PSScriptRoot
if ($AfterPid -gt 0) { Wait-Process -Id $AfterPid -ErrorAction SilentlyContinue }
if (Test-Path "results\probe_delusion_ctrl.json") { Write-Output "already done"; exit 0 }
$p = Start-Process -FilePath ".venv\Scripts\python.exe" -ArgumentList @("-u", "probe_delusion_ctrl.py") `
    -RedirectStandardOutput "results\probe_ctrl.log" -RedirectStandardError "results\probe_ctrl.err" `
    -NoNewWindow -PassThru -Wait
Set-Content -Path "results\probe_ctrl.DONE" -Value "exit $($p.ExitCode) $(Get-Date -Format s)"
