param([switch]$Pilot)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$runRoot = Join-Path $projectRoot 'runs/local_4060'
$lockPath = Join-Path $runRoot 'pipeline.lock'
if (Test-Path -LiteralPath $lockPath) {
    $runningId = (Get-Content -LiteralPath $lockPath -Raw | ConvertFrom-Json).pid
    if (Get-Process -Id $runningId -ErrorAction SilentlyContinue) {
        Write-Output "Pipeline already active, PID $runningId."
        exit 0
    }
}
$pausePath = Join-Path $runRoot 'pause.request'
if (Test-Path -LiteralPath $pausePath) { Remove-Item -LiteralPath $pausePath }
$pythonPath = Join-Path $projectRoot 'work/venv_cuda/Scripts/python.exe'
$stdoutPath = Join-Path $projectRoot 'work/local_pipeline_stdout.log'
$stderrPath = Join-Path $projectRoot 'work/local_pipeline_stderr.log'
$runArguments = @('-X', 'utf8', '-u', '-m', 'scripts.run_local')
if ($Pilot) { $runArguments += '--pilot' }
$job = Start-Process -FilePath $pythonPath -ArgumentList $runArguments -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
Write-Output "Started local pipeline, launcher PID $($job.Id)."
Write-Output "State: $runRoot/pipeline.json"
