# 接收试跑开关，使用本机 CUDA 环境在后台启动实验。
param([switch]$Pilot)
$ErrorActionPreference = 'Stop'
# 根据脚本位置定位项目、运行目录和流水线锁。
$projectRoot = Split-Path -Parent $PSScriptRoot
$runRoot = Join-Path $projectRoot 'runs/local_4060'
$lockPath = Join-Path $runRoot 'pipeline.lock'
# 已有锁对应的进程仍在运行时返回，避免重复启动。
if (Test-Path -LiteralPath $lockPath) {
    $runningId = (Get-Content -LiteralPath $lockPath -Raw | ConvertFrom-Json).pid
    if (Get-Process -Id $runningId -ErrorAction SilentlyContinue) {
        Write-Output "Pipeline already active, PID $runningId."
        exit 0
    }
}
# 移除上一轮暂停请求，使本次启动可以从保存的进度继续。
$pausePath = Join-Path $runRoot 'pause.request'
if (Test-Path -LiteralPath $pausePath) { Remove-Item -LiteralPath $pausePath }
# 准备解释器、日志文件和启动参数。
$pythonPath = Join-Path $projectRoot 'work/venv_cuda/Scripts/python.exe'
$stdoutPath = Join-Path $projectRoot 'work/local_pipeline_stdout.log'
$stderrPath = Join-Path $projectRoot 'work/local_pipeline_stderr.log'
$runArguments = @('-X', 'utf8', '-u', '-m', 'scripts.run_local')
if ($Pilot) { $runArguments += '--pilot' }
# 在隐藏窗口中启动，标准输出和错误输出分别保存。
$job = Start-Process -FilePath $pythonPath -ArgumentList $runArguments -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
Write-Output "Started local pipeline, launcher PID $($job.Id)."
Write-Output "State: $runRoot/pipeline.json"
