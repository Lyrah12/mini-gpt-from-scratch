# 定位当前实验目录，暂停请求由 Python 流水线在更新边界读取。
$projectRoot = Split-Path -Parent $PSScriptRoot
$runRoot = Join-Path $projectRoot 'runs/local_4060'
if (-not (Test-Path -LiteralPath $runRoot)) { throw 'No local run exists.' }
# 创建请求文件后等待流水线完成当前步、保存检查点并退出。
New-Item -ItemType File -Path (Join-Path $runRoot 'pause.request') -Force | Out-Null
Write-Output 'Pause requested. Wait for pipeline.json phase=paused before shutting down.'
