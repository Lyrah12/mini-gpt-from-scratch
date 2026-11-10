$projectRoot = Split-Path -Parent $PSScriptRoot
$runRoot = Join-Path $projectRoot 'runs/local_4060'
if (-not (Test-Path -LiteralPath $runRoot)) { throw 'No local run exists.' }
New-Item -ItemType File -Path (Join-Path $runRoot 'pause.request') -Force | Out-Null
Write-Output 'Pause requested. Wait for pipeline.json phase=paused before shutting down.'
