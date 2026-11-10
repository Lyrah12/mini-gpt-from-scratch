$projectRoot = Split-Path -Parent $PSScriptRoot
$runRoot = Join-Path $projectRoot 'runs/local_4060'
$statePath = Join-Path $runRoot 'pipeline.json'
if (-not (Test-Path -LiteralPath $statePath)) { Write-Output 'Pipeline has not started.'; exit 0 }
$state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
$state | Format-List
if ($state.phase -in @('pretrain','sft','scratch')) {
    $metrics = Join-Path $runRoot ($state.phase + '/metrics.jsonl')
    if (Test-Path -LiteralPath $metrics) { Get-Content -LiteralPath $metrics -Tail 2 }
}
