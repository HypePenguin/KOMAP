param([switch]$Resume)
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$runnerArguments = @('-u', '-B', '-X', 'utf8', (Join-Path $projectRoot 'src\original_split_retrain.py'))
if ($Resume) { $runnerArguments += '--resume' }
Push-Location -LiteralPath $projectRoot
try {
    & (Join-Path $projectRoot '.venv\Scripts\python.exe') @runnerArguments
    if ($LASTEXITCODE -ne 0) { throw "Original source split training failed: $LASTEXITCODE" }
    & (Join-Path $projectRoot '.venv\Scripts\python.exe') -B -X utf8 (Join-Path $projectRoot 'src\build_original_split_report.py')
    if ($LASTEXITCODE -ne 0) { throw 'Original source split report failed' }
} finally {
    Pop-Location
}
