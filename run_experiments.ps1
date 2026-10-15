$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$activeTraining = @(Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python' -and $_.CommandLine -like '*notes_recovered.py*--resume*' })
if ($activeTraining.Count -gt 0) { throw 'Corrected training is already running. Check notes_recovered/execution.log; do not start a duplicate.' }
$pythonExe = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$outputDirectory = Join-Path $PSScriptRoot 'notes_recovered'
New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null
& $pythonExe -X utf8 -u "$PSScriptRoot\src\notes_recovered.py" --resume 2>&1 | Tee-Object -FilePath "$outputDirectory\execution.log" -Append
if ($LASTEXITCODE -ne 0) { throw "Corrected training failed; check notes_recovered\execution.log. Exit=$LASTEXITCODE" }
& $pythonExe -X utf8 "$PSScriptRoot\src\audit_results.py" --runs-dir "$outputDirectory\runs" --split-file "$PSScriptRoot\configs\split_group_fold0.json"
if ($LASTEXITCODE -ne 0) { throw "Independent corrected-result audit failed. Exit=$LASTEXITCODE" }
& $pythonExe -X utf8 "$PSScriptRoot\src\build_report.py"
if ($LASTEXITCODE -ne 0) { throw "Report update failed. Exit=$LASTEXITCODE" }
Write-Output 'Corrected five-run training, inference and independent audit complete. Export the verified updated native Sheet to refresh the XLSX.'
