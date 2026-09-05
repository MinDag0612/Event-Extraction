param([switch]$Cpu, [switch]$SmokeOnly, [switch]$Resume)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
Push-Location $projectRoot
try {
    $pythonPath = Join-Path $projectRoot '.venv-oneie/Scripts/python.exe'
    foreach ($datasetName in @('BKEE', 'GENEVA', 'RAMS', 'VHE')) {
        $smokeArguments = @('-m', 'src.oneie.train', '--dataset', $datasetName, '--smoke')
        if ($Cpu) { $smokeArguments += '--cpu' }
        & $pythonPath @smokeArguments
        if ($LASTEXITCODE -ne 0) { throw "Smoke test failed: $datasetName" }
    }
    if (-not $SmokeOnly) {
        foreach ($datasetName in @('BKEE', 'GENEVA', 'RAMS', 'VHE')) {
            $trainArguments = @('-m', 'src.oneie.train', '--dataset', $datasetName)
            if ($Cpu) { $trainArguments += '--cpu' }
            if ($Resume) { $trainArguments += '--resume' }
            & $pythonPath @trainArguments
            if ($LASTEXITCODE -ne 0) { throw "Training failed: $datasetName" }
        }
    }
} finally {
    Pop-Location
}
