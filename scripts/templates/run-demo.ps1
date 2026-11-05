param([string]$Scenario = 'search', [string]$PythonExecutable = '')
$ErrorActionPreference = 'Stop'
$traceRoot = $PSScriptRoot
$tracePython = if ($PythonExecutable) { $PythonExecutable } else { Join-Path $traceRoot '.venv/Scripts/python.exe' }
if ($PythonExecutable -and -not (Test-Path -LiteralPath $tracePython)) { throw 'The supplied Python executable is missing.' }
if (-not (Test-Path -LiteralPath $tracePython)) {
    & py -3.11 -m venv (Join-Path $traceRoot '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.11 is required to create the demo environment.' }
    $traceRuntime = Get-Content -LiteralPath (Join-Path $traceRoot 'ENVIRONMENT.json') -Raw | ConvertFrom-Json
    $traceTorchSpec = 'torch==' + $traceRuntime.torch_cpu_install_version
    & $tracePython -m pip install $traceTorchSpec --index-url https://download.pytorch.org/whl/cpu
    if ($LASTEXITCODE -ne 0) { throw 'CPU PyTorch installation failed.' }
    & $tracePython -m pip install -r (Join-Path $traceRoot 'requirements-runtime.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Pinned runtime dependency installation failed.' }
    $traceWheel = Get-ChildItem -LiteralPath (Join-Path $traceRoot 'dist') -Filter '*.whl' | Select-Object -First 1
    & $tracePython -m pip install $traceWheel.FullName
    if ($LASTEXITCODE -ne 0) { throw 'TRACE installation failed.' }
}
& $tracePython -I -m trace_ranking demo --bundle $traceRoot --scenario $Scenario
if ($LASTEXITCODE -ne 0) { throw 'TRACE demo did not complete.' }
