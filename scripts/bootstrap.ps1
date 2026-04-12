$ErrorActionPreference = "Stop"
$env:PYTHONPATH = (Resolve-Path (Join-Path $PSScriptRoot "..\\src")).Path
python -m recsys_lab.cli.main bootstrap-check
