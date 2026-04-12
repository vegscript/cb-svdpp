param(
  [Parameter(Mandatory = $true)]
  [string]$DatasetConfig,
  [string]$Dtype = "float32",
  [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = (Resolve-Path (Join-Path $PSScriptRoot "..\\src")).Path

$args = @(
  "-m",
  "recsys_lab.cli.main",
  "prepare-dataset",
  $DatasetConfig,
  "--dtype",
  $Dtype
)

if ($Overwrite) {
  $args += "--overwrite"
}

python @args
