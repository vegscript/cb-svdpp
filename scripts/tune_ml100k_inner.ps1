param(
  [Parameter(Mandatory = $true)]
  [string]$TuningConfig,
  [Parameter(Mandatory = $true)]
  [string]$ProcessedManifest,
  [string]$RuntimeConfig = "configs/runtime/base.yaml",
  [string]$DeviceConfig = "configs/runtime/devices/local_i5_2500k_24gb.yaml"
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = (Resolve-Path (Join-Path $PSScriptRoot "..\\src")).Path

$args = @(
  "-m",
  "recsys_lab.cli.main",
  "tune-ml100k-inner",
  $TuningConfig,
  $ProcessedManifest,
  "--runtime-config",
  $RuntimeConfig,
  "--device-config",
  $DeviceConfig
)

python @args
