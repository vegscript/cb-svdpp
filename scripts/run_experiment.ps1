param(
  [Parameter(Mandatory = $true)]
  [string]$ExperimentConfig,
  [string]$RuntimeConfig = "configs/runtime/base.yaml",
  [string]$DatasetConfig,
  [string]$ModelConfig,
  [string]$DeviceConfig
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = (Resolve-Path (Join-Path $PSScriptRoot "..\\src")).Path

$args = @(
  "-m",
  "recsys_lab.cli.main",
  "run-experiment",
  $ExperimentConfig,
  "--runtime-config",
  $RuntimeConfig,
  "--dry-run"
)

if ($DatasetConfig) {
  $args += @("--dataset-config", $DatasetConfig)
}
if ($ModelConfig) {
  $args += @("--model-config", $ModelConfig)
}
if ($DeviceConfig) {
  $args += @("--device-config", $DeviceConfig)
}

python @args
