param(
  [Parameter(Mandatory = $true)]
  [string]$Model,
  [Parameter(Mandatory = $true)]
  [string]$ProcessedManifest,
  [Parameter(Mandatory = $true)]
  [string]$ModelConfig,
  [string]$RuntimeConfig = "configs/runtime/base.yaml",
  [string]$DeviceConfig = "configs/runtime/devices/local_i5_2500k_24gb.yaml",
  [string]$ModelSeeds = "1,2,3",
  [string]$BenchmarkManifestPaths = ""
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = (Resolve-Path (Join-Path $PSScriptRoot "..\\src")).Path

$args = @(
  "-m",
  "recsys_lab.cli.main",
  "benchmark-ml100k-paper-multiseed",
  $Model,
  $ProcessedManifest,
  $ModelConfig,
  "--runtime-config",
  $RuntimeConfig,
  "--device-config",
  $DeviceConfig,
  "--model-seeds",
  $ModelSeeds
)

if ($BenchmarkManifestPaths -ne "") {
  $args += @(
    "--benchmark-manifest-paths",
    $BenchmarkManifestPaths
  )
}

python @args
