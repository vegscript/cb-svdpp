param(
  [Parameter(Mandatory = $true)]
  [string]$ProcessedManifest,
  [string]$ModelConfig = "configs/models/svdpp.yaml",
  [string]$RuntimeConfig = "configs/runtime/base.yaml",
  [string]$DeviceConfig = "configs/runtime/devices/local_i5_2500k_24gb.yaml",
  [double]$TrainRatio = 0.8,
  [double]$ValidationRatio = 0.1,
  [int]$SplitSeed = 1,
  [int]$ModelSeed = 1
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = (Resolve-Path (Join-Path $PSScriptRoot "..\\src")).Path

python -m recsys_lab.cli.main train-svdpp `
  $ProcessedManifest `
  --model-config $ModelConfig `
  --runtime-config $RuntimeConfig `
  --device-config $DeviceConfig `
  --train-ratio $TrainRatio `
  --validation-ratio $ValidationRatio `
  --split-seed $SplitSeed `
  --model-seed $ModelSeed
