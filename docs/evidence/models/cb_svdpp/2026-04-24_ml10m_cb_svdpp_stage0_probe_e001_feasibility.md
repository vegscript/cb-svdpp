# Evidence Note

- date: `2026-04-24`
- scope: `ml10m` single-epoch clustering-model feasibility probe
- model: `cb_svdpp`
- config: `configs/models/tuned/ml10m_cb_svdpp_stage0_probe_e001.yaml`
- git_commit: `c1d2e1d270311cde2b89f9eb2f6a5b2096d13bce`
- git_dirty: `false`

## Purpose

Measure whether the clustering-based `cb_svdpp` path can complete a bounded
`ml10m` run on the local `local_i5_2500k_24gb` device profile after the
canonical CLI/cache wiring was repaired.

## Inputs

- processed manifest:
  `data/processed/ml10m/ml10m_benchmark_random_v1_explicit_v1_float32_manifest.json`
- model config:
  `configs/models/tuned/ml10m_cb_svdpp_stage0_probe_e001.yaml`
- runtime config:
  `configs/runtime/base.yaml`
- device config:
  `configs/runtime/devices/local_i5_2500k_24gb.yaml`
- split family:
  `benchmark_random_v1`
- split contract:
  `train_ratio=0.8`, `validation_ratio=0.1`
- split seed:
  `1`
- model seed:
  `1`
- split cache:
  auto policy, effective disabled
- training index cache:
  disabled

## Artifacts

- run directory:
  `artifacts/runs/2026-04-24T173014Z_ml10m_cb_svdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001/`
- run manifest:
  `artifacts/runs/2026-04-24T173014Z_ml10m_cb_svdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001/run_manifest.json`
- metrics:
  `artifacts/runs/2026-04-24T173014Z_ml10m_cb_svdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001/metrics.json`
- run manifest validation:
  `valid`

## Method

- fix the canonical `train-cb-svdpp` CLI path before running the probe
- commit and push the CLI fix and explicit `ml10m` probe config before
  execution
- verify `main` is clean and synchronized before execution
- run one bounded epoch to measure local feasibility without silently treating
  the probe as a full-transfer benchmark
- run the canonical wrapper command:
  `.\scripts\train_cb_svdpp.ps1 data\processed\ml10m\ml10m_benchmark_random_v1_explicit_v1_float32_manifest.json --model-config configs\models\tuned\ml10m_cb_svdpp_stage0_probe_e001.yaml --runtime-config configs\runtime\base.yaml --device-config configs\runtime\devices\local_i5_2500k_24gb.yaml --split-family benchmark_random_v1 --train-ratio 0.8 --validation-ratio 0.1 --split-seed 1 --model-seed 1 --disable-training-index-cache`

## Readout

- status:
  `completed`
- Git dirty:
  `false`
- interactions:
  `10000054`
- train rows:
  `8000065`
- validation rows:
  `1000005`
- test rows:
  `999984`
- epochs:
  `1`
- user clusters:
  `80`
- item clusters:
  `80`
- `r_star` density:
  `0.834375`
- induction train RMSE:
  `0.872239`
- train RMSE:
  `0.865231`
- validation RMSE:
  `0.872094`
- test RMSE:
  `0.871385`
- cluster induction wall clock seconds:
  `78.425572`
- main training wall clock seconds:
  `422.423619`
- effective fit time seconds:
  `500.849191`
- inference wall clock seconds:
  `15.853330`
- peak memory MB:
  `12730.062500`
- model size MB:
  `105.085598`
- ratings per second train:
  `18938.488849`
- ratings per second inference:
  `630785.716702`

## Interpretation

This run proves that the current `cb_svdpp` implementation can complete one
bounded `ml10m` clustering-model probe on the local device with a valid run
manifest and a clean Git snapshot.

This is not a final benchmark anchor. It is single-seed, single-epoch,
not tuned on `ml10m`, and not matched in training budget to the completed
`ml10m biased_mf` feasibility baseline. Therefore the RMSE values are
diagnostic only and must not be used for a final model-comparison claim.

The peak memory readout stays below the documented 80 percent RAM guardrail for
the local 24 GB profile, but the measured effective fit time is already about
8.35 minutes for one epoch plus one-epoch induction. Full-transfer feasibility
still requires an explicit bounded decision, because changing the epoch budget
also changes the induction model budget in this runner.

## Decision

- Mark `ml10m cb_svdpp` as `single_epoch_feasibility`.
- Keep final `ml10m` model-comparison claims blocked.
- Do not run a full `ml10m cb_svdpp` transfer on the local device unless the
  plan explicitly accepts a multi-hour local run or moves the full transfer to a
  stronger device.
- The next lower-risk publish-scope step is the data-first `ml20m` path, unless
  a deliberate decision is made to spend the time budget on deeper `ml10m`
  clustering probes.
