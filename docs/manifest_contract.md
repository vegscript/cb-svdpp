# Manifest Contract

## Purpose

Dieses Dokument definiert die kanonischen Manifest-Anforderungen fuer:

- Run-Artefakte
- Benchmark-Artefakte

Manifeste sind der minimale maschinenlesbare Nachweis, dass ein Ergebnis
eindeutig auf eine konkrete Umgebung, Konfiguration, Datengrundlage und
Ausfuehrung zurueckgefuehrt werden kann.

## Why Manifests Are Mandatory

Ohne Manifest ist ein Artefakt spaeter oft nicht mehr sauber beantwortbar in
Bezug auf:

- welches Dataset genau verwendet wurde
- welcher Split benutzt wurde
- welche Seeds liefen
- welches Modell gemeint war
- welches Device-Profil galt
- welches Kommando den Lauf erzeugte
- welche Dateien den Readout tragen

Deshalb gilt:

- Jeder produktive Run braucht ein Manifest.
- Jeder produktive Benchmark braucht ein Manifest.

## Canonical Locations

### Run manifest

```text
artifacts/runs/<run_dir>/run_manifest.json
```

### Benchmark manifest

```text
artifacts/benchmarks/<benchmark_dir>/benchmark_manifest.json
```

## Schema Source Of Truth

Die maschinenlesbaren kanonischen Schemata liegen in:

- [schema/reporting/run_manifest.schema.json](../schema/reporting/run_manifest.schema.json)
- [schema/reporting/benchmark_manifest.schema.json](../schema/reporting/benchmark_manifest.schema.json)

Wenn Text und Schema auseinanderlaufen, muss der Konflikt aufgeloest werden.

## Run Manifest Requirements

Ein Run-Manifest muss mindestens enthalten:

- `manifest_version`
- `kind`
- `generated_at_utc`
- `run_id`
- `status`
- `command`
- `cwd`
- `git.commit`
- `git.dirty`
- `dataset.short_name`
- `dataset.split_family`
- `model.name`
- `runtime.device_profile`
- `runtime.python_version`
- `runtime.dtype`
- `runtime.threading`
- `seeds`
- `artifacts.config_snapshot`
- `artifacts.metrics`
- `artifacts.stdout_log`

Zulaessige Datensatz-Kurzformen fuer Run-Manifeste:

- `ml100k`
- `ml1m`
- `ml10m`
- `ml20m`
- `ml_latest_small` nur als lokaler POC, nie als offizieller Benchmark-Ersatz

Optional, aber dringend empfohlen:

- `dataset.source`
- `dataset.manifest_ref`
- `cb_semantics.alpha`
- `cb_semantics.cluster_contribution_enabled`
- `cb_semantics.cb_claim_eligible`
- `runtime.platform`
- `runtime.hostname`
- `runtime.device_profile_contract`
- `profiling.profile_version`
- `profiling.stages`
- `caches.<cache_name>.status`
- `caches.<cache_name>.cache_fingerprint_sha256`
- `caches.<cache_name>.manifest`
- `dependencies.lockfile`
- `dependencies.environment_hash`
- `timing.started_at_utc`
- `timing.finished_at_utc`

If a run uses persisted local caches, `caches` must report whether each cache
was `hit`, `miss`, or `disabled`. Cache manifests are lineage evidence only;
they do not create speed, scalability, or quality claims without a benchmark.

## Benchmark Manifest Requirements

Ein Benchmark-Manifest muss mindestens enthalten:

- `manifest_version`
- `kind`
- `generated_at_utc`
- `benchmark_id`
- `status`
- `benchmark_scope`
- `command`
- `cwd`
- `git.commit`
- `git.dirty`
- `runtime.device_profile`
- `runtime.python_version`
- `runtime.dtype`
- `runtime.threading`
- `inputs.run_ids`
- `artifacts.summary`

Optional, aber dringend empfohlen:

- `measurement.time_metric`
- `measurement.time_metric_semantics`
- `measurement.sample_unit`
- `measurement.warmup_policy`
- `measurement.warmup_sample_count`
- `measurement.measured_sample_count`
- `inputs.run_manifest_paths`
- `inputs.benchmark_ids`
- `inputs.benchmark_manifest_paths`
- `inputs.model_seeds`
- `inputs.split_seeds`
- `artifacts.tables`
- `artifacts.figures`
- `artifacts.stdout_log`
- `timing.started_at_utc`
- `timing.finished_at_utc`

Hinweis:

- Fuer `completed` Benchmarks muessen `inputs.run_ids` und idealerweise auch
  `inputs.run_manifest_paths` vollstaendig befuellt sein.
- Fuer Multi-Seed-Benchmarks muessen `inputs.benchmark_ids`,
  `inputs.benchmark_manifest_paths` und `inputs.model_seeds` vollstaendig
  befuellt sein.
- Fuer `benchmark_random_v1`-Aggregationen ueber mehrere Split-Seeds soll
  zusaetzlich `inputs.split_seeds` vollstaendig befuellt sein.
- Wenn ein Benchmark einen `measurement`-Block fuehrt, beschreibt dieser die
  kanonische Zeitmetrik, deren Semantik, die Warmup-Politik und die Zahl der
  gemessenen Samples maschinenlesbar.
- Fuer `started`, `failed` oder `cancelled` Benchmarks duerfen diese Listen leer
  sein, wenn der Prozess vor dem ersten gueltigen Run endet.
- Seed-Benchmarks, die in einem Multi-Seed-Readout aggregiert werden, muessen
  einen identischen `git.commit`- und `git.dirty`-Zustand teilen.
- Wenn Benchmark-Zusammenfassungen Zeitmetriken enthalten, muessen deren
  Semantiken dokumentiert sein; bei clustering-basierten Modellen duerfen
  notwendige train-only Vorstufen nicht still ausgeblendet werden.

## Identity Rules

### `run_id`

`run_id` muss mit dem Run-Ordnernamen konsistent sein.

Empfohlene Form:

```text
YYYY-MM-DDTHHMMSSZ_<dataset>_<model>_<profile>_<seedset>
```

### `benchmark_id`

`benchmark_id` muss mit dem Benchmark-Ordnernamen konsistent sein.

Empfohlene Form:

```text
YYYY-MM-DDTHHMMSSZ_<dataset>_<scope>_<profile>
```

## Status Values

Kanonische Statuswerte:

- `started`
- `completed`
- `failed`
- `cancelled`

Freitext-Statuswerte sind nicht erlaubt.

## Path Rules

- Manifest-Pfade muessen repo-relativ oder artefakt-relativ eindeutig sein.
- Keine lokalen absoluten Host-Pfade in Artefaktfeldern.
- Pfade muessen mit den Naming-Conventions kompatibel sein.

## Immutability Rules

- Ein abgeschlossenes Manifest wird nicht still ueberschrieben.
- Korrekturen erzeugen einen neuen Run oder Benchmark.
- Nachtraegliche Annotationen duerfen nur hinzugefuegt werden, wenn die
  urspruengliche Ausfuehrung unveraendert bleibt.

## Claim Rules

Ohne Manifest ist ein Artefakt:

- nicht benchmark-faehig
- nicht report-faehig
- nicht stark claim-faehig

Mit Manifest ist ein Artefakt noch nicht automatisch gueltig, aber zumindest
eindeutig zuordenbar.
