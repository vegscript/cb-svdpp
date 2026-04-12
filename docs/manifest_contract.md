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

- [schema/reporting/run_manifest.schema.json](G:/Meine Ablage/docs/WU/Semester_4/5255 Applications of Data Science/group_ass/final-project/schema/reporting/run_manifest.schema.json)
- [schema/reporting/benchmark_manifest.schema.json](G:/Meine Ablage/docs/WU/Semester_4/5255 Applications of Data Science/group_ass/final-project/schema/reporting/benchmark_manifest.schema.json)

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
- `runtime.platform`
- `runtime.hostname`
- `dependencies.lockfile`
- `dependencies.environment_hash`
- `timing.started_at_utc`
- `timing.finished_at_utc`

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

- `inputs.run_manifest_paths`
- `artifacts.tables`
- `artifacts.figures`
- `artifacts.stdout_log`
- `timing.started_at_utc`
- `timing.finished_at_utc`

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
