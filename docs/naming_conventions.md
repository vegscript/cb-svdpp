# Naming Conventions

## Purpose

Dieses Dokument definiert die kanonischen Benennungsregeln fuer Dateien,
Verzeichnisse, Artefakte, Evidence-Writeups, Konfigurationen und
experimentbezogene Outputs in diesem Repo.

Ziel ist:

- lexikographisch sortierbare Namen
- sofort lesbare Semantik
- keine mehrdeutigen Artefakte
- keine `final_v2_really`-Dateinamen
- saubere Portabilitaet ueber Betriebssysteme hinweg

## Global Naming Rules

Diese Regeln gelten repo-weit, sofern nicht explizit anders angegeben:

- nur ASCII
- nur Kleinbuchstaben fuer Dateinamen
- Woerter mit `_` trennen, nicht mit Leerzeichen
- keine Leerzeichen
- keine Umlaute
- keine Sonderzeichen ausser `_`, `-` und `.`
- keine Namen wie `tmp`, `new`, `final`, `latest`, `misc`, `stuff`
- keine inkrementellen Suffixe wie `_final2`, `_new3`, `_fixed`

## Date Prefix Rules

Wenn Dateien zeitlich oder experimentell relevant sind, beginnen sie mit einem
kanonischen Praefix.

Es gibt zwei Standardformen:

### Date-only prefix

Fuer statische Writeups, Entscheidungen und Evidence-Dateien:

```text
YYYY-MM-DD_<slug>.md
```

Beispiel:

```text
2026-04-12_cb_objective_gap_analysis.md
```

### Timestamp prefix

Fuer Run-Artefakte, Benchmarks und generierte Outputs:

```text
YYYY-MM-DDTHHMMSSZ_<slug>
```

Beispiel:

```text
2026-04-12T181530Z_ml20m_svdpp_seed004
```

## Timezone Rule

- Fuer Run-, Benchmark- und maschinell erzeugte Artefakte wird UTC verwendet.
- Fuer datierte Markdown-Evidence-Dateien ist `YYYY-MM-DD` zulaessig.
- Wenn ein Readout im Inhalt lokale Zeit verwendet, muss die Zeitzone explizit
  genannt sein.

## Slug Rules

Ein `slug` ist:

- kurz
- beschreibend
- stabil
- ASCII-only
- lowercase
- mit `_` getrennt

Gute Beispiele:

- `ml100k_biased_mf_baseline`
- `cb_asvdpp_alpha_sweep`
- `split_leakage_check`

Schlechte Beispiele:

- `test`
- `new`
- `final_version`
- `results_better`

## Canonical Directory Naming

### `configs/`

Konfigurationsdateien folgen:

```text
<family>/<name>.yaml
```

Beispiele:

- `configs/data/movielens_100k.yaml`
- `configs/models/asymmetric_svd.yaml`
- `configs/runtime/devices/local_i5_2500k_24gb.yaml`

### `artifacts/runs/`

Jeder Run bekommt ein eigenes Verzeichnis:

```text
artifacts/runs/YYYY-MM-DDTHHMMSSZ_<dataset>_<model>_<profile>_<seedset>/
```

Beispiel:

```text
artifacts/runs/2026-04-12T181530Z_ml1m_svdpp_local_i5_2500k_s001-005/
```

Pflichtinhalte eines Run-Ordners:

- `run_manifest.json`
- `config_snapshot.yaml`
- `metrics.json`
- `stdout.log`
- optionale weitere Artefakte

### `artifacts/benchmarks/`

Benchmark-Ordner folgen:

```text
artifacts/benchmarks/YYYY-MM-DDTHHMMSSZ_<dataset>_<scope>_<profile>/
```

Beispiel:

```text
artifacts/benchmarks/2026-04-12T194000Z_ml20m_train_throughput_local_i5_2500k/
```

### `artifacts/figures/`

Abbildungen folgen:

```text
YYYY-MM-DD_<topic>_<dataset>_<model>.png
```

Beispiel:

```text
2026-04-12_rmse_vs_k_ml100k_cb_svdpp.png
```

### `artifacts/debug/`

Debug-Artefakte folgen:

```text
YYYY-MM-DDTHHMMSSZ_<topic>_<context>
```

Debug-Dateien sind nie Source of Truth fuer Ergebnisclaims.

## Evidence File Naming

Evidence-Dateien folgen:

```text
docs/evidence/<track>/<workstream>/YYYY-MM-DD_<slug>.md
```

Wenn `docs/evidence/` spaeter eingefuehrt wird, ist dies die kanonische Form.

Bis dahin gilt fuer methodische Spezialnotizen:

```text
docs/methodology/YYYY-MM-DD_<slug>.md
```

Wichtig:

- Datum immer am Anfang
- ein Thema pro Datei
- keine undatierten Evidence-Dateien

## Data Artifact Naming

Abgeleitete Datendateien unter `data/interim` oder `data/processed` muessen
mindestens diese Felder im Namen oder im benachbarten Manifest sichtbar machen:

- dataset
- version oder split-family
- preprocessing-family
- optional dtype

Beispiel:

```text
data/processed/ml20m_benchmark_random_v1_float32.parquet
```

Wenn der Dateiname zu lang wird, muss eine benachbarte Manifest-Datei die
fehlenden Metadaten tragen.

## Model Artifact Naming

Persistierte Modellgewichte oder Parameter-Snapshots folgen:

```text
YYYY-MM-DDTHHMMSSZ_<dataset>_<model>_<profile>_<seed>_<stage>.npz
```

Beispiel:

```text
2026-04-12T201000Z_ml10m_svdpp_local_i5_2500k_s003_epoch020.npz
```

## Seed Naming

Seeds werden kanonisch als:

```text
s001
s002
s003
```

geschrieben.

Seed-Spannen werden geschrieben als:

```text
s001-005
```

Keine uneinheitlichen Formen wie `seed1`, `seed_1`, `01`, `five_seeds`.

## Dataset Short Names

Kanonische Kurzformen:

- `ml100k`
- `ml1m`
- `ml10m`
- `ml20m`

Lokaler POC-Ausnahmefall:

- `ml_latest_small`

Keine Mischformen wie:

- `movielens100k`
- `movie_lens_100k`
- `ml-100k`

## Model Short Names

Kanonische Kurzformen:

- `biased_mf`
- `svdpp`
- `asymmetric_svd`
- `asvdpp`
- `cb_svdpp`
- `cb_asvdpp`

## Reserved Suffixes

Erlaubte stabile Suffixe:

- `_manifest`
- `_metrics`
- `_config_snapshot`
- `_summary`
- `_report`
- `_plot`
- `_epochNNN`

Nicht erlaubte informelle Suffixe:

- `_new`
- `_better`
- `_final`
- `_final2`
- `_last`
- `_real_final`

## Manifest Requirement

Wenn ein Dateiname nicht alle relevanten Metadaten tragen kann, muss im selben
Ordner ein Manifest liegen.

Mindestens fuer Runs und Benchmarks ist ein Manifest Pflicht.

## Immutability Rule

- Run-Ordner werden nach Abschluss nicht umbenannt.
- Ergebnisartefakte werden nicht ueberschrieben, sondern neu erzeugt.
- Korrekturen erzeugen neue datierte Artefakte.

## Compression And Export Naming

Exports folgen:

```text
YYYY-MM-DDTHHMMSSZ_<scope>_<dataset>_<model>_<profile>.<ext>
```

Beispiele:

- `2026-04-12T211500Z_benchmark_ml20m_cb_svdpp_local_i5_2500k.csv`
- `2026-04-12T211500Z_metrics_ml1m_svdpp_local_i5_2500k.json`

## Constraint Summary

Wenn ein neuer Dateityp eingefuehrt wird, muessen mindestens diese Fragen
beantwortet sein:

1. Beginnt der Name mit Datum oder Timestamp?
2. Ist der Name ASCII-only und lowercase?
3. Ist der Datensatz klar erkennbar?
4. Ist das Modell klar erkennbar?
5. Ist Seed oder Seed-Set klar erkennbar, falls relevant?
6. Ist der Device-Kontext klar erkennbar, falls relevant?
7. Gibt es ein Manifest, wenn der Name nicht ausreicht?
