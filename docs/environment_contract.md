# Environment Contract

## Purpose

Dieses Dokument definiert die kanonische Runtime-, Setup- und
Portabilitaetsumgebung des Repos.

## Canonical Toolchain

- Python: `3.10.x`
- Environment und Locking: `uv`
- Build- und Dependency-Quelle: `pyproject.toml` und `uv.lock`

Der lokale Referenzrechner laeuft aktuell mit `Python 3.10.7`.
Der aktuelle kanonische `uv`-Toolchain-Nachweis verwendet `uv 0.11.7`.

## Canonical Setup Path

Ein neues Geraet soll ueber einen kanonischen Setup-Pfad einsatzbereit werden.

Der Zielpfad ist:

```bash
uv sync --extra dev
```

Wrapper-Skripte duerfen spaeter denselben Pfad kapseln, aber keine zweite
Dependency-Wahrheit einfuehren.

## Forbidden Environment Practices

- kein `pip install` als stiller lokaler Sonderpfad
- kein zweites Environment-Management als paralleler Standard
- keine lokal manuell gesetzten Hardcoded-Pfade im Produktionscode
- keine implizite Thread-Konfiguration ohne dokumentiertes Device-Profil

## Device Profiles

Device-Profile sind die kanonische Stelle fuer:

- Thread-Anzahl
- BLAS-/OpenMP-Thread-Anzahl
- Default-Dtype
- Speicher- und Cache-Pfade
- RAM-Guardrail fuer claim-eligible Runs
- Benchmark-Schalter
- spaetere GPU-Schalter

## Model Config Contract

Produktive CLI-, Benchmark- und Tuning-Pfade duerfen Modellkonfigurationen nur
ueber die Pydantic-`ModelProfile`-Schemas und den `ModelAdapter` bauen.

Der zulaessige Produktionspfad ist:

```text
YAML payload -> validate_model_config_payload -> ModelAdapter.build_model_config
```

Interne Modell-Config-Dataclasses duerfen Defaults fuer fokussierte Tests und
direkte Modell-Unit-Tests behalten. Diese Defaults sind keine produktive
Config-Quelle. Produktive Pfade duerfen deshalb keine Modell-Dataclass direkt
aus YAML-Dicts konstruieren und keine `.get(..., default)`- oder
`.setdefault(...)`-Fallbacks fuer Modellparameter verwenden.

Abgeleitete Benchmark- oder Tuning-Kandidaten muessen nach jeder
programmatischen Override-Erzeugung erneut mit `validate_model_config_payload`
validiert werden. Felder wie `init_std`, `training_backend`, `lambda_xC` und
`lambda_yC` muessen explizit im Schema stehen und duerfen nicht still implizit
ergaenzt werden.

Lokale Referenzhardware:

- CPU: `Intel Core i5-2500K`
- Threads: `4`
- RAM: `24 GB`
- GPU: `GT 1030 2 GB`
- SSD: `Samsung SSD 850 EVO 500GB`
- HDD: `TOSHIBA DT01ACA300 3TB`

Default-Regeln fuer dieses Profil:

- CPU-first
- `float32` als Performance-Default
- `float64` fuer Referenz- und Validierungslaufe
- aktive Caches und Artefakte auf SSD
- grosse Rohdaten oder Archive optional auf HDD

## Environment Variables

Wenn Pfade oder Device-Verhalten per Environment konfiguriert werden, dann nur
ueber klar benannte Variablen, zum Beispiel:

- `RECSYS_DATA_ROOT`
- `RECSYS_ARTIFACT_ROOT`
- `RECSYS_CACHE_ROOT`
- `RECSYS_DEVICE_PROFILE`

Keine undokumentierten lokalen Zusatzvariablen.

Wenn `RECSYS_CACHE_ROOT` verwendet wird, dann ist dies der kanonische Ort fuer
abgeleitete, rebuildbare Runtime-Caches wie split-spezifische Trainingsindizes.
Solche Caches duerfen niemals die kanonischen Datenartefakte unter
`data/processed/` ersetzen.

## Threading Contract

Benchmark-Laufe muessen ihre Thread-Konfiguration explizit dokumentieren.

Mindestens relevant:

- `OMP_NUM_THREADS`
- `MKL_NUM_THREADS`
- `OPENBLAS_NUM_THREADS`
- `NUMEXPR_NUM_THREADS`

Wenn Threading relevant fuer einen Claim ist, muss es im Benchmark-Readout
stehen.

## Claim-Eligible Device Profile Contract

Ein Device-Profil ist fuer claim-eligible Runs nur zulaessig, wenn der
maschinelle Runtime-Contract besteht.

Pflichtfelder:

- `metadata.status` darf nicht `draft`, `template` oder `placeholder` sein
- `device_profile.name`
- `device_profile.compute_class`
- `device_profile.cpu_model`
- `device_profile.logical_threads`
- `device_profile.physical_cores`
- `device_profile.ram_gb`
- `device_profile.gpu_enabled`
- `storage.cache_preference`
- `storage.archive_preference`
- `threading.omp_num_threads`
- `threading.blas_threads`
- `resource_limits.ram_guardrail_fraction`
- `precision.default_dtype`
- `precision.reference_dtype`

Preflight:

```powershell
.venv\Scripts\python.exe -m recsys_lab.cli.main validate-runtime-profile configs\runtime\devices\local_i5_2500k_24gb.yaml --claim-eligible
```

Das generische `hpc_cpu.yaml` bleibt ein Draft-Template und muss mit
`--claim-eligible` fehlschlagen, solange CPU-, Thread- und RAM-Felder nicht
konkret fuer einen Cluster ersetzt wurden.

## Portability Contract

Jeder produktive Pfad muss:

- auf einem neuen Geraet ohne manuelle Quellcodeaenderung laufen koennen
- ohne absolute lokale Dateipfade auskommen
- ohne `it runs on my machine`-Sonderbehandlung auskommen

Wenn ein Schritt lokalspezifisch ist, muss er:

- im Device-Profil abbildbar sein
- dokumentiert sein
- die Portabilitaet des Kernpfads nicht zerstoeren

## Process Hygiene Contract

Das Repo soll keine unnoetigen lang laufenden Prozesse als Normalfall
erzeugen.

Deshalb gilt:

- Hilfsprozesse nach Task-Ende schliessen
- keine stillen Watcher oder lokalen Server ohne ausdruecklichen Bedarf
- keine vermeidbaren Parallelhosts offen halten
- bei laengeren Runs auf Speicherverbrauch achten

Operativer Default:

- einmalskriptartige Kommandos statt dauerhafter Session-Prozesse
- grossere Datenpfade bevorzugt streaming-, chunk- oder dateibasiert statt
  unnoetig speicherresident
- keine blind aggressive Prozessbereinigung, wenn die Prozessherkunft nicht
  klar ist

## Benchmark Mode

Performance-Benchmarks duerfen nur unter explizit dokumentierten
Rahmenbedingungen berichtet werden:

- Device-Profil
- Python-Version
- Dtype
- Thread-Konfiguration
- Dataset und Split
- Seed-Politik
- Warmup oder kein Warmup

Ohne diese Angaben ist ein Performance-Claim unvollstaendig.
