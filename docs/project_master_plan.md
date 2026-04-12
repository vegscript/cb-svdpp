# Project Master Plan

Stand: 2026-04-12

## 1. Projektziel

Ziel ist eine konzeptionell und methodisch integre Reproduktion des Papers
`Clustering-Based Factorized Collaborative Filtering` mit einem sauberen,
modularen, skalierbaren und reproduzierbaren Engineering-Setup.

Wir bauen nicht einfach "ein recommender script", sondern eine Forschungs- und
Benchmarking-Plattform fuer die folgende Modellleiter:

1. `BiasedMF`
2. `SVD++`
3. `ASVD++`
4. `CB-SVD++`
5. `CB-ASVD++`

Die Plattform muss:

- mehrere MovieLens-Datasets einheitlich verarbeiten
- mathematisch sauber dokumentiert sein
- methodisch korrekt benchmarken
- auf dem lokalen Zielgeraet performant laufen
- auf neue Geraete portierbar sein
- mit einem einzigen Setup-Befehl auf einem neuen Geraet einsatzbereit werden

## 2. Feststehende Entscheidungen

### 2.1 Datasets

Offizieller Scope:

- `MovieLens 100K`
- `MovieLens 1M`
- `MovieLens 10M`
- `MovieLens 20M`

### 2.2 Primare Zielhardware

Default-Optimierung erfolgt fuer den aktuellen lokalen PC:

- CPU: `Intel Core i5-2500K @ 3.30 GHz`
- Cores / Threads: `4 / 4`
- RAM: `24 GB`
- GPU: `NVIDIA GeForce GT 1030, 2 GB`
- Storage:
  - `Samsung SSD 850 EVO 500GB`
  - `TOSHIBA DT01ACA300 3TB HDD`
- Python: `3.10.7`

Konsequenz:

- Architektur ist `CPU-first`
- GPU ist nicht Teil des Default-Hot-Path
- aktive Daten, Caches und Artefakte muessen auf SSD liegen
- Archive und rohe Downloads koennen auf HDD liegen

### 2.3 Performance-Ziel

Wir optimieren gleichzeitig fuer:

- `RMSE`
- Training Throughput
- Inference Throughput
- Speicherverbrauch
- Reproduzierbarkeit

Wichtig:

Dieses Projekt zielt auf `state-of-the-art engineering quality innerhalb dieser
Modellfamilie`, nicht auf einen unfairen Vergleich mit modernen Deep Learning
Recommendern ausserhalb des Paper-Scopes.

### 2.4 Entwicklungsprinzip

Wir implementieren erst dann, wenn fuer jede Schicht klar ist:

- mathematische Formel
- Zielobjektiv
- Regularisierung
- Update-Regel
- Dateneingang
- erwartete Outputs
- Benchmark-Kriterien

## 3. Terminologie und Modellnamen

Um Unklarheiten zu vermeiden, verwenden wir intern nur diese kanonischen Namen:

- `biased_mf`
- `svdpp`
- `asvdpp`
- `cb_svdpp`
- `cb_asvdpp`

Hinweis:

Das `CB` im Paper bedeutet hier `clustering-based`, nicht `content-based`.

## 4. Architekturprinzipien

## 4.1 Nicht verhandelbar

- keine Data Leakage
- keine Modelllogik in Notebooks
- keine verstreuten Ad-hoc-Skripte im Repo-Root
- keine Black-Box-Recommender-Library fuer die Kernmodelle
- keine stillen Abweichungen vom Paper
- keine ungeplanten Dateiausgaben ausserhalb klarer Artefaktordner

## 4.2 Zielbild

Das Repo ist eine `konfigurationsgetriebene Forschungsplattform`, kein loses
Skript-Sammelsurium.

Jeder Run muss eindeutig bestimmt sein durch:

- Dataset
- Split-Protokoll
- Seed
- Modellvariante
- Hyperparameter
- Device-Profil
- Precision-Profil
- Commit-Stand

## 5. Technologie-Stack

## 5.1 Sprache

- `Python` als Projekt- und Forschungssprache

Begruendung:

- fuer Uni-Projekt sinnvoll
- schnell genug fuer Orchestrierung, Daten-Pipeline und Benchmarking
- performant genug fuer diese Modellfamilie, wenn der Hot-Path korrekt gebaut ist

## 5.2 Kernbibliotheken

Pflicht-Stack:

- `numpy` fuer dichte numerische Arrays
- `scipy` fuer sparse Strukturen und lineare Algebra
- `numba` fuer beschleunigte Trainingskernels
- `scikit-learn` fuer `KMeans` und `MiniBatchKMeans`
- `pyarrow` fuer Spaltenformate und Parquet
- `polars` fuer schnelles tabellarisches Preprocessing
- `pydantic` fuer strikte Konfigurationsvalidierung
- `PyYAML` fuer Konfigurationsdateien
- `threadpoolctl` zur Kontrolle von BLAS/OpenMP-Threads
- `psutil` fuer Speicher- und Laufzeitmessung

Dev-Stack:

- `pytest`
- `hypothesis`
- `ruff`
- `mypy`
- `pre-commit`

CLI / Orchestrierung:

- `typer`

## 5.3 Was wir bewusst nicht als Default verwenden

- `pandas` nicht im Hot-Path
- `Spark` nicht in Phase 1
- `PyTorch` oder `TensorFlow` nicht fuer die Kernmodelle
- `Jupyter` nicht fuer Kernlogik
- keine GPU-first Architektur

Begruendung:

MovieLens bis 20M ist fuer eine saubere Single-Node-CPU-Architektur beherrschbar.
Verteilte Infrastruktur wuerde die methodische und technische Komplexitaet frueh
erhoehen, ohne in Phase 1 einen realen Vorteil zu garantieren.

## 6. Setup- und Portabilitaetsstrategie

Ziel:

Ein neues Geraet soll mit einem Befehl einsatzbereit werden.

Entscheidung:

- Paket- und Environment-Management ueber `uv`
- Projektkonfiguration ueber `pyproject.toml`
- gelockte Abhaengigkeiten ueber `uv.lock`

Geplanter Setup-Befehl:

```bash
uv sync --extra dev
```

Ergaenzend:

- `scripts/bootstrap.ps1`
- `scripts/bootstrap.sh`

Diese Wrapper duerfen spaeter lokale Checks, Pfadvalidierung und Device-Profile
automatisieren, ohne die Kernstruktur zu verkomplizieren.

## 7. Daten- und Storage-Strategie

## 7.1 Datenhaltung

Wir trennen strikt:

- `raw`: heruntergeladene Originaldaten
- `interim`: normalisierte und remappte Zwischendaten
- `processed`: trainingsfertige Arrays und Sparse-Strukturen
- `artifacts`: Ergebnisse, Logs, Plots, Benchmarks

## 7.2 Performance-Regeln

- IDs werden frueh auf kompakte Integer remappt
- Standard fuer IDs: `int32`
- Standard fuer Ratings/Faktoren im lokalen Performance-Profil: `float32`
- Referenzprofil fuer numerische Validierung: `float64`
- keine DataFrame-Zugriffe im Trainingsinneren
- contiguous arrays und cache-freundliche Speicherlayouts
- user->items und item->users Indizes werden explizit vorbereitet

## 7.3 Out-of-core und verteilte Verarbeitung

Fuer MovieLens bis 20M ist `optimiertes Single-Node In-Memory Training` der
Default.

Trotzdem wird die Datenpipeline so entworfen, dass spaeter moeglich ist:

- memory-mapped Reads
- chunked preprocessing
- Parquet-basierte Zwischenspeicherung

Nicht Teil von Phase 1:

- verteiltes Training
- Spark-basierte Kernlogik

## 8. Modellarchitektur

Zentrale Anforderung:

Die Modelle muessen `komponierbar` sein, sodass dieselben mathematischen Bausteine
in mehreren Modellen wiederverwendet werden und keine Formel still dupliziert oder
 leicht unterschiedlich implementiert wird.

## 8.1 Wiederverwendbare Bausteine

1. `global_mean`
2. `user_bias`
3. `item_bias`
4. `user_embedding`
5. `item_embedding`
6. `implicit_feedback_aggregator`
7. `explicit_feedback_aggregator`
8. `cluster_assignment`
9. `cluster_embedding`
10. `cluster_mix(alpha)`

## 8.2 Modellkomposition

### `biased_mf`

- Bias-Block
- direkte User-/Item-Latents

### `svdpp`

- `biased_mf`
- implizite Rueckmeldungen ueber `y_j`

### `asvdpp`

- `svdpp`
- expliziter Item-seitiger Feedback-Block ueber `x_j`

### `cb_svdpp`

- `svdpp`
- Cluster-Latents fuer User und Items
- Cluster-Mix mit `alpha`

### `cb_asvdpp`

- `asvdpp`
- Cluster-Latents fuer User und Items
- Cluster-Mix mit `alpha`

## 8.3 Kritische Regel

Wenn ein mathematischer Term in zwei Modellen derselbe sein soll, dann muss er
aus demselben Code-Baustein stammen. Keine Copy-Paste-Formeln in mehreren Dateien.

## 9. Evaluationsprotokoll

## 9.1 Primare Metriken

- `RMSE`
- `train_time_total`
- `train_time_per_epoch`
- `ratings_per_second_train`
- `ratings_per_second_inference`
- `peak_memory_mb`

## 9.2 Reproduktionsregeln

- kein Ergebnis aus nur einem Seed berichten
- Kernvergleiche ueber mehrere Seeds
- Mittelwert und Standardabweichung speichern
- hyperparameter tuning nur auf Validation
- Testset nur fuer finale Berichte

## 9.3 Split-Politik

Paper-faithful Ebene:

- fuer `MovieLens 100K` moeglichst nah am Paper-Protokoll

Kanonische Benchmark-Ebene:

- einheitliche, klar definierte Split-Strategie ueber alle MovieLens-Varianten
- random oder zeitbasiert, aber niemals implizit gemischt

Standardannahme fuer Phase 1:

- randomisierte, seed-kontrollierte Splits fuer vergleichbare Ablationen

Spaetere Erweiterung:

- zeitbasierte Splits fuer realistischere Generalisierung

## 10. Repo-Struktur

```text
final-project/
|-- README.md
|-- pyproject.toml
|-- uv.lock
|-- .gitignore
|-- .editorconfig
|-- configs/
|   |-- runtime/
|   |   |-- base.yaml
|   |   `-- devices/
|   |       |-- local_i5_2500k_24gb.yaml
|   |       `-- hpc_cpu.yaml
|   |-- data/
|   |   |-- movielens_100k.yaml
|   |   |-- movielens_1m.yaml
|   |   |-- movielens_10m.yaml
|   |   `-- movielens_20m.yaml
|   |-- models/
|   |   |-- biased_mf.yaml
|   |   |-- svdpp.yaml
|   |   |-- asvdpp.yaml
|   |   |-- cb_svdpp.yaml
|   |   `-- cb_asvdpp.yaml
|   `-- experiments/
|       |-- paper_reproduction.yaml
|       `-- benchmark_matrix.yaml
|-- docs/
|   |-- repo_governance.md
|   |-- environment_contract.md
|   |-- data_and_split_contract.md
|   |-- evaluation_protocol.md
|   |-- project_master_plan.md
|   |-- math/
|   |   |-- notation.md
|   |   |-- objective_functions.md
|   |   `-- update_rules.md
|   |-- methodology/
|   |   |-- splits_and_leakage.md
|   |   |-- evaluation_protocol.md
|   |   `-- deviations_from_paper.md
|   `-- architecture/
|       |-- repo_structure.md
|       |-- data_pipeline.md
|       `-- model_composition.md
|-- src/
|   `-- recsys_lab/
|       |-- cli/
|       |-- config/
|       |-- data/
|       |-- math/
|       |-- components/
|       |-- clustering/
|       |-- models/
|       |-- training/
|       |-- evaluation/
|       |-- experiments/
|       `-- utils/
|-- tests/
|   |-- unit/
|   |-- integration/
|   |-- property/
|   `-- regression/
|-- scripts/
|   |-- bootstrap.ps1
|   |-- bootstrap.sh
|   |-- download_movielens.ps1
|   `-- run_experiment.ps1
|-- data/
|   |-- raw/
|   |-- interim/
|   `-- processed/
`-- artifacts/
    |-- runs/
    |-- benchmarks/
    `-- figures/
```

## 11. Device-Profile-System

Das Projekt wird geraetespezifisch konfigurierbar gemacht.

Ein Device-Profil definiert unter anderem:

- Thread-Anzahl
- BLAS/OpenMP-Konfiguration
- Default-Dtype
- Pfade fuer SSD-Cache und HDD-Archiv
- Batch- und Chunk-Groessen
- Logging-Detailgrad
- Benchmark-Modus

Default-Profil fuer den aktuellen PC:

- `threads = 4`
- `blas_threads = 4`
- `default_dtype = float32`
- `gpu_enabled = false`
- `cache_on_ssd = true`

## 12. Qualitaets- und Hygiene-Regeln

- Root bleibt sauber
- keine Ergebnisdateien im Root
- keine Datenfiles in `src/`
- keine Notebooks ausserhalb eines spaeteren klaren `notebooks/`-Ordners
- alle generierten Outputs unter `artifacts/`
- alle temporaren Daten unter `data/interim` oder `artifacts/tmp`
- alle lokalen Pfade nur in ignorierten lokalen Profilen, nie hart kodiert

## 13. Teststrategie

## 13.1 Unit-Tests

- RMSE gegen Handrechnung
- Mapping und Remapping
- Split-Integritaet
- NaN/Inf Guards
- deterministische Seeds

## 13.2 Property-Tests

- gleiche Seeds erzeugen gleiche Initialisierung
- `alpha = 0` reduziert CB-Varianten auf ihre Nicht-CB-Basis
- leere oder invalide Inputs schlagen explizit fehl

## 13.3 Integrations-Tests

- kompletter Mini-Run auf Toy-Daten
- Dataset-Laden -> Split -> Training -> Evaluation -> Artefaktschreiben

## 13.4 Regression-Tests

- kleine Referenz-RMSE-Faelle
- Laufzeitbudgets fuer kritische Pfade

## 14. Phasenplan

### Phase 0: Planung und Governance

- Masterplan festziehen
- Terminologie festziehen
- Repo-Struktur festziehen
- Technologie-Stack festziehen

### Phase 1: Infrastruktur

- Setup
- Config-System
- Datenpipeline fuer MovieLens
- Logging und Artefaktstruktur

### Phase 2: Mathematische Basis

- `biased_mf`
- Referenztests
- Precision-Vergleich `float32` vs `float64`

### Phase 3: Feedback-Modelle

- `svdpp`
- `asvdpp`
- Performance-Benchmarks

### Phase 4: Clustering-Layer

- Latent-Extraktion
- KMeans / MiniBatchKMeans
- Cluster-zu-Cluster Matrix
- `cb_svdpp`
- `cb_asvdpp`

### Phase 5: Benchmarking und Bericht

- Ablationen
- Seed-Wiederholungen
- Laufzeitvergleiche
- finaler Ergebnisbericht

## 15. Jetzt fest beschlossen

Die folgenden Punkte gelten ab jetzt als entschieden:

- Python bleibt die Sprache
- CPU-first Architektur
- `uv` als Setup-Standard
- strikte Konfigurationssteuerung
- modulare Formelbausteine
- keine Kernlogik in Notebooks
- MovieLens 100K bis 20M als offizieller Datensatz-Scope
- RMSE und Performance sind beide erstklassige Zielgroessen

## 16. Was noch nicht final implementiert, aber architektonisch beschlossen ist

- exakte Datei- und Modulnamen koennen beim Implementieren noch minimal angepasst werden
- zeitbasierte Splits sind eine spaetere Erweiterung
- GPU-Support bleibt optional und nicht Teil des lokalen Defaults

## 17. Naechster Schritt

Vor weiterer Implementierung wird dieser Plan als Referenz verwendet.

Die naechste inhaltliche Arbeit sollte sein:

1. die finale Repo-Struktur gegen dieses Dokument und die Governance-Docs abzugleichen
2. die mathematischen Spezifikationen fuer `biased_mf`, `svdpp` und `asvdpp` als eigene Docs zu schreiben
3. erst danach den Implementierungsstart fuer Modellcode freizugeben
