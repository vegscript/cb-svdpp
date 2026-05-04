# Evaluation Protocol

## Purpose

Dieses Dokument definiert die kanonischen Metriken, Claim-Regeln und
Benchmark-Bedingungen des Repos.

## Evaluation Layers

Dieses Repo bewertet Modelle auf zwei Ebenen:

1. `rating prediction`
2. `system performance`

Spaeter kann zusaetzlich eine `ranking evaluation`-Ebene aktiviert werden, wenn
Top-N-Empfehlungen berichtet werden.

## Primary Rating Metrics

- `RMSE`
- `MAE`

Wenn Aussagen ueber das Paper oder ueber rating prediction gemacht werden, ist
`RMSE` die Primaermetrik.

## Ranking Metrics

Sobald Top-N-Empfehlungen evaluiert werden, muessen diese Metriken explizit und
getrennt berichtet werden:

- `Precision@K`
- `Recall@K`
- `NDCG@K`
- `HitRate@K`
- `Coverage`

Rating-Metriken und Ranking-Metriken duerfen nicht vermischt interpretiert
werden.

## System Metrics

- `train_time_total`
- `train_time_per_epoch`
- `ratings_per_second_train`
- `ratings_per_second_inference`
- `peak_memory_mb`
- `model_size_mb`

Wenn `faster`, `lighter`, `more scalable` oder `HPC-grade` behauptet wird,
muessen diese Kennzahlen oder aequivalente Readouts berichtet werden.

## Seed Policy

Es gelten drei Stufen:

- Entwicklungslaeufe: `1` Seed ist zulaessig
- Vergleichslaeufe: mindestens `3` Seeds
- finale Ergebnisclaims: bevorzugt `5` Seeds

Die Zahl der Seeds muss immer im Readout genannt werden.

## Tuning Protocol

Tuning ist eine innere Validierungsprozedur. Es erzeugt keine
Modellvergleichs-Claims und keine Paper-Reproduktionsclaims. Ein Kandidat darf
erst dann als benchmark-relevant behandelt werden, wenn nach der Auswahl ein
separater sauberer Outer-Benchmark mit reserviertem Test-Split dokumentiert
wurde.

### Scope

Aktiv getuned werden nur die CB-Modelle, bei denen die Cluster- und
Alpha-Parameter den Ressourcenbedarf und die Rating-Metriken wesentlich
beeinflussen:

- `cb_svdpp`
- `cb_asvdpp`

Die Nicht-CB-Modelle `biased_mf`, `svdpp`, `asymmetric_svd` und `asvdpp`
werden in diesem begrenzten Protokoll nicht neu getuned. Ihre sichtbaren
Profile bleiben Benchmark- oder Transfer-Profile, bis ein separates
Tuning-Protokoll fuer diese Modellfamilien beschlossen wird.

### Tuned And Fixed Parameters

Getuned werden in den aktiven Grids nur:

- `training.latent_dim`, wenn das aktive Grid diesen Wert explizit variiert
- `training.epochs`, als bewusstes Budgetlimit pro Kandidat
- `clustering.n_user_clusters`
- `clustering.n_item_clusters`
- `clustering.alpha`
- `clustering.algorithm` und `clustering.kmeans_n_init` nur dort, wo sie in
  der aktiven Config explizit festgeschrieben sind

Fix bleiben, sofern die jeweilige aktive Config sie nicht ueberschreibt:

- Modellfamilie und Basisprofil
- Split-Familie
- `model_seed`
- Learning Rate
- Regularisierungsterme
- Dtype
- Device-Profil
- Cluster-Induction-Datenbasis
- Metrikdefinition
- Manifest- und Cache-Semantik

`R_star` bleibt diagnostisch. Es ist kein Tuning-Ziel, kein Koeffizient und
keine Auswahlmetrik.

### Active Candidate Grids

Die aktiven Tuning-Konfigurationen liegen ausschliesslich unter
`configs/experiments/tuning/active/`.

`ml100k_cb_svdpp_g6_validation_grid.yaml`:

- Modell: `cb_svdpp`
- Dataset: `ml100k`
- Split-Familie: `benchmark_random_v1`
- Split Seeds: `1, 2, 3`
- Validation Ratio: `0.1`
- Model Seed: `1`
- Kandidaten: `latent_dim=32`, `epochs=2`, `learning_rate=0.01`,
  Regularisierungsterme `0.02`
- Grid: `n_user_clusters`/`n_item_clusters` in `{32, 64, 80, 100}` und
  `alpha` in `{0.0, 0.025, 0.05}`
- Status: validation-only selection evidence, nicht Benchmark-Anker

`ml1m_cb_svdpp_stage0.yaml`:

- Modell: `cb_svdpp`
- Dataset: `ml1m`
- Split-Familie: `benchmark_random_v1`
- Split Seeds: `1, 2`
- Validation Ratio: `0.1`
- Model Seed: `1`
- Basisprofil: `configs/models/selected/ml1m/ml1m_cb_svdpp_stage0_transfer.yaml`
- Kandidaten: `epochs=2`; Cluster/Alpha-Kandidaten genau wie in der aktiven
  YAML definiert
- Status: Stage0-Selection, nicht finaler Benchmark-Anker

`ml1m_cb_asvdpp_stage0.yaml`:

- Modell: `cb_asvdpp`
- Dataset: `ml1m`
- Split-Familie: `benchmark_random_v1`
- Split Seeds: `1, 2`
- Validation Ratio: `0.1`
- Model Seed: `1`
- Basisprofil:
  `configs/models/selected/ml1m/ml1m_cb_asvdpp_stage0_transfer.yaml`
- Kandidaten: `epochs=2`; Cluster/Alpha-Kandidaten genau wie in der aktiven
  YAML definiert
- Status: Stage0-Selection, nicht finaler Benchmark-Anker

`ml20m_cb_svdpp_g11_lower_memory_validation_grid.yaml`:

- Modell: `cb_svdpp`
- Dataset: `ml20m`
- Split-Familie: `benchmark_random_v1`
- Split Seeds: `1, 2`
- Validation Ratio: `0.1`
- Model Seed: `1`
- Kandidaten: `latent_dim` in `{16, 32}`, `epochs=1`,
  `n_user_clusters`/`n_item_clusters` in `{32, 64}`, `alpha` in `{0.0, 0.025}`,
  `algorithm=kmeans`, `kmeans_n_init=10`
- Resource Gate: lokales `local_i5_2500k_24gb` Profil, maximal `80%` RAM
- Status: blocked/negative-resource reassessment, nicht aktives selected
  profile und kein Modellvergleichs-Claim

### Validation Split

Alle aktiven Tuning-Grids verwenden:

- `split_family: benchmark_random_v1`
- `train_ratio: 0.8`
- `validation_ratio: 0.1`
- die in der jeweiligen aktiven Config definierten `split_seeds`
- `model_seed: 1`

Der Test-Split darf waehrend der Kandidatenwahl nicht ausgewertet werden. Falls
ein Tuning-Lauf technisch trotzdem ein Test-Feld erzeugen koennte, darf dieses
Feld nicht in die Auswahlregel einfliessen und nicht als Ergebnisclaim
berichtet werden.

### Selection Rule

Die Kandidatenwahl erfolgt in dieser Reihenfolge:

1. niedrigste mittlere `validation_rmse`
2. niedrigere `validation_rmse`-Streuung, wenn die Mittelwerte praktisch gleich
   sind
3. niedrigere Trainingszeit, wenn Validierungsqualitaet und Stabilitaet keine
   klare Entscheidung liefern
4. besserer Memory-/Resource-Status, insbesondere keine Guardrail-Verletzung

Resource Gates sind harte Vorbedingungen, wenn eine aktive Config sie definiert.
Ein Kandidat, der die dort definierte RAM-Guardrail verletzt, darf nicht durch
eine bessere Validation-RMSE ausgewaehlt werden.

### Alpha-0 Semantics

`alpha=0` ist ein expliziter Ablationskandidat. Er bedeutet:

- Cluster-Artefakte koennen weiterhin erzeugt und validiert werden
- der Cluster-Kanal traegt keinen gewichteten Beitrag zur Praediktorformel bei
- `alpha=0` ist kein Fehlerfall
- `alpha>0` aktiviert nur den Cluster-Kanal
- weder `alpha=0` noch `alpha>0` machen einen Run automatisch
  `cb_claim_eligible`

Ein CB-Claim braucht weiterhin die im Manifest und in den Evidence-Dateien
geforderten semantischen Nachweise.

### Claim Boundary

Claim-relevant sind nur:

- nachgelagerte Outer-Benchmarks mit reserviertem Test-Split
- sauber dokumentierte Multi-Seed-Readouts
- Runs mit Manifest, Config Snapshot, Metrics und cleanem Claim-Kontext

Nicht claim-relevant sind:

- einzelne Tuning-Kandidaten
- validation-only Selection Runs
- `alpha=0` Ablationen
- Resource-Gate-Fehlschlaege
- blocked/negative-resource Evidence
- Entwicklungs- oder Stage0-Runs ohne separaten Outer-Benchmark

Negative oder blockierte Runs muessen dokumentiert werden, wenn sie eine
Richtung disqualifizieren oder den lokalen Ressourcenrahmen begrenzen. Sie
duerfen aber nicht als schlechtere Modellqualitaet interpretiert werden, wenn
der Run primaer durch Ressourcenstatus begrenzt war.

## Reporting Rules

Ein Ergebnisbericht muss mindestens enthalten:

- Dataset
- Split-Familie
- Modellname
- relevante Hyperparameter
- Seed-Anzahl
- Dtype
- Device-Profil
- Metriken

Wenn moeglich zusaetzlich:

- Mittelwert
- Standardabweichung
- Laufzeit
- Peak Memory

## Claim Rules

### `paper-faithful`

Nur erlaubt wenn:

- Daten- und Split-Protokoll konsistent zum Paper sind oder Abweichungen klar
  dokumentiert wurden
- Formel und Optimierungsverfahren konsistent sind
- Metrikdefinition konsistent ist

### `reproduced`

Nur erlaubt wenn:

- ein plausibler, dokumentierter Reproduktionslauf existiert
- die Abweichung zum Referenzwert klar berichtet wird

### `improved`

Nur erlaubt wenn:

- eine Baseline mit gleichem Setup vorliegt
- der Vergleich fair ist
- die Metrikverbesserung im Readout sichtbar ist

### `faster`

Nur erlaubt wenn:

- gleiche oder fair vergleichbare Rahmenbedingungen vorliegen
- Device-Profil, Threads und Dtype gleich oder dokumentiert unterschiedlich sind

## Benchmark Conditions

Ein reproduzierbarer Benchmark muss explizit nennen:

- Device-Profil
- Python-Version
- Threads
- Dtype
- Dataset
- Split
- Seeds
- Warmup-Regel
- exaktes Kommando oder Run-Profil
- welche Zeitmetrik verglichen wird

Empfohlen ist zusaetzlich ein maschinenlesbarer `measurement`-Block im
Benchmark-Manifest mit:

- `time_metric`
- `time_metric_semantics`
- `sample_unit`
- `warmup_policy`
- `warmup_sample_count`
- `measured_sample_count`

Fuer vergleichbare Fit-Zeit-Claims gilt:

- bei Standard-MF-Modellen ist `training_wall_clock_seconds` die kanonische
  Fit-Zeit
- bei clustering-basierten Modellen muss die vergleichbare Fit-Zeit alle
  train-only Vorstufen enthalten, die fuer das finale Modell notwendig sind
- wenn Clusterinduktion separat gemessen wird, ist die Benchmark-Fit-Zeit die
  Summe aus Clusterinduktion und Haupttraining
- eine Zeitangabe, die Clusterinduktion bei `cb_*`-Modellen ausblendet, ist
  nicht fair vergleichbar mit nicht-clustering-basierten Modellen

Wenn ein Benchmark bestehende Run-Artefakte wiederverwendet, muss zusaetzlich
klar sein:

- dass das Repo clean war
- dass Git-Commit und effektiv geladene Config-Inhalte identisch waren
- welche Folds neu gerechnet und welche wiederverwendet wurden
- welche Quell-Benchmark-Manifeste in einen Multi-Seed-Readout eingeflossen sind
- dass alle aggregierten Seed-Benchmarks denselben Git-Commit und denselben
  Dirty-Status teilen

In einem dirty Workspace sind wiederverwendete Benchmark-Folds nicht
benchmark-final.

## Negative Results

Negative Resultate muessen dokumentiert werden, wenn sie:

- eine Richtung disqualifizieren
- einen vermeintlichen Performance-Gewinn widerlegen
- eine numerische Instabilitaet zeigen
- eine paper-faithful Annahme falsifizieren

Scheitern ist ein Ergebnis, wenn es sauber nachgewiesen ist.
