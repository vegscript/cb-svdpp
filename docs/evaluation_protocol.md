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

- Hyperparameter-Tuning auf `ml100k` erfolgt nicht auf `paper_faithful_ml100k_v1`
  direkt, sondern nur auf `paper_faithful_ml100k_inner_v1`.
- Die offiziellen aeusseren Test-Folds `u1.test` bis `u5.test` bleiben fuer die
  finale Bewertung reserviert.
- Tuning-Runs duerfen keine Test-RMSE aus den aeusseren offiziellen Folds
  berichten oder fuer Kandidatenwahl verwenden.
- Wenn aus Laufzeitgruenden nur eine Teilmenge der aeusseren Folds fuer Tuning
  genutzt wird, muss dies explizit als `stage` oder `development` gekennzeichnet
  werden.

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

Wenn ein Benchmark bestehende Run-Artefakte wiederverwendet, muss zusaetzlich
klar sein:

- dass das Repo clean war
- dass Git-Commit und effektiv geladene Config-Inhalte identisch waren
- welche Folds neu gerechnet und welche wiederverwendet wurden

In einem dirty Workspace sind wiederverwendete Benchmark-Folds nicht
benchmark-final.

## Negative Results

Negative Resultate muessen dokumentiert werden, wenn sie:

- eine Richtung disqualifizieren
- einen vermeintlichen Performance-Gewinn widerlegen
- eine numerische Instabilitaet zeigen
- eine paper-faithful Annahme falsifizieren

Scheitern ist ein Ergebnis, wenn es sauber nachgewiesen ist.
