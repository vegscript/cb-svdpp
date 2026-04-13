# Data And Split Contract

## Purpose

Dieses Dokument definiert die kanonische Datenstruktur, die Datenzonen und die
Split-Regeln fuer alle Experimente in diesem Repo.

## Supported Datasets

Offizieller Scope:

- `MovieLens 100K`
- `MovieLens 1M`
- `MovieLens 10M`
- `MovieLens 20M`

Andere Datasets duerfen spaeter ergaenzt werden, aber nicht still unter
denselben Benchmark-Namen laufen.

Lokale POC-Datensaetze duerfen zusaetzlich verwendet werden, muessen aber klar
als nicht benchmark-eligibel markiert werden.

## Data Zones

Die Datenzonen sind strikt getrennt:

- `data/raw/`: heruntergeladene Originaldaten
- `data/interim/`: validierte, normalisierte und remappte Zwischenstaende
- `data/processed/`: trainingsfertige Artefakte
- `artifacts/`: Ergebnisse und Benchmarks, keine kanonischen Datenquellen

Keine Datenzustandsmischung zwischen diesen Zonen.

## Manifest Requirement

Jeder produktive Datensatzpfad braucht ein Manifest oder aequivalente Metadaten
mit mindestens:

- Dataset-Name
- Version
- Quelle
- Download-Datum
- Dateiliste
- Checksummen oder Fingerprints
- Zeilenanzahl oder Interaktionsanzahl
- Rating-Skala

Ohne Manifest ist ein Datensatz nicht benchmark-faehig.

## Canonical Interaction Schema

Intern sollen Interaktionen spaetestens nach dem Preprocessing kanonisch
vorliegen als:

- `user_idx`: kompakter Integer-Index, `int32`
- `item_idx`: kompakter Integer-Index, `int32`
- `rating`: numerischer Rating-Wert
- `timestamp`: optional, aber falls vorhanden explizit typisiert

Roh-IDs und gemappte IDs muessen nachvollziehbar verknuepfbar bleiben.

## Mapping Rules

- User- und Item-IDs werden frueh auf kompakte, dichte Integer abgebildet.
- Die Mapping-Tabellen muessen reproduzierbar sein.
- Kein stiller Wechsel der ID-Zuordnung zwischen Runs mit derselben
  Preprocessing-Version.

## Split Families

Dieses Repo kennt nur explizit benannte Split-Familien.

### `paper_faithful`

Ziel:

- so nah wie vertretbar am Paper-Protokoll

Anwendung:

- primaer fuer `MovieLens 100K`

Aktueller Repo-Vertrag:

- `paper_faithful_ml100k_v1` verwendet die offiziellen im GroupLens-Paket
  enthaltenen Split-Dateien `u1.base/u1.test` bis `u5.base/u5.test`
- diese Splits liefern nur `train` und `test`
- es wird in diesem Pfad keine stille zusaetzliche Validation aus dem Training
  herausgeschnitten
- wenn Hyperparameter-Tuning benoetigt wird, muss dafuer ein eigener,
  explizit dokumentierter Tuning-Pfad verwendet werden

### `paper_faithful_ml100k_inner_v1`

Ziel:

- leakagesicheres Hyperparameter-Tuning auf Basis der offiziellen `ml100k`
  Trainingssplits

Repo-Vertrag:

- der aeussere Testpfad bleibt der offizielle `u1.test` bis `u5.test`-Pfad
- fuer Tuning wird nur der jeweilige offizielle `uX.base`-Pfad verwendet
- aus `uX.base` wird ein innerer `train/validation`-Split mit
  Train-Coverage-Regel erzeugt
- die offizielle aeussere Testpartition wird in Tuning-Runs nicht ausgewertet
- Tuning-Ergebnisse aus diesem Pfad duerfen nur zur Kandidatenauswahl dienen,
  nicht als finale Benchmark-Claims

### `benchmark_random_v1`

Ziel:

- dataset-uebergreifend einheitliche, seed-kontrollierte Vergleichbarkeit

### `benchmark_time_v1`

Ziel:

- spaetere realistischere Generalisierung

Status:

- noch nicht kanonisch aktiv

## Split Rules

- Train, Validation und Test muessen disjunkt sein.
- Hyperparameter-Tuning erfolgt nur auf Validation.
- Test wird nur fuer finale Bewertung verwendet.
- Dieselbe Split-Definition muss fuer vergleichbare Modelllaeufe gleich bleiben.
- Der Split-Seed ist Bestandteil des Experiment-Profils.

## Leakage Rules

Nicht erlaubt:

- Normalisierung mit Teststatistiken
- Clustering auf Validation oder Test fuer finale Claims
- Hyperparameter-Wahl auf Test-RMSE
- indirekte Nutzung von Testinformationen ueber Cache-Reuse oder gemischte
  Preprocessing-Artefakte

## Clustering Rules

Da die CB-Modelle Cluster aus gelernten Latents ableiten, gelten zusaetzlich:

- Latents fuer Clustering werden nur aus dem Trainingspfad gewonnen
- Cluster-Zahl wird auf Validation entschieden, nicht auf Test
- Cluster-Assignments und Cluster-Matrizen muessen versionierbar und
  reproduzierbar sein

## Cold-Start And OOV Rules

Unbekannte User oder Items muessen explizit behandelt werden.

Es muss klar dokumentiert sein:

- ob sie im Benchmark ausgeschlossen werden
- ob ein Fallback verwendet wird
- wie sich dies auf Metriken auswirkt

Keine stillen Drops oder impliziten NaN-Faelle.
