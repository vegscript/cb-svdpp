# Architektur

## Leitidee

Die Architektur folgt einer expliziten Layer-Logik:

1. Ein gemeinsames Daten- und Evaluationsfundament.
2. Ein baseline latent-factor Modell.
3. Ein implizites Feedback-Layer (`SVD++`).
4. Ein Cluster-/Kategorie-Layer gemaess Paper.

Jeder Layer muss isoliert aktivierbar und in Ablationen messbar sein.

## Kernmodule

### `recsys_lab.core`

- gemeinsame Typen
- Konfigurationen
- Modellprotokolle
- Serialisierungs- und Reproduktions-Hilfen

### `recsys_lab.data`

- kanonisches Interaction-Schema
- Train/Validation/Test-Splits
- spaetere Dataset-Adapter

### `recsys_lab.metrics`

- RMSE
- MAE
- spaeter Ranking-Metriken fuer Top-N

### `recsys_lab.models`

- `BiasedMFRecommender`
- `SVDPPRecommender`
- `CBSVDPPRecommender`
- spaeter `CBASVDPPRecommender`

### `recsys_lab.clustering`

- Clusterer-Interface
- KMeans auf gelernten Item-/User-Latents
- Bildung der Cluster-Interaktionsmatrix `R*`

### `recsys_lab.experiments`

- Experiment-Spezifikation
- Runner
- Reproduzierbarkeits-Hooks
- spaeter Artefakt-Logging

## Nicht verhandelbare Designregeln

- Kein Modell liest direkt rohe CSVs.
- Kein Modell erzeugt eigene Slices oder Metriken.
- Clustering ist ein eigener Pipeline-Schritt und kein Seiteneffekt des Trainings.
- Alle Modellvarianten konsumieren dieselbe `InteractionDataset`-Abstraktion.
- Jede mathematische Erweiterung hat eine eigene Konfigurationsklasse.

## Paper Mapping

### Layer 0: `BiasedMF`

Papier-Gleichung:

`r_ui = q_i^T p_u + b_i + b_u`

Nutzen:

- baseline
- initiale User-/Item-Latents fuer spaeteres Clustering

### Layer 1: `SVD++`

Papier-Gleichung:

`r_ui = q_i^T (p_u + |N(u)|^-1/2 * sum_j y_j) + b_i + b_u`

Nutzen:

- implizites Nutzerverhalten
- direkter Vergleich gegen baseline

### Layer 2: `CB-SVD++`

Papier-Idee:

- lerne Latents mit baseline MF
- cluster Users und Items im Latent-Raum
- konstruiere Cluster-zu-Cluster Mittelwertmatrix `R*`
- fuehre Cluster-Latents in den Praediktor ein

Wesentliche Zusatzparameter:

- `alpha`: Mix zwischen individueller und Cluster-Ebene
- `n_user_clusters`
- `n_item_clusters`

## Warum dieser Aufbau

Damit koennen wir spaeter fuer jedes Dataset dieselbe Frage beantworten:

- Was bringt reine Faktorisierung?
- Was bringt implizites Feedback darueber hinaus?
- Was bringt die Cluster-Erweiterung zusaetzlich?
- Ist der Gewinn robust ueber Seeds, Split-Strategien und Datasets?
