# Kernel Cost Anatomy V1

## Zweck

Kernel Cost Anatomy V1 macht den Trainingsblock `fit_model` strukturell
erklaerbar. Performance Forensics V1 misst, dass `fit_model` in produktiven
Unified-Runs ein dominanter Zeitblock sein kann. Dieses Profil zerlegt diesen
Block nicht durch Microtimer in Numba-Kernels, sondern beschreibt die Arbeit,
die der Trainingskernel pro Run strukturell leisten muss.

Jeder Unified-Run schreibt zusaetzlich:

```text
artifacts/runs/<run_id>/kernel_profile.json
```

`metrics.json` enthaelt eine kompakte Referenz auf dieses Artefakt.
`run_manifest.json` enthaelt `artifacts.kernel_profile`.

## Warum fit_model Zerlegt Wird

Die Stage `fit_model` umfasst pro Modell die eigentliche Trainingsschleife ueber
Ratings und Epochen. Je nach Modell kommen pro Rating weitere Zugriffe auf
User-Historien, explicit-feedback-Indizes oder Cluster-Historien hinzu. Zwei
Runs koennen daher dieselbe Anzahl Ratings und Epochen haben, aber sehr
unterschiedliche strukturelle Kernelarbeit verursachen.

Kernel Cost Anatomy V1 dokumentiert diese Struktur, damit spaetere
Optimierungsarbeit nicht nur an einer Gesamtzeit haengt.

## Geschaetzte Kosten

Das Profil enthaelt:

- `epoch_durations_seconds`: bereits im Modell gemessene Epoch-Dauern.
- `ratings_per_second_by_epoch`: Trainingsratings pro gemessener Epoche.
- `history_structure.implicit`: Summary des implicit user history index.
- `history_structure.explicit`: Summary des explicit feedback index.
- `history_structure.cluster`: Summary des user cluster history index.
- `estimated_kernel_work.rating_updates`: `train_rows * epochs`.
- `estimated_kernel_work.implicit_history_visits`: train-row-weighted implicit history visits ueber alle Epochen.
- `estimated_kernel_work.explicit_history_visits`: train-row-weighted explicit feedback visits ueber alle Epochen.
- `estimated_kernel_work.cluster_history_visits`: train-row-weighted cluster history visits ueber alle Epochen.
- `estimated_kernel_work.estimated_factor_touches`: heuristische, konsistente Schaetzung der beruehrten Latent-Factor-Komponenten.

History-Summaries enthalten nur Aggregate wie `total_edges`, `mean_len`,
Perzentile und `max_len`. Arrays werden nicht in JSON serialisiert.

## Zeit vs. Arbeit

Gemessene Zeit und geschaetzte Arbeit sind verschiedene Groessen.

`epoch_durations_seconds` und abgeleitete Ratios sind Wall-Clock-Messungen aus
dem Trainingslauf. Sie enthalten die tatsaechliche lokale Ausfuehrungszeit fuer
den jeweiligen Run.

`estimated_factor_touches` ist dagegen eine strukturelle Heuristik. Sie nutzt
konstante Touch-Faktoren fuer Basis-Ratingupdates, implicit history, explicit
feedback und cluster history. Diese Zahl ist kein CPU-Instruktionszaehler und
kein direkter Speicherbandbreiten-Messwert. Sie dient dazu, Runs und Modelle mit
derselben Rechenlogik konsistent einzuordnen.

## Keine Optimierung, Kein Claim

Kernel Cost Anatomy V1 optimiert nichts:

- keine Modellformel wird geaendert
- keine Numba-Schleife wird umgeschrieben
- keine Hyperparameter werden angepasst
- keine zusaetzlichen Trainingslaeufe werden gestartet
- keine Performance-Claims werden erzeugt

Die Daten sind Mess- und Anatomieartefakte. Aussagen wie schneller, besser oder
skalierbarer brauchen weiterhin separate Benchmarks und Claim-Gates.

## CSV Erzeugen

Kernel-Profile mehrerer Run-Ordner koennen gesammelt werden mit:

```powershell
.venv\Scripts\python.exe scripts\collect_kernel_profiles.py --runs-dir artifacts\runs --output-dir artifacts\reports
```

Der Collector schreibt:

```text
artifacts/reports/kernel_cost_anatomy.csv
```

Die CSV enthaelt pro Run:

- Dataset, Modell und Run-ID
- Epochen, Latent-Dimension und Trainingszeilen
- totale Fit-Zeit aus den Epoch-Dauern
- mittlere Sekunden pro Epoche
- implicit, explicit und cluster history visits
- geschaetzte Factor-Touches
- Sekunden pro Million geschaetzter Factor-Touches

## Nutzung Fuer Spaetere Optimierung

Die Profile helfen, spaetere Arbeit zu priorisieren:

- Wenn `rating_updates` dominiert, liegt der Fokus auf dem Basis-Updatepfad.
- Wenn `implicit_history_visits` dominiert, muss der implicit-history Zugriff
  untersucht werden.
- Wenn `explicit_history_visits` dominiert, ist der explicit-feedback Pfad der
  naechste Kandidat.
- Wenn `cluster_history_visits` hoch ist, sollte der Cluster-History Pfad
  isoliert betrachtet werden.
- Wenn die Sekunden pro Million geschaetzter Factor-Touches zwischen Modellen
  stark auseinanderlaufen, braucht der konkrete Kernelpfad eine separate
  Messung.

Diese Ableitung ist eine Untersuchungsreihenfolge, keine Optimierungszusage.
