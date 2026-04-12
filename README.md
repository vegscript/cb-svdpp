# Recommender System Reproduction Lab

Dieses Repository ist als modularer Forschungs- und Engineering-Stack fuer eine
konzeptionell integre Reproduktion von:

- `biased matrix factorization`
- `SVD++`
- `Asymmetric-SVD`
- `ASVD++` im Sinne der Poster-Familie
- clustering-based Erweiterungen aus dem Paper

geplant.

Ziele:

- saubere Trennung zwischen `paper-faithful reproduction` und `engineering extensions`
- dataset-agnostische Evaluation ueber mehrere Benchmarks
- mathematische und methodische Nachvollziehbarkeit
- reproduzierbare Experimente ohne Notebook-Monolithen

## Geplanter Modellpfad

1. `BiasedMF`: baseline factorization mit Bias-Terms.
2. `SVD++`: implicit feedback ueber Item-History.
3. `Asymmetric-SVD`: Koren 2008, item-seitige Nutzerreprasentation ohne freien User-Vektor.
4. `ASVD++`: `SVD++` plus expliziter item-seitiger Feedback-Block gemaess Poster.
5. `CB-SVD++`: Cluster-Level Erweiterung des Posters.
6. `CB-ASVD++`: Cluster-Level Erweiterung des Posters fuer die `ASVD++`-Familie.

## Struktur

- `docs/`: Architektur, Forschungsprotokoll, Reproduktionsrichtlinien
- `src/recsys_lab/`: Produktionscode
- `tests/`: Unit- und Integritaetstests

## Grundsatz

Neue Modelle werden nur aufgenommen, wenn sie:

- eine klare mathematische Spezifikation haben
- ueber dieselben Daten-Splits vergleichbar sind
- mit identischen Metriken und mehrfachen Seeds evaluiert werden

## Naechste Schritte

1. Datensatz-Adapter fuer `MovieLens100k` und weitere Benchmarks bauen.
2. `BiasedMF` voll implementieren.
3. `SVD++` voll implementieren.
4. `Asymmetric-SVD` und `ASVD++` mathematisch und implementatorisch absichern.
5. Clustering-Pipeline auf gelernten Latent-Vektoren aufsetzen.
6. Paper-Tabellen und Abbildungen so weit wie moeglich reproduzieren.
