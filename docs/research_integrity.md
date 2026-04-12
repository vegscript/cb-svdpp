# Forschungs- und Integritaetsprotokoll

## Absolute Prioritaeten

- mathematische Integritaet
- methodische Integritaet
- Reproduzierbarkeit
- vergleichbare Ablationen
- saubere Trennung zwischen Paper-Reproduktion und eigener Erweiterung

## Was zusaetzlich extrem wichtig ist

### 1. Klare Reproduktionsstufen

Jedes Ergebnis muss exakt einer Stufe zuordenbar sein:

- `paper-faithful`: so nah wie moeglich an Formeln, Datenprotokoll und Metrik
- `paper-inspired`: gleiche Idee, aber moderne oder pragmatische Anpassungen
- `extended`: neue Varianten, die explizit nicht mehr Reproduktion sind

Wenn diese Trennung fehlt, werden Resultate wissenschaftlich schnell wertlos.

### 2. Data leakage prevention

- kein Clustering auf Testdaten
- keine Normalisierung mit Teststatistiken
- keine Hyperparameter-Wahl auf Test-RMSE
- keine impliziten Signale aus unzulaessigen Splits

### 3. Repeated runs und Unsicherheit

- Ergebnisse nie aus nur einem Seed berichten
- Mittelwert und Standardabweichung ueber mehrere Seeds
- wenn moeglich Signifikanztest fuer Kernvergleiche

### 4. Exakte Experiment-Spezifikation

Pro Run muessen gespeichert werden:

- Dataset-Version
- Split-Strategie
- Seed
- Hyperparameter
- Modellvariante
- Laufzeit
- Metriken

### 5. Numerische Integritaet

- explizite Regularisierungsterms dokumentieren
- Update-Regeln versionieren
- Lernraten und Initialisierung fixieren
- harte Guards gegen NaN/Inf

### 6. Invarianten- und Sanity-Tests

Mindestens diese Tests sollten spaeter vorhanden sein:

- RMSE-Berechnung gegen Handrechnung
- Split-Disjunktheit
- Reproduzierbarkeit bei gleichem Seed
- keine unbekannten User-/Item-IDs im Train-Loop
- Clustering nur auf trainierten Latents
- `alpha = 0` reduziert CB-Modell auf Basismodell

### 7. HPC und Skalierung ehrlich behandeln

HPC am SOTA bedeutet hier nicht nur "schnell", sondern:

- klare Komplexitaetsannahmen
- messbare Laufzeiten
- kontrollierter Speicherverbrauch
- spaetere Optimierung der Hotspots statt frueher Premature Optimization

Fuer die erste saubere Version sind Korrektheit und Reproduzierbarkeit wichtiger als aggressive Micro-Optimierung.

### 8. Dokumentierte Abweichungen vom Paper

Sobald ihr etwas aendert, muss es in einer Tabelle stehen:

- was wurde geaendert
- warum wurde es geaendert
- welche Auswirkung auf Vergleichbarkeit ist zu erwarten

## Empfohlene Arbeitsweise

1. `BiasedMF` sauber und voll testen.
2. Erst dann `SVD++`.
3. Erst dann Clustering-Schritt auf trainierten Latents.
4. Erst dann `CB-SVD++`.
5. Erst danach `CB-ASVD++` oder weitere Erweiterungen.

## Definition von "enterprise grade" in diesem Projekt

- klare Modulgrenzen
- konfigurationsgetrieben
- testbar ohne UI/Notebook
- keine versteckten Seiteneffekte
- reproduzierbare CLI oder Runner
- dokumentierte Artefakte und Ergebnisse
