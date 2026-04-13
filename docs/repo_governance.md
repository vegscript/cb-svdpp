# Repo Governance

## Purpose

Dieses Dokument definiert die Arbeits- und Strukturregeln fuer das Repo. Ziel
ist eine Forschungscodebasis, die gleichzeitig:

- akademisch nachvollziehbar
- methodisch sauber
- technisch portierbar
- benchmarkbar
- langfristig wartbar

ist.

## Governance Priorities

Die Prioritaeten dieses Repos sind:

1. mathematische Korrektheit
2. methodische Korrektheit
3. Reproduzierbarkeit
4. Performance auf realer Zielhardware
5. modulare Architektur
6. Repo-Hygiene

Wenn diese Ziele in Konflikt geraten, gewinnt nicht der schnellste Workaround,
sondern die sauberste nachweisbare Loesung.

## Canonical Documents

Die folgenden Dateien sind kanonisch:

- `README.md`: Einstieg und praktische Nutzung
- `docs/project_master_plan.md`: Projektziel, Scope, Architekturleitbild
- `docs/repo_governance.md`: Arbeits- und Strukturregeln
- `docs/environment_contract.md`: Runtime-, Setup- und Portabilitaetsregeln
- `docs/data_and_split_contract.md`: Datenschema, Datenfluss und Split-Regeln
- `docs/evaluation_protocol.md`: Metriken, Claims, Benchmark-Regeln
- `docs/naming_conventions.md`: Benennungsregeln fuer Artefakte, Evidence und Outputs
- `docs/manifest_contract.md`: maschinenlesbare Nachweisregeln fuer Runs und Benchmarks
- `docs/report/report_contract.md`: Regeln fuer den kontinuierlich gepflegten Einreichungs-Report

Kein anderes Dokument darf dieselben Regeln parallel und widerspruechlich
definieren.

## Active Areas

- `src/`: produktiver Code
- `tests/`: Tests und Guardrails
- `configs/`: kanonische Konfiguration
- `scripts/`: reproduzierbare Hilfspfade
- `docs/`: kanonische Dokumentation
- `data/`: Datenzonen
- `artifacts/`: generierte Ergebnisse

## Non-Canonical Areas

- Scratch-Dateien sind keine Repo-Struktur.
- Ad-hoc-Notebooks sind keine Source of Truth.
- Root-Logs, Root-Dumps und Root-Ergebnisdateien sind unzulaessig.

## Definition Of Done

Eine relevante Aenderung ist nur dann `done`, wenn:

- die betroffene kanonische Dokumentation weiterhin korrekt ist
- der Claim der Aenderung nachweisbar ist
- die naechstliegenden Tests ausgefuehrt wurden
- neue Artefakte an einem kanonischen Ort liegen
- keine zweite Wahrheit fuer dieselbe Logik entstanden ist

## Acceptable Claims

Die folgenden Begriffe sind reserviert und duerfen nur mit Nachweis verwendet
werden:

- `fixed`
- `improved`
- `faster`
- `more memory efficient`
- `reproduced`
- `paper-faithful`
- `scalable`
- `ready`

Ohne Nachweis ist die korrekte Formulierung:

- `proposed`
- `implemented but not fully validated`
- `hypothesized`
- `partially verified`

## Documentation Discipline

- Dokumentation soll Regeln und Entscheidungen zentralisieren, nicht streuen.
- Wenn eine fruehere Datei durch eine spaetere kanonische Datei ersetzt wird,
  muss die fruehere Datei klar als superseded markiert oder entsprechend
  reduziert werden.
- Abweichungen vom Paper werden nicht in Commit-Messages versteckt, sondern an
  einem kanonischen Ort dokumentiert.
- Benennungsregeln werden nicht pro Unterordner neu erfunden, sondern folgen
  `docs/naming_conventions.md`.

## Structure Discipline

- Eine Verantwortung soll moeglichst einen kanonischen Ort haben.
- Keine halbaktiven Parallelpfade fuer dasselbe Modell.
- Keine Modellmathematik in mehreren Modulen mit leicht verschiedenen Formeln.
- Keine Imports aus provisorischen oder lokalen Helper-Dateien, wenn dafuer ein
  kanonischer Modulort vorgesehen ist.

## Benchmark Reuse Discipline

- Benchmark-Reuse ist nur erlaubt, wenn ein bestehender Run inhaltlich mit dem
  aktuellen Benchmark-Auftrag identisch ist.
- Mindestbedingungen fuer Reuse sind:
  - gleicher Git-Commit
  - cleanes Repo
  - gleiche effektiv geladene Config-Inhalte
  - gleiches Dataset-Manifest
  - gleiche Split-Familie und gleicher Fold
  - gleicher Seed
  - gleiches Dtype- und Device-Profil
- In einem dirty Workspace werden keine frueheren Runs fuer Benchmarks
  wiederverwendet.

## Review Questions

Vor dem Mergen oder vor dem Abschliessen eines groesseren Schritts muessen diese
Fragen mit `ja` beantwortbar sein:

1. Ist klar, welche Datei fuer diese Wahrheit autoritativ ist?
2. Ist klar, auf welchem Dataset und Split die Aussage basiert?
3. Ist klar, ob der Schritt paper-faithful, paper-inspired oder extended ist?
4. Ist der Claim test- oder benchmarkseitig belegt?
5. Ist die Aenderung spaeter auf einem neuen Geraet reproduzierbar?
