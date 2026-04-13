# AGENTS

Diese Datei richtet sich an Coding- und Research-Agenten, die direkt in diesem
Repo arbeiten.

## Purpose

Dieses Repo ist eine Forschungs- und Engineering-Plattform fuer die
methodisch saubere Reproduktion und Erweiterung clustering-basierter
faktorisierter Recommender-Systeme.

Diese `AGENTS.md` regelt das Arbeitsverhalten im Repo. Sie ist kein Ersatz fuer
die mathematischen Spezifikationen oder fuer das Evaluationsprotokoll.

## Working Standard

Dieses Repo arbeitet mit hohem Nachweis-, Hygiene- und Strukturanspruch:

- kein Claim ohne pruefbaren Nachweis
- keine Aussage wie `better`, `faster`, `correct`, `paper-faithful`,
  `reproduced`, `scalable` oder `ready` ohne Test, Benchmark oder klaren Readout
- keine stillen methodischen Wechsel ohne datierte Dokumentation
- keine zweite Wahrheit fuer Formeln, Splits, Konfigurationen oder Benchmarks
- keine Schattenpfade, Duplikate oder halbaktiven Parallelimplementierungen

Wenn ein Schritt nicht belegt ist, ist er Hypothese, nicht Ergebnis.

## First Read

Vor groesseren Aenderungen kurz lesen:

- [README.md](G:/Meine Ablage/docs/WU/Semester_4/5255 Applications of Data Science/group_ass/final-project/README.md)
- [docs/project_master_plan.md](G:/Meine Ablage/docs/WU/Semester_4/5255 Applications of Data Science/group_ass/final-project/docs/project_master_plan.md)
- [docs/repo_governance.md](G:/Meine Ablage/docs/WU/Semester_4/5255 Applications of Data Science/group_ass/final-project/docs/repo_governance.md)
- [docs/environment_contract.md](G:/Meine Ablage/docs/WU/Semester_4/5255 Applications of Data Science/group_ass/final-project/docs/environment_contract.md)
- [docs/data_and_split_contract.md](G:/Meine Ablage/docs/WU/Semester_4/5255 Applications of Data Science/group_ass/final-project/docs/data_and_split_contract.md)
- [docs/evaluation_protocol.md](G:/Meine Ablage/docs/WU/Semester_4/5255 Applications of Data Science/group_ass/final-project/docs/evaluation_protocol.md)
- [docs/naming_conventions.md](G:/Meine Ablage/docs/WU/Semester_4/5255 Applications of Data Science/group_ass/final-project/docs/naming_conventions.md)
- [docs/manifest_contract.md](G:/Meine Ablage/docs/WU/Semester_4/5255 Applications of Data Science/group_ass/final-project/docs/manifest_contract.md)
- [docs/report/report_contract.md](G:/Meine Ablage/docs/WU/Semester_4/5255 Applications of Data Science/group_ass/final-project/docs/report/report_contract.md)
- [docs/methodology/deviations_from_paper.md](G:/Meine Ablage/docs/WU/Semester_4/5255 Applications of Data Science/group_ass/final-project/docs/methodology/deviations_from_paper.md)

Wenn ein Task Modellmathematik oder Update-Regeln betrifft, muessen vor
Implementierung zusaetzlich die einschlaegigen mathematischen Spezifikationen in
`docs/math/` gelesen oder zuerst erstellt werden.

## Repo Boundaries

- Aktiver Code lebt in `src/`.
- Aktive Tests leben in `tests/`.
- Aktive Doku lebt in `docs/`.
- Aktive Skripte leben in `scripts/`.
- Konfigurationen leben in `configs/`.
- Daten leben nur in `data/`.
- Generierte Outputs leben nur in `artifacts/`.
- Keine Modelllogik in Notebooks.
- Keine Artefakte, Logs, Dumps, Benchmarks oder Scratch-Dateien im Repo-Root.

## Source Of Truth Rules

- `docs/project_master_plan.md` ist die kanonische Projektleitplanke.
- `docs/repo_governance.md` ist die kanonische Governance-Quelle.
- `docs/environment_contract.md` ist die kanonische Umgebungs- und
  Reproduktionsquelle.
- `docs/data_and_split_contract.md` ist die kanonische Daten- und Split-Quelle.
- `docs/evaluation_protocol.md` ist die kanonische Evaluations- und
  Claim-Quelle.
- `docs/manifest_contract.md` ist die kanonische Manifest-Quelle.
- Mathematik, Objectives und Update-Regeln duerfen nur an einem kanonischen Ort
  dokumentiert sein.
- Wenn mehrere Dateien dieselbe Wahrheit zu beschreiben scheinen, muss die
  Dopplung aufgeloest werden.

## Mathematical Integrity

- Kein Modell darf implementiert oder veraendert werden ohne:
  - klar definierte Notation
  - Praediktorformel
  - Zielobjektiv
  - Regularisierung
  - Update-Regel oder Optimierungsverfahren
  - definierte Spezialfaelle und Reduktionen
- Wenn zwei Modelle denselben mathematischen Term teilen, muss derselbe
  Code-Baustein verwendet werden.
- Copy-Paste-Mathematik in mehreren Modulen ist nicht akzeptabel.
- Ein Modell darf nicht als `paper-faithful` bezeichnet werden, wenn auch nur
  eine relevante Abweichung undokumentiert ist.

## Methodological Integrity

- Keine Data Leakage.
- Kein Hyperparameter-Tuning auf Testdaten.
- Kein Clustering auf Validation- oder Testdaten, wenn Cluster im Training oder
  in finalen Claims verwendet werden.
- Kein stiller Wechsel von Split-Strategie, Metrikdefinition, Dtype,
  Device-Profil oder Seed-Politik.
- `paper-faithful`, `paper-inspired` und `extended` muessen explizit getrennt
  bleiben.
- Jede Ergebnisbehauptung muss auf ein klar benanntes Dataset, Split-Protokoll,
  Seed-Set und Modellprofil referenzieren.

## Proof Discipline

Jede relevante Aenderung braucht Nachweis proportional zum Risiko:

- Bugfix: reproduzierbarer Fehlerfall oder Regressionstest
- Modell- oder Methodenwechsel: Vergleich gegen Baseline
- Performance-Aenderung: Vorher/Nachher-Benchmark
- Daten-, Split- oder Mapping-Aenderung: Manifest-, Schema- oder Split-Nachweis
- numerische Aenderung: Precision- oder Stabilitaetsvergleich

Ohne solchen Nachweis darf eine Aenderung nicht als `improved`, `fixed`,
`faster`, `more scalable`, `paper-faithful` oder `ready` bezeichnet werden.

Negative Resultate sind explizit festzuhalten, wenn sie eine Richtung
disqualifizieren.

## Evidence And Artifact Discipline

- Entscheidungsrelevante Artefakte gehoeren nach `artifacts/`.
- Benchmark-Artefakte gehoeren nach `artifacts/benchmarks/`.
- Lauf-Artefakte gehoeren nach `artifacts/runs/`.
- Abbildungen gehoeren nach `artifacts/figures/`.
- Alle relevanten Artefakte folgen den kanonischen Namen aus
  `docs/naming_conventions.md`.
- Produktive Runs und Benchmarks ohne Manifest sind nicht claim-faehig.
- Grosse rebuildbare Zwischenprodukte gehoeren nicht in den Repo-Root.
- Wenn ein Task eine Entscheidung absichert, muss klar dokumentiert sein:
  - welches Kommando lief
  - auf welchem Device-Profil
  - mit welchem Dataset und Split
  - mit welchen Seeds
  - welche Kennzahlen entstanden
- Bestehende Run-Artefakte duerfen fuer einen neuen Benchmark nur dann
  wiederverwendet werden, wenn:
  - das Repo nicht dirty ist
  - der Git-Commit uebereinstimmt
  - die effektiv geladenen Config-Inhalte uebereinstimmen
  - Dataset-, Split-, Seed- und Dtype-Kontrakt uebereinstimmen
- In einem dirty Workspace ist Benchmark-Reuse unzulaessig.

## Testing Discipline

- Nach jeder isolierten Codeaenderung muessen die naechstliegenden fokussierten
  Tests ausgefuehrt werden.
- Tests muessen den Claim der Aenderung absichern.
- Numerische Pfade brauchen Guardrails gegen stille Regressionen.
- Determinismus und Seed-Verhalten muessen explizit getestet werden, wenn ein
  Pfad reproduzierbar sein soll.
- Wenn eine vollstaendige Verifikation zu teuer ist, muss klar benannt werden,
  was getestet wurde und was offen bleibt.

## Environment Discipline

- `pyproject.toml` und spaeter `uv.lock` sind die kanonische
  Dependency-Wahrheit.
- Keine manuell installierten Abhaengigkeiten ausserhalb des kanonischen
  Setup-Pfads.
- Device-spezifische Unterschiede werden ueber Profile in `configs/` gesteuert,
  nicht ueber hart kodierte lokale Sonderfaelle.
- Python ist die Orchestrierungsschicht. Hotpaths duerfen nur dann als reine
  Python-Loops bestehen bleiben, wenn Messung zeigt, dass es ausreicht.

## Process Hygiene

- Keine lang laufenden Hilfsprozesse offen lassen, wenn sie fuer den aktuellen
  Task nicht mehr benoetigt werden.
- Keine stillen Hintergrundserver, Watcher, Hosts oder REPLs offen lassen.
- Vor und nach laengeren Schritten auf offensichtliche Altlasten achten, wenn
  der Task Prozesse startet.
- Prozesse nur dann beenden, wenn sie klar diesem Repo-Task zuordenbar sind.
- Keine generischen System- oder Nutzerprozesse blind terminieren.
- Arbeit so planen, dass die lokale RAM-Nutzung moeglichst unter der
  80-Prozent-Marke bleibt.
- Wenn ein Schritt diese Grenze voraussichtlich verletzt, muss vorher ein
  kleinerer oder gestreamter Pfad bevorzugt werden.

## HPC And Performance Discipline

- Performance-Claims brauchen Messung.
- Vermutungen ueber `schneller` oder `skalierbarer` reichen nicht.
- In Hotpaths bevorzugen:
  - algorithmische Verbesserung vor Mikro-Optimierung
  - vektorisierte, sparse-aware oder kompilierte Pfade
  - kontrollierte Speicherbelegung
  - klare Datenfluesse ohne unnoetiges Vollmaterialisieren
- Keine stille Performance- oder Speicherregression akzeptieren, nur weil ein
  Pfad funktional korrekt ist.

## Change Style

- Kleine, gerichtete Aenderungen bevorzugen.
- Keine breitflaechigen Reformatierungen neben fachlichen Aenderungen.
- Keine neue zweite Konfigurationsebene aufmachen.
- Neue Hilfslogik in kanonischen Modulen unterbringen, nicht als ad-hoc Kopie.
- Repo-Dokumentation standardmaessig in ASCII halten.

## Safe Defaults

- Wenn unklar ist, ob ein Claim ausreichend belegt ist: konservativ formulieren.
- Wenn unklar ist, ob ein Artefakt dauerhaft relevant ist: in `artifacts/`
  behalten, nicht im Root ablegen.
- Wenn unklar ist, ob eine Modellabweichung vom Paper relevant ist: als
  Abweichung dokumentieren.
- Wenn mathematische Spezifikation und Implementierung auseinanderlaufen:
  Spezifikation und Code muessen vor weiteren Claims wieder synchronisiert
  werden.
