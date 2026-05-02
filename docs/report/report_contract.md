# Report Contract

## Purpose

Dieses Dokument definiert den kanonischen Arbeitsvertrag fuer den
einzureichenden Projekt-Report.

Der Report ist:

- ein akademischer Projektbericht
- kontinuierlich gepflegt
- evidence-backed
- klar gegliedert
- auf Einreichbarkeit optimiert

Der Report ist nicht:

- ein Arbeitslog
- ein Dump aller Benchmarks
- ein changelogartiges Journal
- eine Vollkopie der technischen Repo-Dokumentation

## Core Principle

Der Report wird waehrend des Projekts fortlaufend gepflegt, aber nur mit
verdichteten, einreichungsrelevanten Inhalten.

Rohwissen lebt in:

- `artifacts/`
- Evidence-Dateien
- technischen Governance- und Math-Dokumenten

Der Report enthaelt nur:

- was gemacht wurde
- warum es gemacht wurde
- wie es methodisch begruendet ist
- welche belastbaren Resultate daraus folgen

## Canonical Files

Die kanonischen Report-Dateien sind:

- [docs/report/report_contract.md](G:/Meine Ablage/docs/WU/Semester_4/5255 Applications of Data Science/group_ass/final-project/docs/report/report_contract.md)
- [docs/report/project_report.md](G:/Meine Ablage/docs/WU/Semester_4/5255 Applications of Data Science/group_ass/final-project/docs/report/project_report.md)

Kein anderes Dokument darf still als parallel gepflegter zweiter Report wachsen.

## Report Writing Standard

Der Report soll:

- akademisch sein
- knapp bleiben
- in sauberem, neutralem Stil schreiben
- nur evidence-backed Aussagen enthalten
- sich an Recommender- und Data-Science-Standards orientieren

Der Report soll nicht:

- jeden Zwischenschritt erzaehlen
- jeden fehlgeschlagenen Versuch im Haupttext ausbreiten
- Implementierungsdetails ohne analytischen Mehrwert auswalzen
- Rohlogs, Tabellen-Dumps oder unbearbeitete Konsolenausgaben enthalten

## Recommended Report Structure

Der Report folgt dieser Grundstruktur:

1. Titel
2. Abstract
3. Introduction
4. Research Objective And Reproduction Scope
5. Data
6. Methodology
7. System Design And Implementation
8. Experimental Setup
9. Results
10. Discussion
11. Limitations
12. Conclusion
13. References

Optional:

- Appendix

## Section Expectations

### Abstract

Kurzfassung von:

- Problem
- Ansatz
- Datasets
- Modelle
- wichtigste Ergebnisse

### Introduction

- Problemraum
- Relevanz von Recommender-Systemen
- Kontext der Paper-Reproduktion

### Research Objective And Reproduction Scope

- was reproduziert werden soll
- was konzeptionell statt exakt reproduziert wird
- welche Claims bewusst nicht gemacht werden

### Data

- welche MovieLens-Datasets verwendet werden
- warum diese gewaehlt wurden
- Nutzbarkeitspruefung
- Datenqualitaet
- Preprocessing-Entscheidungen
- warum z. B. Parquet verwendet wird

### Methodology

- mathematische Modellleiter
- Evaluation-Protokoll
- Hyperparameter-Strategie
- Alpha-Sweeps und andere relevante Untersuchungen

### System Design And Implementation

- modulare Repo-Struktur
- Device-Profile
- Performance-Orientierung
- Portabilitaet
- Hygieneregeln, soweit sie fuer den Bericht relevant sind

### Experimental Setup

- Datasets
- Splits
- Seeds
- Hardware-Profil
- Software-Umgebung
- Metriken

### Results

- Modellvergleiche auf gleichen Datasets
- tabellarische Kernresultate
- geeignete Grafiken
- keine ungefilterten Rohbenchmark-Dumps

### Discussion

- Interpretation der Resultate
- Unterschiede zwischen Modellen
- Bedeutung von Clustering und `alpha`
- methodische Einordnung

### Limitations

- Grenzen des Papers
- Grenzen der Reproduktion
- offene Punkte wie unter-spezifizierte CB-Optimierung

### Conclusion

- kompakte Zusammenfassung
- wichtigste Erkenntnisse

## Continuous Maintenance Rules

Der Report wird kontinuierlich gepflegt, aber nur bei material changes.

Ein Update in den Report ist gerechtfertigt, wenn:

- eine finale oder vorlaeufig stabile Dataset-Entscheidung getroffen wurde
- ein Preprocessing-Schritt kanonisch beschlossen wurde
- eine Modellfamilie mathematisch geschlossen wurde
- eine stabile Benchmark-Serie vorliegt
- eine Grafik final oder nahezu final ist
- eine relevante methodische Abweichung fuer den finalen Bericht bedeutsam ist

Ein Update ist nicht gerechtfertigt, wenn:

- nur lokale Debug-Arbeit stattfand
- ein instabiler Zwischenwert vorliegt
- ein Benchmark nur einmal ohne Einordnung lief
- eine Entscheidung noch nicht belastbar ist

## Report Update Style

Neue Informationen werden nicht als Tagebuch hinten angehaengt.

Stattdessen gilt:

- bestehende Sektion aktualisieren
- Formulierung integrieren
- veraltete Aussage ersetzen
- nur stabile Abbildungen und Tabellen aufnehmen

Der Report soll immer wie ein zusammenhaengender Bericht lesbar bleiben.

## Evidence Boundary

Der Report darf sich auf Evidence stuetzen, aber nicht zu Evidence mutieren.

Deshalb:

- Rohzahlen bleiben in `artifacts/`
- Detailbeweise bleiben in Evidence-Dateien oder Benchmarks
- der Report zitiert und verdichtet diese Ergebnisse

## Figure Policy

Der Report soll nur sinnvolle Grafiken enthalten.

Geeignete Inhalte:

- Modellvergleich ueber Datasets
- RMSE- und Laufzeitplots
- Alpha-Sweeps
- Clusterzahl-Sweeps
- Architekturdiagramme
- Datenflussdiagramme

Nicht geeignet:

- jede kleine Debug-Visualisierung
- mehrere fast identische Abbildungen ohne neue Aussage
- rohe Screenhots statt sauberer Grafiken

## Mermaid Policy

Mermaid-Diagramme sind ausdruecklich erlaubt fuer:

- Repo-Architektur
- Datenpipeline
- Trainingsfluss
- Modellkomposition

Sie sollen nur verwendet werden, wenn sie einen echten Erklaerungsgewinn bringen.

## Table Policy

Tabellen im Report sollen:

- klein genug zum Lesen sein
- dieselben Metriken konsistent berichten
- klar beschriftet sein
- gleiche Modellfamilien fair vergleichen

Keine ausufernden Tabellen mit jeder einzelnen Experimentvariante im Haupttext.

## Citation Policy

Der Report verwendet eine konsistente Zitierweise mit Referenzliste am Ende.

Empfohlener Standard:

- numerische Zitate im Text, zum Beispiel `[1]`, `[2]`
- vollstaendige Referenzen in `References`

Mindestens zu zitieren sind:

- das Zielpaper
- Koren 2008
- MovieLens-Datensatzquelle
- weitere zentrale Quellen, die methodisch oder technisch benutzt werden

## Claim Discipline In The Report

Im Report duerfen starke Begriffe nur mit entsprechender Evidenz erscheinen:

- `improved`
- `faster`
- `more scalable`
- `reproduced`
- `paper-faithful`

Wenn ein Punkt noch offen ist, muss der Report das offen benennen.

## Length Discipline

Der Report darf nicht unkontrolliert wachsen.

Daher gilt:

- nur final relevante Inhalte in den Haupttext
- Details in Appendix oder Evidence auslagern
- eine neue Grafik oder Tabelle nur aufnehmen, wenn sie eine klare neue Aussage
  traegt
- pro Abschnitt auf Verdichtung achten

## Writing Workflow

Empfohlener Arbeitszyklus:

1. technische Arbeit und Evidence erzeugen
2. relevante, stabile Erkenntnis identifizieren
3. entsprechende Report-Sektion aktualisieren
4. veraltete Formulierung ersetzen
5. Referenzen und Abbildungen sauber nachziehen

Der Report wird also kontinuierlich gepflegt, aber nie roh fortgeschrieben.

## Definition Of Report-Ready

Ein Abschnitt ist report-ready, wenn:

- die Aussage stabil ist
- die Aussage evidence-backed ist
- der Text akademisch formuliert ist
- keine offensichtlich temporaren Platzhalter mehr enthalten sind
- die zugehoerigen Referenzen vorhanden sind
