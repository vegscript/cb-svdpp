# CB Design Decision

Stand: 2026-04-13

## Purpose

Dieses Dokument schliesst die offene Designluecke fuer die erste
implementierbare clustering-based Modellfamilie in diesem Repo.

Die Primaerquelle ist das RecSys-2013-Poster von Mirbakhsh und Ling. Das Poster
gibt die CB-Praediktoren an und beschreibt, dass:

- auf den Trainingsdaten zuerst `BMF` trainiert wird
- die resultierenden User- und Item-Latents geclustert werden
- `KMeans` verwendet wird
- `R_star` als durchschnittliche Ratingmatrix zwischen User- und Item-Clustern
  konstruiert wird

Das Poster spezifiziert jedoch nicht vollstaendig:

- die genaue regularisierte Zielfunktion
- die exakten SGD-Updates
- ob `R_star` selbst nochmals faktorisiert wird
- ob Cluster-Zuweisungen waehrend des CB-Trainings neu berechnet werden

## Accepted Repo Contract

Die v1-Implementierung der CB-Familie verwendet den folgenden kanonischen
Vertrag.

### 1. Cluster induction

- Auf jedem Train/Validation/Test-Split wird zuerst nur auf `train` ein
  `biased_mf`-Modell fit gemacht.
- Fuer User-Cluster werden die trainierten User-Latents `p_u` verwendet.
- Fuer Item-Cluster werden die trainierten Item-Latents `q_i` verwendet.
- User- und Item-Cluster werden separat mit `KMeans` gelernt.
- Die Cluster-Zuweisungen `a(u)` und `b(i)` bleiben waehrend des spaeteren
  CB-Trainings fix.

### 2. Role of `R_star`

- `R_star` wird nur aus Trainingsratings berechnet.
- Fuer jedes Clusterpaar `(a, b)` wird der arithmetische Mittelwert aller
  beobachteten Trainingsratings gebildet, deren User zu `a` und deren Item zu
  `b` gehoert.
- Leere Clusterpaare werden ueber eine separate Count-Matrix gekennzeichnet.
- In v1 ist `R_star` ein Diagnose-, Analyse- und Nachweisartefakt.
- In v1 gibt es keine zweite Faktorierungsstufe und keinen zusaetzlichen
  Loss-Term auf `R_star`.

### 3. Optimization contract

- Der publizierte gemischte Praediktor bleibt erhalten.
- Cluster-Latents `p_C`, `q_C`, `y_C` und `x_C` werden als normale trainierbare
  Parameter im Hauptobjective gefuehrt.
- Die individuellen Parameter und die Cluster-Parameter werden gemeinsam ueber
  dasselbe Rating-Residual aktualisiert.
- `alpha` ist in v1 ein fester Hyperparameter pro Run und wird nicht gelernt.
- Die v1-CB-Familie verwendet dieselben bereits akzeptierten Repo-Contracts wie
  ihre nicht-clustering-basierten Basismodelle, einschliesslich `D-003` fuer
  die detachierten expliziten Residualgewichte in `asvdpp`-abgeleiteten
  Modellen.

## Why this contract was chosen

- Er ist nah genug an der Primaerquelle, um den publizierten CB-Praediktor
  ehrlich zu erhalten.
- Er fuehrt keine unbewiesene zusaetzliche Modellstufe fuer `R_star` ein.
- Er ist reproduzierbar, implementierbar und sauber testbar.
- Er haelt Train-only-Disziplin fuer Clusterbildung und `R_star` strikt ein.
- Er erzwingt klare Claim-Grenzen statt stiller Interpretationen.

## Claim boundary

Die erste Repo-Implementierung von `cb_svdpp` und `cb_asvdpp` darf nur als

- `paper-inspired CB implementation`
- `source-grounded predictor with repo-defined optimization`

bezeichnet werden.

Nicht zulaessig sind ohne weitere Primaerquellen oder Herleitung Claims wie:

- `exact paper reproduction`
- `exact optimizer-faithful reproduction`
- `fully specified source implementation`

## Operational consequences

- Jeder CB-Run muss im Run-Evidence-Dokument explizit nennen:
  - welches Basismodell fuer die Clusterinduktion verwendet wurde
  - wie viele User- und Item-Cluster verwendet wurden
  - dass `R_star` nur aus `train` berechnet wurde
  - dass Cluster-Zuweisungen im CB-Training fix waren
- Wenn spaeter eine zweite, `R_star`-getriebene Optimierungsvariante gebaut
  wird, ist das eine neue dokumentierte Abweichung und keine stille Erweiterung
  von v1.
