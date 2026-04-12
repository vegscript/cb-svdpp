# Deviations From Paper

## Purpose

Dieses Dokument ist die kanonische Stelle fuer alle relevanten Abweichungen von
den Quellpapern.

Keine Abweichung darf nur in Code, Commit-Message oder Chat-Wissen existieren.

## Source Papers

Primare Quellen fuer dieses Repo:

1. Yehuda Koren, `Factorization Meets the Neighborhood: a Multifaceted
   Collaborative Filtering Model`, KDD 2008.
2. Nima Mirbakhsh, Charles X. Ling, `Clustering-Based Factorized Collaborative
   Filtering`, RecSys 2013 Poster.

## Current Deviation Register

### D-001: `biased_mf` statt `SVD`

- Status: accepted
- Scope: terminology
- Source: Koren 2008
- Decision:
  Das Repo verwendet den Namen `biased_mf` statt pauschal `svd`, um die in
  Recommender-Systemen uebliche regularisierte Faktorisierung mit Bias-Terms von
  der linearen Algebra-SVD zu unterscheiden.
- Rationale:
  Das ist terminologisch praeziser und reduziert Verwechslungsgefahr.
- Claim impact:
  Keine methodische Abweichung, nur praezisere Benennung.

### D-002: separates `asymmetric_svd` und `asvdpp`

- Status: accepted
- Scope: model naming
- Source: Koren 2008, Mirbakhsh and Ling 2013
- Decision:
  Das Repo fuehrt `asymmetric_svd` und `asvdpp` als getrennte Familien.
- Rationale:
  Koren 2008 `Asymmetric-SVD` hat keinen freien User-Vektor `p_u`.
  Das Poster verwendet fuer eine spaetere, erweiterte Familie die Bezeichnung
  `Asymmetric-SVD++`. Diese Modelle duerfen im Repo nicht still vermischt
  werden.
- Claim impact:
  Jede Ergebnisdarstellung muss explizit sagen, welche Familie gemeint ist.

### D-003: explizite Feedback-Residuals werden im SGD-Schritt detachiert

- Status: accepted
- Scope: optimization contract for `asymmetric_svd` and `asvdpp`
- Source: Koren 2008 under-specifies the exact optimizer details for practical
  SGD with bias-coupled residual terms
- Decision:
  Im ersten Implementierungspfad werden die Groessen
  `w_uj = r_uj - b_uj` innerhalb des inneren SGD-Schritts als konstante
  Residualgewichte behandelt.
- Rationale:
  Das ergibt einen klaren, stabilen und reproduzierbaren Update-Vertrag.
  Eine vollstaendige Ableitung durch die Bias-Abhaengigkeit bleibt moeglich,
  waere aber ein eigener Optimizer-Variant.
- Claim impact:
  Diese Modelle duerfen nicht als exakt optimizer-faithful oder exakt
  paper-faithful bezeichnet werden. Zulaessig sind nur Claims, die den
  detachierten Repo-Optimizer explizit benennen.

### D-004: CB-Optimierung ist im Poster unter-spezifiziert

- Status: accepted
- Scope: `cb_svdpp`, `cb_asvdpp`
- Source: Mirbakhsh and Ling 2013 Poster
- Observation:
  Das Poster gibt Praediktoren fuer CB-Modelle an, spezifiziert aber nicht
  vollstaendig:
  - die exakte regularisierte Zielfunktion
  - die Rolle von `R*` in der Optimierung
  - die exakten Update-Regeln
- Decision:
  Das Repo verwendet fuer die erste implementierbare CB-Familie einen
  expliziten v1-Trainingsvertrag:
  - Cluster-Zuweisungen werden aus train-only `biased_mf`-Latents abgeleitet.
  - User- und Item-Cluster werden separat mit `KMeans` gelernt.
  - Cluster-Zuweisungen bleiben waehrend des eigentlichen CB-Trainings fix.
  - `R_star` wird ausschliesslich aus Trainingsratings als Cluster-Mean-Matrix
    plus Cluster-Count-Matrix berechnet.
  - `R_star` dient in v1 als Diagnose- und Nachweisartefakt, nicht als
    zusaetzliche Optimierungsaufgabe.
  - Cluster-Latents `p_C`, `q_C`, `y_C` und `x_C` werden direkt im
    Hauptobjective mit den individuellen Parametern trainiert.
  - `alpha` ist in v1 ein fester Hyperparameter und wird nicht mitoptimiert.
- Rationale:
  Diese Entscheidung erhaelt den publizierten Praediktor, vermeidet eine freie
  Erfindung einer zweiten, im Poster nicht belegten Faktorierungsstufe fuer
  `R_star`, und fuehrt zu einem klaren, reproduzierbaren und testbaren
  Implementierungsvertrag.
- Claim impact:
  `cb_svdpp` und `cb_asvdpp` duerfen in diesem Repo als source-grounded
  predictor mit repo-defined optimization bezeichnet werden. Sie sind nicht
  fuer Claims wie `exact paper reproduction` oder `exact optimizer-faithful
  reproduction` freigegeben.
