# Notation

## Purpose

Dieses Dokument ist die kanonische Notationsquelle fuer die Modellfamilien in
diesem Repo.

## Source Grounding

Die Notation lehnt sich an:

1. Koren 2008 fuer `biased_mf`, `svdpp` und `asymmetric_svd`
2. Mirbakhsh and Ling 2013 fuer `cb_svdpp`, `cb_asvdpp` und die dortige
   `Asymmetric-SVD++`-Benennung

an.

## Naming Note

Dieses Repo unterscheidet strikt:

- `asymmetric_svd`: Koren 2008, kein freier User-Vektor `p_u`
- `asvdpp`: Poster-Familie `SVD++` plus expliziter item-seitiger Feedback-Block

Diese Begriffe duerfen nicht vermischt werden.

## Core Sets

- `U`: Menge aller Nutzer
- `I`: Menge aller Items
- `K`: Menge aller beobachteten Ratings, also `K = {(u, i) | r_ui ist bekannt}`
- `R(u)`: Menge aller explizit von Nutzer `u` bewerteten Items
- `N(u)`: Menge aller impliziten Rueckmeldungen von Nutzer `u`

Default-Regel fuer dieses Repo:

- Wenn nur Rating-Daten vorliegen, dann ist im paper-faithful Profil
  `N(u) = R(u)`.
- Wenn spaeter echte implizite Ereignisse verfuegbar sind, muss das Profil dies
  explizit dokumentieren.

## Ratings And Baselines

- `r_ui`: beobachtetes Rating von Nutzer `u` fuer Item `i`
- `r_hat_ui`: vorhergesagtes Rating
- `mu`: globaler Rating-Mittelwert auf dem Trainingssatz
- `b_u`: Nutzer-Bias
- `b_i`: Item-Bias
- `b_ui = mu + b_u + b_i`: baseline estimate

## Latent Factors

- `f`: Latent-Dimension
- `p_u in R^f`: freier Nutzer-Vektor
- `q_i in R^f`: Item-Vektor
- `y_j in R^f`: impliziter Item-Faktor
- `x_j in R^f`: expliziter item-seitiger Faktor

## Normalization Terms

- `norm_R(u) = |R(u)|^(-1/2)` wenn `|R(u)| > 0`, sonst `0`
- `norm_N(u) = |N(u)|^(-1/2)` wenn `|N(u)| > 0`, sonst `0`

Diese Normalisierung wird fuer alle Modelle kanonisch verwendet.

## Residual Weights For Explicit Feedback Blocks

Fuer Modelle mit explizitem item-seitigem Feedback-Block definieren wir:

- `w_uj = r_uj - b_uj`

mit

- `b_uj = mu + b_u + b_j`

Wichtig:

Im ersten Implementierungspfad wird `w_uj` innerhalb eines inneren SGD-Schritts
als detachierte Groesse behandelt. Siehe
[docs/methodology/deviations_from_paper.md](../methodology/deviations_from_paper.md)
und
[objective_functions.md](objective_functions.md).

## Cluster Notation

- `A`: Menge der User-Cluster
- `B`: Menge der Item-Cluster
- `a(u) in A`: Cluster-Zuordnung von Nutzer `u`
- `b(i) in B`: Cluster-Zuordnung von Item `i`
- `n_A = |A|`: Anzahl User-Cluster
- `n_B = |B|`: Anzahl Item-Cluster
- `alpha in [0, 1]`: Mischkoeffizient zwischen individueller und Cluster-Ebene

Cluster-Level Faktoren:

- `p_C[a] in R^f`: Cluster-Nutzervektor fuer User-Cluster `a`
- `q_C[b] in R^f`: Cluster-Itemvektor fuer Item-Cluster `b`
- `y_C[b] in R^f`: Cluster-Implizitvektor fuer Item-Cluster `b`
- `x_C[b] in R^f`: Cluster-Explizitvektor fuer Item-Cluster `b`

## Cluster Rating Matrix

Das Poster fuehrt eine Cluster-zu-Cluster Matrix `R_star` ein. In diesem Repo
definieren wir:

- `R_star[a, b] = mean({r_ui | a(u) = a, b(i) = b, (u, i) in K_train})`

Wichtig:

- `R_star` wird ausschliesslich aus Trainingsdaten gebildet.
- Die exakte Rolle von `R_star` in der Optimierung der CB-Modelle ist im Poster
  nicht vollstaendig spezifiziert.

## Parameter Families

Zur Vermeidung stiller Regularisierungsvermischung benutzen wir diese Namen:

- `lambda_b` fuer Bias-Terme
- `lambda_p` fuer freie Nutzervektoren
- `lambda_q` fuer Itemvektoren
- `lambda_y` fuer implizite Faktoren
- `lambda_x` fuer explizite item-seitige Faktoren
- `lambda_pC`, `lambda_qC`, `lambda_yC`, `lambda_xC` fuer Cluster-Faktoren

Paper-faithful Profile duerfen diese Werte binden, aber die Notation bleibt
getrennt.
