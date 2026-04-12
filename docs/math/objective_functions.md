# Objective Functions

## Purpose

Dieses Dokument definiert die kanonischen Zielfunktionen der Modellfamilien in
diesem Repo.

## Global Conventions

Alle Objectives in diesem Dokument gelten auf dem Trainingssatz `K_train`.

Wir definieren den Fehlerterm generell als:

```text
e_ui = r_ui - r_hat_ui
```

Die generische Loss-Form ist:

```text
J = sum_{(u, i) in K_train} e_ui^2 + regularization
```

## 1. Bias Baseline

```text
J_bias =
sum_{(u, i) in K_train} (r_ui - mu - b_u - b_i)^2
+ lambda_b * (sum_u b_u^2 + sum_i b_i^2)
```

Diese Stufe ist fuer Analysen nuetzlich, aber nicht die primare Basismodellstufe
des Repos.

## 2. `biased_mf`

Praediktor:

```text
r_hat_ui = mu + b_u + b_i + p_u^T q_i
```

Objective:

```text
J_bmf =
sum_{(u, i) in K_train} (r_ui - mu - b_u - b_i - p_u^T q_i)^2
+ lambda_b * (sum_u b_u^2 + sum_i b_i^2)
+ lambda_p * sum_u ||p_u||_2^2
+ lambda_q * sum_i ||q_i||_2^2
```

Paper grounding:

- entspricht der regularisierten Faktorisierungsform aus Koren 2008

Paper-faithful Spezialisierung:

- `lambda_b = lambda_p = lambda_q = lambda_3` ist erlaubt

## 3. `svdpp`

Praediktor:

```text
z_u_svdpp = p_u + norm_N(u) * sum_{j in N(u)} y_j
r_hat_ui = mu + b_u + b_i + q_i^T z_u_svdpp
```

Objective:

```text
J_svdpp =
sum_{(u, i) in K_train} (r_ui - mu - b_u - b_i - q_i^T z_u_svdpp)^2
+ lambda_b * (sum_u b_u^2 + sum_i b_i^2)
+ lambda_p * sum_u ||p_u||_2^2
+ lambda_q * sum_i ||q_i||_2^2
+ lambda_y * sum_j ||y_j||_2^2
```

Source status:

- der Praediktor ist in Koren 2008 explizit gegeben
- die exakte regularisierte Zielfunktion wird dort fuer `SVD++` nicht in voller
  Laenge ausformuliert
- die obige Zielfunktion ist der kanonische Repo-Vertrag

## 4. `asymmetric_svd`

Praediktor:

```text
z_u_asym =
norm_R(u) * sum_{j in R(u)} w_uj * x_j
+ norm_N(u) * sum_{j in N(u)} y_j

r_hat_ui = mu + b_u + b_i + q_i^T z_u_asym
```

mit

```text
w_uj = r_uj - (mu + b_u + b_j)
```

Objective:

```text
J_asym =
sum_{(u, i) in K_train} (r_ui - mu - b_u - b_i - q_i^T z_u_asym)^2
+ lambda_b * (sum_u b_u^2 + sum_i b_i^2)
+ lambda_q * sum_i ||q_i||_2^2
+ lambda_x * sum_j ||x_j||_2^2
+ lambda_y * sum_j ||y_j||_2^2
```

Source status:

- der Praediktor und eine regularisierte Loss-Familie sind in Koren 2008
  angegeben
- fuer den praktischen SGD-Vertrag wird in diesem Repo `w_uj` innerhalb eines
  inneren Schritts detachiert

Claim boundary:

- optimizer-faithful Claims sind erst nach finaler Entscheidung in
  `deviations_from_paper.md` zulaessig

## 5. `asvdpp`

Praediktor:

```text
z_u_asvdpp =
p_u
+ norm_N(u) * sum_{j in N(u)} y_j
+ norm_R(u) * sum_{j in R(u)} w_uj * x_j

r_hat_ui = mu + b_u + b_i + q_i^T z_u_asvdpp
```

Objective:

```text
J_asvdpp =
sum_{(u, i) in K_train} (r_ui - mu - b_u - b_i - q_i^T z_u_asvdpp)^2
+ lambda_b * (sum_u b_u^2 + sum_i b_i^2)
+ lambda_p * sum_u ||p_u||_2^2
+ lambda_q * sum_i ||q_i||_2^2
+ lambda_x * sum_j ||x_j||_2^2
+ lambda_y * sum_j ||y_j||_2^2
```

Source status:

- diese Familie folgt dem Poster-Praediktor fuer `Asymmetric-SVD++`
- das Poster spezifiziert keine vollstaendige regularisierte Zielfunktion
- die obige Zielfunktion ist der kanonische Repo-Vertrag

## 6. `cb_svdpp`

Gemischte Faktoren:

```text
q_mix_i = (1 - alpha) * q_i + alpha * q_C[b(i)]
p_mix_u = (1 - alpha) * p_u + alpha * p_C[a(u)]
y_mix_j = (1 - alpha) * y_j + alpha * y_C[b(j)]
```

Praediktor:

```text
z_u_cb_svdpp = p_mix_u + norm_N(u) * sum_{j in N(u)} y_mix_j
r_hat_ui = mu + b_u + b_i + q_mix_i^T z_u_cb_svdpp
```

Objective:

```text
J_cb_svdpp =
sum_{(u, i) in K_train} (r_ui - mu - b_u - b_i - q_mix_i^T z_u_cb_svdpp)^2
+ lambda_b  * (sum_u b_u^2 + sum_i b_i^2)
+ lambda_p  * sum_u ||p_u||_2^2
+ lambda_q  * sum_i ||q_i||_2^2
+ lambda_y  * sum_j ||y_j||_2^2
+ lambda_pC * sum_a ||p_C[a]||_2^2
+ lambda_qC * sum_b ||q_C[b]||_2^2
+ lambda_yC * sum_b ||y_C[b]||_2^2
```

Source status:

- der gemischte Praediktor folgt dem Poster
- das Poster spezifiziert die Optimierung nicht vollstaendig
- dieses Objective ist ein expliziter Repo-Vertrag, nicht automatisch eine
  exakte Paper-Reproduktion

## 7. `cb_asvdpp`

Gemischte Faktoren:

```text
q_mix_i = (1 - alpha) * q_i + alpha * q_C[b(i)]
p_mix_u = (1 - alpha) * p_u + alpha * p_C[a(u)]
y_mix_j = (1 - alpha) * y_j + alpha * y_C[b(j)]
x_mix_j = (1 - alpha) * x_j + alpha * x_C[b(j)]
```

Praediktor:

```text
z_u_cb_asvdpp =
p_mix_u
+ norm_N(u) * sum_{j in N(u)} y_mix_j
+ norm_R(u) * sum_{j in R(u)} w_uj * x_mix_j

r_hat_ui = mu + b_u + b_i + q_mix_i^T z_u_cb_asvdpp
```

Objective:

```text
J_cb_asvdpp =
sum_{(u, i) in K_train} (r_ui - mu - b_u - b_i - q_mix_i^T z_u_cb_asvdpp)^2
+ lambda_b  * (sum_u b_u^2 + sum_i b_i^2)
+ lambda_p  * sum_u ||p_u||_2^2
+ lambda_q  * sum_i ||q_i||_2^2
+ lambda_x  * sum_j ||x_j||_2^2
+ lambda_y  * sum_j ||y_j||_2^2
+ lambda_pC * sum_a ||p_C[a]||_2^2
+ lambda_qC * sum_b ||q_C[b]||_2^2
+ lambda_xC * sum_b ||x_C[b]||_2^2
+ lambda_yC * sum_b ||y_C[b]||_2^2
```

Source status:

- der gemischte Praediktor folgt dem Poster
- die Optimierungsform ist im Poster nicht vollstaendig gegeben
- dieses Objective ist der kanonische Repo-Vertrag fuer die erste
  implementierbare Fassung

## 8. CB-Specific Constraint

Fuer alle CB-Objectives gilt:

- Cluster-Assignments werden ausserhalb des SGD-Schritts erzeugt und waehrend
  des Trainings als fix betrachtet.
- `R_star` wird ausschliesslich aus Trainingsdaten berechnet.
- Wenn spaeter eine zusaetzliche Auxiliary-Loss auf `R_star` eingefuehrt wird,
  ist das eine neue `extended` Modellfamilie und keine stille Aenderung.
