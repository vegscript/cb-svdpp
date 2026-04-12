# Update Rules

## Purpose

Dieses Dokument definiert die kanonischen SGD-Update-Regeln fuer die
implementierbaren Modellfamilien in diesem Repo.

## General Rule

Fuer einen beobachteten Trainingspunkt `(u, i)` gilt:

```text
e_ui = r_ui - r_hat_ui
```

Mit Lernrate `eta` werden Parameter durch Gradientenabstieg auf die
regularisierte Fehlerfunktion aktualisiert.

Wichtig:

- Bei gekoppelten Updates wird der fuer die Ableitung benoetigte alte Zustand
  explizit zwischengespeichert.
- Wenn ein Modell `w_uj` verwendet, wird diese Groesse im ersten
  Implementierungspfad im inneren Schritt als detachiert behandelt.

## 1. `biased_mf`

Praediktor:

```text
r_hat_ui = mu + b_u + b_i + p_u^T q_i
```

Updates:

```text
b_u <- b_u + eta * (e_ui - lambda_b * b_u)
b_i <- b_i + eta * (e_ui - lambda_b * b_i)

let p_old = p_u
let q_old = q_i

p_u <- p_u + eta * (e_ui * q_old - lambda_p * p_old)
q_i <- q_i + eta * (e_ui * p_old - lambda_q * q_old)
```

## 2. `svdpp`

Zwischenrepraesentation:

```text
z_u = p_u + norm_N(u) * sum_{j in N(u)} y_j
```

Updates:

```text
b_u <- b_u + eta * (e_ui - lambda_b * b_u)
b_i <- b_i + eta * (e_ui - lambda_b * b_i)

let p_old = p_u
let q_old = q_i

p_u <- p_u + eta * (e_ui * q_old - lambda_p * p_old)
q_i <- q_i + eta * (e_ui * z_u - lambda_q * q_old)

for j in N(u):
    y_j <- y_j + eta * (e_ui * norm_N(u) * q_old - lambda_y * y_j)
```

## 3. `asymmetric_svd`

Zwischenrepraesentation:

```text
z_u =
norm_R(u) * sum_{j in R(u)} w_uj * x_j
+ norm_N(u) * sum_{j in N(u)} y_j
```

Updates:

```text
b_u <- b_u + eta * (e_ui - lambda_b * b_u)
b_i <- b_i + eta * (e_ui - lambda_b * b_i)

let q_old = q_i

q_i <- q_i + eta * (e_ui * z_u - lambda_q * q_old)

for j in R(u):
    x_j <- x_j + eta * (e_ui * norm_R(u) * w_uj * q_old - lambda_x * x_j)

for j in N(u):
    y_j <- y_j + eta * (e_ui * norm_N(u) * q_old - lambda_y * y_j)
```

Wichtig:

- Diese Updates folgen dem detachierten `w_uj`-Vertrag.
- Eine exakte Bias-Rueckfuehrung durch `w_uj` waere ein anderer Optimizer und
  muss separat dokumentiert werden.

## 4. `asvdpp`

Zwischenrepraesentation:

```text
z_u =
p_u
+ norm_N(u) * sum_{j in N(u)} y_j
+ norm_R(u) * sum_{j in R(u)} w_uj * x_j
```

Updates:

```text
b_u <- b_u + eta * (e_ui - lambda_b * b_u)
b_i <- b_i + eta * (e_ui - lambda_b * b_i)

let p_old = p_u
let q_old = q_i

p_u <- p_u + eta * (e_ui * q_old - lambda_p * p_old)
q_i <- q_i + eta * (e_ui * z_u - lambda_q * q_old)

for j in R(u):
    x_j <- x_j + eta * (e_ui * norm_R(u) * w_uj * q_old - lambda_x * x_j)

for j in N(u):
    y_j <- y_j + eta * (e_ui * norm_N(u) * q_old - lambda_y * y_j)
```

## 5. `cb_svdpp`

Gemischte Faktoren:

```text
q_mix_i = (1 - alpha) * q_i + alpha * q_C[b(i)]
p_mix_u = (1 - alpha) * p_u + alpha * p_C[a(u)]
y_mix_j = (1 - alpha) * y_j + alpha * y_C[b(j)]

z_u = p_mix_u + norm_N(u) * sum_{j in N(u)} y_mix_j
```

Updates:

```text
b_u <- b_u + eta * (e_ui - lambda_b * b_u)
b_i <- b_i + eta * (e_ui - lambda_b * b_i)

let q_old = q_i
let qC_old = q_C[b(i)]
let p_old = p_u
let pC_old = p_C[a(u)]
let qmix_old = q_mix_i
let z_old = z_u

q_i      <- q_i      + eta * ((1 - alpha) * e_ui * z_old - lambda_q  * q_old)
q_C[b(i)]<- q_C[b(i)]+ eta * (alpha       * e_ui * z_old - lambda_qC * qC_old)

p_u      <- p_u      + eta * ((1 - alpha) * e_ui * qmix_old - lambda_p  * p_old)
p_C[a(u)]<- p_C[a(u)]+ eta * (alpha       * e_ui * qmix_old - lambda_pC * pC_old)

for j in N(u):
    y_j      <- y_j      + eta * ((1 - alpha) * e_ui * norm_N(u) * qmix_old - lambda_y  * y_j)
    y_C[b(j)]<- y_C[b(j)]+ eta * (alpha       * e_ui * norm_N(u) * qmix_old - lambda_yC * y_C[b(j)])
```

Implementierungshinweis:

- In der tatsaechlichen Implementierung muessen Mehrfachtreffer desselben
  Cluster-Index innerhalb `N(u)` sauber aggregiert oder bewusst mehrfach
  angewendet werden. Die gewaehlte Variante muss dokumentiert sein.

## 6. `cb_asvdpp`

Gemischte Faktoren:

```text
q_mix_i = (1 - alpha) * q_i + alpha * q_C[b(i)]
p_mix_u = (1 - alpha) * p_u + alpha * p_C[a(u)]
y_mix_j = (1 - alpha) * y_j + alpha * y_C[b(j)]
x_mix_j = (1 - alpha) * x_j + alpha * x_C[b(j)]

z_u =
p_mix_u
+ norm_N(u) * sum_{j in N(u)} y_mix_j
+ norm_R(u) * sum_{j in R(u)} w_uj * x_mix_j
```

Updates:

```text
b_u <- b_u + eta * (e_ui - lambda_b * b_u)
b_i <- b_i + eta * (e_ui - lambda_b * b_i)

let q_old = q_i
let qC_old = q_C[b(i)]
let p_old = p_u
let pC_old = p_C[a(u)]
let qmix_old = q_mix_i
let z_old = z_u

q_i      <- q_i      + eta * ((1 - alpha) * e_ui * z_old - lambda_q  * q_old)
q_C[b(i)]<- q_C[b(i)]+ eta * (alpha       * e_ui * z_old - lambda_qC * qC_old)

p_u      <- p_u      + eta * ((1 - alpha) * e_ui * qmix_old - lambda_p  * p_old)
p_C[a(u)]<- p_C[a(u)]+ eta * (alpha       * e_ui * qmix_old - lambda_pC * pC_old)

for j in N(u):
    y_j      <- y_j      + eta * ((1 - alpha) * e_ui * norm_N(u) * qmix_old - lambda_y  * y_j)
    y_C[b(j)]<- y_C[b(j)]+ eta * (alpha       * e_ui * norm_N(u) * qmix_old - lambda_yC * y_C[b(j)])

for j in R(u):
    x_j      <- x_j      + eta * ((1 - alpha) * e_ui * norm_R(u) * w_uj * qmix_old - lambda_x  * x_j)
    x_C[b(j)]<- x_C[b(j)]+ eta * (alpha       * e_ui * norm_R(u) * w_uj * qmix_old - lambda_xC * x_C[b(j)])
```

## 7. Claim Boundary

Die hier definierten Update-Regeln sind der kanonische Implementierungsvertrag
dieses Repos.

Sie duerfen nur dann als exakt paper-faithful bezeichnet werden, wenn:

- die Quellen diese Regeln vollstaendig tragen oder
- die Abweichung explizit und akzeptiert in
  `docs/methodology/deviations_from_paper.md` dokumentiert ist
