# Regressão Logística

**Prioridade: Essencial**

Apesar do nome “regressão”, Logistic Regression é um dos modelos clássicos de **classificação**. Ela é um baseline forte porque produz scores/probabilidades, é rápida e seus coeficientes têm interpretação útil.

## 1. O que faz?

Modela a probabilidade de uma classe, normalmente binária.

Primeiro calcula um score linear:

\[
z = \beta_0 + \beta_1x_1 + \cdots + \beta_px_p
\]

Depois aplica a função sigmoide:

\[
P(y=1|x)=\frac{1}{1+e^{-z}}
\]

O resultado fica entre 0 e 1.

## 2. Intuição

Features movem o **log-odds** da classe positiva. Depois transformamos esse score em probabilidade.

A classe final depende de um threshold:

```text
probabilidade 0.82
threshold 0.50
→ classe 1
```

Mas 0.5 não é uma lei. O threshold deve refletir custos e capacidade operacional.

## 3. Quando usar?

- classificação binária/multiclasse;
- baseline principal para dados tabulares;
- quando interpretação de coeficientes importa;
- problemas aproximadamente lineares no log-odds;
- quando você quer probabilidades relativamente bem comportadas.

## 4. Vantagens

- rápida;
- forte baseline;
- regularização embutida;
- probabilidades disponíveis;
- interpretação de direção dos coeficientes.

## 5. Limitações

- fronteira linear no espaço transformado;
- interações e não linearidades precisam de feature engineering;
- coeficientes podem ser instáveis com multicolinearidade;
- scaling é recomendado com regularização.

## 6. Código correto com pipeline

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("model", LogisticRegression(C=1.0, max_iter=3000))
])

pipe.fit(X_train, y_train)
pred = pipe.predict(X_test)
prob = pipe.predict_proba(X_test)[:, 1]

print(classification_report(y_test, pred))
print("ROC-AUC:", roc_auc_score(y_test, prob))
```

## Odds e odds ratio

Se:

\[
\log\frac{p}{1-p}=\beta_0+\beta_1x
\]

então `exp(β1)` é um odds ratio associado a +1 unidade em `x`, mantendo as outras variáveis constantes.

```python
import numpy as np
coef = pipe.named_steps["model"].coef_[0]
print(np.exp(coef))
```

Com features padronizadas, +1 unidade equivale a aproximadamente +1 desvio padrão, não +1 unidade original.

## Regularização e `C`

No scikit-learn, `C` é o **inverso** da força de regularização:

```text
C pequeno → regularização forte
C grande  → regularização fraca
```

Isso costuma confundir iniciantes.

## Class imbalance

```python
LogisticRegression(class_weight="balanced")
```

Pode ajudar, mas não substitui escolha de métrica e threshold. Avalie Precision/Recall/PR-AUC.

## Experimente threshold

```python
import numpy as np

for t in [0.2, 0.4, 0.5, 0.7, 0.8]:
    pred_t = (prob >= t).astype(int)
    print(t, pred_t.sum())
```

Depois calcule precision e recall em cada threshold.

## Anki

- Por que Logistic Regression é classificação?
- Qual a função da sigmoide?
- O que `C` pequeno significa no scikit-learn?
- Por que threshold 0.5 não é obrigatório?
- Qual a diferença entre `predict` e `predict_proba`?
