# ROC-AUC, PR-AUC e threshold

**Prioridade: Essencial**

Algumas métricas avaliam o **ranking dos scores** em vários thresholds, em vez de uma única decisão binária.

## Score antes da classe

Muitos classificadores produzem:

```text
cliente A → 0.92
cliente B → 0.71
cliente C → 0.48
cliente D → 0.13
```

O threshold converte score em decisão. Separar **qualidade do ranking** de **política de decisão** é um conceito poderoso.

## ROC Curve

A ROC traça:

- eixo Y: True Positive Rate = Recall;
- eixo X: False Positive Rate = FP / (FP + TN).

Cada ponto corresponde a um threshold.

```python
from sklearn.metrics import RocCurveDisplay, roc_auc_score

RocCurveDisplay.from_predictions(y_test, prob)
print(roc_auc_score(y_test, prob))
```

## Intuição da ROC-AUC

Uma interpretação útil: probabilidade de um positivo aleatório receber score maior que um negativo aleatório, sob condições usuais e sem empates complicadores.

- 0.5 → ranking próximo do aleatório;
- 1.0 → separação perfeita;
- abaixo de 0.5 → ranking invertido ou problema.

## PR Curve

Precision–Recall foca diretamente performance sobre a classe positiva.

```python
from sklearn.metrics import PrecisionRecallDisplay, average_precision_score

PrecisionRecallDisplay.from_predictions(y_test, prob)
print(average_precision_score(y_test, prob))
```

Em classe positiva rara, PR costuma ser mais informativa sobre o trade-off operacional entre encontrar positivos e manter qualidade dos alertas.

## ROC-AUC alta não escolhe threshold

Um modelo pode ter excelente ranking, mas a operação ainda precisa definir:

- quantos alertas comporta;
- custo de FP e FN;
- benefício por TP;
- prevalência atual;
- requisitos regulatórios/negócio.

## Escolha de threshold por custo

Exemplo simplificado:

```text
TP = +R$ 100 evitados
FP = -R$ 5 de custo operacional
FN = -R$ 100 perdidos
```

Você pode calcular valor por threshold e escolher o que maximiza a função de utilidade, em vez de maximizar F1 por hábito.

```python
import numpy as np
from sklearn.metrics import confusion_matrix

for t in np.linspace(0.05, 0.95, 19):
    pred_t = (prob >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred_t).ravel()
    valor = 100*tp - 5*fp - 100*fn
    print(round(t,2), valor)
```

## Calibração

Um modelo é calibrado se, entre exemplos com score próximo de 0.8, aproximadamente 80% são positivos — simplificando a ideia.

Ranking e calibração são diferentes:

- ROC-AUC pode ser alta mesmo com probabilidades mal calibradas;
- uma política que usa valores absolutos de probabilidade pode exigir calibração.

Ferramentas:

```python
from sklearn.calibration import CalibrationDisplay
CalibrationDisplay.from_predictions(y_test, prob, n_bins=10)
```

## Prevalência muda

Precision depende da prevalência da classe. Se a frequência de fraude muda em produção, precision para o mesmo modelo/threshold pode mudar, mesmo que características de ranking permaneçam semelhantes.

## Anki

- O que ROC-AUC avalia conceitualmente?
- Por que PR-AUC é especialmente útil com classe positiva rara?
- Por que AUC alta não define threshold?
- Ranking bom implica calibração boa?
- Como custo de FP/FN pode orientar threshold?
