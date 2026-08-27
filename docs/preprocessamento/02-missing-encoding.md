# Missing values e encoding

**Prioridade: Essencial**

Dados tabulares reais raramente chegam prontos. Duas classes de problemas aparecem cedo: valores ausentes e variáveis categóricas.

## Missing values

Um `NaN` não significa sempre a mesma coisa. Pode representar:

- informação realmente desconhecida;
- campo não aplicável;
- falha de coleta;
- ausência estrutural;
- evento que ainda não ocorreu.

Antes de imputar, entenda o mecanismo.

## Estratégias comuns

### Numéricas

- mediana: robusta a outliers;
- média: simples, mas sensível a extremos;
- valor constante: útil quando ausência tem significado operacional;
- modelos de imputação: mais sofisticados, nem sempre necessários.

```python
from sklearn.impute import SimpleImputer

imp = SimpleImputer(strategy="median")
X_num = imp.fit_transform(X_num)
```

### Categóricas

```python
SimpleImputer(strategy="most_frequent")
```

ou uma categoria explícita:

```python
SimpleImputer(strategy="constant", fill_value="NAO_INFORMADO")
```

## Missing indicator

Às vezes, **o fato de estar ausente é informativo**.

```python
SimpleImputer(strategy="median", add_indicator=True)
```

Isso cria indicadores binários para ausência.

!!! warning "Imputação também aprende"
    Mediana, média e moda devem ser estimadas no treino, não no dataset inteiro. Coloque o imputer dentro de um `Pipeline`.

## Encoding

Modelos precisam de representação numérica. Uma variável como:

```text
plano = bronze, prata, ouro
```

não pode ser enviada como texto puro para a maioria dos estimadores do scikit-learn.

## One-hot encoding

Cria uma coluna por categoria.

```text
plano_bronze plano_prata plano_ouro
1            0           0
0            1           0
0            0           1
```

```python
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder(handle_unknown="ignore")
```

`handle_unknown="ignore"` evita quebrar quando produção contém categoria não vista no treino.

## Ordinal encoding

Se existe ordem real:

```text
baixo < médio < alto
```

podemos mapear para valores ordenados. Mas não invente ordem onde ela não existe.

```python
from sklearn.preprocessing import OrdinalEncoder

enc = OrdinalEncoder(categories=[["baixo", "medio", "alto"]])
```

## O perigo do label encoding arbitrário

Mapear:

```text
SP=0, RJ=1, MG=2
```

pode sugerir ao modelo uma ordem e distância que não existem. Para categorias nominais, one-hot é um baseline muito mais seguro.

## Cardinalidade alta

Uma coluna com milhares de categorias pode explodir dimensionalidade no one-hot. Alternativas incluem:

- agrupar categorias raras;
- hashing;
- target encoding com validação cuidadosamente protegida contra leakage;
- embeddings em modelos apropriados.

No 80/20, comece perguntando se aquela coluna realmente agrega valor e se categorias raras podem ser agrupadas.

## `ColumnTransformer`

Datasets misturam numéricas e categóricas. O `ColumnTransformer` permite pipelines diferentes por grupo.

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

num_cols = ["idade", "renda"]
cat_cols = ["estado", "plano"]

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

pre = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols)
])
```

Esse objeto pode entrar em um `Pipeline` junto com o modelo.

## Perguntas de recuperação

1. Por que imputação pode causar leakage?
2. Quando uma flag de missing pode ajudar?
3. Por que one-hot é mais seguro que mapear categorias nominais para 0,1,2?
4. O que `handle_unknown="ignore"` resolve?
5. Qual o papel do `ColumnTransformer`?
