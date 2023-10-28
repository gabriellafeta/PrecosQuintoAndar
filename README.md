# PrecosQuintoAndar
Comparando algoritmos de regressão utilizando base extraída do Quinto Andar para prever preços de alugueis da cidade de Belo Horizonte.

## Importando os pacote

Os pacotes utilziados foram:

```
import pandas as pd
import openpyxl
import matplotlib.pyplot as plt
import numpy as np
import math
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsRegressor
import seaborn as sns

```


## Análise exploratória

Através de uma análise manual das colunas do dataset extraído da [Apify](https://apify.com/) foi possível correlacionar com o [Método de precificação de aluguel do Quinto Andar](https://mkt.quintoandar.com.br/quanto-cobrar-de-aluguel/) e encontrar diretamente as colunas de maior interesse:

```
bairro = ["houseInfo/address/neighborhood"]
banheiros = ["houseInfo/bathrooms"]
área = ["houseInfo/area"]
quartos = ["houseInfo/bedrooms"]
condomínio = ["houseInfo/condoPrice"]
aluguel = ["houseInfo/rentPrice"]

```
O dataset inicial continha 997 linhas, mas ele foi tratado retirando as linhas em branco e considerando apenas o município de Belo Horizonte

```
base_quinto_andar.dropna(axis=0, how="all", inplace=True)
base_quinto_andar.dropna(axis=1, how="all", inplace=True)

base_quinto_andar_v2 = base_quinto_andar.loc[base_quinto_andar["houseInfo/address/city"] == "Belo Horizonte"]
```

Então, foram calculados os valores máximo, mínimo e médio para averiguar outliers relativos à área:

```
print(max(base_quinto_andar_v2["houseInfo/area"]))
-> 30000
print(min(base_quinto_andar_v2["houseInfo/area"]))
-> 15
print(np.mean(base_quinto_andar_v2["houseInfo/area"]))
-> 130
```
Assim, o dataset foi tratado para retirar este outlier:

```
remover_area = base_quinto_andar_v2[base_quinto_andar_v2["houseInfo/area"] < 30000]
base_quinto_andar_v3 = remover_area
```
Pela visualização gráfica é possível ver a nova distribuição:
```
base_quinto_andar_v3["houseInfo/area"].hist(bins=10)
```
<img src="área pós tratamento.png"
   width="500"
     height="400">

     
Da mesma forma veremos os outliers de preço de aluguel por inspeção visual:
```
base_quinto_andar_v3["houseInfo/rentPrice"].hist(bins=10)
```

<img src="preço aluguel antes.png"
   width="500"
     height="400">

  nota-se grande correção entre área e preço de aluguel, mas para fins de simplificação removeremos aqueles cujo aluguel superam 7 mil reais mensais:

```
remover_rent = base_quinto_andar_v2[base_quinto_andar_v2["houseInfo/rentPrice"] < 7000]
remover_rent["houseInfo/rentPrice"].hist(bins=10)
```
<img src="Captura de tela 2023-10-27 221354.png"
   width="500"
     height="400">

O dataset final ficou com 939 linhas. Por fim, mostrarei o preço médio de aluguel para os top 10 bairros em relação a número de empreendimentos:

```
bqa_tratada = remover_rent


media_preco_bairros = bqa_tratada.groupby("houseInfo/address/neighborhood")["houseInfo/rentPrice"].mean()
contagem_bairros = bqa_tratada["houseInfo/address/neighborhood"].value_counts()
top_10_bairros = contagem_bairros.head(10).index
media_preco_top_10_bairros = media_preco_bairros[top_10_bairros]

media_preco_top_10_bairros_sorted = media_preco_top_10_bairros.sort_values(ascending=False)

plt.figure(figsize=(15, 10))
sns.barplot(x=media_preco_top_10_bairros_sorted.index, y=media_preco_top_10_bairros_sorted.values, palette="viridis")
plt.xticks(rotation=45)
plt.xlabel("Bairro")
plt.ylabel("Média de Preço de Aluguel")
plt.title("Média de Preço de Aluguel nos 10 Bairros com Mais Itens")
plt.show()

```

<img src="bairros.png"
   width="500"
     height="400">

Percebe-se que o número de lugares para alugar na base é diretamente proporcional ao preço do aluguel. O Bairro Gutierrez possui maior número de empreendimentos listados e preço médio de aluguel de R$ 3500,00.

## Seleção de atributo

A seleção de atributo terá como base o [Ganho de informação]()
