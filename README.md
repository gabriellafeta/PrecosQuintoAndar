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

```


