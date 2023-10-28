# Importando pacotes
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

# conhecendo a base de dados

base_quinto_andar = pd.read_excel("dataset_quinto-andar-scraper_2023-10-22_17-44-21-900.xlsx")
# print(base_quinto_andar.head())
# print(base_quinto_andar.shape)

# Removendo linhas e colunas em branco
# Serão considerados apenas os empreendimentos em BH

base_quinto_andar.dropna(axis=0, how="all", inplace=True)
base_quinto_andar.dropna(axis=1, how="all", inplace=True)

base_quinto_andar_v2 = base_quinto_andar.loc[base_quinto_andar["houseInfo/address/city"] == "Belo Horizonte"]

# Vamos averiguar os outliers relativos a preço e area

# print(max(base_quinto_andar_v2["houseInfo/area"]))
# print(min(base_quinto_andar_v2["houseInfo/area"]))
# print(np.mean(base_quinto_andar_v2["houseInfo/area"]))

# base_quinto_andar_v2["houseInfo/area"].hist(bins=10)

# O empreendimento de 30000 de aluguel causa uma grande distorção
remover_area = base_quinto_andar_v2[base_quinto_andar_v2["houseInfo/area"] < 30000]
base_quinto_andar_v3 = remover_area

# base_quinto_andar_v3["houseInfo/area"].hist(bins=10)

# A grande maioria dos empreendimentos tem até 100 metros quadrados, o que era esperado
# Agora iremos averiguar outliers em relação ao preço

# base_quinto_andar_v3["houseInfo/rentPrice"].hist(bins=10)

# Consideraremos até 7mil reais

remover_rent = base_quinto_andar_v2[base_quinto_andar_v2["houseInfo/rentPrice"] < 7000]

remover_rent["houseInfo/rentPrice"].hist(bins=10)
bqa_tratada = remover_rent

# A seleção de atributo terá como metodologia as diretrizes do quinto andar
# E a métrica do ganho de informação

def entropia(col):
    entropia = 0
    col_cte = Counter(col)

    for classe in col.unique():
        prob = col_cte[classe] / len(col)
        entropia -= prob * math.log2(prob)

    return entropia


def ganho_de_informacao(col):
    col_cte = Counter(col)
    entropia_ponderada = 0

    for classe in col.unique():
        prob_label = col_cte[classe] / len(col)
        entropia_ponderada -= prob_label * prob_label * math.log2(prob_label)

    GI = entropia(col) - entropia_ponderada

    return GI

# Criando uma coluna para especificar faixas relativas as áreas e preço de condomínio

bqa_tratada_v2 = bqa_tratada.copy()

limites = [0, 30, 60, 90, 150, 300, 800]
valores_area = ['0-30', '31-60', '61-90', '91 - 150', '151 - 300', 'acima de 300']
bqa_tratada_v2.loc[:, "Faixa Área"] = pd.cut(bqa_tratada["houseInfo/area"], bins=limites, labels=valores_area, include_lowest=True)

limites2 = [0, 300, 600, 1200, 2000, 3312]
valores_condo = ['0-300', '301-600', '601-1200', '1201-2000','Acima de 2000']
bqa_tratada_v2.loc[:, "Faixa condominio"] = pd.cut(bqa_tratada["houseInfo/condoPrice"],bins=limites2, labels=valores_condo, include_lowest=True)


# Calculando os ganhos de informação para as caracteristicas mais relevantes do quinto andar

bairro = bqa_tratada_v2["houseInfo/address/neighborhood"]
banheiros = bqa_tratada_v2["houseInfo/bathrooms"]
area = bqa_tratada_v2["Faixa Área"]
quartos = bqa_tratada_v2["houseInfo/bedrooms"]
condo = bqa_tratada_v2["Faixa condominio"]

# Produzindo uma matriz de Ganhos de informação

GI_matriz = {
    "Bairro": ganho_de_informacao(bairro),
    "Banheiros": ganho_de_informacao(banheiros),
    "Área": ganho_de_informacao(area),
    "Quartos": ganho_de_informacao(quartos),
    "Condomínio": ganho_de_informacao(condo)
}

# Monstando o dataset que será aplicado aos modelos

atributos = {
    "Bairro": bqa_tratada_v2["houseInfo/address/neighborhood"],
    "Banheiros": bqa_tratada_v2["houseInfo/bathrooms"],
    "Área": bqa_tratada_v2["Faixa Área"],
    "Quartos": bqa_tratada_v2["houseInfo/bedrooms"],
    "Condomínio": bqa_tratada_v2["Faixa condominio"],
    "Preço aluguel": bqa_tratada_v2["houseInfo/rentPrice"]
}

atributos_df = pd.DataFrame(atributos)

# Começando a aplicar os modelos de machine learning

X = atributos_df[["Banheiros", "Área", "Quartos", "Condomínio"]]
X_dummies = pd.get_dummies(X, columns=["Banheiros", "Área", "Quartos", "Condomínio"])
y = atributos_df["Preço aluguel"]

# Modelo RandomForestRegressor

X_train, X_test, y_train, y_test = train_test_split(X_dummies, y, test_size=0.2, random_state=42)

modelo_1 = RandomForestRegressor(n_estimators=100, random_state=42)
modelo_1.fit(X_train, y_train)

y_pred = modelo_1.predict(X_test)

diferencas = y_test - y_pred
diferencas_absolutas = np.abs(diferencas)
diferenca_percentual = 100 * np.abs(y_test - y_pred) / y_test
desvio_padrao = np.std(diferenca_percentual)

resultado = pd.DataFrame({
    'Real': y_test,
    'Previsto': y_pred,
    'Diferença': diferencas,
    'Diferença Absoluta': diferencas_absolutas,
    'Diferença Percentual': diferenca_percentual,
    'Desvio padrão': desvio_padrao
})

# print(resultado)
print("Raiz do Erro Quadrático Médio (RMSE):", np.sqrt(mean_squared_error(y_test, y_pred)))
print("Média da Diferença Percentual:", np.mean(resultado["Diferença Percentual"]))

# Modelo RegressãoLinear com validação cruzada

X2 = atributos_df[["Banheiros", "Área", "Quartos", "Condomínio"]]
X_dummies2 = pd.get_dummies(X2, columns=["Banheiros", "Área", "Quartos", "Condomínio"])
y2 = atributos_df["Preço aluguel"]

modelo_2 = LinearRegression()
cv = KFold(n_splits=5, shuffle=True, random_state=42)
y_pred2 = cross_val_predict(modelo_2, X_dummies2, y2, cv=cv)

diferencas2 = y2 - y_pred2
diferencas_absolutas2 = np.abs(diferencas2)
diferenca_percentual2 = 100 * np.abs(y2 - y_pred2) / y2
desvio_padrao2 = np.std(diferenca_percentual2)

resultado2 = pd.DataFrame({
    'Real': y2,
    'Previsto': y_pred2,
    'Diferença': diferencas2,
    'Diferença Absoluta': diferencas_absolutas2,
    'Diferença Percentual': diferenca_percentual2
})

condicao = (resultado2['Previsto'] >= 1) & (resultado2['Previsto'] <= 100000)
resultado_filtrado = resultado2[condicao]

print("Raiz do Erro Quadrático Médio (RMSE) após filtragem:", np.sqrt(mean_squared_error(resultado_filtrado['Real'], resultado_filtrado['Previsto'])))
print("Média da Diferença Percentual após filtragem:", np.mean(resultado_filtrado['Diferença Percentual']))

# Modelo RegressãoLinear com KNNRegressor

modelo_3 = KNeighborsRegressor(n_neighbors=5)

X3 = atributos_df[["Banheiros", "Área", "Quartos", "Condomínio"]]
X_dummies3 = pd.get_dummies(X3, columns=["Banheiros", "Área", "Quartos", "Condomínio"])
y3 = atributos_df["Preço aluguel"]

cv3 = KFold(n_splits=5, shuffle=True, random_state=42)
y_pred3 = cross_val_predict(modelo_3, X_dummies3, y3, cv=cv3)

diferencas3 = y3 - y_pred3
diferencas_absolutas3 = np.abs(diferencas3)
diferenca_percentual3 = 100 * np.abs(y3 - y_pred3) / y3
desvio_padrao3 = np.std(diferenca_percentual3)

resultado3 = pd.DataFrame({
    'Real': y3,
    'Previsto': y_pred3,
    'Diferença': diferencas3,
    'Diferença Absoluta': diferencas_absolutas3,
    'Diferença Percentual': diferenca_percentual3
})

print("Raiz do Erro Quadrático Médio (RMSE):", np.sqrt(mean_squared_error(y3, y_pred3)))
print("Média da Diferença Percentual:", np.mean(resultado3["Diferença Percentual"]))