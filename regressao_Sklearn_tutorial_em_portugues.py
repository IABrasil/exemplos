#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 15:41:38 2017

@author: ubuntu1604
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 09:44:42 2017

@author: Gleber Tacio Teixeira
"""

import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler




input_file = "seuDiretorio/Concrete_Data.csv"

# Importando dados do arquivo CSV, com delimitador default sendo virgula
df = pd.read_csv(input_file, header = 0)

# Removendo as colunas sem dados numericos
df = df._get_numeric_data()

# criando um numpy array com os valores numericos para usar como input no scikit-learn
df_array = df.as_matrix()

X = df_array[:, 1:]  # selecionando colunas de 1 até quantas existirem, como Features (variáveis independentes)
y = df_array[:, 0]   # selecionando coluna 0 (primeira coluna) como Label (variável dependente, Target)

#Criando um dataset de treino e um de teste a partir das amostras no arquivo, com separação aleatória
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


##############  Regressão Linear  ###################################################


# Criando um objeto regressor linear
regr = linear_model.LinearRegression()

# Treinando o modelo com os dados de treinamento
regr.fit(X_train, y_train)

# Fazendo predições com os dados de teste
y_pred = regr.predict(X_test)


# Avaliando o modelo pelo MSE (Mean squared error)
print()
print("MSE da Regressão Linear: %.2f"
      % mean_squared_error(y_test, y_pred))
# Avaliando o modelo pelo R2, variance score: 1 é a predição perfeita
print('Variance score: %.2f \n' % r2_score(y_test, y_pred))

################## Randon forests Regressor #############################################



# Criando um objeto regressor com Random Forest
randForest = RandomForestRegressor(100)

# Treinando o modelo com os dados de treinamento
randForest.fit(X_train, y_train)

# Fazendo predições com os dados de teste
y_pred = randForest.predict(X_test)

# Avaliando o modelo pelo MSE (Mean squared error)
print("MSE para a Árvore de Decisão: %.2f"
      % mean_squared_error(y_test, y_pred))
# Avaliando o modelo pelo R2, variance score: 1 é a predição perfeita
print('Variance score: %.2f \n' % r2_score(y_test, y_pred))

##################    MLP   REGRESSOR #########################



# Criando um objeto regressor MLP

nn = MLPRegressor(
    hidden_layer_sizes=(300,200,100),  activation='relu', solver='lbfgs', alpha=0.001, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
    random_state=0, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
    early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

#Criando um padronizador de escala
scaler = StandardScaler()
#Gerando um padronizador de escalas com os dados de treino
scaler.fit(X_train)

#Aplicando as transformações aos dados
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

print ("Processando treinamento da rede......\n")

# Treinando o modelo com os dados de treinamento
nn.fit(X_train, y_train)


# Fazendo predições com os dados de teste
y_pred = nn.predict(X_test)

# Avaliando o modelo pelo MSE (Mean squared error)
print("MSE para a Multi Layer Perceptron: %.2f"
      % mean_squared_error(y_test, y_pred))
# Avaliando o modelo pelo R2, variance score: 1 é a predição perfeita
print('Variance score: %.2f \n' % r2_score(y_test, y_pred))



'''
Dataset utilizado

Fonte:
    
https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/


Concrete Compressive Strength 

---------------------------------

Data Type: multivariate
 
Abstract: Concrete is the most important material in civil engineering. The 
concrete compressive strength is a highly nonlinear function of age and 
ingredients. These ingredients include cement, blast furnace slag, fly ash, 
water, superplasticizer, coarse aggregate, and fine aggregate.

---------------------------------

Sources: 

  Original Owner and Donor
  Prof. I-Cheng Yeh
  Department of Information Management 
  Chung-Hua University, 
  Hsin Chu, Taiwan 30067, R.O.C.
  e-mail:icyeh@chu.edu.tw
  TEL:886-3-5186511

  Date Donated: August 3, 2007
 
---------------------------------

Data Characteristics:
    
The actual concrete compressive strength (MPa) for a given mixture under a 
specific age (days) was determined from laboratory. Data is in raw form (not scaled). 

Summary Statistics: 

Number of instances (observations): 1030
Number of Attributes: 9
Attribute breakdown: 8 quantitative input variables, and 1 quantitative output variable
Missing Attribute Values: None

---------------------------------

Variable Information:

Given is the variable name, variable type, the measurement unit and a brief description. 
The concrete compressive strength is the regression problem. The order of this listing 
corresponds to the order of numerals along the rows of the database. 

Name -- Data Type -- Measurement -- Description

Cement (component 1) -- quantitative -- kg in a m3 mixture -- Input Variable
Blast Furnace Slag (component 2) -- quantitative -- kg in a m3 mixture -- Input Variable
Fly Ash (component 3) -- quantitative -- kg in a m3 mixture -- Input Variable
Water (component 4) -- quantitative -- kg in a m3 mixture -- Input Variable
Superplasticizer (component 5) -- quantitative -- kg in a m3 mixture -- Input Variable
Coarse Aggregate (component 6) -- quantitative -- kg in a m3 mixture -- Input Variable
Fine Aggregate (component 7) -- quantitative -- kg in a m3 mixture -- Input Variable
Age -- quantitative -- Day (1~365) -- Input Variable
Concrete compressive strength -- quantitative -- MPa -- Output Variable 
---------------------------------
'''





