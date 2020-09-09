# ============================================================================
# Librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Machine Learning
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# ============================================================================
# Lectura de datos 
data = pd.read_csv("DATA/train.csv")

# Se verifica los valores para los cuales se tienen NaN
# print(data.isna().sum())
# Luego de esto se debe hacer ingenieria de parametros para poder rellenar
# estos espacion en blanco y asi poder entrenar con seguridad cualquier modelo
# se reemplaza valores vacios por el promedio de edad del conjunto de datos
data[data.columns[5]].fillna(value=round(data.Age.mean()),inplace=True)
# Se reemplaza los valores vacios por U, elección arbitraria
data.Cabin.fillna(value="U",inplace=True)
data[data.columns[10]] = data[data.columns[10]].apply(lambda x: x[0])
# por utlimo la variable embarked tiene 2 posiciones vacias las reemplazamos
# arbitrariamente con S, aunque puede ser a elección
data[data.columns[10]].fillna(value="S",inplace=True)

# Se considera que el nombre no aporta ninguna decision sustancial dado que 
# solamente sirve como identificador y esa funcion la cumple prefectamente
# la variabe (PassengerID), por tanto se elimina de la base de datos. Por otra
# parte tambien la variable (Ticket) tiene valores unicos que no aportan nada
# al modelo, se elimina también 
data = data.drop([data.columns[3],data.columns[8]], axis=1)

# Existen en la base de datos variables categoricas que deben codificarse
# Para este caso se utiliza la codificación más simple, variables dummy
# se identifican las variables categoricas
Pclass = data.columns[2]
Sex = data.columns[3]
SibSp = data.columns[5]
Parch = data.columns[6]
Cabin = data.columns[8]
Embarked = data.columns[9]

data = pd.get_dummies(data, columns=[Pclass, Sex, SibSp, Parch,Cabin, 
                      Embarked])

# Variables para el modelo, variable objetivo
Survival = data.columns[1]
X = data.drop([Survival], axis=1)
y = data[Survival]




X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)

# Consrucción del modelo
Modelo_1 = BernoulliNB()
Modelo_1.fit(X_train,y_train)
y_pred = Modelo_1.predict(X_test)
acc = accuracy_score(y_test,y_pred)
print(acc*100)

