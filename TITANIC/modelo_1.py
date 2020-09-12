#============================================================================
# Librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Machine Learning
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import (train_test_split,
                                     GridSearchCV)
from sklearn.ensemble import (RandomForestClassifier,
                              StackingClassifier)
from sklearn.metrics import (accuracy_score,
                             recall_score,
                             f1_score, precision_score,
                             plot_confusion_matrix,
                             roc_curve, auc)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore",
                        category=sklearn.exceptions.UndefinedMetricWarning)

# ============================================================================
# Lectura de datos
data = pd.read_csv("DATA/train.csv")
test = pd.read_csv("DATA/test.csv")
pid = test.PassengerId

# columnas a eliminar
name_data = data.columns[3]
ticket_data = data.columns[8]

name_test = test.columns[2]
ticket_test = test.columns[7]


# Se considera que el nombre no aporta ninguna decision sustancial dado que
# solamente sirve como identificador y esa funcion la cumple prefectamente
# la variabe (PassengerID), por tanto se elimina de la base de datos. Por otra
# parte tambien la variable (Ticket) tiene valores unicos que no aportan nada
# al modelo, se elimina también

data = data.drop([name_data, ticket_data], axis=1)
test = test.drop([name_test, ticket_test], axis=1)


# Se verifica los valores para los cuales se tienen NaN

# print(test.isna().sum())
Age_test = test.columns[3]
Fare_test = test.columns[6]
Cabin_test = test.columns[7]

# print(data.isna().sum())
Age_data = data.columns[4]
Cabin_data = data.columns[8]
Embarked_data = data.columns[9]


# Luego de esto se debe hacer ingenieria de parametros para poder rellenar
# estos espacion en blanco y asi poder entrenar con seguridad cualquier modelo
# se reemplaza valores vacios por el promedio de edad del conjunto de datos

data[Age_data].fillna(value=round(data.Age.mean()), inplace=True)
test[Age_test].fillna(value=round(test.Age.mean()), inplace=True)

# Se reemplaza los valores vacios por U, elección arbitraria

test.Cabin.fillna(value="U", inplace=True)
data.Cabin.fillna(value="U", inplace=True)

# por utlimo la variable embarked tiene 2 posiciones vacias las reemplazamos
# arbitrariamente con S, aunque puede ser a elección

data[Embarked_data].fillna(value="S", inplace=True)

# Para test se rellena con el promedio de datos en la variables Fare

test[Fare_test].fillna(test.Fare.mean(), inplace=True)

# Variables para el modelo, variable objetivo
Survival = data.columns[1]
X = data.drop([Survival], axis=1)
y = data[Survival]


# Existen en la base de datos variables categoricas que deben codificarse
# Para este caso se utiliza la codificación más simple, variables dummy
# se identifican las variables categoricas , ahora se manejan matrices de
# numpy para poder utilizar la clase OneHotEncoder que permite codificar
# datos nuevos en el modelo para prediciones
X = X.to_numpy()
Y = y.to_numpy()
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(X)
X = enc.transform(X).toarray()
# Pclass = data.columns[2]
# Sex = data.columns[3]
# SibSp = data.columns[5]
# Parch = data.columns[6]
# Cabin = data.columns[8]
# Embarked = data.columns[9]

# data = pd.get_dummies(data, columns=[Pclass, Sex, SibSp, Parch, Cabin,
#                       Embarked])

#

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=11)

# # Consrucción del modelo
# # Se consideran tres modelos de clasificacion
clfs = [BernoulliNB(alpha=0.8),
        LogisticRegression(solver="liblinear"),
        RandomForestClassifier(# bootstrap=True,
                               # ccp_alpha=0.0,
                               # criterion="gini",
                               # max_depth=4,
                               # max_features="auto",
                               # max_leaf_nodes=5,
                               # max_samples=None,
                               # min_impurity_decrease=0.0,
                               # min_impurity_split=None,
                               random_state=11)]

# # compilacion de los modelos
for c in clfs:
    c.fit(X_train, y_train)


# # Construccion de una funcion para medir metricas
def scores(clfs):
    columns = ["Name", "Accuracy", "Precision",
               "Recall", "F1-Score"]
    metricas = pd.DataFrame([], columns=columns)
    for cls in clfs:
        stats = {}
        pred_test = cls.predict(X_test)
        pred_train = cls.predict(X_train)
        stats.update({'training Accuracy': accuracy_score(y_train, pred_train),
                      'test Accuracy': accuracy_score(y_test, pred_test),
                      'Name': type(cls).__name__,
                      'Recall': recall_score(y_test, pred_test),
                      'F1-Score': f1_score(y_test, pred_test),
                      'Precision': precision_score(y_test, pred_test)})
        metricas = metricas.append(stats, ignore_index=True)
    return metricas


# # Función para el plot de las matrices de confusión
def confusion_matrix(clfs):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(11, 3.5))

    for cls, ax in zip(clfs, axes.flatten()):
        plot_confusion_matrix(cls,
                              X_test,
                              y_test,
                              ax=ax,
                              cmap='Reds',
                              display_labels=y_test)
    ax.title.set_text(type(cls).__name__)
    plt.tight_layout()
    plt.savefig("Matriz_de _confusion")

# Activar para guradar la matriz de confusionde los modelos por aparte
# confusion_matrix(clfs)

# Medicion de las metrica para los tres modelos
print(scores(clfs).iloc[0])
print("")
print(scores(clfs).iloc[1])
print("")
print(scores(clfs).iloc[2])
print("")
# Construccion del meta modelo

# Encontrar los hiperparametros optimos de cada modelo con validacion
# cruzada
r = 11
scoring = {"Accuracy": "accuracy",
           "f1_Score": "f1",
           "Precision": "precision",
           "Recall": "recall"}
# Seleccion hiperparametros arbol de decision
# ============================================================================
parametros_decision_tree = {"max_depth": np.arange(2, 20)}
model_desicion_tree = DecisionTreeClassifier(class_weight='balanced',
                                             random_state=r
                                             )
model_desicion_tree = GridSearchCV(model_desicion_tree,
                                   parametros_decision_tree,
                                   cv=10,
                                   scoring=scoring,
                                   refit="Accuracy"
                                   )
model_desicion_tree.fit(X_train, y_train)
print("Nombre modelo: ", type(model_desicion_tree).__name__)
print("Mejores parametros: {}".format(model_desicion_tree.best_params_))
print("Mejor puntaje: {}".format(model_desicion_tree.best_score_))
print(" ")
# ============================================================================

# Seleccion hiperparametros Random Forest
# ============================================================================
parametros_Random_forest = {"n_estimators": np.arange(1, 20),
                            "max_depth": np.arange(1, 11, 2),
                            "max_samples": np.arange(1, 11, 2)}
model_random_forest = RandomForestClassifier(bootstrap=True,
                                             ccp_alpha=0.0,
                                             criterion="entropy",
                                             random_state=r
                                             )
model_random_forest = GridSearchCV(model_random_forest,
                                   parametros_Random_forest,
                                   cv=10,
                                   scoring=scoring,
                                   refit="Accuracy")
model_random_forest.fit(X_train, y_train)
print("Nombre modelo: ", type(model_random_forest).__name__)
print("Mejores parametros: {}".format(model_random_forest.best_params_))
print("Mejor puntaje: {}".format(model_random_forest.best_score_))
print(" ")
# ============================================================================










clfs = [("Bernoulli", BernoulliNB(alpha=0.8)),
        ("Arbol Decision", DecisionTreeClassifier(max_depth=11,
                                                  class_weight="balanced",
                                                  random_state=r)),
        ("LR", LogisticRegression(solver="liblinear",
                                  class_weight="balanced",
                                  random_state=r)),
        ("Forest", RandomForestClassifier(bootstrap=True,
                                          ccp_alpha=0.0,
                                          criterion="entropy",
                                          max_depth=9,
                                          n_estimators=20,
                                          max_samples=10,
                                          random_state=r))]

clf = StackingClassifier(estimators=clfs,
                         final_estimator=LogisticRegression())


clf.fit(X_train, y_train)
pred_test = clf.predict(X_test)
pred_train = clf.predict(X_train)
print("Name: ", type(clf).__name__)
print("training Accuracy: ", accuracy_score(y_train, pred_train))
print("test Accuracy: ", accuracy_score(y_test, pred_test))

print('Recall: ', recall_score(y_test, pred_test))
print('F1-Score: ', f1_score(y_test, pred_test))
print('Precision: ', precision_score(y_test, pred_test))

# Grafica de la matriz de confusion del metamodelo
plot_confusion_matrix(clf, X_test,
                      y_test, cmap="Blues",
                      display_labels=y_test)
plt.savefig("Metamodelo")
plt.show()

# grafica de la curva ROC del metamodelo
frp, tpr, _ = roc_curve(y_test, pred_test)
roc_auc = auc(frp, tpr)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(frp, tpr, "b", label="curva ROC (area = %0.3f)" % roc_auc)
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de verdadero Positivo')
plt.grid(color='k', alpha=0.5, linestyle='dashed', linewidth=0.9)
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.savefig("curva_ROC")


# Resultado para someter a la competencia
test_encode = enc.transform(test).toarray()
output = pd.DataFrame({"PassengerID": pid,
                       "Survived": clf.predict(test_encode)})
output.to_csv("submission.csv", index=False)
print("Resultados generados")
