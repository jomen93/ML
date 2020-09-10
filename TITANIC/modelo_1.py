# ============================================================================
# Librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Machine Learning
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import (train_test_split,
                                     KFold)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier,
                              VotingClassifier,
                              StackingClassifier)
from sklearn.metrics import (accuracy_score, auc, roc_auc_score,
                             recall_score, log_loss, roc_curve,
                             f1_score, precision_score,
                             plot_confusion_matrix)
from sklearn.linear_model import LogisticRegression

# ============================================================================
# Lectura de datos
data = pd.read_csv("DATA/train.csv")

# Se verifica los valores para los cuales se tienen NaN
# print(data.isna().sum())
# Luego de esto se debe hacer ingenieria de parametros para poder rellenar
# estos espacion en blanco y asi poder entrenar con seguridad cualquier modelo
# se reemplaza valores vacios por el promedio de edad del conjunto de datos
data[data.columns[5]].fillna(value=round(data.Age.mean()), inplace=True)
# Se reemplaza los valores vacios por U, elección arbitraria
data.Cabin.fillna(value="U", inplace=True)
data[data.columns[10]] = data[data.columns[10]].apply(lambda x: x[0])
# por utlimo la variable embarked tiene 2 posiciones vacias las reemplazamos
# arbitrariamente con S, aunque puede ser a elección
data[data.columns[10]].fillna(value="C", inplace=True)

# Se considera que el nombre no aporta ninguna decision sustancial dado que
# solamente sirve como identificador y esa funcion la cumple prefectamente
# la variabe (PassengerID), por tanto se elimina de la base de datos. Por otra
# parte tambien la variable (Ticket) tiene valores unicos que no aportan nada
# al modelo, se elimina también
data = data.drop([data.columns[3], data.columns[8]], axis=1)

# Existen en la base de datos variables categoricas que deben codificarse
# Para este caso se utiliza la codificación más simple, variables dummy
# se identifican las variables categoricas
Pclass = data.columns[2]
Sex = data.columns[3]
SibSp = data.columns[5]
Parch = data.columns[6]
Cabin = data.columns[8]
Embarked = data.columns[9]

data = pd.get_dummies(data, columns=[Pclass, Sex, SibSp, Parch, Cabin,
                      Embarked])

# Variables para el modelo, variable objetivo
Survival = data.columns[1]
X = data.drop([Survival], axis=1)
y = data[Survival]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Consrucción del modelo
# Se consideran tres modelos de clasificacion
clfs = [BernoulliNB(alpha=0.8),
        LogisticRegression(solver="liblinear"),
        RandomForestClassifier(random_state=11)]

# compilacion de los modelos
for c in clfs:
    c.fit(X_train, y_train)


# Construccion de una funcion para medir metricas
def scores(clfs):
    columns = ["Name", "Accuracy", "Precision",
               "Recall", "F1-Score"]
    metricas = pd.DataFrame([], columns=columns)
    for cls in clfs:
        stats = {}
        pred = cls.predict(X_test)
        stats.update({'Accuracy': accuracy_score(y_test, pred),
                      'Name': type(cls).__name__,
                      'Recall': recall_score(y_test, pred),
                      'F1-Score': f1_score(y_test, pred),
                      'Precision': precision_score(y_test, pred)})
        metricas = metricas.append(stats, ignore_index=True)
    return metricas


# Función para el plot de las matrices de confusión
def confusion_matrix(clfs):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 3))

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
    plt.show()

confusion_matrix(clfs)

# Medicion de las metrica para los tres modelos
print(scores(clfs).iloc[0])
print("")
print(scores(clfs).iloc[1])
print("")
print(scores(clfs).iloc[2])
print("")
# Construccion del meta modelo
clfs = [("Bernoulli", BernoulliNB(alpha=0.8)),
        ("LR", LogisticRegression(solver="liblinear")),
        ("Forest", RandomForestClassifier(random_state=11))]

clf = StackingClassifier(estimators=clfs,
      final_estimator=RandomForestClassifier(random_state=11))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf.fit(X_train, y_train)
pred = clf.predict(X_test)
print("Name: ", type(clf).__name__)
print("Accuracy: ", accuracy_score(y_test, pred))
print('Recall: ', recall_score(y_test, pred))
print('F1-Score: ', f1_score(y_test, pred))
print('Precision: ', precision_score(y_test, pred))

plot_confusion_matrix(clf, X_test,
                      y_test, cmap="Blues",
                      display_labels=y_test)
plt.savefig("Metamodelo")
plt.show()



print("Hola Jose David")
