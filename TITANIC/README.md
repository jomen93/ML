# Titanic

<div aling="center">
El hundimiento del Titanic fue uno de los naufragio más grandes de la historia. El 15 de abril de 1912, durante su viaje inaugural, considerado coo insumergible, se hundió tras chocar con un iceberg. Desafortunadamente, no habia suficientes botes salvavidas para todos a bordo, lo que resultó en la muerte de 1502 de los 224 pasajeros y la triupalción. 

Si bien hubo algún elemento de suerte involucrado en sobrevivir, parece que algunos grupos de personas tenían más probabilidades de sobrevivir que otros. Se quiere en este repositorio crear un modelo predictivo que responda la pregunta:
</div>

### ¿Qué tipo de personas tenían más probabilidad de sobrevivir?

<div aling="center">
las bases de datos se tienen en dos grupos, uno de entrenamiento **train.csv**y otro de validación **test.csv**. Este conjunto de datos se utiliza para construir diferentes modelos de aprendizaje automático.
</div>

## Diccionario de datos 

| Variable | Definición |Llave
| ------------- | ------------- |-------------- |
| survival      | Sobreviviente              | 0 = No, 1 = si
| pclass        | Clase de tiquete           | 1 = 1ra, 2 = 2da, 3 = 3ra |
| sex           | Genero                     |  |
| Age           | Edad en años               |  |
| sibsp         | # de hermanas/cónyugues    |  |
| parch         | # padres/hijos             |  |
| ticket        | Número de tiquete          |  |
| fare          | Tarifa de Pasajero         |  |
| cabin         | Numero de Cabina           |a |
| embarked      | Puerto de embarcación      | C = Cherburgo Q = Queenstown, S = Southampton |



## Notas de las variables

**pclass:** una referencia  del nivel socioeconomico 
1ra ------> Alto
2da ------> Medio
3ra ------> Bajo

**Edad:** la edad es fracciónaria si es menor que 1. LA edad estimada se representa por xx.5

**sibsp:** El conjunto de datos define las relaciones familiares de la siguiente manera 

Hermano = hermano, hermana, hermanastro hermanastra
Cónyugue = Esposo, Esposa (Se ignoran los amates, novios , etc)

**parch:** El conjunto de datos define las relaciones familiares de la siguiente manera 

Padre = Madre, Padre
Niño = hija, hijo, hijastra, hijastro
Algunos niños viajaban solo con la niñera, por lo tanto la variable en ese caso toma el valor de cero (parch = 0)

La base de datos está desbalanceada respecto del genero
| Hombre | Mujer |
| 65%    |  35%  |



