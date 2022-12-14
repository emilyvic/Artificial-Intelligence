# -*- coding: utf-8 -*-
"""Sin Framework - Emilia_Jácome-A00828347.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1mR6XGK0iJwqSNgVdJQ8ejnUG2dT3PMeP

Emilia Victoria Jácome Iñiguez

A00828347

**MOMENTO DE RETROALIMENTACIÓN: IMPLEMENTACIÓN DE UNA TÉCNICA DE APRENDIZAJE MÁQUINA SIN EL USO DE UN FRAMEWORK**

#***Preliminar***

##***Conexión a Drive***
"""

from google.colab import drive
drive.mount('/content/drive')

cd /content/drive/MyDrive/Inteligencia Artificial Avanzada/Datasets

"""##***Cargar los datos a data frames***"""

import pandas as pd
columns = ["fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density", "pH", "sulphates", "alcohol", "quality"]
df = pd.read_csv('winequality-white.csv') #generar el dataframe con los datos de wine

df.columns = columns

df

df.describe()

"""###***Convertir el tipo de dato a float***"""

df.astype(float).dtypes

df.info()

"""#***Preparación***

##*Separar el dataset en entrenamiento y validación*

Las variables independientes (x) serán: el nivel de acidez volátil en el vino, el nivel de pH, el nivel de sulphatos y el nivel de cloruros, mientras que la variable dependiente (y) será el nivel de alcohol en el vino.

Por lo que se estará intentando determinar el nivel de alcohol en función del las variables independientes declaradas enteriormente en una muestra de 4898 vinos.
"""

x_ = df.volatile_acidity
x2 = df.pH # Columnas volatile_acidity, pH
x3 = df.sulphates
x4 = df.chlorides
#x_ = df.drop(["quality"], axis=1)
y_ = df.alcohol

#Datos de entrenamiento
x_train = x_.iloc[:3918]
x2_train = x2.iloc[:3918]
x3_train = x3.iloc[:3918]
x4_train = x4.iloc[:3918]

y_train = y_.iloc[:3918]

#Datos de validación
x_validate = x_.iloc[3918:]
x2_validate = x2.iloc[3918:]
x3_validate = x3.iloc[3918:]
x4_validate = x4.iloc[3918:]

y_validate = y_.iloc[3918:]

df_test = pd.DataFrame(y_validate.iloc[:50,])
df_test.reset_index(inplace=True)

df_x_test = pd.DataFrame(x_validate.iloc[:50])
df_x_test.reset_index(inplace=True)

df_x2_test = pd.DataFrame(x2_validate.iloc[:50])
df_x2_test.reset_index(inplace=True)

df_x3_test = pd.DataFrame(x3_validate.iloc[:50])
df_x3_test.reset_index(inplace=True)

df_x4_test = pd.DataFrame(x4_validate.iloc[:50])
df_x4_test.reset_index(inplace=True)

"""##*Definición de parámetros e hiperparámetros*"""

alpha = 0.00001 
n_train = len(y_train)
n_validate = len(y_validate)

"""# **Implementación de Modelos de Regresión Lineal**

##***Orden 1***

Entrenamiento del modelo de **regresión lineal** de Orden 1 (x) con variables:

*Variable Independiente (x):*
* x1 = Nivel de Acidez

*Variable Dependiente (y):*
* Nivel de Alcohol

###Entrenamiento
"""

import math
h   = lambda x,theta: theta[0]+theta[1]*x #

theta = [9,11] # Cambiar dependiendo del orden del modelo (un theta para cada dimensión de nuestros datos + 1)

for idx in range(10000):
  acumDelta = []
  acumDeltaX = []
  for x_i, y_i in zip(x_train,y_train):
    acumDelta.append(h(x_i,theta)-y_i)
    acumDeltaX.append((h(x_i,theta)-y_i)*x_i)

  #Sumatorias de las J's
  sJt0 = sum(acumDelta)  
  sJt1 = sum(acumDeltaX)

  #Cálculo de thetas
  theta[0] = theta[0]-alpha/n_train*sJt0
  theta[1] = theta[1]-alpha/n_train*sJt1

print(theta)

"""###*Imprimir Predicciones*
Se corren 50 predicciones para validar la salida del modelo
"""

y_preds = []
for i in range (3918,3968):
  y_pred = theta[0]+theta[1]*x_validate[i]
  y_preds.append(y_pred)
df_pred1 =pd.DataFrame()

df_pred1["volatile_acidity"] = df_x_test["volatile_acidity"]
df_pred1["pred_alcohol"] = pd.DataFrame(y_preds)

df_pred1["real_alcohol"] = df_test["alcohol"]
df_pred1["dif"] = abs(df_pred1["pred_alcohol"] - df_pred1["real_alcohol"])
df_pred1

"""###*Validación*

####MSE

Cálculo de error con la función de costo basada en el Mean Square Error
"""

# Validación Error Cuadrático Medio (MSE)
j_i = lambda x,y,theta: (y-h(x,theta))**2 #MSE
acumDelta = []
for x_i, y_i in zip(x_validate,y_validate):
  acumDelta.append(j_i(x_i,y_i,theta))  

sDelta = sum(acumDelta)  
J_validate = 1/(2*n_validate)*sDelta


# Training
acumDelta = []
for x_i, y_i in zip(x_train,y_train):
  acumDelta.append(j_i(x_i,y_i,theta))  

sDelta = sum(acumDelta)  
J_train = 1/(2*n_train)*sDelta



print("Error J Validación con MSE: ", J_validate)
print("Error J Entrenamiento con MSE: ", J_train)
print("Valores de theta con MSE: ", theta)

"""A través de esta validación, se puede ver que los errores son mínimos con los siguientes valores iniciales de theta:
* theta 0: 9
* thata 1: 11

Ambos errores están por debajo de 3 puntos, por lo que el margen es muy pequeño y esto demuestra un alto nivel de precisión en el modelo.

####RMSE

Cálculo de error con la función de costo basada en el Root Mean Square Error
"""

# Validación con RMSE
j_i2 = lambda x,y,theta: math.sqrt((y-h(x,theta))**2) #RMSE

acumDelta = []
for x_i, y_i in zip(x_validate,y_validate):
  acumDelta.append(j_i2(x_i,y_i,theta))  

sDelta = sum(acumDelta)  
J_validate = 1/(2*n_validate)*sDelta


# Training
acumDelta = []
for x_i, y_i in zip(x_train,y_train):
  acumDelta.append(j_i2(x_i,y_i,theta))  

sDelta = sum(acumDelta)  
J_train = 1/(2*n_train)*sDelta


print("Error J Validación con RMSE: ", J_validate)
print("Error J Entrenamiento con RMSE: ", J_train)
print("Valores de theta con RMSE: ", theta)

"""Para el caso del RMSE, se puede ver que los errores son aún más pequeños en comparación al MSE ya que se encuentran por debajo de 0.9 puntos lo que indica una alta calidad en la validación y el entrenamiento del modelo.

####MAE

Cálculo de error con la función de costo basada en el Mean Absolute Error
"""

# Validación con MAE
j_i4 = lambda x,y,theta: abs((y-h(x,theta))) #MAE

acumDelta = []
for x_i, y_i in zip(x_validate,y_validate):
  acumDelta.append(j_i4(x_i,y_i,theta))  

sDelta = sum(acumDelta)  
J_validate = 1/(2*n_validate)*sDelta


# Training
acumDelta = []
for x_i, y_i in zip(x_train,y_train):
  acumDelta.append(j_i4(x_i,y_i,theta))  

sDelta = sum(acumDelta)  
J_train = 1/(2*n_train)*sDelta


print("Error J Validación con MAE: ", J_validate)
print("Error J Entrenamiento con MAE: ", J_train)
print("Valores de theta con MAE: ", theta)

"""Finalmente, para el caso del MAE se puede ver que los errores de validación y de entrenamiento continúan por debajo de 0.9 puntos, lo cuál indica una alta calidad en estos procesos.

##***Orden 2***

Entrenamiento del modelo de **regresión lineal** de Orden 2 (x2) con variables:

*Variable Independiente (x):*
* x1 = Nivel de Acidez
* x2 = 
Nivel de pH

*Variable Dependiente (y):*
* Nivel de Alcohol

###Entrenamiento
"""

theta = [1,0,3] # Agregar un elemento 

h2 = lambda x,theta,x2: theta[0]+theta[1]*x+theta[2]*x2
alpha = 0.0000001

for idx in range(10000):
  acumDelta = []
  acumDeltaX = []
  acumDeltaX2 = []
  for x_i, y_i, x2_i in zip(x_train,y_train,x2_train): # Agregar la nueva dimensión
    acumDelta.append(h2(x_i,theta,x2_i)-y_i)
    acumDeltaX.append((h2(x_i,theta,x2_i)-y_i)*x_i)    
    acumDeltaX2.append((h2(x_i,theta,x2_i)-y_i)*x2_i) # Acumular para el nuevo theta    

  sJt0 = sum(acumDelta)  
  sJt1 = sum(acumDeltaX)
  sJt2 = sum(acumDeltaX2)
  theta[0] = theta[0]-alpha/n_train*sJt0
  theta[1] = theta[1]-alpha/n_train*sJt1
  theta[2] = theta[2]-alpha/n_train*sJt2 # ACtualizar el nuevo theta

print(theta)

"""###*Imprimir Predicciones*
Se corren 50 predicciones para validar la salida del modelo
"""

y_preds = []

for i in range (3918,3968):
  y2_pred = theta[0]+theta[1]*x_validate[i]+theta[2]*x2_validate[i]
  y_preds.append(y2_pred)

df_pred2 = pd.DataFrame()

df_pred2["volatile_acidity"] = df_x_test["volatile_acidity"]
df_pred2["pH"] = df_x2_test["pH"]

df_pred2["pred_alcohol"] = pd.DataFrame(y_preds)

df_pred2["real_alcohol"] = df_test["alcohol"]
df_pred2["dif"] = abs(df_pred2["pred_alcohol"] - df_pred2["real_alcohol"])

df_pred2

"""###*Validación*

####MSE

Cálculo de error con la función de costo basada en el Mean Square Error
"""

j2_i = lambda x,y,theta,x2: (h2(x,theta,x2)-y)**2

# Validación
acumDelta = []
for x_i, y_i,x2_i in zip(x_validate,y_validate,x2_validate): #Agregar la nueva dimensión
  acumDelta.append(j2_i(x_i,y_i,theta,x2_i))  
  #Acumular para el nuevo theta
sDelta = sum(acumDelta)  
J_validate = 1/(2*n_validate)*sDelta


# Training
acumDelta = []
for x_i, y_i,x2_i in zip(x_train,y_train,x2_train):
  acumDelta.append(j2_i(x_i,y_i,theta,x2_i))  

sDelta = sum(acumDelta)  
J_train = 1/(2*n_train)*sDelta

print("Error J Validación con MSE: ", J_validate)
print("Error J Entrenamiento con MSE: ", J_train)
print("Valores de theta con MSE: ", theta)

"""A través de esta validación, se puede ver que los errores son mínimos con los siguientes valores iniciales de theta:
* theta 0: 1
* theta 1: 0
* theta 2: 3

Ambos errores están por debajo de 0.9 puntos, por lo que el margen es extremadamente pequeño y esto demuestra un alto nivel de precisión en el modelo.

####RMSE

Cálculo de error con la función de costo basada en el Root Mean Square Error
"""

j2_i2 = lambda x,y,theta,x2: math.sqrt((h2(x,theta,x2)-y)**2)

# Validación
acumDelta = []
for x_i, y_i,x2_i in zip(x_validate,y_validate,x2_validate): #Agregar la nueva dimensión
  acumDelta.append(j2_i2(x_i,y_i,theta,x2_i))  
  #Acumular para el nuevo theta
sDelta = sum(acumDelta)  
J_validate = 1/(2*n_validate)*sDelta


# Training
acumDelta = []
for x_i, y_i,x2_i in zip(x_train,y_train,x2_train):
  acumDelta.append(j2_i2(x_i,y_i,theta,x2_i))  

sDelta = sum(acumDelta)  
J_train = 1/(2*n_train)*sDelta



print("Error J Validación con RMSE: ", J_validate)
print("Error J Entrenamiento con RMSE: ", J_train)
print("Valores de theta con RMSE: ", theta)

"""Ambos errores están por debajo de 0.6 puntos, por lo que el margen es extremadamente pequeño y esto demuestra un alto nivel de calidad en la validación y el entrenamiento.

####MAE

Cálculo de error con la función de costo basada en el Mean Absolute Error
"""

j2_i4 = lambda x,y,theta,x2: abs((h2(x,theta,x2)-y)) ##MAE

# Validación
acumDelta = []
for x_i, y_i,x2_i in zip(x_validate,y_validate,x2_validate): #Agregar la nueva dimensión
  acumDelta.append(j2_i4(x_i,y_i,theta,x2_i))  
  #Acumular para el nuevo theta
sDelta = sum(acumDelta)  
J_validate = 1/(2*n_validate)*sDelta


# Training
acumDelta = []
for x_i, y_i,x2_i in zip(x_train,y_train,x2_train):
  acumDelta.append(j2_i4(x_i,y_i,theta,x2_i))  

sDelta = sum(acumDelta)  
J_train = 1/(2*n_train)*sDelta



print("Error J Validación con MAE: ", J_validate)
print("Error J Entrenamiento con MAE: ", J_train)
print("Valores de theta con MAE: ", theta)

"""Ambos errores están por debajo de 0.9 puntos, por lo que el margen es extremadamente pequeño y esto demuestra un alto nivel de calidad en la validación y el entrenamiento.

##***Orden 4***
Entrenamiento del modelo de **regresión lineal** de Orden 4 (x) con variables:

*Variable Independiente (x):*
* x1= Nivel de Acidez
* x2= Nivel de pH
* x3= Nivel de Sulfatos
* x4= Nivel de cloruros

*Variable Dependiente (y):*
* Nivel de Alcohol

###Entrenamiento
"""

theta = [1,1,1,4,65] # Agregar un elemento 

h4 = lambda x,theta,x2,x3,x4: theta[0]+theta[1]*x+theta[2]*x2+theta[3]*x3+theta[4]*x4
alpha = 0.00001

for idx in range(1000):
  acumDelta = []
  acumDeltaX = []
  acumDeltaX2 = []
  acumDeltaX3 = []
  acumDeltaX4 = []
  for x_i, y_i, x2_i, x3_i, x4_i in zip(x_train,y_train,x2_train, x3_train, x4_train): # Agregar la nueva dimensión
    acumDelta.append(h4(x_i,theta,x2_i,x3_i,x4_i)-y_i)
    acumDeltaX.append((h4(x_i,theta,x2_i,x3_i,x4_i)-y_i)*x_i)    
    acumDeltaX2.append((h4(x_i,theta,x2_i,x3_i,x4_i)-y_i)*x2_i) # Acumular para el nuevo theta   
    acumDeltaX3.append((h4(x_i,theta,x2_i,x3_i,x4_i)-y_i)*x3_i) # Acumular para el nuevo theta   
    acumDeltaX4.append((h4(x_i,theta,x2_i,x3_i,x4_i)-y_i)*x4_i) # Acumular para el nuevo theta    

  sJt0 = sum(acumDelta)  
  sJt1 = sum(acumDeltaX)
  sJt2 = sum(acumDeltaX2)
  sJt3 = sum(acumDeltaX3)
  sJt4 = sum(acumDeltaX4)

  theta[0] = theta[0]-alpha/n_train*sJt0
  theta[1] = theta[1]-alpha/n_train*sJt1
  theta[2] = theta[2]-alpha/n_train*sJt2 # ACtualizar el nuevo theta
  theta[3] = theta[3]-alpha/n_train*sJt3 # ACtualizar el nuevo theta
  theta[4] = theta[4]-alpha/n_train*sJt4 # ACtualizar el nuevo theta

print(theta)

"""###*Imprimir Predicciones*
Se corren 50 predicciones para validar la salida del modelo
"""

y_preds = []
for i in range (3918,3968):
  y4_pred = theta[0]+theta[1]*x_validate[i]+theta[2]*x2_validate[i]+theta[3]*x3_validate[i]+theta[4]*x4_validate[i]
  y_preds.append(y4_pred)
  df_pred4 = pd.DataFrame()

  df_pred4["volatile_acidity"] = df_x_test["volatile_acidity"]
  df_pred4["pH"] = df_x2_test["pH"]
  df_pred4["sulphates"] = df_x3_test["sulphates"]
  df_pred4["chlorides"] = df_x4_test["chlorides"]

  df_pred4["pred_alcohol"] = pd.DataFrame(y_preds)

  df_pred4["real_alcohol"] = df_test["alcohol"]
  df_pred4["dif"] = abs(df_pred4["pred_alcohol"] - df_pred4["real_alcohol"])

df_pred4

"""###*Validación*

####MSE

Cálculo de error con la función de costo basada en el Mean Square Error
"""

j4_i = lambda x,y,theta,x2,x3,x4: (h4(x,theta,x2,x3,x4)-y)**2

# Validación
acumDelta = []
for x_i, y_i,x2_i, x3_i, x4_i in zip(x_validate,y_validate,x2_validate,x3_validate,x4_validate): #Agregar la nueva dimensión
  acumDelta.append(j4_i(x_i,y_i,theta,x2_i,x3_i,x4_i))  
  #Acumular para el nuevo theta
sDelta = sum(acumDelta)  
J_validate = 1/(2*n_validate)*sDelta


# Training
acumDelta = []
for x_i, y_i,x2_i, x3_i, x4_i in zip(x_train,y_train,x2_train,x3_train,x4_train):
  acumDelta.append(j4_i(x_i,y_i,theta,x2_i,x3_i,x4_i))  

sDelta = sum(acumDelta)  
J_train = 1/(2*n_train)*sDelta



print("Error J Validación con MSE: ", J_validate)
print("Error J Entrenamiento con MSE: ", J_train)
print("Valores de theta con MSE: ", theta)

"""A través de esta validación, se puede ver que los errores son mínimos con los siguientes valores iniciales de theta:
* theta 0: 1
* theta 1: 1
* theta 2: 1
* theta 3: 4
* theta 4: 65

Ambos errores están por debajo de 4 puntos, por lo que el margen es medio y esto demuestra un nivel de precisión medio en el modelo.

####RMSE

Cálculo de error con la función de costo basada en el Root Mean Square Error
"""

j4_i2 = lambda x,y,theta,x2,x3,x4: math.sqrt((h4(x,theta,x2,x3,x4)-y)**2)

# Validación
acumDelta = []
for x_i, y_i,x2_i, x3_i, x4_i in zip(x_validate,y_validate,x2_validate,x3_validate,x4_validate): #Agregar la nueva dimensión
  acumDelta.append(j4_i2(x_i,y_i,theta,x2_i,x3_i,x4_i))  
  #Acumular para el nuevo theta
sDelta = sum(acumDelta)  
J_validate = 1/(2*n_validate)*sDelta


# Training
acumDelta = []
for x_i, y_i,x2_i, x3_i, x4_i in zip(x_train,y_train,x2_train,x3_train,x4_train):
  acumDelta.append(j4_i2(x_i,y_i,theta,x2_i,x3_i,x4_i))  

sDelta = sum(acumDelta)  
J_train = 1/(2*n_train)*sDelta



print("Error J Validación con RMSE: ", J_validate)
print("Error J Entrenamiento con RMSE: ", J_train)
print("Valores de theta con RMSE: ", theta)

"""Ambos errores están por debajo de 1 punto, por lo que el margen es medio y esto demuestra un nivel de precisión alto en la calidad del entrenamiento y validación del modelo.

####MAE

Cálculo de error con la función de costo basada en el Mean Absolute Error
"""

j4_i3 = lambda x,y,theta,x2,x3,x4: abs((h4(x,theta,x2,x3,x4)-y)) #MAE

# Validación
acumDelta = []
for x_i, y_i,x2_i, x3_i, x4_i in zip(x_validate,y_validate,x2_validate,x3_validate,x4_validate): #Agregar la nueva dimensión
  acumDelta.append(j4_i3(x_i,y_i,theta,x2_i,x3_i,x4_i))  
  #Acumular para el nuevo theta
sDelta = sum(acumDelta)  
J_validate = 1/(2*n_validate)*sDelta


# Training
acumDelta = []
for x_i, y_i,x2_i, x3_i, x4_i in zip(x_train,y_train,x2_train,x3_train,x4_train):
  acumDelta.append(j4_i3(x_i,y_i,theta,x2_i,x3_i,x4_i))  

sDelta = sum(acumDelta)  
J_train = 1/(2*n_train)*sDelta



print("Error J Validación con MAE: ", J_validate)
print("Error J Entrenamiento con MAE: ", J_train)
print("Valores de theta con MAE: ", theta)

"""Ambos errores están por debajo de 4 puntos, por lo que el margen es medio y esto demuestra un nivel de precisión alto en la calidad del entrenamiento y validación del modelo.

#**Conclusión:**

El mejor modelo que se utilizó para predecir el nivel de alcohol a partir de una muestra de vinos fue el modelo de regresión lineal de grado 2 ya que es el modelo que presentó el mayor nivel de precisión en sus predicciones a partir de las métricas de desempeño calculadas. 

En lo que respecta al MSE, esta métrica muestra el promedio de los cuadrados de los errores por lo que, aunque esta métrica permite medir el nivel de cambio lineal que ocurre en los datos, el hecho de elevar las diferencias al cuadrado tiende a magnificar/inflar los errores y por ende, datos muy alejados logran impactar mayormente al resultado general. Por tal motivo, se utiliza el RMSE el cuál quita este efecto de inflación al obtener la raíz cuadrada del MSE. Tal fue el caso del modelo de Orden 2 ya que si bien los MSE's de entrenamiento y validación estuvieron por debajo de 0.9, al obtener la métrica de los RMSE's estos valores fueron aún más bajos con una diferencia de 0.2 debido al efecto de la inflación del cuadrado de los errores, el cuál es un margen extremadamente bueno de error. Finalmente, se corroboraron estos valores con la métrica del MAE en la cuál las puntuaciones aumentan linealmente con el error promedio de los valores del error absoluto. Por ende, los valores del MAE coincidieron con los del RMSE lo que quiere decir que independientemente del signo de los errores, estos se encuentrán por debajo de 0.6, lo que indica un alto nivel de precisión en las predicciones. 

Dentro del contexto del problema, este error es relevante ya que lo que se está intentando predecir es el nivel de alcohol que tiene un vino dadas su nivel de acidez y su pH, y dado que el nivel de alcohol oscila entre un valor de 8 a 14.2 con una desviación estándar de 1.23, es conveniente que nuestro modelo tenga un error mucho menor al valor de desviación estándar para tener una precisión alta y que los valores de alcohol que se obtengan en futuras observaciones sean lo más cercanos a los reales.
"""
