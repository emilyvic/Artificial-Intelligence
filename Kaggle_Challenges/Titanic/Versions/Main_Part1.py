#Importar librerías
import pandas as pd 
import numpy as np


#Definir los dataframes
df_train = pd.read_csv('train.csv') #generar el dataframe con los datos del titanic
df_test = pd.read_csv('test.csv') #generar el dataframe con los datos del titanic
df_gs = pd.read_csv('gender_submission.csv') #generar el dataframe con los datos del titanic

#Unir los dataframes de train y test
df = df_train.append(df_test, ignore_index= True) #ver función de concat (pandas)

#Exploración de Datos

##Mostrar las columnas del dataset
df.columns

##Mostrar dimensiones del dataset
df.shape

##Resumen del dataframe
df.info()

##Ver el tipo de Datos
df.dtypes

##Mostrar los primeros 10 reglones
df.head(10)


##Descripción de Variables Numéricas
df.describe()

##Descripción de variables Cualitativas
df.describe(include='O')

#Limpieza de Datos

##Checar valores no nulos
df.notnull().sum()

##Checar valores nulos
df.isnull().sum()

##Checar blancos
df.isna().sum()

##Eliminar Registros con valores nulos o NaN d
df.dropna(axis=0,how='any',inplace=True)

##Verificar que no existan valores nulos
df.isnull().sum()

#Columna que se va a predecir##
#df.groupby('Survived').size()

#Visualizacion 
Footer
© 2022 GitHub, Inc.
