This is my portfolio for the Machine Learning Module of my Advanced Artificial Inteligence for Data Science course!

*TASK DESCRIPTION:

In this deliverable I implemented a Machine Learning (ML) Algorithm without the use of a Machine Learning and/or estadistical framework/library to determine the alcohol quantity 
in a wine sample. For the development of the testing part, I generated 3 models based on linear regression, each one with a different degree equation. Being 1-degree, 2-dregree 
and 4-degree algorithms respectively. In the preparation part, I made sure to separate the dataset in two groups as training and validation in order to test the precision of the 
algorithm with real results. Then, I tested the implementation of the model with the validation portion of my dataset and printed out some predictions as a sample. 

*DATASET USED: 

  *Name:* Winequality.csv
  
  Source: https://archive.ics.uci.edu/ml/datasets/wine+quality
  
  Original Source: Paulo Cortez, University of Minho, Guimarães, Portugal, http://www3.dsi.uminho.pt/pcortez
    A. Cerdeira, F. Almeida, T. Matos and J. Reis, Viticulture Commission of the Vinho Verde Region(CVRVV), Porto, Portugal
    @2009
   
  Length: 4898

*VARIABLES:

  1-degree Model: 
      Feature Variable (x):
          * x1 = Acidity level

       Predictor Variable (y):*
          * Alcohol level

  2-degree Model:
        Feature Variable (x):
          * x1 = Acidity level
          * x2 = pH

       Predictor Variable (y):*
          * Alcohol level
  4-degree Model:
        Feature Variable (x):
          * x1 = Acidity level
          * x2 = pH
          * x3 = sulphates
          * x4 = chlorides

       Predictor Variable (y):*
          * Alcohol level

*PREDICTIONS:

Model 1:

| volatile_acidity | pred_alcohol | real_alcohol | dif |
| ------------- | ------------- | ------------- |------------- |
0.35|12.682355|14.2|1.517645|
0.31|12.244429|13.2|0.955571|
0.25|11.587539|11.2|0.387539|
0.36|12.791837|10.2|2.591837|
0.31|12.244429|12.8|0.555571|
0.31|12.244429|10.1|2.144429|
0.31|12.244429|10.1|2.144429|
0.22|11.259094|10.3|0.959094|
0.14|10.383241|9.9|0.483241|
0.22|11.259094|11.5|0.240906|
0.32|12.35391|9|3.35391|
0.32|12.35391|9|3.35391|
0.28|11.915984|12.1|0.184016|
0.3|12.134947|13.3|1.165053|

Model 2:

| volatile_acidity | pH | pred_alcohol | real_alcohol | dif |
| ------------- | ------------- | ------------- |------------- | ------------- |
0.35|3.12|10.358484|14.2|3.841516|
0.31|3|9.998539|13.2|3.201461|
0.25|3.04|10.118522|11.2|1.081478|
0.36|3.14|10.418475|10.2|0.218475|
0.31|3.09|10.268499|12.8|2.531501|
0.31|2.95|9.848561|10.1|0.251439|
0.31|2.95|9.848561|10.1|0.251439|
0.22|2.83|9.488616|10.3|0.811384|
0.14|3.19|10.568459|9.9|0.668459|
0.22|3.15|10.448474|11.5|1.051526|
0.32|3.18|10.538458|9|1.538458|
0.32|3.18|10.538458|9|1.538458|
0.28|3.03|10.088526|12.1|2.011474|
0.3|3.14|10.418477|13.3|2.881523|

Model 4:

| volatile_acidity | pH | sulphates | chlorides | pred_alcohol | real_alcohol | dif |
| ------------- | ------------- | ------------- |------------- | ------------- | ------------- | ------------- |
0.35|3.12|0.4|0.037|8.588462|14.2|5.611538|
0.31|3|0.4|0.035|8.294482|13.2|4.905518|
0.25|3.04|0.53|0.05|9.771173|11.2|1.428827|
0.36|3.14|0.57|0.057|10.599866|10.2|0.399866|
0.31|3.09|0.36|0.038|8.422215|12.8|4.377785|
0.31|2.95|0.39|0.045|8.852826|10.1|1.247174|
0.31|2.95|0.39|0.045|8.852826|10.1|1.247174|
0.22|2.83|0.31|0.031|7.408369|10.3|2.891631|
0.14|3.19|0.33|0.056|9.404861|9.9|0.495139|
0.22|3.15|0.31|0.046|8.713697|11.5|2.786303|
0.32|3.18|0.56|0.063|10.951008|9|1.951008|
0.32|3.18|0.56|0.063|10.951008|9|1.951008|
0.28|3.03|0.41|0.03|8.010413|12.1|4.089587|
0.3|3.14|0.41|0.036|8.534017|13.3|4.765983|

			
*TRAINING-VALIDATE DATA SEPARATION:

  *For the model implementation, it was necessary to separate the dataset into two subsets:
      * Training set
      * Validation set
   For that reason a ration of 80:20 was considered to divide the dataset. Meaning that 3918 data were asigned to the training set 
   and 980 data were assigned to the validation set.

*TRAINING QUALITY:

The best model implemented to predict the alcohol level in the wine sample was the 2-degree linear regression model since it had the best level of accuracy
from the other models implemented. Meaning that the predictions made by this model in particular were closer to the real data in comparison to the one´s made
with other models. The performance metric used to measure this accuracy level was the Mean Square Error (MSE). The feature variables used in this model in particular 
were:
* Acidity level
* pH level

En lo que respecta al MSE, esta métrica muestra el promedio de los cuadrados de los errores por lo que, aunque esta métrica permite medir el nivel de cambio lineal 
que ocurre en los datos, el hecho de elevar las diferencias al cuadrado tiende a magnificar/inflar los errores y por ende, datos muy alejados logran impactar 
mayormente al resultado general. Por tal motivo, se utiliza el RMSE el cuál quita este efecto de inflación al obtener la raíz cuadrada del MSE. Tal fue el caso del 
modelo de Orden 2 ya que si bien los MSE's de entrenamiento y validación estuvieron por debajo de 0.9, al obtener la métrica de los RMSE's estos valores fueron aún 
más bajos con una diferencia de 0.2 debido al efecto de la inflación del cuadrado de los errores, el cuál es un margen extremadamente bueno de error. Finalmente, se 
corroboraron estos valores con la métrica del MAE en la cuál las puntuaciones aumentan linealmente con el error promedio de los valores del error absoluto. Por ende, 
los valores del MAE coincidieron con los del RMSE lo que quiere decir que independientemente del signo de los errores, estos se encuentrán por debajo de 0.6, 
lo que indica un alto nivel de precisión en las predicciones. 

Dentro del contexto del problema, este error es relevante ya que lo que se está intentando predecir es el nivel de alcohol que tiene un vino dadas su nivel de 
acidez y su pH, y dado que el nivel de alcohol oscila entre un valor de 8 a 14.2 con una desviación estándar de 1.23, es conveniente que nuestro modelo tenga un 
error mucho menor al valor de desviación estándar para tener una precisión alta y que los valores de alcohol que se obtengan en futuras observaciones sean lo más 
cercanos a los reales.


*PERFORMANCE METRICS:
  * MSE (Mean Square Error):
  The MSE is an estimator that measures the average square error between the estimator and the prediction.
  Measures the difference between the prediction and the actual value of the distribution and is an accuracy measurement to
  determine how accurate the predictions were made based on how distant they were from the actual value. It takes into account the
  variance as well as the standard deviation of the dataset. It was for great value to make sure the dataset didn´t have many outlier predictions with
  a disproportional error, since the MSE puts on a great amount of weight into those errors.
  
  * RMSE (Root Mean Square Error):
    Heuristically that RMSE it represents a normalized distance between the vector of predicted values and the vector of observed values that is being rescaled
    according to the size of observations. Since the MSE can sometimes increase the effect of the biggest errors, the RMSE is al alternative error to ignore the
    inflation effect from elevating the error to the square. Plus it helps to consider the standard deviation σ of a typical observed value from our model’s prediction, 
    assuming that our observed data can be decomposed as:
    observed value = predicted value + predictably distributed random noise with mean zero.
    
  * MAE (Mean Absolute Error):
      
  
  *Loss Function:
  A loss function in Machine Learning is a measure of how accurately the  ML model is able to predict the expected outcome (the ground truth). 
  The loss function will take two items as input: 
  * the output value of our model 
  * the ground truth expected value. 
  The output of the loss function is called the loss which is a measure of how well our model did at predicting the outcome. A high value for the loss 
  means the model performed very poorly. A low value for the loss means our model performed very well.
  
  For the three models implemented, the loss function output was minimun as the initial values of theta were changed in order to minimize the error. 

*GOOGLE COLAB URL:
To access the original code in Google Colab: https://colab.research.google.com/drive/1mR6XGK0iJwqSNgVdJQ8ejnUG2dT3PMeP?usp=sharing 
